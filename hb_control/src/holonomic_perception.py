#!/usr/bin/env python3
"""
This Python file runs a ROS 2 node named localization_node which publishes the position of crates and a holonomic drive robot.
This node subscribes to the following topics:
 SUBSCRIPTIONS
 /camera/image_raw
 /camera/camera_info
 /crates_pose
 /bot_pose
"""
import math
import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge
from sensor_msgs.msg import Image,CameraInfo
from hb_interfaces.msg import Pose2D, Poses2D

class PoseDetector(Node):
    def __init__(self):
        super().__init__('localization_node')
        
        # Initialize CvBridge for image conversion
        self.bridge = CvBridge()
        
        # ---------- PARAMETERS ----------
        self.crates_marker_length = 0.05  # Set marker size in meters
        self.bots_marker_length = 0.05    # Set bot marker size in meters
        self.aruco_dict_name = 'DICT_4X4_50'  # Choose ArUco dictionary
        
        # ---------- TOPICS ----------
        self.image_sub = self.create_subscription(Image, "/camera/image_raw", self.image_callback, 10)
        self.crate_poses_pub = self.create_publisher(Poses2D, '/crate_pose', 10)
        self.bot_poses_pub = self.create_publisher(Poses2D, '/bot_pose', 10)
        self.camera_info_sub = self.create_subscription(CameraInfo,"/camera/camera_info",self.camera_info_callback,10)
        
        # ---------- CAMERA PARAMETERS ----------
        self.camera_matrix = None  # load camera intrinsics (3x3 matrix)
        self.dist_coeffs = None    # load distortion coefficients (1x5 array)
        
        # ---------- IMAGE MATRICES ----------
        self.pixel_matrix = [None]*4  # derive pixel points matrix [[x1,y1], [x2,y2], ...]
        self.world_matrix = [[0,0],[2438.4,0.0],[0.0,2438.4],[2438.4,2438.4]]  # derive world points matrix [[x1,y1], [x2,y2], ...]
        self.H_matrix = None    # compute homography matrix using cv2.findHomography
        
        # ---------- ARUCO SETUP ----------
        # Initialize ArUco detector
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        self.aruco_params = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)
        
        self.get_logger().info('PoseDetector initialized')
        
    def camera_info_callback(self,msg):
        self.camera_matrix = np.array(msg.k,dtype=np.float32).reshape(3,3)
        self.dist_coeffs = np.array(msg.d,dtype=np.float32)

    def check_corner_ids(self, ids):
        """Check if all four corner markers (1, 3, 5, 7) are detected"""
        if ids is None:
            return False
        detected_ids = set(ids.flatten())
        required_corners = {1, 3, 5, 7}
        return required_corners.issubset(detected_ids)

    def pixel_to_world(self, pixel_x, pixel_y):
        """
        - Calculate the H_matrix using: use cv2.findHomography
        - Convert the pixel coordinates into real world coordinates using: cv2.perspectiveTransform(src_pts, self.H_matrix)
        """
        # Implement pixel to world coordinate conversion
        # Step 1: Ensure H_matrix is computed
        # Step 2: Create pixel point in correct format for cv2.perspectiveTransform
        # Step 3: Apply transformation and return world coordinates
        
        if self.H_matrix is None:
            # print("NO HM")
            return None , None
        
        src_point = np.array([[[pixel_x,pixel_y]]],dtype=np.float32)
        world_pts = cv2.perspectiveTransform(src_point,self.H_matrix)
        # print(world_pts)
        
        world_x = world_pts[0,0][0]
        world_y = world_pts[0,0][1]
        
        center_x = 1219.2  #2438.4/2
        center_y = 1219.2
        
        offset_x = world_x - center_x
        offset_y = world_y - center_y
        
        scale_factor = 1.0142
        corrected_offset_x = offset_x / scale_factor
        corrected_offset_y = offset_y / scale_factor
        
        world_x_corrected = center_x + corrected_offset_x
        world_y_corrected = center_y + corrected_offset_y
        
        return world_x_corrected, world_y_corrected
            
        # return world_pts[0,0][0], world_pts[0,0][1]

    def image_callback(self, msg):
        """
        Callback function for the image subscriber.
        Main Steps:
        1) Convert ROS Image -> cv image using CvBridge
        2) Undistort the image using camera intrinsics
        3) Detect all the markers in the world (cv2.aruco.drawDetectedMarkers)
        4) Derive the Pixel Matrix and the World Matrix using Corner Markers
        5) Compute the Homography Matrix (cv2.findHomography)
        5) Convert center pixel of crates marker and bot markers to world coordinates
        6) Using OpenCV calculate the yaw angle of each marker (cv2.aruco.estimatePoseSingleMarkers)
        7) Convert the yaw angle as per the new coordinate system
        8) Publish the bot pose and crate poses using the given custom message type
        """
        try:
            # Step 1: Convert ROS Image -> cv image using CvBridge
            # Use self.bridge.imgmsg_to_cv2() to convert ROS image to OpenCV format
            
            self.image = self.bridge.imgmsg_to_cv2(msg,desired_encoding="bgr8")
            
            # Step 2: Undistort the image using camera intrinsics
            # Use cv2.undistort() with camera_matrix and dist_coeffs
            # Convert to grayscale for marker detection
            
            undistorted_img = cv2.undistort(self.image,self.camera_matrix,self.dist_coeffs)
            gray_img = cv2.cvtColor(undistorted_img,cv2.COLOR_BGR2GRAY)
            gray_img = cv2.GaussianBlur(gray_img, (3, 3), 0)
            
            # Step 3: Detect all the markers in the world
            # Use self.detector.detectMarkers() to find ArUco markers
            # Use cv2.aruco.drawDetectedMarkers() to visualize detected markers
            
            corners,ids,rejected = self.detector.detectMarkers(gray_img)
            final_img = cv2.aruco.drawDetectedMarkers(undistorted_img,corners,ids)    
                 
            # cv2.imshow("aruco_markers", final_img)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            
            # Step 4: Derive the Pixel Matrix and the World Matrix using Corner Markers
            # Identify corner markers (IDs 1, 3, 5, 7)
            # Extract their pixel coordinates and map to known world coordinates
            self.pixel_matrix = [None]*4
            if ids is not None:
             for i,id in enumerate(ids.flatten()):
                if id == 1:
                    self.pixel_matrix[0] = corners[i][0][0]
                elif id == 3:
                    self.pixel_matrix[1] = corners[i][0][1]
                elif id == 5:
                    self.pixel_matrix[2] = corners[i][0][3]
                elif id == 7:
                    self.pixel_matrix[3] = corners[i][0][2]
                # elif id == 9:
                #     print(f"9 - - - -{corners[i][0]}")                    
                    
            # Step 5: Compute the Homography Matrix
            # Use cv2.findHomography() with pixel and world points
            
            if all(p is not None for p in self.pixel_matrix) and ids is not None and self.check_corner_ids(ids):
                pixel_points = np.array(self.pixel_matrix, dtype=np.float32).reshape(-1, 1, 2)
                world_points = np.array(self.world_matrix, dtype=np.float32).reshape(-1, 1, 2)
                self.H_matrix, mask = cv2.findHomography(pixel_points, world_points, cv2.RANSAC, 5.0)
            else:
                if self.H_matrix is None:
                    self.get_logger().warning("Insufficient points for homography")
                            
            # Step 6: Convert center pixel of markers to world coordinates
            # For each detected marker (excluding corner markers):
            #       - Calculate center pixel coordinate
            #       - Use pixel_to_world() to convert to world coordinates
            
            self.bot_poses = []
            self.crate_poses= []
            if ids is not None:
              for i,id in enumerate(ids.flatten()):
                if id not in [1,3,5,7]:
                    x, y  = np.mean(corners[i][0],axis=0)
                    world_x,world_y = self.pixel_to_world(x,y)
                    # print(f" x -- {world_x}  y -- {world_y}")
            
            # Step 7: Calculate yaw angle of each marker
            # Use cv2.aruco.estimatePoseSingleMarkers() or any other method to get rotation vectors
            # If you are going ahead with it, convert rotation vector to rotation matrix using cv2.Rodrigues()
            # Extract yaw angle from rotation matrix
            
            # rvec , tvec, _ = cv2.aruco.estimatePoseSingleMarkers(corners,self.bots_marker_length,self.camera_matrix,self.dist_coeffs)
            
                    marker_length = 0.05 #cause both are same
                    
                    obj_points = np.array([
                                 [-marker_length/2,  marker_length/2, 0],  # top-left
                                 [ marker_length/2,  marker_length/2, 0],  # top-right
                                 [ marker_length/2, -marker_length/2, 0],  # bottom-right
                                 [-marker_length/2, -marker_length/2, 0]   # bottom-left
                             ], dtype=np.float32)
              
                    success, rvec, tvec = cv2.solvePnP(obj_points, corners[i][0], self.camera_matrix, self.dist_coeffs)
                    
                    # rvec , tvec , _ = cv2.aruco.estimatePoseSingleMarkers()
                    
                    if success and world_x is not None:
                     rot_mat ,_ = cv2.Rodrigues(rvec)
                     yaw = np.arctan2(rot_mat[1, 0], rot_mat[0, 0])
                     h_rotation = np.arctan2(self.H_matrix[1, 0], self.H_matrix[0, 0])
                     yaw_world = yaw - h_rotation
                     yaw_deg = np.degrees(yaw_world)
                     if yaw_deg < 0:
                         yaw_deg += 360
                         
                    #  print(f"{id}    {yaw_deg}")
                     world_x_m = world_x/1000
                     world_y_m = world_y/1000
                
                     cv2.putText(undistorted_img, f"x:{world_x:.2f} y:{world_y:.2f} yaw:{yaw_deg:.2f}", 
                                (int(x), int(y-20)), cv2.FONT_HERSHEY_SIMPLEX, 
                                0.4, (0, 255, 0), 1)
                                 
            # Step 8: Separate and publish poses
            # Create separate dictionaries for bot_poses and crate_poses
            # Call publish_crate_poses() and publish_bot_poses()
                     
                     pose = {"id":id, "x":world_x, "y":world_y ,"w":yaw_deg}
                     if id in [0,2,4]:
                         self.bot_poses.append(pose)
                     else:
                         self.crate_poses.append(pose)
                
            
            self.publish_crate_poses(self.crate_poses)
            self.publish_bot_poses(self.bot_poses)
            
            # Display the image with detected markers
            resized_img = cv2.resize(undistorted_img, (1200, 900)) 
            cv2.imshow('Detected Markers', resized_img)
            # cv2.imshow('Detected Markers', undistorted_img)
            cv2.waitKey(1)
            
        except Exception as e:
            self.get_logger().error(f'Error processing image: {str(e)}')

    def publish_crate_poses(self, poses):
        """
        - Convert python pose dictionary -> message (Poses2D)
        - self.crate_poses_pub.publish(msg)
        """
        # Create Poses2D message
        # For each pose in poses list:
        #       - Create Pose2D message
        #       - Set id, x, y, w fields
        #       - Append to poses message
        # Publish the message
        msg = Poses2D()
        for pose in poses:
            pose_msg = Pose2D()
            pose_msg.id = int(pose["id"])
            pose_msg.x = float(pose["x"]) if pose["x"] is not None else 0.0
            pose_msg.y = float(pose["y"]) if pose["y"] is not None else 0.0
            pose_msg.w = float(pose["w"])
            msg.poses.append(pose_msg)
        self.crate_poses_pub.publish(msg)
        
    def publish_bot_poses(self, poses):
        """
        - Convert python pose dictionary -> message (Poses2D)
        - self.bot_poses_pub.publish(msg)
        """
        # Create Poses2D message
        # For each pose in poses list:
        #       - Create Pose2D message
        #       - Set id, x, y, w fields
        #       - Append to poses message
        # Publish the message
        
        msg = Poses2D()
        for pose in poses:
            pose_msg = Pose2D()
            pose_msg.id = int(pose["id"])
            pose_msg.x = float(pose["x"]) if pose["x"] is not None else 0.0
            pose_msg.y = float(pose["y"]) if pose["y"] is not None else 0.0
            pose_msg.w = float(pose["w"])
            msg.poses.append(pose_msg)
        self.bot_poses_pub.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    pose_detector = PoseDetector()
    try:
        rclpy.spin(pose_detector)
    except KeyboardInterrupt:
        pass
    finally:
        pose_detector.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()