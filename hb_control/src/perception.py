#!/usr/bin/env python3
"""
This Python file runs a ROS 2 node named localization_node which publishes the position of crates and a holonomic drive robot.
This node reads directly from the camera instead of subscribing to a topic.
"""
import math
import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge
from sensor_msgs.msg import Image,CameraInfo
from hb_interfaces.msg import Pose2D, Poses2D
import subprocess

class PoseDetector(Node):
    def __init__(self):
        super().__init__('localization_node')
        
        # Initialize CvBridge for image conversion
        self.bridge = CvBridge()
        
        # ---------- CAMERA SETUP ----------
        self.CAMERA_ID = 'video2'  # Change this to match your camera
        
        # Configure camera using v4l2-ctl
        try:
            subprocess.run(
                ["v4l2-ctl", "-d", f"/dev/{self.CAMERA_ID}", 
                 "--set-fmt-video=width=1920,height=1080,pixelformat=MJPG"],
                check=False
            )
            subprocess.run(
                ["v4l2-ctl", "-d", f"/dev/{self.CAMERA_ID}", "-c", "auto_exposure=1"],
                check=False
            )
        except Exception as e:
            self.get_logger().warning(f"v4l2-ctl configuration failed: {e}")
        
        # Open camera
        self.cap = cv2.VideoCapture(int(self.CAMERA_ID[-1]))
        
        # Set MJPG format
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        self.cap.set(cv2.CAP_PROP_FOURCC, fourcc)
        
        if not self.cap.isOpened():
            raise RuntimeError("Failed to open camera")
        
        # Set resolution
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        
        self.get_logger().info(f"Camera opened: {self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)}x{self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}")
        
        # ---------- PARAMETERS ----------
        self.crates_marker_length = 0.05
        self.bots_marker_length = 0.05
        self.aruco_dict_name = 'DICT_4X4_50'
        
        # ---------- TOPICS ----------
        # No image subscription needed anymore!
        self.crate_poses_pub = self.create_publisher(Poses2D, '/crate_pose', 10)
        self.bot_poses_pub = self.create_publisher(Poses2D, '/bot_pose', 10)
        
        # Create a timer to read from camera at ~30 FPS
        self.timer = self.create_timer(0.033, self.camera_loop)  # ~30 Hz
        
        # ---------- CAMERA PARAMETERS ----------
        self.camera_matrix = np.array(
            [[1274.830015, 0.000000, 935.895473],
             [0.000000, 1273.945201, 579.772489],
             [0.000000, 0.000000, 1.0]],
            dtype=np.float32
        )
        
        self.dist_coeffs = np.array(
            [0.007847, -0.071989, 0.000015, -0.003323, 0.0],
            dtype=np.float32
        )

        # ---------- PERSISTENT CORNER STORAGE ----------
        self.corner_positions = {
            1: None,
            3: None,
            5: None,
            7: None
        }
        
        self.world_coords = {
            1: [0, 0],
            3: [2438.4, 0.0],
            5: [0.0, 2438.4],
            7: [2438.4, 2438.4]
        }
        
        self.H_matrix = None
        self.homography_computed = False
        
        # ---------- ARUCO SETUP ----------
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        self.aruco_params = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)
        
        self.get_logger().info('PoseDetector initialized with direct camera input')

    def camera_loop(self):
        """Read from camera and process image"""
        ret, frame = self.cap.read()
        
        if not ret:
            self.get_logger().error("Failed to capture frame")
            return
        
        # Process the frame (same as image_callback)
        self.process_frame(frame)

    def update_corner_positions(self, ids, corners):
        """Update the stored positions of corner markers whenever they're detected"""
        if ids is None:
            return
        
        for i, id in enumerate(ids.flatten()):
            if id in self.corner_positions:
                if id == 1:
                    self.corner_positions[1] = corners[i][0][0]
                elif id == 3:
                    self.corner_positions[3] = corners[i][0][1]
                elif id == 5: 
                    self.corner_positions[5] = corners[i][0][3]
                elif id == 7:
                    self.corner_positions[7] = corners[i][0][2]
                
                self.get_logger().debug(f"Updated corner marker {id} position")

    def all_corners_detected(self):
        """Check if all four corner markers have been detected at least once"""
        return all(pos is not None for pos in self.corner_positions.values())

    def compute_homography(self):
        """Compute homography matrix using stored corner positions"""
        if not self.all_corners_detected():
            return False
        
        pixel_points = []
        world_points = []
        
        for marker_id in sorted(self.corner_positions.keys()):
            pixel_points.append(self.corner_positions[marker_id])
            world_points.append(self.world_coords[marker_id])
        
        pixel_points = np.array(pixel_points, dtype=np.float32).reshape(-1, 1, 2)
        world_points = np.array(world_points, dtype=np.float32).reshape(-1, 1, 2)
        
        self.H_matrix, mask = cv2.findHomography(pixel_points, world_points, cv2.RANSAC, 5.0)
        
        if not self.homography_computed:
            self.get_logger().info("Homography matrix computed successfully!")
            self.homography_computed = True
        
        return True

    def pixel_to_world(self, pixel_x, pixel_y):
        """Convert pixel coordinates to world coordinates using homography matrix"""
        if self.H_matrix is None:
            return None, None
        
        src_point = np.array([[[pixel_x, pixel_y]]], dtype=np.float32)
        world_pts = cv2.perspectiveTransform(src_point, self.H_matrix)
        
        world_x = world_pts[0, 0][0]
        world_y = world_pts[0, 0][1]
        
        center_x = 1219.2
        center_y = 1219.2
        
        offset_x = world_x - center_x
        offset_y = world_y - center_y
        
        scale_factor = 1.03
        corrected_offset_x = offset_x / scale_factor
        corrected_offset_y = offset_y / scale_factor
        
        world_x_corrected = center_x + corrected_offset_x
        world_y_corrected = center_y + corrected_offset_y
        
        return world_x_corrected, world_y_corrected

    def process_frame(self, frame):
        """Main image processing (renamed from image_callback)"""
        try:
            # Undistort the image
            undistorted_img = cv2.undistort(frame, self.camera_matrix, self.dist_coeffs)
            gray_img = cv2.cvtColor(undistorted_img, cv2.COLOR_BGR2GRAY)
            
            # Detect all markers
            corners, ids, rejected = self.detector.detectMarkers(gray_img)
            final_img = cv2.aruco.drawDetectedMarkers(undistorted_img, corners, ids)
            
            # Update corner positions
            self.update_corner_positions(ids, corners)
            
            # Compute homography
            self.compute_homography()
            
            # Display status
            status_text = f"Corners detected: {sum(1 for p in self.corner_positions.values() if p is not None)}/4"
            cv2.putText(undistorted_img, status_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            if self.homography_computed:
                cv2.putText(undistorted_img, "Homography: OK", (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                cv2.putText(undistorted_img, "Homography: Waiting...", (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Process non-corner markers
            self.bot_poses = []
            self.crate_poses = []
            
            if ids is not None and self.H_matrix is not None:
                for i, id in enumerate(ids.flatten()):
                    if id not in [1, 3, 5, 7]:
                        x, y = np.mean(corners[i][0], axis=0)
                        world_x, world_y = self.pixel_to_world(x, y)
                        
                        if world_x is None:
                            continue
                        
                        marker_length = 0.05
                        obj_points = np.array([
                            [-marker_length/2,  marker_length/2, 0],
                            [ marker_length/2,  marker_length/2, 0],
                            [ marker_length/2, -marker_length/2, 0],
                            [-marker_length/2, -marker_length/2, 0]
                        ], dtype=np.float32)
                        
                        success, rvec, tvec = cv2.solvePnP(obj_points, corners[i][0], 
                                                          self.camera_matrix, self.dist_coeffs)
                        
                        if success:
                            rot_mat, _ = cv2.Rodrigues(rvec)
                            yaw = np.arctan2(rot_mat[1, 0], rot_mat[0, 0])
                            h_rotation = np.arctan2(self.H_matrix[1, 0], self.H_matrix[0, 0])
                            yaw_world = yaw - h_rotation
                            yaw_deg = np.degrees(yaw_world)
                            if yaw_deg < 0:
                                yaw_deg += 360
                            
                            cv2.putText(undistorted_img, f"x:{world_x:.2f} y:{world_y:.2f} yaw:{yaw_deg:.2f}", 
                                       (int(x), int(y-20)), cv2.FONT_HERSHEY_SIMPLEX, 
                                       0.4, (0, 255, 0), 1)
                            
                            pose = {"id": id, "x": world_x, "y": world_y, "w": yaw_deg}
                            if id in [0, 2, 4]:
                                self.bot_poses.append(pose)
                            else:
                                self.crate_poses.append(pose)
            
            # Publish poses
            self.publish_crate_poses(self.crate_poses)
            self.publish_bot_poses(self.bot_poses)
            
            # Display the image
            resized_img = cv2.resize(undistorted_img, (1500, 1000))
            cv2.imshow('Detected Markers', resized_img)
            cv2.waitKey(1)
            
        except Exception as e:
            self.get_logger().error(f'Error processing frame: {str(e)}')

    def publish_crate_poses(self, poses):
        """Publish crate poses"""
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
        """Publish bot poses"""
        msg = Poses2D()
        for pose in poses:
            pose_msg = Pose2D()
            pose_msg.id = int(pose["id"])
            pose_msg.x = float(pose["x"]) if pose["x"] is not None else 0.0
            pose_msg.y = float(pose["y"]) if pose["y"] is not None else 0.0
            pose_msg.w = float(pose["w"])
            msg.poses.append(pose_msg)
        self.bot_poses_pub.publish(msg)
    
    def __del__(self):
        """Cleanup when node is destroyed"""
        if hasattr(self, 'cap'):
            self.cap.release()
        cv2.destroyAllWindows()

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