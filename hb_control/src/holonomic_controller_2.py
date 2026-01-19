#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
from hb_interfaces.msg import Pose2D
from hb_interfaces.msg import Poses2D
from hb_interfaces.msg import BotCmdArray
from hb_interfaces.msg import BotCmd
from linkattacher_msgs.srv import AttachLink 
from linkattacher_msgs.srv import DetachLink
import numpy as np
from scipy.optimize import linear_sum_assignment
import math
import time
import json

class PID:
    def __init__(self, kp, ki, kd, max_out=1.0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.max_out = max_out
        self.integral = 0.0
        self.prev_error = 0.0

    def compute(self, error, dt):
          
        self.integral +=error*dt
        
        self.der = (error-self.prev_error)/dt
        
        self.prev_error = error
        
        output = (self.kp*error)+(self.integral*self.ki)+(self.der*self.kd)
    
        output = np.clip(output,-self.max_out,self.max_out)
        
        return output
    
    def reset(self):
        self.integral = 0.0
        self.prev_error = 0.0

class HolonomicPIDController(Node):
    def __init__(self):
        super().__init__('holonomic_pid_controller_2') 

        self.bot_id = 0
        self.current_pose = [None]*3
        self.dt = None
        self.last_time = self.get_clock().now()
        self.error_tol = 5.0   
        self.max_vel = 75.0   
        
        self.d1_zone = [1020, 1410, 1075, 1355]
        self.docking_zone = [1218.0,205.0,0.0]
        
        self.reset = True
        self.crate_reached = False
        # self.crate_picked = False
        self.in_d1 = False
        # self.crate_detached = False
        self.in_dz = False
        
        self.crate_colour = ["red","green","blue"]
        
        self.pid_params = {
            'x': {'kp': 4.5, 'ki': 0.005, 'kd': 0.3, 'max_out': self.max_vel},
            'y': {'kp': 4.5, 'ki': 0.005, 'kd': 0.3, 'max_out': self.max_vel},
            'theta': {'kp': 1.8, 'ki': 0.02, 'kd': 0.0, 'max_out': 35.0}
        }

        self.pid_x = PID(**self.pid_params['x'])
        self.pid_y = PID(**self.pid_params['y'])
        self.pid_theta = PID(**self.pid_params['theta'])
        
        self.bot_sub = self.create_subscription(Poses2D,"/bot_pose",self.pose_cb,10)
        self.crate_sub = self.create_subscription(Poses2D,"/crate_pose",self.crate_cb,10)

        self.vel_pub = self.create_publisher(BotCmdArray, '/bot_cmd', 10)
        
        self.crate_pose = [0.0,0.0,0.0]
        self.crate_id = None
        
        self.rc = 185.0
        self.mc = 10.2
        
        self.attach_success = False
        self.detach_success = False
        
        self.error_avg = []
        
        
        self.attach_client = self.create_client(AttachLink,"/attach_link")
        while not self.attach_client.wait_for_service(timeout_sec=1.0):
            print("attach service not available")
        
        self.detach_client = self.create_client(DetachLink,"/detach_link")
        while not self.detach_client.wait_for_service(timeout_sec=1.0):
            print("detach service not available")
        
        self.attach_req = AttachLink.Request()
        self.detach_req = DetachLink.Request()
        
        self.timer = self.create_timer(0.5, self.control_cb)  

    def pose_cb(self, msg):
        for pose in msg.poses:
            if pose.id == 0:
                self.current_pose[0] = pose.x
                self.current_pose[1] = pose.y
                self.current_pose[2] = pose.w
                
    def crate_cb(self, msg):
        for pose in msg.poses:
                self.crate_id = pose.id
                self.crate_pose[0] = pose.x
                self.crate_pose[1] = pose.y
                self.crate_pose[2] = pose.w
                
    def pid_reset(self):
        self.pid_x.reset()
        self.pid_y.reset()
        self.pid_theta.reset()
        
    def control_cb(self):

        if None in self.current_pose:
            return
    
        now = self.get_clock().now()
        self.dt = (now - self.last_time).nanoseconds / 1e9
        if self.dt <= 0:
            return
        self.last_time = now
        
        if not self.crate_reached:   
            target_x = self.crate_pose[0]  
            target_y = self.crate_pose[1]   
            error_x = target_x - self.current_pose[0]
            error_y = target_y - self.current_pose[1]
            target_theta = math.atan2(error_y, error_x) - (math.pi/2) 
            # target_theta = self.crate_pose[2] 
            dist,theta = self.go_go(target_x,target_y,target_theta,error_thresthold = 145,angle_threshold = 0.1,use_a_t = True)
            
            if dist<140.0 and abs(theta)<0.1:
            # if abs(theta)<0.2:
                self.crate_reached = True
                self.pid_reset() 
                self.pick_crate()
                
        if self.attach_success and not self.in_d1:
            # wheel_vel = [0, 0.0, 0.0, 0.0, 45.0, 45.0]
            # self.publish_wheel_velocities(wheel_vel)
            
            target_x = (self.d1_zone[0] + self.d1_zone[1])/2
            target_y = (self.d1_zone[2] + self.d1_zone[3])/2
            # target_theta = self.crate_pose[2] 
            dist,theta = self.go_go(target_x,target_y,0.0,error_thresthold = 50,angle_threshold = 0.0,use_a_t = False)
            
            if dist<50:
                self.in_d1 = True
                self.pid_reset() 
                self.place_crate()
                
        if self.detach_success and not self.in_dz:
            target_x = self.docking_zone[0]
            target_y = self.docking_zone[1]
            target_theta = self.docking_zone[2] 
            dist,theta = self.go_go(target_x,target_y,target_theta,error_thresthold = 20,angle_threshold = 0.15,use_a_t = True)
            
            if dist<20 and abs(theta)<0.1:
                self.in_dz = True
                print("reached")


    def go_go(self, target_x, target_y, target_theta, error_thresthold, angle_threshold, use_a_t):
    
       error_x = target_x - self.current_pose[0]
       error_y = target_y - self.current_pose[1]
    #    target_theta = math.atan2(error_y, error_x) - (math.pi/2)
       theta_robot = np.radians(self.current_pose[2])
       error_theta = theta_robot - target_theta
       error_theta = math.atan2(math.sin(error_theta), math.cos(error_theta))
       
       dist = math.sqrt(error_x**2 + error_y**2)
       
       if dist > error_thresthold:
           
           vx_global = self.pid_x.compute(error_x, self.dt)
           vy_global = self.pid_y.compute(error_y, self.dt)
           
           vx = vx_global * math.cos(theta_robot) + vy_global * math.sin(theta_robot)
           vy = -vx_global * math.sin(theta_robot) + vy_global * math.cos(theta_robot)
           vtheta = 0.0
           
           self.get_logger().info(f"1---------------------------------------------------------") 
       elif abs(error_theta) > angle_threshold and use_a_t: 
           vx = vy = 0.0
           pid_output = self.pid_theta.compute(error_theta, self.dt)
           if pid_output >= 0:
               vtheta = max(pid_output, 7.0)
           else:
               vtheta = -max(abs(pid_output), 7.0)
           
           self.get_logger().info(f"2---------------------------------------------------------")
       else:
           vx = 0.0
           vy = 0.0
           vtheta = 0.0
       
       self.get_logger().info(f"dist -- {dist}")
    #    self.get_logger().info(f"error_x ={error_x}, error_y ={error_y}")
    #    self.get_logger().info(f"targetx ={target_x}, targety ={target_y}")
       self.get_logger().info(f"target={target_theta:.2f}, robot={theta_robot:.2f}, error={error_theta:.2f}")
    #    self.get_logger().info(f"vx - {vx}   vy - {vy}   vtheta - {vtheta}")
   
       alpha_deg = np.array([30, 150, 270])
       alpha_rad = np.radians(alpha_deg)
   
       M = np.array([[np.cos(alpha_rad[0] + np.pi/2), np.cos(alpha_rad[1] + np.pi/2), np.cos(alpha_rad[2] + np.pi/2)],
                     [np.sin(alpha_rad[0] + np.pi/2), np.sin(alpha_rad[1] + np.pi/2), np.sin(alpha_rad[2] + np.pi/2)],
                     [1, 1, 1]])
       M_inv = np.linalg.inv(M)
       
       vel = np.array([[vx], [vy], [vtheta]])
       s = np.dot(M_inv, vel)
       
       wheel_vel = [0, s[0][0], s[1][0], s[2][0], 0.0, 0.0]
       self.publish_wheel_velocities(wheel_vel)
       
       return dist, error_theta
   
    def pick_crate(self):
        wheel_vel = [0, 0.0, 0.0, 0.0, 90.0, 90.0]
        self.publish_wheel_velocities(wheel_vel)
        time.sleep(4.0)
        if not self.attach_success:
           self.send_attach_req(self.crate_id)
           
    
    def place_crate(self):
        wheel_vel = [0, 0.0, 0.0, 0.0, 90.0, 90.0]
        self.publish_wheel_velocities(wheel_vel)
        time.sleep(4.0)
        if not self.detach_success:
           self.send_detach_req(self.crate_id)
        

    def publish_wheel_velocities(self, wheel_vel):
        msg = BotCmdArray()
        cmd = BotCmd()
        cmd.id = int(wheel_vel[0])
        cmd.m1 = wheel_vel[1]
        cmd.m2 = wheel_vel[2]
        cmd.m3 = wheel_vel[3]
        cmd.base = wheel_vel[4]
        cmd.elbow = wheel_vel[5]
        msg.cmds.append(cmd)
        self.vel_pub.publish(msg)
        
    def send_attach_req(self, id):
        
        data_dict = {
            "model1_name": "hb_crystal",
            "link1_name": "arm_link_2",
            "model2_name": f"crate_{self.crate_colour[id%3]}_{id}",
            "link2_name": f"box_link_{id}"
        }
        
        self.attach_req.data = json.dumps(data_dict)
        future = self.attach_client.call_async(self.attach_req)
        future.add_done_callback(self.handle_attach)
    
    def send_detach_req(self, id):
        
        data_dict = {
            "model1_name": "hb_crystal",
            "link1_name": "arm_link_2",
            "model2_name": f"crate_{self.crate_colour[id%3]}_{id}",
            "link2_name": f"box_link_{id}"
        }
        
        self.detach_req.data = json.dumps(data_dict)
        future = self.detach_client.call_async(self.detach_req)
        future.add_done_callback(self.handle_detach)    
        
    def handle_attach(self,future):
        try:
            response = future.result()
            print(f"{response.message}")
            self.attach_success = response.success
            if self.attach_success:
                print("attached ----------------------a---------------------------a-----------------------a------------------------a-----------------------a")
                self.pid_reset() 
                wheel_vel = [0, 0.0, 0.0, 0.0, 60.0, 60.0]
                self.publish_wheel_velocities(wheel_vel)
            else:
                print("trying to attach again XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
                self.pick_crate()
        except Exception as e:
            print(f"Failed to attach: {e}")
            
    def handle_detach(self,future):
        try:
            response = future.result()
            print(f"{response.message}")
            self.detach_success = response.success
            if self.detach_success:
                print("detached -------------------d-------------------d------------------d-----------------------d------------------------d------------------------------d")
                self.pid_reset() 
            else:
                print("trying to detach again XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
                self.place_crate()
        except Exception as e:
            print(f"Failed to detach: {e}")
        

def main(args=None):
    rclpy.init(args=args)
    controller = HolonomicPIDController()
    rclpy.spin(controller)
    controller.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
