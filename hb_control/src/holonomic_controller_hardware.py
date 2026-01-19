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
from std_msgs.msg import Bool
from geometry_msgs.msg import Pose2D

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
        super().__init__('holonomic_controller_hardware') 

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
        self.in_d1 = False
        self.in_dz = False
        
        self.crate_colour = ["red","green","blue"]
        
        self.pid_params = {
            'x': {'kp': 3.5, 'ki': 0.00, 'kd': 0.15, 'max_out': self.max_vel},
            'y': {'kp': 3.5, 'ki': 0.00, 'kd': 0.15 , 'max_out': self.max_vel},
            'theta': {'kp': 1.8, 'ki': 0.0, 'kd': 0.1, 'max_out': 35.0}
        }

        self.pid_x = PID(**self.pid_params['x'])
        self.pid_y = PID(**self.pid_params['y'])
        self.pid_theta = PID(**self.pid_params['theta'])
        
        self.bot_sub = self.create_subscription(Poses2D,"/bot_pose",self.pose_cb,10)
        self.crate_sub = self.create_subscription(Poses2D,"/crate_pose",self.crate_cb,10)
        self.ir_state_sub = self.create_subscription(Bool,"/ir_sensor_state",self.ir_callback,10)
        self.error_pub = self.create_publisher(Pose2D,"/pid_errors",10)

        self.vel_pub = self.create_publisher(BotCmdArray, '/bot_cmd', 10)
        
        self.crate_pose = [0.0,0.0,0.0]
        self.crate_id = None
        
        self.rc = 185.0
        self.mc = 10.2
        
        self.attach_success = False
        self.detach_success = False
        
        self.error_avg = []
        
        self.arm = 90.0
        self.solenoid_on = 0.0
        self.ir_state = False
        
        self.prev_dist = 0.0
        
        self.timer = self.create_timer(0.5, self.control_cb)  

    def ir_callback(self, msg):
        self.ir_state = msg.data

    def pose_cb(self, msg):
        for pose in msg.poses:
            if pose.id == 0:
                self.current_pose[0] = pose.x
                self.current_pose[1] = pose.y
                self.current_pose[2] = pose.w
                
    def crate_cb(self, msg):
        for pose in msg.poses:
            if pose.id == 12:
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
            target_theta = math.atan2(error_y, error_x) - (math.pi/2) - np.radians(3.0)
            # target_theta = math.atan2(error_y, error_x)
            print(f"target theta {target_theta}")
            dist,theta = self.go_to_target(target_x,target_y ,target_theta,error_thresthold = 250.0,angle_threshold = 0.10,use_a_t = True)
            
            if dist<250.0 and abs(theta)<0.10:
                print("crate reached")
                self.crate_reached = True
                wheel_vel = [0, 0.0, 0.0, 0.0, self.arm, self.solenoid_on]
                self.publish_wheel_velocities(wheel_vel)
                self.pid_reset() 
                self.pick_crate()
                return
                
        if self.attach_success and not self.in_d1:
            
            target_x = (self.d1_zone[0] + self.d1_zone[1])/2
            target_y = (self.d1_zone[2] + self.d1_zone[3])/2
            error_x = target_x - self.current_pose[0]
            error_y = target_y - self.current_pose[1]
            target_theta = math.atan2(error_y, error_x) - (math.pi/2) 
            dist,theta = self.go_to_target(target_x,target_y - 100.0,target_theta,error_thresthold = 150,angle_threshold = 0.4,use_a_t = True)
            
            if dist<150 and abs(theta)<0.4:
                self.in_d1 = True
                self.pid_reset() 
                self.place_crate()
                
        if self.detach_success and not self.in_dz:
            target_x = self.docking_zone[0]
            target_y = self.docking_zone[1]
            target_theta = self.docking_zone[2] 
            dist,theta = self.go_to_target(target_x,target_y,target_theta,error_thresthold = 75,angle_threshold = 0.25,use_a_t = True)
            
            if dist<75 and abs(theta)<0.25:
                self.in_dz = True
                print("reached")


    def go_to_target(self, target_x, target_y, target_theta, error_thresthold, angle_threshold, use_a_t):
    
       error_x = target_x - self.current_pose[0]
       error_y = target_y - self.current_pose[1]
       theta_robot = np.radians(self.current_pose[2])
    #    error_theta = theta_robot - target_theta 
       error_theta = target_theta-theta_robot 
       error_theta = math.atan2(math.sin(error_theta), math.cos(error_theta)) 
    #    theta = self.current_pose[2]
       
       dist = math.sqrt(error_x**2 + error_y**2)
       
       if dist - self.prev_dist > 100 and self.prev_dist != 0.0:
           self.prev_dist = dist
       else:
           self.prev_dist = dist
       
       if dist > error_thresthold:
           
           vx_global = self.pid_x.compute(error_x, self.dt)
           vy_global = self.pid_y.compute(error_y, self.dt)
           
           vx = vx_global * math.cos(theta_robot) + vy_global * math.sin(theta_robot) 
           vy = -vx_global * math.sin(theta_robot) + vy_global * math.cos(theta_robot)
           vtheta = 0.0
        #    pid_output = self.pid_theta.compute(error_theta, self.dt)
        #    if pid_output >= 0:
        #        vtheta = max(pid_output, 20.0)
        #    else:
        #        vtheta = max(abs(pid_output), 20.0)
           print(f"dist correction -- {dist}")
           
       elif abs(error_theta) > angle_threshold and use_a_t: 
           vx = vy = 0.0
           pid_output = self.pid_theta.compute(error_theta, self.dt)
           if pid_output >= 0:
               vtheta = max(pid_output, 40.0)
           else:
               vtheta = max(abs(pid_output), 40.0)
           print(f"yaw correction -- {error_theta}")
       else:
           vx = 0.0
           vy = 0.0
           vtheta = 0.0
       
       self.get_logger().info(f"dist = {dist} , error_theta ={error_theta:.2f}")
    #    self.get_logger().info(f"error_x = {error_x}, error_y = {error_y}, error_theta ={error_theta:.2f}")
    #    self.get_logger().info(f"targetx ={target_x}, targety ={target_y}")
    #    self.get_logger().info(f"target={target_theta:.2f}, robot={theta_robot:.2f}, error={error_theta:.2f}")
    #    self.get_logger().info(f"vx - {vx}   vy - {vy}   vtheta - {vtheta}")
   
       alpha_deg = np.array([30, 150, 270])
       alpha_rad = np.radians(alpha_deg)
   
       M = np.array([[np.cos(alpha_rad[0] + np.pi/2), np.cos(alpha_rad[1] + np.pi/2), np.cos(alpha_rad[2] + np.pi/2)],
                     [np.sin(alpha_rad[0] + np.pi/2), np.sin(alpha_rad[1] + np.pi/2), np.sin(alpha_rad[2] + np.pi/2)],
                     [1, 1, 1]])
       M_inv = np.linalg.inv(M)
       
       vel = np.array([[vx], [vy], [vtheta]])
       s = np.dot(M_inv, vel)
       
       wheel_vel = [0, s[0][0], s[1][0], s[2][0], self.arm, self.solenoid_on]
       self.publish_wheel_velocities(wheel_vel)
       
       self.error_pub.publish(Pose2D(x=error_x,y=error_y,theta=error_theta))
       return dist, error_theta
   
    def pick_crate(self):
        self.solenoid_on = 1.0
        self.arm = 75.0
        wheel_vel = [0, 0.0, 0.0, 0.0, self.arm, self.solenoid_on]
        self.publish_wheel_velocities(wheel_vel)
        time.sleep(3.0)
        self.arm = 90.0
        wheel_vel = [0, 0.0, 0.0, 0.0, self.arm, self.solenoid_on]
        self.publish_wheel_velocities(wheel_vel)
        self.arm = 110.0
        wheel_vel = [0, 0.0, 0.0, 0.0, self.arm, self.solenoid_on]
        self.publish_wheel_velocities(wheel_vel)
        
        self.attach_success = True
           
    
    def place_crate(self):
        self.solenoid_on = 0.0
        self.arm = 90.0
        wheel_vel = [0, 0.0, 0.0, 0.0, self.arm, self.solenoid_on]
        self.publish_wheel_velocities(wheel_vel)
        time.sleep(2.0)
        self.arm = 110.0
        wheel_vel = [0, 0.0, 0.0, 0.0, self.arm, self.solenoid_on]
        self.publish_wheel_velocities(wheel_vel)


        self.detach_success = True
        

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
        
def main(args=None):
    rclpy.init(args=args)
    controller = HolonomicPIDController()
    rclpy.spin(controller)
    controller.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
