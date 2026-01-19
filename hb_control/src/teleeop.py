#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from hb_interfaces.msg import Poses2D, BotCmdArray, BotCmd
from geometry_msgs.msg import Twist
import numpy as np

class HolonomicPIDController(Node):
    def __init__(self):
        super().__init__('holonomic_pid_controller_2') 

        self.bot_id = 0
        self.current_pose = [None]*3
        
        self.bot_sub = self.create_subscription(Poses2D, "/bot_pose", self.pose_cb, 10)
        self.cmd_vel = self.create_subscription(Twist, "/cmd_vel", self.go_go, 10)
        self.vel_pub = self.create_publisher(BotCmdArray, '/bot_cmd', 10)

    def pose_cb(self, msg):
        for pose in msg.poses:
            if pose.id == 0:
                self.current_pose[0] = pose.x
                self.current_pose[1] = pose.y
                self.current_pose[2] = pose.w
                
    def go_go(self, msg):
        # Check if we have pose data
        # if self.current_pose[2] is None:
        #     self.get_logger().warn("No pose data yet, cannot transform velocity")
        #     return
        
        # Get commanded velocities (world frame)
        vx_world = msg.linear.x * 75
        vy_world = msg.linear.y * 75
        vtheta = msg.angular.z * 75
        
        # Get robot orientation
        # theta_robot = np.radians(self.current_pose[2])
        theta_robot = 30
        
        # Transform to robot body frame
        vx_robot = vx_world * np.cos(theta_robot) + vy_world * np.sin(theta_robot)
        vy_robot = -vx_world * np.sin(theta_robot) + vy_world * np.cos(theta_robot)
        
        # Wheel configuration for holonomic drive
        alpha_deg = np.array([30, 150, 270])
        alpha_rad = np.radians(alpha_deg)
        
        M = np.array([
            [np.cos(alpha_rad[0] + np.pi/2), np.cos(alpha_rad[1] + np.pi/2), np.cos(alpha_rad[2] + np.pi/2)],
            [np.sin(alpha_rad[0] + np.pi/2), np.sin(alpha_rad[1] + np.pi/2), np.sin(alpha_rad[2] + np.pi/2)],
            [1, 1, 1]
        ])
        
        M_inv = np.linalg.inv(M)
        vel = np.array([[vx_robot], [vy_robot], [vtheta]])
        s = np.dot(M_inv, vel)
        
        wheel_vel = [0, s[0][0], s[1][0], s[2][0], 0.0, 0.0]
        self.publish_wheel_velocities(wheel_vel)
        
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