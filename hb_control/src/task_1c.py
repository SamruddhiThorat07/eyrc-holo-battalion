#!/usr/bin/env python3

'''
This Python file runs a ROS 2 node of name holonomic_pid_controller which holds the position of a holonomic robot
and drives it through a series of predefined goals using PID controllers on [x, y, θ].

This node publishes and subscribes to the following topics:

        PUBLICATIONS                               SUBSCRIPTIONS
        /forward_velocity_controller/commands      /bot_pose

Instead of defining separate variables for each PID axis, lists/dictionaries are used.
For example: pid_params['x'], pid_params['y'], pid_params['theta'], etc.

Code modularity and clarity are maintained to make tuning and extension easier.
'''

# ---------------------- Import Required Libraries ----------------------------
import rclpy
from rclpy.node import Node
# import hb_interface messages
from hb_interfaces.msg import Pose2D
from hb_interfaces.msg import Poses2D
import numpy as np
import math
import time
from hb_interfaces.msg import BotCmdArray, BotCmd
from geometry_msgs.msg import Pose2D

# ---------------------- PID Controller Class --------------------------------
class PID:
    def __init__(self, kp, ki, kd, max_out=1.0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.max_out = max_out
        self.integral = 0.0
        self.prev_error = 0.0

    def compute(self, error, dt):
#-----------------------------PID Compute Steps--------------------------------------------------------------
        # 1. Accumulate the error over time for the Integral term
        # 2. Compute the change in error for the Derivative term
        # 3. Calculate the PID output:
        # 4. Store the current error for use in the next iteration
        # 5. Limit (clip) the output between [-max_out, +max_out] to avoid unsafe velocities
#------------------------------------------------------------------------------------------------------------
                
        self.integral +=error*dt
        
        self.der = (error-self.prev_error)/dt
        
        self.prev_error = error
        
        output = (self.kp*error)+(self.integral*self.ki)+(self.der*self.kd)
    
        output = np.clip(output,-self.max_out,self.max_out)
        
        return output
    
    def reset(self):
        self.integral = 0.0
        self.prev_error = 0.0

# ---------------------- Main Node Class -------------------------------------
class HolonomicPIDController(Node):
    def __init__(self):
        super().__init__('holonomic_pid_controller')  # initializing ros node

        self.bot_id = 9
        self.current_pose = [None]*3
        self.goals_reached = 0
        self.dt = None
        self.last_time = self.get_clock().now()
        self.error_tol = 5.0   
        self.max_vel = 75.0 
        
        # ---------------- Goal Definitions ----------------

        # List of waypoints [(x, y, yaw_deg)]
        self.goals = [
            (820, 920, 0),
            (820, 1520, 0),
            (1620, 1520, 0),
            (1620, 920, 0),
            (820, 920, 0),
        ]

        #----------------DO NOT CHNAGE----------------------

        # ---------------- PID Parameters ----------------
        self.pid_params = {
            'x': {'kp': 2.5, 'ki': 0.00, 'kd': 0.0, 'max_out': self.max_vel},
            'y': {'kp': 2.5, 'ki': 0.00, 'kd': 0.0, 'max_out': self.max_vel},
            'theta': {'kp': 1.8, 'ki': 0.0, 'kd': 0.0, 'max_out': 35.0}
        }

        # Initialize PIDs
        self.pid_x = PID(**self.pid_params['x'])
        self.pid_y = PID(**self.pid_params['y'])
        self.pid_theta = PID(**self.pid_params['theta'])

        # ---------------- ROS 2 Publishers & Subscribers ----------------
        
        # Write a subscriber for /bot_pose
        self.bot_sub = self.create_subscription(Poses2D,"/bot_pose",self.pose_cb,10)

        self.publisher = self.create_publisher(
            BotCmdArray, '/bot_cmd', 10
        )
        
        self.error_pub = self.create_publisher(Pose2D,"/pid_errors",10)
        
        # ---------------- Timer for Control Loop ----------------
        self.timer = self.create_timer(0.03, self.control_cb)  # ~30ms = 33 Hz

        self.get_logger().info(f'Holonomic PID Controller started. Goals: {self.goals}')


    # ---------------- Subscriber Callback ----------------
    def pose_cb(self, msg):
        for pose in msg.poses:
            if pose.id == 0:
                self.current_pose[0] = pose.x
                self.current_pose[1] = pose.y
                self.current_pose[2] = pose.w
        
    # ---------------- Control Loop ----------------
    def control_cb(self):
        
        if None in self.current_pose:
            return
        
        # Time delta
        now = self.get_clock().now()
        dt = (now - self.last_time).nanoseconds / 1e9
        if dt <= 0:
            return
        self.last_time = now

        # Current robot pose
        
        x_pos = self.current_pose[0]
        y_pos = self.current_pose[1]
        theta_pos = self.current_pose[2]

        # If all goals are reached → stop
        
        if self.goals_reached > 4:
            self.publish_wheel_velocities([0,0.0, 0.0, 0.0,90.0, 0.0])
            print("reached")
            return

        # Current target goal
        else:
            target_x = self.goals[self.goals_reached][0]  
            target_y = self.goals[self.goals_reached][1]  
            target_theta = self.goals[self.goals_reached][2]      

        # Errors
        error_x = target_x - self.current_pose[0]
        error_y = target_y - self.current_pose[1]
        error_theta = 0.0
        
        # PID outputs        
        # vx = self.pid_x.compute(error_x,dt)  
        # vy = self.pid_y.compute(error_y,dt)
        
        self.error_pub.publish(Pose2D(x=error_x,y=error_y,theta=error_theta))
        
        vx_global = self.pid_x.compute(error_x, dt)
        vy_global = self.pid_y.compute(error_y, dt)
        
        vx = vx_global * math.cos(target_theta) + vy_global * math.sin(target_theta)
        vy = -vx_global * math.sin(target_theta) + vy_global * math.cos(target_theta)
        vtheta = self.pid_theta.compute(error_theta,dt)

        # Convert to wheel velocities (custom equations)
        alpha_deg = np.array([30,150,270])
        alpha_rad = np.radians(alpha_deg)

        M = np.array([[np.cos(alpha_rad[0]+ np.pi/2), np.cos(alpha_rad[1] + np.pi/2), np.cos(alpha_rad[2]+np.pi/2)],[np.sin(alpha_rad[0]+np.pi/2), np.sin(alpha_rad[1]+np.pi/2), np.sin(alpha_rad[2]+np.pi/2)],[1,1,1]])
        M_inv = np.linalg.inv(M)
        vel = np.array([[vx], [vy], [vtheta]])
        s = np.dot(M_inv, vel)
        
        wheel_vel = [0, s[0][0], s[1][0], s[2][0], 90.0, 0.0]
        # wheel_vel = s.flatten()[[1, 0, 2]]
        wheel_vel = np.clip(wheel_vel, -self.max_vel, self.max_vel)
        self.publish_wheel_velocities(wheel_vel)
        # Goal check
        dist = math.sqrt(error_x**2+error_y**2)
        if dist<self.error_tol:
            print(f"{self.goals[self.goals_reached]} reached ")
            self.goals_reached +=1
            self.pid_x.reset()
            self.pid_y.reset()
            self.pid_theta.reset()


    # ---------------- Publisher ----------------
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
        self.publisher.publish(msg)


# ---------------------- Main Function -------------------------------------
def main(args=None):
    rclpy.init(args=args)
    controller = HolonomicPIDController()
    rclpy.spin(controller)
    controller.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
