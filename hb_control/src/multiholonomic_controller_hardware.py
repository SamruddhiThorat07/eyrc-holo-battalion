#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from hb_interfaces.msg import Pose2D,Poses2D,BotCmdArray,BotCmd
import numpy as np
from scipy.optimize import linear_sum_assignment
import math
import time
from std_msgs.msg import Bool
from hb_interfaces.msg import BotIrState

# STATES

# IDLE
# GO_TO_CRATE
# PICK_UP
# GO_TO_DROP_ZONE
# DROP
# check if any crate is left(not a state)  if yes then repeat
# GO_TO_DOCKING_ZONE

# BOTS              PRIORITY
# 0 - Crystal          3
# 2 - Frostbite        2    
# 4 - Glacio           1

# DROP_ZONE
# Red → D1
# Blue → D3
# Green → D2


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

class MultiHolonomicController(Node):
    def __init__(self):
        super().__init__('multiholonomic_controller_hardware') 
        
        self.bot_id = [0,1,2]
        self.dt = None
        self.last_time = self.get_clock().now()
        self.error_tol = 5.0   
        self.max_vel = 90.0   
                
        self.pid_params = {
            'x': {'kp': 3.5, 'ki': 0.0, 'kd': 0.15, 'max_out': self.max_vel},
            'y': {'kp': 3.5, 'ki': 0.0, 'kd': 0.15, 'max_out': self.max_vel},
            'theta': {'kp': 1.8, 'ki': 0.0, 'kd': 0.1, 'max_out': 35.0} 
        }
        
        
        self.pid_controllers = {
            0: {'x': PID(**self.pid_params['x']), 'y': PID(**self.pid_params['y']), 'theta': PID(**self.pid_params['theta'])},
            1: {'x': PID(**self.pid_params['x']), 'y': PID(**self.pid_params['y']), 'theta': PID(**self.pid_params['theta'])},
            2: {'x': PID(**self.pid_params['x']), 'y': PID(**self.pid_params['y']), 'theta': PID(**self.pid_params['theta'])}
        }
                
        self.d_zone = [
            [1020, 1410, 1075, 1355] ,  #red
            [675, 965, 1920, 2115] ,  #green
            [1470, 1762, 1920, 2115]   #blue
        ]
        self.d_zone_occupied = [False,False,False]
        self.d_zone_assigned = [False,False,False]
        self.d_zone_bot = [None, None, None]
        
        self.docking_zone = [
            [1218.0,205.0,0.0],
            [1568.0,202.0,0.0] ,
            [864.0,204.0,0.0]
        ]
        
        
        self.STATE = ["IDLE","IDLE","IDLE"]
        self.bot_pose = [[None]*3, [None]*3, [None]*3]
        self.bot_assigned_crate = [None, None, None] 
        self.bot_name = {
            0: "hb_crystal",
            1: "hb_frostbite", 
            2: "hb_glacio"
        }
        self.bot_priority = {0: 3, 1: 2, 2: 1}
        self.arm_pos = [90.0,90.0,90.0]
        self.solenoid = [0.0,0.0,0.0]
        
        self.current_crates = set()
        self.completed_crates = set()
        self.crates = {}
        
        self.crate_reached = [False, False, False]
        self.attach_success = [False, False, False]
        self.detach_success = [False, False, False]
        self.in_drop_zone = [False, False, False]
        self.in_docking_zone = [False, False, False]
        
        self.crate_colour = ["red","green","blue"]
        
        self.pick_start_time = [None, None, None]
        self.drop_start_time = [None, None, None]
        self.drop_wait_count = [0, 0, 0]
        self.wait_duration = 2.5
        
        self.ir_state = [False,False,False]
        
        alpha_deg = np.array([30, 150, 270])
        alpha_rad = np.radians(alpha_deg)
    
        M = np.array([[np.cos(alpha_rad[0] + np.pi/2), np.cos(alpha_rad[1] + np.pi/2), np.cos(alpha_rad[2] + np.pi/2)],
                      [np.sin(alpha_rad[0] + np.pi/2), np.sin(alpha_rad[1] + np.pi/2), np.sin(alpha_rad[2] + np.pi/2)],
                      [1, 1, 1]])
        self.M_inv = np.linalg.inv(M)
        
        self.bot_sub = self.create_subscription(Poses2D,"/bot_pose",self.pose_cb,10)
        self.crate_sub = self.create_subscription(Poses2D,"/crate_pose",self.crate_cb,10)
        self.ir_sub = self.create_subscription(BotIrState,"/ir_sensor_state",self.ir_callback,10)
        self.vel_pub = self.create_publisher(BotCmdArray, '/bot_cmd', 10)
                        
        self.timer = self.create_timer(0.33, self.control_cb)  
        
        print("MultiHolonomicController Initialized")
        
    #------------------------------------------------------------------------------------------------------
        
    def check_crate_in_zone(self, crate_id):
        
        if crate_id not in self.crates:
            return False
        
        x, y, w = self.crates[crate_id]
        zone_id = crate_id % 3  
        zone = self.d_zone[zone_id]
        margin = 15  
        if (zone[0] + margin < x < zone[1] - margin) and (zone[2] + margin < y < zone[3] - margin):
            self.d_zone_occupied[zone_id] = True
            return True
        else:
            return False  
                
    def assign_crates(self):
        
        currently_assigned = [cid for cid in self.bot_assigned_crate if cid is not None]
          
        unassigned_crates = [
           crate_id for crate_id in self.crates.keys() 
           if crate_id not in self.completed_crates 
           and crate_id not in currently_assigned       
        ]
        
        if not unassigned_crates:
           return  
      
        idle_bots = [bot_id for bot_id in [0, 1, 2] if self.STATE[bot_id] == "IDLE"]
        
        if not idle_bots:
           return  
       
        cost_matrix = []
        for bot_id in idle_bots:
            bot_costs = []
            for crate_id in unassigned_crates:
                bot_x, bot_y = self.bot_pose[bot_id][0], self.bot_pose[bot_id][1]
                crate_x, crate_y = self.crates[crate_id][0], self.crates[crate_id][1]
                
                distance = math.sqrt((crate_x - bot_x)**2 + (crate_y - bot_y)**2)
                bot_costs.append(distance)
            cost_matrix.append(bot_costs)
            
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
            
        for row, col in zip(row_ind, col_ind):
            bot_id = idle_bots[row]
            crate_id = unassigned_crates[col]
            self.bot_assigned_crate[bot_id] = crate_id
            drop_zone_id = crate_id % 3
            self.d_zone_assigned[drop_zone_id] = True
            self.d_zone_bot[drop_zone_id] = bot_id
            self.STATE[bot_id] = "GO_TO_CRATE"
            
            print(f"Assigned Crate {crate_id} to Bot {bot_id}")

    def pose_cb(self, msg):
        for pose in msg.poses:
                bot_id = int(pose.id/2)
                self.bot_pose[bot_id] = [pose.x,pose.y,pose.w]
                # print(f"{bot_id}------------------------{self.bot_pose[bot_id]}")
                
    def crate_cb(self, msg):
        for pose in msg.poses:
            if pose.id in [12,14,30]:
                self.crates[pose.id] = [pose.x,pose.y,pose.w]
        # print(self.current_crates)
        
    def ir_callback(self,msg):
        bot_id = int(msg.id/2)
        self.ir_state[bot_id] = msg.state
                
    def pid_reset(self, bot_id):
        self.pid_controllers[bot_id]['x'].reset()
        self.pid_controllers[bot_id]['y'].reset()
        self.pid_controllers[bot_id]['theta'].reset()
        
    def control_cb(self):

        # print("Control CB Triggered")
        if any(pose[0] is None for pose in self.bot_pose):
            # print("Waiting for all bot poses...")
            return
    
        now = self.get_clock().now()
        self.dt = (now - self.last_time).nanoseconds / 1e9
        if self.dt <= 0:
            return
        self.last_time = now
        
        self.assign_crates()
        
        for bot_id in [0,1,2]:
            
            state = self.STATE[bot_id]
              
            if state == "IDLE":
                print(f"Bot {bot_id} in IDLE")
                wheel_vel = [bot_id, 0.0, 0.0, 0.0, self.arm_pos[bot_id], self.solenoid[bot_id]]
                self.publish_wheel_velocities(wheel_vel)
              
            elif state == "GO_TO_CRATE":
                print(f"Bot {bot_id} GO_TO_CRATE {self.bot_assigned_crate[bot_id]}")
                crate_id = self.bot_assigned_crate[bot_id]
                if crate_id is None or crate_id not in self.crates:
                    self.STATE[bot_id] = "IDLE"
                    continue
                
                target_x, target_y, crate_theta = self.crates[crate_id]
  
                error_x = target_x - self.bot_pose[bot_id][0]
                error_y = target_y - self.bot_pose[bot_id][1]
                target_theta = math.atan2(error_y, error_x) - (math.pi/2) - np.radians(3.0)
                dist,theta = self.go_to_target(bot_id,target_x,target_y,target_theta,error_threshold = 235,angle_threshold = 0.1,use_target_theta = True)
                
                if dist<225.0 and abs(theta)<0.1:
                    # if self.ir_state[bot_id]:
                       self.STATE[bot_id] = "PICK_UP"
                       self.pid_reset(bot_id)
                    # else:
                    #     wheel_vel = [bot_id, -50.0, 50.0, 0.0, self.arm_pos[bot_id], self.solenoid[bot_id]]
                    #     self.publish_wheel_velocities(wheel_vel)
                    #     # print(f"Bot {bot_id} WAITING for IR Sensor")
                    #     print(f"retreating {bot_id}")
                                              
            elif state == "PICK_UP":
                print(f"Bot {bot_id} PICK_UP")
                if not self.crate_reached[bot_id]:
                    self.pick_crate(bot_id)
                    self.crate_reached[bot_id] = True
                else:
                     if not self.attach_success[bot_id]:
                         self.pick_crate(bot_id)
                
                if self.attach_success[bot_id]:
                    self.STATE[bot_id] = "GO_TO_DROP_ZONE"
                    self.pid_reset(bot_id)
               
            elif state == "GO_TO_DROP_ZONE":
                crate_id = self.bot_assigned_crate[bot_id]
                drop_zone_id = crate_id % 3  # 0=red, 1=green, 2=blue
                
                zone = self.d_zone[drop_zone_id]
                target_x = (zone[0] + zone[1]) / 2
                target_y = (zone[2] + zone[3]) / 2
                error_x = target_x - self.bot_pose[bot_id][0]
                error_y = target_y - self.bot_pose[bot_id][1]
                target_theta = math.atan2(error_y, error_x) - (math.pi/2) 
                
                if self.d_zone_assigned[drop_zone_id] and self.d_zone_bot[drop_zone_id] != bot_id:
                    print(f"Bot {bot_id} WAITING - Zone {drop_zone_id} taken by {self.d_zone_bot[drop_zone_id]}")
                    wheel_vel = [bot_id, 0.0, 0.0, 0.0, self.arm_pos[bot_id], self.solenoid[bot_id]]
                    self.publish_wheel_velocities(wheel_vel)
                    continue
                else:
                    print(f"Bot {bot_id} GO_TO_DROP_ZONE {drop_zone_id}")
                
                dist, theta = self.go_to_target(bot_id, target_x, target_y, target_theta, 
                                             error_threshold=250, angle_threshold=0.5, use_target_theta=True)
                
                if dist < 250 and abs(theta)<0.6:
                    self.STATE[bot_id] = "DROP"
                    self.pid_reset(bot_id)
                   
            elif state == "DROP":
                print(f"Bot {bot_id} DROP")
                if not self.in_drop_zone[bot_id]:
                    self.place_crate(bot_id)
                    self.in_drop_zone[bot_id] = True
                else:
                    if not self.detach_success[bot_id]:
                       self.place_crate(bot_id)
                    
                if self.detach_success[bot_id]:
                    self.drop_wait_count[bot_id] += 1
                    if self.drop_wait_count[bot_id] >= 15:
                        crate_id = self.bot_assigned_crate[bot_id]
                        if self.check_crate_in_zone(crate_id):
                            print(f"Bot {bot_id}: Crate {crate_id} placed correctly")
                            self.completed_crates.add(crate_id)
                            print("crate raw pose:", crate_id, self.crates[crate_id])
                            print("zone bounds:", self.d_zone[crate_id%3])
                        else:
                            print(f"Bot {bot_id}: Crate {crate_id} NOT in zone")
                            print("crate raw pose:", crate_id, self.crates[crate_id])
                            print("zone bounds:", self.d_zone[crate_id%3])
                    
                        self.bot_assigned_crate[bot_id] = None
                        self.crate_reached[bot_id] = False
                        self.attach_success[bot_id] = False
                        self.detach_success[bot_id] = False
                        self.in_drop_zone[bot_id] = False
                        
                        currently_assigned = sum(1 for cid in self.bot_assigned_crate if cid is not None)
                        total_handled = len(self.completed_crates) + currently_assigned
                        all_crates_known = len(self.crates)
                        
                        if total_handled >= all_crates_known and all_crates_known > 0:
                            self.STATE[bot_id] = "GO_TO_DOCKING_ZONE"
                        else:
                            self.STATE[bot_id] = "IDLE"
                        
                        print(f"Handled: {total_handled}/{all_crates_known} → Dock: {total_handled >= all_crates_known}")
                        
                        drop_zone_id = crate_id % 3
                        self.d_zone_assigned[drop_zone_id] = False
                        self.d_zone_bot[drop_zone_id] = None
                        self.pid_reset(bot_id)
                    
            elif state == "GO_TO_DOCKING_ZONE":
                print(f"Bot {bot_id} GO_TO_DOCKING_ZONE")
                dock = self.docking_zone[bot_id]
                dist, theta_error = self.go_to_target(bot_id, dock[0], dock[1], dock[2],
                                                       error_threshold=50, angle_threshold=0.25,use_target_theta = True)
                
                if dist < 50 and abs(theta_error) < 0.25:
                    self.in_docking_zone[bot_id] = True
                    self.STATE[bot_id] = "COMPLETE"
                    wheel_vel = [bot_id, 0.0, 0.0, 0.0, self.arm_pos[bot_id], self.solenoid[bot_id]]
                    self.publish_wheel_velocities(wheel_vel)
   
    def go_to_target(self, bot_id , target_x, target_y, target_theta, error_threshold, angle_threshold, use_target_theta):
    
       error_x = target_x - self.bot_pose[bot_id][0]
       error_y = target_y - self.bot_pose[bot_id][1]
       theta_robot = np.radians(self.bot_pose[bot_id][2])
       error_theta = target_theta - theta_robot
       error_theta = math.atan2(math.sin(error_theta), math.cos(error_theta))
       
       dist = math.sqrt(error_x**2 + error_y**2)
       
       if dist > error_threshold:
           
           vx_global = self.pid_controllers[bot_id]['x'].compute(error_x, self.dt)
           vy_global = self.pid_controllers[bot_id]['y'].compute(error_y, self.dt)
           
           vx = vx_global * math.cos(theta_robot) + vy_global * math.sin(theta_robot)
           vy = -vx_global * math.sin(theta_robot) + vy_global * math.cos(theta_robot)
           vtheta = 0.0
           if dist < (error_threshold+50):
               vx = vx*0.4
               vy = vy*0.4
           
       elif abs(error_theta) > angle_threshold and use_target_theta: 
           vx = vy = 0.0
           pid_output = self.pid_controllers[bot_id]['theta'].compute(error_theta, self.dt)
           if pid_output >= 0:
               vtheta = max(pid_output, 40.0)
           else:
               vtheta = -max(abs(pid_output), 40.0)
           
       else:
           vx = 0.0
           vy = 0.0
           vtheta = 0.0
       
    #    self.get_logger().info(f"dist -- {dist}")
    #    self.get_logger().info(f"error_x ={error_x}, error_y ={error_y}")
    #    self.get_logger().info(f"targetx ={target_x}, targety ={target_y}")
    #    self.get_logger().info(f"target={target_theta:.2f}, robot={theta_robot:.2f}, error={error_theta:.2f}")
    #    self.get_logger().info(f"botid - {bot_id} vx - {vx}   vy - {vy}   vtheta - {vtheta}")
                    
       vel = np.array([[vx], [vy], [vtheta]])
       s = np.dot(self.M_inv, vel)
       
       wheel_vel = [bot_id, s[0][0], s[1][0], s[2][0], self.arm_pos[bot_id], self.solenoid[bot_id]]
       self.publish_wheel_velocities(wheel_vel)
       
       return dist, error_theta
   
    def pick_crate(self,bot_id):      
        if self.pick_start_time[bot_id] is None:
            self.pick_start_time[bot_id] = time.time()
            self.arm_pos[bot_id] = 70.0    
            self.solenoid[bot_id] = 1.0  
            wheel_vel = [bot_id, 0.0, 0.0, 0.0, self.arm_pos[bot_id], self.solenoid[bot_id]] 
            self.publish_wheel_velocities(wheel_vel)
            return
        elapsed = time.time() - self.pick_start_time[bot_id]
    
        if elapsed >= self.wait_duration:
            self.arm_pos[bot_id] = 110.0 
            self.publish_wheel_velocities([bot_id, 0.0, 0.0, 0.0, self.arm_pos[bot_id], self.solenoid[bot_id]])
            self.attach_success[bot_id] = True
            self.pick_start_time[bot_id] = None
               
    def place_crate(self,bot_id):  
        if self.drop_start_time[bot_id] is None:
            self.drop_start_time[bot_id] = time.time()    
            self.arm_pos[bot_id] = 80.0    
            self.solenoid[bot_id] = 0.0        
            wheel_vel = [bot_id, 0.0, 0.0, 0.0, self.arm_pos[bot_id], self.solenoid[bot_id]]  
            self.publish_wheel_velocities(wheel_vel)
            self.arm_pos[bot_id] = 110.0    
            self.solenoid[bot_id] = 0.0        
            wheel_vel = [bot_id, -30.0, 30.0, 0.0, self.arm_pos[bot_id], self.solenoid[bot_id]]  
            self.publish_wheel_velocities(wheel_vel)
            return
        elapsed = time.time() - self.drop_start_time[bot_id]
    
        if elapsed >= self.wait_duration:
            self.arm_pos[bot_id] = 110.0 
            self.publish_wheel_velocities([bot_id, 0.0, 0.0, 0.0, self.arm_pos[bot_id], self.solenoid[bot_id]])
            self.detach_success[bot_id] = True
            self.drop_start_time[bot_id] = None

    def publish_wheel_velocities(self, wheel_vel):
        msg = BotCmdArray()
        cmd = BotCmd()
        cmd.id = int(wheel_vel[0])*2
        cmd.m1 = wheel_vel[1]
        cmd.m2 = wheel_vel[2]
        cmd.m3 = wheel_vel[3]
        cmd.base = wheel_vel[4]
        cmd.elbow = wheel_vel[5]
        msg.cmds.append(cmd)
        self.vel_pub.publish(msg)
        
def main(args=None):
    rclpy.init(args=args)
    controller = MultiHolonomicController()
    rclpy.spin(controller)
    controller.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()