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
        
        # Waypoints for navigating to docking zone
        self.docking_waypoints = [
            [690, 690], [1040, 690], [1395, 690],
            [1750, 690], [1750, 1045], [1750, 1400],
            [1750, 1750], [1395, 1750], [1040, 1750],
            [690, 1750], [690, 1400], [690, 1045]
        ]
        
        # Track current waypoint for each bot
        self.current_waypoint_index = [None, None, None]
        self.waypoint_path = [[], [], []]
        
        # Track crate visits per bot
        self.crate_visit_count = {0: {}, 1: {}, 2: {}}  # bot_id: {crate_id: visit_count}
        
        self.STATE = ["IDLE","IDLE","IDLE"]
        self.bot_pose = [[None]*3, [None]*3, [None]*3]
        self.bot_assigned_crate = [None, None, None] 
        self.bot_name = {
            0: "hb_crystal",
            1: "hb_frostbite", 
            2: "hb_glacio"
        }
        self.bot_priority = {0: 3, 1: 2, 2: 1}
        self.arm_pos = [100.0,100.0,100.0]
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
        self.angle_correction = [2.0,-5.0,2.0]  # Adjusted based on testing
        self.dist_threshold = [210.0,215.0,210.0]
        
        self.offset = 50.0
        
        # Collision avoidance
        self.collision_threshold = 225.0  # mm
        
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
    
    def is_point_in_drop_zone(self, x, y, margin=100):
        """Check if a point is inside any drop zone with margin"""
        for zone in self.d_zone:
            if (zone[0] - margin < x < zone[1] + margin) and \
               (zone[2] - margin < y < zone[3] + margin):
                return True
        return False
    
    def distance(self, x1, y1, x2, y2):
        """Calculate Euclidean distance between two points"""
        return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    
    def check_collision_and_priority(self, bot_id, target_x, target_y):
        """
        Check if bot is too close to another bot.
        Returns: (should_stop, should_reverse)
        - (False, False): Continue normally
        - (True, False): Stop in place
        - (False, True): Move backward
        """
        bot_x, bot_y = self.bot_pose[bot_id][0], self.bot_pose[bot_id][1]
        
        for other_id in [0, 1, 2]:
            if other_id == bot_id:
                continue
            
            if self.STATE[other_id] in ["IDLE", "COMPLETE"]:
                continue
                
            other_x, other_y = self.bot_pose[other_id][0], self.bot_pose[other_id][1]
            bot_distance = self.distance(bot_x, bot_y, other_x, other_y)
            
            if bot_distance < self.collision_threshold:
                my_dist_to_target = self.distance(bot_x, bot_y, target_x, target_y)
                
                other_target_x, other_target_y = None, None
                if self.STATE[other_id] == "GO_TO_CRATE":
                    other_crate_id = self.bot_assigned_crate[other_id]
                    if other_crate_id and other_crate_id in self.crates:
                        other_target_x, other_target_y = self.crates[other_crate_id][0], self.crates[other_crate_id][1]
                elif self.STATE[other_id] == "GO_TO_DROP_ZONE":
                    other_crate_id = self.bot_assigned_crate[other_id]
                    if other_crate_id:
                        drop_zone_id = other_crate_id % 3
                        zone = self.d_zone[drop_zone_id]
                        other_target_x = (zone[0] + zone[1]) / 2
                        other_target_y = (zone[2] + zone[3]) / 2
                elif self.STATE[other_id] == "GO_TO_DOCKING_ZONE":
                    dock = self.docking_zone[other_id]
                    other_target_x, other_target_y = dock[0], dock[1]
                
                if other_target_x is not None:
                    other_dist_to_target = self.distance(other_x, other_y, other_target_x, other_target_y)
                    
                    # The bot farther from its target should move backward
                    if my_dist_to_target > other_dist_to_target:
                        return False, True  # Move backward
        
        return False, False  # Continue normally

    def move_backward(self, bot_id):
        """Move bot backward away from collision"""
        # Get current position and calculate backward direction
        bot_x, bot_y = self.bot_pose[bot_id][0], self.bot_pose[bot_id][1]
        theta_robot = np.radians(self.bot_pose[bot_id][2])
        
        # Move backward in robot's local -x direction
        vx = -40.0  # Negative for backward
        vy = 0.0
        vtheta = 0.0
        
        vel = np.array([[vx], [vy], [vtheta]])
        s = np.dot(self.M_inv, vel)
        
        wheel_vel = [bot_id, s[0][0], s[1][0], s[2][0], self.arm_pos[bot_id], self.solenoid[bot_id]]
        self.publish_wheel_velocities(wheel_vel)
        
    def get_drop_target(self, crate_id, zone):
        """Returns (x, y) slightly offset so two crates fit side-by-side"""
        center_x = (zone[0] + zone[1]) / 2
        center_y = (zone[2] + zone[3]) / 2
        
        # Decide left/right or top/bottom depending on your zone orientation
        offset = self.offset  # mm — tune this (≈ half crate width + margin)
        
        if crate_id in [12, 13]:   # first crate of each color → left / bottom
            return center_x - offset, center_y
        else:                      # 30, 16 right / top
            return center_x + offset, center_y
    
    def find_waypoint_path(self, bot_id, start_x, start_y, goal_x, goal_y):
        """Find the best waypoint path from start to goal avoiding drop zones"""
        
        # Find nearest waypoint to start position
        min_dist_start = float('inf')
        start_wp_idx = 0
        for i, wp in enumerate(self.docking_waypoints):
            if not self.is_point_in_drop_zone(wp[0], wp[1]):
                dist = self.distance(start_x, start_y, wp[0], wp[1])
                if dist < min_dist_start:
                    min_dist_start = dist
                    start_wp_idx = i
        
        # Find nearest waypoint to goal position
        min_dist_goal = float('inf')
        goal_wp_idx = 0
        for i, wp in enumerate(self.docking_waypoints):
            if not self.is_point_in_drop_zone(wp[0], wp[1]):
                dist = self.distance(goal_x, goal_y, wp[0], wp[1])
                if dist < min_dist_goal:
                    min_dist_goal = dist
                    goal_wp_idx = i
        
        # Create path along waypoints (going in the shorter direction around the loop)
        path = []
        num_waypoints = len(self.docking_waypoints)
        
        # Calculate distance going clockwise vs counter-clockwise
        if start_wp_idx <= goal_wp_idx:
            clockwise_dist = goal_wp_idx - start_wp_idx
            counter_clockwise_dist = num_waypoints - clockwise_dist
        else:
            counter_clockwise_dist = start_wp_idx - goal_wp_idx
            clockwise_dist = num_waypoints - counter_clockwise_dist
        
        # Choose shorter path
        if clockwise_dist <= counter_clockwise_dist:
            # Go clockwise
            idx = start_wp_idx
            while idx != goal_wp_idx:
                if not self.is_point_in_drop_zone(self.docking_waypoints[idx][0], 
                                                   self.docking_waypoints[idx][1]):
                    path.append(self.docking_waypoints[idx])
                idx = (idx + 1) % num_waypoints
        else:
            # Go counter-clockwise
            idx = start_wp_idx
            while idx != goal_wp_idx:
                if not self.is_point_in_drop_zone(self.docking_waypoints[idx][0], 
                                                   self.docking_waypoints[idx][1]):
                    path.append(self.docking_waypoints[idx])
                idx = (idx - 1) % num_waypoints
        
        # Add final waypoint if it's safe
        if not self.is_point_in_drop_zone(self.docking_waypoints[goal_wp_idx][0], 
                                           self.docking_waypoints[goal_wp_idx][1]):
            path.append(self.docking_waypoints[goal_wp_idx])
        
        return path
        
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
        
        filtered_crates = []
        crate_proximity_threshold = 500.0  # mm
        
        for crate_id in unassigned_crates:
            cx, cy = self.crates[crate_id][0], self.crates[crate_id][1]
            too_close = False
            skip_this_crate = False
            
            # Check against already assigned crates
            for assigned_id in currently_assigned:
                if assigned_id in self.crates:
                    ax, ay = self.crates[assigned_id][0], self.crates[assigned_id][1]
                    dist = self.distance(cx, cy, ax, ay)
                    if dist < crate_proximity_threshold:
                        too_close = True
                        break
            
            # Check against other filtered crates
            if not too_close:
                for other_id in filtered_crates:
                    ox, oy = self.crates[other_id][0], self.crates[other_id][1]
                    dist = self.distance(cx, cy, ox, oy)
                    if dist < crate_proximity_threshold:
                        # Compare IDs - keep higher value
                        if crate_id > other_id:
                            # Remove the lower ID crate, keep this one
                            filtered_crates.remove(other_id)
                        else:
                            # Skip this crate, keep the higher ID one
                            skip_this_crate = True
                        break
            
            if not too_close and not skip_this_crate:
                filtered_crates.append(crate_id)
        
        unassigned_crates = filtered_crates
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
            
            # REMOVED: Don't assign drop zone here - do it after pickup!
            # drop_zone_id = crate_id % 3
            # self.d_zone_assigned[drop_zone_id] = True
            # self.d_zone_bot[drop_zone_id] = bot_id
            
            # Initialize visit count for this crate if not exists
            if crate_id not in self.crate_visit_count[bot_id]:
                self.crate_visit_count[bot_id][crate_id] = 0
            
            # Increment visit count
            self.crate_visit_count[bot_id][crate_id] += 1
            
            # Check if this is a repeat visit (2nd or more)
            if self.crate_visit_count[bot_id][crate_id] >= 2:
                # Use waypoint navigation for repeat visits
                bot_x, bot_y = self.bot_pose[bot_id][0], self.bot_pose[bot_id][1]
                crate_x, crate_y = self.crates[crate_id][0], self.crates[crate_id][1]
                self.waypoint_path[bot_id] = self.find_waypoint_path(bot_id, bot_x, bot_y, crate_x, crate_y)
                self.current_waypoint_index[bot_id] = 0
                print(f"Bot {bot_id} using waypoint path to crate {crate_id} (visit #{self.crate_visit_count[bot_id][crate_id]})")
            else:
                # First visit - go directly
                self.waypoint_path[bot_id] = []
                self.current_waypoint_index[bot_id] = None
                print(f"Bot {bot_id} going directly to crate {crate_id} (first visit)")
            
            self.STATE[bot_id] = "GO_TO_CRATE"
            print(f"Assigned Crate {crate_id} to Bot {bot_id}")


    def pose_cb(self, msg):
        for pose in msg.poses:
                bot_id = int(pose.id/2)
                self.bot_pose[bot_id] = [pose.x,pose.y,pose.w]
                # print(f"{bot_id}------------------------{self.bot_pose[bot_id]}")
                
    def crate_cb(self, msg):
        for pose in msg.poses:
            if pose.id in [12,13,30,16]:
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
                
                # In GO_TO_CRATE state:
                should_stop, should_reverse = self.check_collision_and_priority(bot_id, target_x, target_y)
                
                if should_reverse:
                    print(f"Bot {bot_id} REVERSING - collision avoidance")
                    self.move_backward(bot_id)
                    continue
                elif should_stop:
                    print(f"Bot {bot_id} STOPPED - collision avoidance")
                    wheel_vel = [bot_id, 0.0, 0.0, 0.0, self.arm_pos[bot_id], self.solenoid[bot_id]]
                    self.publish_wheel_velocities(wheel_vel)
                    continue
                
                # Check if using waypoint navigation
                if self.waypoint_path[bot_id] and self.current_waypoint_index[bot_id] < len(self.waypoint_path[bot_id]):
                    # Navigate through waypoints
                    current_wp = self.waypoint_path[bot_id][self.current_waypoint_index[bot_id]]
                    wp_x, wp_y = current_wp[0], current_wp[1]
                    
                    error_x = wp_x - self.bot_pose[bot_id][0]
                    error_y = wp_y - self.bot_pose[bot_id][1]
                    target_theta = math.atan2(error_y, error_x) - (math.pi/2)
                    
                    dist, theta_error = self.go_to_target(bot_id, wp_x, wp_y, target_theta,
                                                           error_threshold=80, angle_threshold=0.3, 
                                                           use_target_theta=True, slow_down=False)
                    
                    if dist < 80:
                        self.current_waypoint_index[bot_id] += 1
                        print(f"Bot {bot_id} reached waypoint {self.current_waypoint_index[bot_id]}/{len(self.waypoint_path[bot_id])}")
                        self.pid_reset(bot_id)
                else:
                    # Direct navigation to crate (either first visit or all waypoints completed)
                    error_x = target_x - self.bot_pose[bot_id][0]
                    error_y = target_y - self.bot_pose[bot_id][1]
                    target_theta = math.atan2(error_y, error_x) - (math.pi/2) - np.radians(self.angle_correction[bot_id])
                    dist, theta = self.go_to_target(bot_id, target_x, target_y, target_theta,
                                                     error_threshold=self.dist_threshold[bot_id], 
                                                     angle_threshold=0.1, use_target_theta=True, slow_down=True)
                    
                    if dist < self.dist_threshold[bot_id] and abs(theta) < 0.1:
                        self.STATE[bot_id] = "PICK_UP"
                        self.pid_reset(bot_id)
                                              
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
                # target_x = (zone[0] + zone[1]) / 2
                # target_y = (zone[2] + zone[3]) / 2
                
                target_x, target_y = self.get_drop_target(crate_id, zone)
                
                if self.d_zone_assigned[drop_zone_id] and self.d_zone_bot[drop_zone_id] != bot_id:
                    print(f"Bot {bot_id} WAITING - Zone {drop_zone_id} taken by {self.d_zone_bot[drop_zone_id]}")
                    wheel_vel = [bot_id, 0.0, 0.0, 0.0, self.arm_pos[bot_id], self.solenoid[bot_id]]
                    self.publish_wheel_velocities(wheel_vel)
                    continue
                else:
                    print(f"Bot {bot_id} GO_TO_DROP_ZONE {drop_zone_id}")
                
                should_stop, should_reverse = self.check_collision_and_priority(bot_id, target_x, target_y)
                
                if should_reverse:
                    print(f"Bot {bot_id} REVERSING - collision avoidance")
                    self.move_backward(bot_id)
                    continue
                elif should_stop:
                    print(f"Bot {bot_id} STOPPED - collision avoidance")
                    wheel_vel = [bot_id, 0.0, 0.0, 0.0, self.arm_pos[bot_id], self.solenoid[bot_id]]
                    self.publish_wheel_velocities(wheel_vel)
                    continue
                
                error_x = target_x - self.bot_pose[bot_id][0]
                error_y = target_y - self.bot_pose[bot_id][1]
                target_theta = math.atan2(error_y, error_x) - (math.pi/2)
                
                dist, theta = self.go_to_target(bot_id, target_x, target_y, target_theta, 
                                             error_threshold=200, angle_threshold=0.2, 
                                             use_target_theta=True, slow_down=True)
                
                if dist < 200 and abs(theta) < 0.2:
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
                        self.drop_wait_count[bot_id] = 0
                        
                        currently_assigned = sum(1 for cid in self.bot_assigned_crate if cid is not None)
                        total_handled = len(self.completed_crates) + currently_assigned
                        all_crates_known = len(self.crates)
                        
                        if total_handled >= all_crates_known and all_crates_known > 0:
                            self.STATE[bot_id] = "GO_TO_DOCKING_ZONE"
                            # Calculate waypoint path when transitioning to docking
                            dock = self.docking_zone[bot_id]
                            bot_x, bot_y = self.bot_pose[bot_id][0], self.bot_pose[bot_id][1]
                            self.waypoint_path[bot_id] = self.find_waypoint_path(bot_id, bot_x, bot_y, dock[0], dock[1])
                            self.current_waypoint_index[bot_id] = 0
                            print(f"Bot {bot_id} waypoint path: {len(self.waypoint_path[bot_id])} waypoints")
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
                
                target_x = dock[0]
                target_y = dock[1]
                
                should_stop, should_reverse = self.check_collision_and_priority(bot_id, target_x, target_y)
                
                if should_reverse:
                    print(f"Bot {bot_id} REVERSING - collision avoidance")
                    self.move_backward(bot_id)
                    continue
                elif should_stop:
                    print(f"Bot {bot_id} STOPPED - collision avoidance")
                    wheel_vel = [bot_id, 0.0, 0.0, 0.0, self.arm_pos[bot_id], self.solenoid[bot_id]]
                    self.publish_wheel_velocities(wheel_vel)
                    continue
                
                # If waypoint path exists and not completed
                if self.waypoint_path[bot_id] and self.current_waypoint_index[bot_id] < len(self.waypoint_path[bot_id]):
                    # Navigate through waypoints
                    current_wp = self.waypoint_path[bot_id][self.current_waypoint_index[bot_id]]
                    wp_x, wp_y = current_wp[0], current_wp[1]
                    
                    # Calculate target theta towards waypoint
                    error_x = wp_x - self.bot_pose[bot_id][0]
                    error_y = wp_y - self.bot_pose[bot_id][1]
                    target_theta = math.atan2(error_y, error_x) - (math.pi/2)
                    
                    dist, theta_error = self.go_to_target(bot_id, wp_x, wp_y, target_theta,
                                                           error_threshold=80, angle_threshold=0.3, 
                                                           use_target_theta=True, slow_down=False)
                    
                    # Move to next waypoint when current one is reached
                    if dist < 80:
                        self.current_waypoint_index[bot_id] += 1
                        print(f"Bot {bot_id} reached waypoint {self.current_waypoint_index[bot_id]}/{len(self.waypoint_path[bot_id])}")
                        self.pid_reset(bot_id)
                else:
                    # All waypoints completed or no waypoints, go directly to dock
                    dist, theta_error = self.go_to_target(bot_id, dock[0], dock[1], dock[2],
                                                           error_threshold=50, angle_threshold=0.25, 
                                                           use_target_theta=True, slow_down=True)
                    
                    if dist < 50 and abs(theta_error) < 0.25:
                        self.in_docking_zone[bot_id] = True
                        self.STATE[bot_id] = "COMPLETE"
                        wheel_vel = [bot_id, 0.0, 0.0, 0.0, self.arm_pos[bot_id], self.solenoid[bot_id]]
                        self.publish_wheel_velocities(wheel_vel)
                        print(f"Bot {bot_id} DOCKED!")
                        
    def go_to_target(self, bot_id , target_x, target_y, target_theta, error_threshold, angle_threshold, use_target_theta,slow_down):
    
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
           if dist < (error_threshold+50) and slow_down:
               vx = vx*0.4
               vy = vy*0.4
           
       elif abs(error_theta) > angle_threshold and use_target_theta: 
           vx = vy = 0.0
           pid_output = self.pid_controllers[bot_id]['theta'].compute(error_theta, self.dt)
           if pid_output >= 0:
               vtheta = max(pid_output, 25.0)
           else:
               vtheta = -max(abs(pid_output), 25.0)
           
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
            self.arm_pos[bot_id] = 80.0    
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
            self.publish_wheel_velocities(wheel_vel)
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

