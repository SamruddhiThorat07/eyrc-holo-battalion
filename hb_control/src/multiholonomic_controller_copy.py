#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from hb_interfaces.msg import Pose2D,Poses2D,BotCmdArray,BotCmd
from linkattacher_msgs.srv import AttachLink,DetachLink
import numpy as np
from scipy.optimize import linear_sum_assignment
import math
import time
import json
import heapq
from std_msgs.msg import Bool


# STATES
# IDLE
# GO_TO_CRATE
# PICK_UP
# GO_TO_DROP_ZONE
# DROP
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

class HolonomicPIDController(Node):
    def __init__(self):
        super().__init__('holonomic_pid_controller_2') 
        
        self.bot_id = [0,1,2]
        self.dt = None
        self.last_time = self.get_clock().now()
        self.error_tol = 5.0   
        self.max_vel = 100.0   
        
        self.pid_params = {
            'x': {'kp': 6.25, 'ki': 0.005, 'kd': 0.3, 'max_out': self.max_vel},
            'y': {'kp': 6.25, 'ki': 0.005, 'kd': 0.3, 'max_out': self.max_vel},
            'theta': {'kp': 3.5, 'ki': 0.025, 'kd': 0.0, 'max_out': 75.0}
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
        
        # ============ GRID & PATHFINDING SETUP ============
        self.arena_size = 2438.4
        self.grid_size = 10
        self.cell_size = self.arena_size / self.grid_size  # 243.84mm per cell
        self.grid = np.zeros((self.grid_size, self.grid_size), dtype=int)  # 0=free, 1=obstacle
        
        # Waypoint tracking for each bot
        self.bot_waypoints = [[], [], []]  # List of (x,y) waypoints for each bot
        self.bot_current_waypoint_index = [0, 0, 0]  # Current waypoint index
        self.waypoint_reached_threshold = 60.0  # mm
        
        # Mark drop zones as static obstacles (for now)
        self.mark_drop_zones_as_obstacles()
        
        self.STATE = ["IDLE","IDLE","IDLE"]
        self.bot_pose = [[None]*3, [None]*3, [None]*3]
        self.bot_assigned_crate = [None, None, None] 
        self.bot_name = {
            0: "hb_crystal",
            1: "hb_frostbite", 
            2: "hb_glacio"
        }
        self.bot_priority = {0: 3, 1: 2, 2: 1}
        self.base = [0.0,0.0,0.0]
        self.elbow = [0.0,0.0,0.0]
        
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
        self.wait_duration = 4.5 
        
        self.bot_sub = self.create_subscription(Poses2D,"/bot_pose",self.pose_cb,10)
        self.crate_sub = self.create_subscription(Poses2D,"/crate_pose",self.crate_cb,10)

        self.vel_pub = self.create_publisher(BotCmdArray, '/bot_cmd', 10)
        
        self.attach_client = self.create_client(AttachLink,"/attach_link")
        while not self.attach_client.wait_for_service(timeout_sec=1.0):
            print("attach service not available")
        
        self.detach_client = self.create_client(DetachLink,"/detach_link")
        while not self.detach_client.wait_for_service(timeout_sec=1.0):
            print("detach service not available")
        
        self.attach_req = AttachLink.Request()
        self.detach_req = DetachLink.Request()
        
        self.timer = self.create_timer(0.33, self.control_cb)
        
        print("Controller initialized with A* pathfinding!")
        print(f"Grid: {self.grid_size}x{self.grid_size}, Cell size: {self.cell_size:.2f}mm")
        
    # ============ GRID & PATHFINDING METHODS ============
    
    def world_to_grid(self, x, y):
        """Convert world coordinates (mm) to grid indices"""
        grid_x = int(x / self.cell_size)
        grid_y = int(y / self.cell_size)
        grid_x = max(0, min(self.grid_size - 1, grid_x))
        grid_y = max(0, min(self.grid_size - 1, grid_y))
        return grid_x, grid_y
    
    def grid_to_world(self, grid_x, grid_y):
        """Convert grid indices to world coordinates (center of cell)"""
        x = (grid_x + 0.5) * self.cell_size
        y = (grid_y + 0.5) * self.cell_size
        return x, y
    
    def mark_drop_zones_as_obstacles(self):
        """Mark all drop zones as obstacles in the grid"""
        for zone in self.d_zone:
            x_min, x_max, y_min, y_max = zone
            gx_min, gy_min = self.world_to_grid(x_min, y_min)
            gx_max, gy_max = self.world_to_grid(x_max, y_max)
            
            for gx in range(gx_min, gx_max + 1):
                for gy in range(gy_min, gy_max + 1):
                    if 0 <= gx < self.grid_size and 0 <= gy < self.grid_size:
                        self.grid[gx][gy] = 1
        
        print("Drop zones marked as obstacles in grid")
    
    def heuristic(self, cell, goal):
        """Diagonal distance heuristic for A*"""
        dx = abs(cell[0] - goal[0])
        dy = abs(cell[1] - goal[1])
        D = 1.0
        D2 = 1.414
        return D * (dx + dy) + (D2 - 2 * D) * min(dx, dy)
    
    def get_neighbors(self, cell):
        """Get valid neighbors (8-connected with diagonals)"""
        x, y = cell
        neighbors = [
            (x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1),  # Orthogonal
            (x + 1, y + 1), (x + 1, y - 1), (x - 1, y + 1), (x - 1, y - 1)  # Diagonal
        ]
        
        valid_neighbors = []
        for nx, ny in neighbors:
            if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size:
                if self.grid[nx][ny] == 0:  # Not obstacle
                    valid_neighbors.append((nx, ny))
        
        return valid_neighbors
    
    def astar(self, start_x, start_y, goal_x, goal_y):
        """A* pathfinding - returns list of (world_x, world_y) waypoints"""
        start = self.world_to_grid(start_x, start_y)
        goal = self.world_to_grid(goal_x, goal_y)
        
        # Check if start/goal valid
        if self.grid[start[0]][start[1]] == 1 or self.grid[goal[0]][goal[1]] == 1:
            print(f"A* failed: start {start} or goal {goal} is obstacle")
            return []
        
        counter = 0
        open_set = []
        heapq.heappush(open_set, (0, counter, start))
        
        came_from = {}
        g_score = {start: 0}
        f_score = {start: self.heuristic(start, goal)}
        closed_set = set()
        
        while open_set:
            current_f, _, current = heapq.heappop(open_set)
            
            if current == goal:
                # Reconstruct path
                grid_path = [current]
                while current in came_from:
                    current = came_from[current]
                    grid_path.append(current)
                grid_path.reverse()
                
                # Convert to world coordinates
                world_path = []
                for gx, gy in grid_path:
                    wx, wy = self.grid_to_world(gx, gy)
                    world_path.append((wx, wy))
                
                return world_path
            
            closed_set.add(current)
            
            for neighbor in self.get_neighbors(current):
                if neighbor in closed_set:
                    continue
                
                # Calculate cost (diagonal = 1.414, orthogonal = 1.0)
                dx = abs(neighbor[0] - current[0])
                dy = abs(neighbor[1] - current[1])
                move_cost = 1.414 if (dx == 1 and dy == 1) else 1.0
                
                tentative_g = g_score[current] + move_cost
                
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + self.heuristic(neighbor, goal)
                    
                    counter += 1
                    heapq.heappush(open_set, (f_score[neighbor], counter, neighbor))
        
        return []  # No path found
    
    def navigate_waypoints(self, bot_id, final_theta=None):
        """
        Navigate through waypoints for a bot
        Returns: (reached_final, target_x, target_y, target_theta)
        """
        # If no waypoints or finished
        if not self.bot_waypoints[bot_id] or \
           self.bot_current_waypoint_index[bot_id] >= len(self.bot_waypoints[bot_id]):
            return True, self.bot_pose[bot_id][0], self.bot_pose[bot_id][1], final_theta or 0.0
        
        # Get current target waypoint
        target_x, target_y = self.bot_waypoints[bot_id][self.bot_current_waypoint_index[bot_id]]
        
        # Distance to current waypoint
        dist = math.sqrt((target_x - self.bot_pose[bot_id][0])**2 + 
                        (target_y - self.bot_pose[bot_id][1])**2)
        
        # If reached waypoint, move to next
        if dist < self.waypoint_reached_threshold:
            self.bot_current_waypoint_index[bot_id] += 1
            print(f"Bot {bot_id}: Reached waypoint {self.bot_current_waypoint_index[bot_id]}/{len(self.bot_waypoints[bot_id])}")
            
            # Check if finished all waypoints
            if self.bot_current_waypoint_index[bot_id] >= len(self.bot_waypoints[bot_id]):
                return True, target_x, target_y, final_theta or 0.0
            
            # Get next waypoint
            target_x, target_y = self.bot_waypoints[bot_id][self.bot_current_waypoint_index[bot_id]]
        
        # Calculate target theta (direction toward waypoint)
        error_x = target_x - self.bot_pose[bot_id][0]
        error_y = target_y - self.bot_pose[bot_id][1]
        
        # Use final_theta only at last waypoint
        is_last_waypoint = (self.bot_current_waypoint_index[bot_id] == len(self.bot_waypoints[bot_id]) - 1)
        if is_last_waypoint and final_theta is not None:
            target_theta = final_theta
        else:
            target_theta = math.atan2(error_y, error_x) - (math.pi/2)
        
        return False, target_x, target_y, target_theta
        
    # ============ ORIGINAL METHODS ============
        
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
            
            # COMPUTE PATH TO CRATE
            crate_x, crate_y = self.crates[crate_id][0], self.crates[crate_id][1]
            bot_x, bot_y = self.bot_pose[bot_id][0], self.bot_pose[bot_id][1]
            waypoints = self.astar(bot_x, bot_y, crate_x, crate_y)
            
            if waypoints:
                self.bot_waypoints[bot_id] = waypoints
                self.bot_current_waypoint_index[bot_id] = 0
                print(f"Assigned Crate {crate_id} to Bot {bot_id}, path: {len(waypoints)} waypoints")
            else:
                print(f"WARNING: No path to crate {crate_id} for bot {bot_id}")

    def pose_cb(self, msg):
        for pose in msg.poses:
                bot_id = int(pose.id/2)
                self.bot_pose[bot_id] = [pose.x,pose.y,pose.w]
                
    def crate_cb(self, msg):
        for pose in msg.poses:
                self.crates[pose.id] = [pose.x,pose.y,pose.w]
                
    def pid_reset(self, bot_id):
        self.pid_controllers[bot_id]['x'].reset()
        self.pid_controllers[bot_id]['y'].reset()
        self.pid_controllers[bot_id]['theta'].reset()
        
    def control_cb(self):
        if any(pose[0] is None for pose in self.bot_pose):
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
                wheel_vel = [bot_id, 0.0, 0.0, 0.0, self.base[bot_id], self.elbow[bot_id]]
                self.publish_wheel_velocities(wheel_vel)
              
            elif state == "GO_TO_CRATE":
                print(f"Bot {bot_id} GO_TO_CRATE {self.bot_assigned_crate[bot_id]}")
                crate_id = self.bot_assigned_crate[bot_id]
                if crate_id is None or crate_id not in self.crates:
                    self.STATE[bot_id] = "IDLE"
                    continue
                
                # Navigate using waypoints
                crate_x, crate_y, crate_theta = self.crates[crate_id]
                error_x = crate_x - self.bot_pose[bot_id][0]
                error_y = crate_y - self.bot_pose[bot_id][1]
                final_theta = math.atan2(error_y, error_x) - (math.pi/2)
                
                reached, target_x, target_y, target_theta = self.navigate_waypoints(bot_id, final_theta)
                
                if not reached:
                    # Still navigating waypoints
                    dist, theta = self.go_to_target(bot_id, target_x, target_y, target_theta,
                                                   error_threshold=60, angle_threshold=0.2, use_target_theta=True)
                else:
                    # Reached final waypoint, do precise positioning
                    dist, theta = self.go_to_target(bot_id, crate_x, crate_y, final_theta,
                                                   error_threshold=145, angle_threshold=0.1, use_target_theta=True)
                    
                    if dist < 145.0 and abs(theta) < 0.1:
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
                    # COMPUTE PATH TO DROP ZONE
                    crate_id = self.bot_assigned_crate[bot_id]
                    drop_zone_id = crate_id % 3
                    zone = self.d_zone[drop_zone_id]
                    target_x = (zone[0] + zone[1]) / 2
                    target_y = (zone[2] + zone[3]) / 2
                    
                    bot_x, bot_y = self.bot_pose[bot_id][0], self.bot_pose[bot_id][1]
                    waypoints = self.astar(bot_x, bot_y, target_x, target_y)
                    
                    if waypoints:
                        self.bot_waypoints[bot_id] = waypoints
                        self.bot_current_waypoint_index[bot_id] = 0
                        print(f"Bot {bot_id}: Path to drop zone, {len(waypoints)} waypoints")
                    
                    self.STATE[bot_id] = "GO_TO_DROP_ZONE"
                    self.pid_reset(bot_id)
               
            elif state == "GO_TO_DROP_ZONE":
                crate_id = self.bot_assigned_crate[bot_id]
                drop_zone_id = crate_id % 3
                
                zone = self.d_zone[drop_zone_id]
                target_x = (zone[0] + zone[1]) / 2
                target_y = (zone[2] + zone[3]) / 2
                error_x = target_x - self.bot_pose[bot_id][0]
                error_y = target_y - self.bot_pose[bot_id][1]
                final_theta = math.atan2(error_y, error_x) - (math.pi/2)
                
                if self.d_zone_assigned[drop_zone_id] and self.d_zone_bot[drop_zone_id] != bot_id:
                    print(f"Bot {bot_id} WAITING - Zone {drop_zone_id} taken by {self.d_zone_bot[drop_zone_id]}")
                    wheel_vel = [bot_id, 0.0, 0.0, 0.0, self.base[bot_id], self.elbow[bot_id]]
                    self.publish_wheel_velocities(wheel_vel)
                    continue
                else:
                    print(f"Bot {bot_id} GO_TO_DROP_ZONE {drop_zone_id}")
                
                # Navigate using waypoints
                reached, wp_target_x, wp_target_y, wp_target_theta = self.navigate_waypoints(bot_id, final_theta)
                
                if not reached:
                    # Still following waypoints
                    dist, theta = self.go_to_target(bot_id, wp_target_x, wp_target_y, wp_target_theta,
                                                   error_threshold=60, angle_threshold=0.3, use_target_theta=True)
                else:
                    # Final approach to drop zone center
                    dist, theta = self.go_to_target(bot_id, target_x, target_y, final_theta,
                                                   error_threshold=160, angle_threshold=0.6, use_target_theta=True)
                    
                    if dist < 160 and abs(theta) < 0.6:
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
                        else:
                            print(f"Bot {bot_id}: Crate {crate_id} NOT in zone")
                    
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
                            # COMPUTE PATH TO DOCKING ZONE
                            dock = self.docking_zone[bot_id]
                            bot_x, bot_y = self.bot_pose[bot_id][0], self.bot_pose[bot_id][1]
                            waypoints = self.astar(bot_x, bot_y, dock[0], dock[1])
                            
                            if waypoints:
                                self.bot_waypoints[bot_id] = waypoints
                                self.bot_current_waypoint_index[bot_id] = 0
                                print(f"Bot {bot_id}: Path to docking, {len(waypoints)} waypoints")
                            
                            self.STATE[bot_id] = "GO_TO_DOCKING_ZONE"
                        else:
                            self.STATE[bot_id] = "IDLE"
                        
                        drop_zone_id = crate_id % 3
                        self.d_zone_assigned[drop_zone_id] = False
                        self.d_zone_bot[drop_zone_id] = None
                        self.pid_reset(bot_id)
                    
            elif state == "GO_TO_DOCKING_ZONE":
                print(f"Bot {bot_id} GO_TO_DOCKING_ZONE")
                dock = self.docking_zone[bot_id]
                
                # Navigate using waypoints
                reached, wp_target_x, wp_target_y, wp_target_theta = self.navigate_waypoints(bot_id, dock[2])
                
                if not reached:
                    # Following waypoints
                    dist, theta_error = self.go_to_target(bot_id, wp_target_x, wp_target_y, wp_target_theta,
                                                         error_threshold=60, angle_threshold=0.2, use_target_theta=True)
                else:
                    # Final docking precision
                    dist, theta_error = self.go_to_target(bot_id, dock[0], dock[1], dock[2],
                                                         error_threshold=20, angle_threshold=0.15, use_target_theta=True)
                    
                    if dist < 20 and abs(theta_error) < 0.1:
                        self.in_docking_zone[bot_id] = True
                        self.STATE[bot_id] = "COMPLETE"
                        wheel_vel = [bot_id, 0.0, 0.0, 0.0, self.base[bot_id], self.elbow[bot_id]]
                        self.publish_wheel_velocities(wheel_vel)
                        print(f"Bot {bot_id} DOCKED!")
   
    def go_to_target(self, bot_id, target_x, target_y, target_theta, error_threshold, angle_threshold, use_target_theta):
        error_x = target_x - self.bot_pose[bot_id][0]
        error_y = target_y - self.bot_pose[bot_id][1]
        theta_robot = np.radians(self.bot_pose[bot_id][2])
        error_theta = theta_robot - target_theta
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
              
        alpha_deg = np.array([30, 150, 270])
        alpha_rad = np.radians(alpha_deg)
    
        M = np.array([[np.cos(alpha_rad[0] + np.pi/2), np.cos(alpha_rad[1] + np.pi/2), np.cos(alpha_rad[2] + np.pi/2)],
                      [np.sin(alpha_rad[0] + np.pi/2), np.sin(alpha_rad[1] + np.pi/2), np.sin(alpha_rad[2] + np.pi/2)],
                      [1, 1, 1]])
        M_inv = np.linalg.inv(M)
        
        vel = np.array([[vx], [vy], [vtheta]])
        s = np.dot(M_inv, vel)
        
        wheel_vel = [bot_id, s[0][0], s[1][0], s[2][0], self.base[bot_id], self.elbow[bot_id]]
        self.publish_wheel_velocities(wheel_vel)
        
        return dist, error_theta
    
    def pick_crate(self,bot_id):      
        if self.pick_start_time[bot_id] is None:
            self.pick_start_time[bot_id] = time.time()
            self.base[bot_id] = 90.0    
            self.elbow[bot_id] = 90.0  
            wheel_vel = [bot_id, 0.0, 0.0, 0.0, self.base[bot_id], self.elbow[bot_id]] 
            self.publish_wheel_velocities(wheel_vel)
            return
        elapsed = time.time() - self.pick_start_time[bot_id]
    
        if elapsed >= self.wait_duration:
            if not self.attach_success[bot_id]:
               crate_id = self.bot_assigned_crate[bot_id]
               self.send_attach_req(bot_id, crate_id)
            self.pick_start_time[bot_id] = None
               
    def place_crate(self,bot_id):  
        if self.drop_start_time[bot_id] is None:
            self.drop_start_time[bot_id] = time.time()    
            self.base[bot_id] = 90.0    
            self.elbow[bot_id] = 90.0        
            wheel_vel = [bot_id, 0.0, 0.0, 0.0, self.base[bot_id], self.elbow[bot_id]]  
            self.publish_wheel_velocities(wheel_vel)
            return
        elapsed = time.time() - self.drop_start_time[bot_id]
    
        if elapsed >= self.wait_duration:
            if not self.detach_success[bot_id]:
               crate_id = self.bot_assigned_crate[bot_id]
               self.send_detach_req(bot_id, crate_id)
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
        
    def send_attach_req(self, bot_id, crate_id):
        
        data_dict = {
            "model1_name": self.bot_name[bot_id],  
            "link1_name": "arm_link_2",
            "model2_name": f"crate_{self.crate_colour[crate_id%3]}_{crate_id}",
            "link2_name": f"box_link_{crate_id}"
        }
        
        self.attach_req.data = json.dumps(data_dict)
        future = self.attach_client.call_async(self.attach_req)
        future.add_done_callback(lambda f: self.handle_attach(f, bot_id))
    
    def send_detach_req(self, bot_id,crate_id):
        
        data_dict = {
            "model1_name": self.bot_name[bot_id],
            "link1_name": "arm_link_2",
            "model2_name": f"crate_{self.crate_colour[crate_id%3]}_{crate_id}",
            "link2_name": f"box_link_{crate_id}"
        }
        
        self.detach_req.data = json.dumps(data_dict)
        future = self.detach_client.call_async(self.detach_req)
        future.add_done_callback(lambda f: self.handle_detach(f, bot_id))
        
    def handle_attach(self, future, bot_id):
        try:
            response = future.result()
            self.attach_success[bot_id] = response.success
            if self.attach_success[bot_id]:
                print(f"Bot {bot_id} attached crate successfully")   
                self.base[bot_id] = 70.0    
                self.elbow[bot_id] = 70.0               
                wheel_vel = [bot_id, 0.0, 0.0, 0.0, self.base[bot_id], self.elbow[bot_id]]  
                self.publish_wheel_velocities(wheel_vel)
            else:
                print(f"Bot {bot_id} attach failed, retrying...")
                self.pick_crate(bot_id)
        except Exception as e:
            print(f"Bot {bot_id} attach error: {e}")
            
    def handle_detach(self, future, bot_id):
        try:
            response = future.result()
            self.detach_success[bot_id] = response.success
            if self.detach_success[bot_id]:
                print(f"Bot {bot_id} detached crate successfully")  
                self.base[bot_id] = 5.0    
                self.elbow[bot_id] = 5.0                     
                wheel_vel = [bot_id, -50.0, 50.0, 0.0, self.base[bot_id], self.elbow[bot_id]]  
                self.publish_wheel_velocities(wheel_vel)
            else:
                print(f"Bot {bot_id} detach failed, retrying...")
                self.place_crate(bot_id)
        except Exception as e:
            print(f"Bot {bot_id} detach error: {e}")
        
def main(args=None):
    rclpy.init(args=args)
    controller = HolonomicPIDController()
    rclpy.spin(controller)
    controller.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()