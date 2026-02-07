#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from hb_interfaces.msg import Pose2D, Poses2D, BotCmdArray, BotCmd
import numpy as np
from scipy.optimize import linear_sum_assignment
import math
import time
from std_msgs.msg import Bool
from hb_interfaces.msg import BotIrState

# =============================================================================
# STATES USED IN STATE MACHINE
# =============================================================================
# IDLE                 - Bot is waiting for task assignment
# GO_TO_CRATE          - Bot is navigating to assigned crate
# PICK_UP              - Bot is picking up crate (arm + solenoid control)
# GO_TO_DROP_ZONE      - Bot is navigating to drop zone with crate
# DROP                 - Bot is placing crate in drop zone
# GO_TO_DOCKING_ZONE   - Bot is returning to docking position
# COMPLETE             - Task finished, bot is docked
# =============================================================================

# =============================================================================
# BOT PRIORITIES & ZONE ASSIGNMENTS
# =============================================================================
# BOTS:
# 0 - Crystal          Priority: 3 (highest)
# 1 - Frostbite        Priority: 2
# 2 - Glacio           Priority: 1 (lowest)
#
# DROP ZONES:
# Red crates   → D1 (zone 0)
# Green crates → D2 (zone 1)
# Blue crates  → D3 (zone 2)
# =============================================================================


class PID:
    """
    PID Controller for position and orientation control
    Used for smooth velocity control to reach targets
    """
    def __init__(self, kp, ki, kd, max_out=1.0):
        self.kp = kp          # Proportional gain
        self.ki = ki          # Integral gain
        self.kd = kd          # Derivative gain
        self.max_out = max_out  # Maximum output limit
        self.integral = 0.0
        self.prev_error = 0.0

    def compute(self, error, dt):
        """Calculate PID output based on error and time step"""
        self.integral += error * dt
        self.der = (error - self.prev_error) / dt
        self.prev_error = error
        output = (self.kp * error) + (self.integral * self.ki) + (self.der * self.kd)
        output = np.clip(output, -self.max_out, self.max_out)
        return output
    
    def reset(self):
        """Reset PID controller (call when switching targets)"""
        self.integral = 0.0
        self.prev_error = 0.0


class MultiHolonomicController(Node):
    """
    Main controller for 3 holonomic robots
    Handles crate detection, assignment, navigation, pickup, and delivery
    """
    
    def __init__(self):
        super().__init__('multiholonomic_controller_hardware') 
        
        # =============================================================================
        # BASIC CONFIGURATION
        # =============================================================================
        self.bot_id = [0, 1, 2]  # IDs for three bots
        self.dt = None  # Time step for PID control
        self.last_time = self.get_clock().now()
        self.error_tol = 5.0   
        self.max_vel = 90.0  # Maximum linear velocity (mm/s)
        
        # =============================================================================
        # PID CONTROLLER SETUP
        # =============================================================================
        # Tuned parameters for hardware (more conservative than simulation)
        self.pid_params = {
            'x': {'kp': 3.5, 'ki': 0.0, 'kd': 0.15, 'max_out': self.max_vel},
            'y': {'kp': 3.5, 'ki': 0.0, 'kd': 0.15, 'max_out': self.max_vel},
            'theta': {'kp': 1.8, 'ki': 0.0, 'kd': 0.1, 'max_out': 35.0}  # deg/s
        }
        
        # Create separate PID controllers for each bot (x, y, theta)
        self.pid_controllers = {
            0: {'x': PID(**self.pid_params['x']), 
                'y': PID(**self.pid_params['y']), 
                'theta': PID(**self.pid_params['theta'])},
            1: {'x': PID(**self.pid_params['x']), 
                'y': PID(**self.pid_params['y']), 
                'theta': PID(**self.pid_params['theta'])},
            2: {'x': PID(**self.pid_params['x']), 
                'y': PID(**self.pid_params['y']), 
                'theta': PID(**self.pid_params['theta'])}
        }
        
        # =============================================================================
        # DROP ZONE DEFINITIONS (mm)
        # =============================================================================
        # Format: [x_min, x_max, y_min, y_max]
        self.d_zone = [
            [1020, 1410, 1075, 1355],   # Red zone (D1)
            [675, 965, 1920, 2115],     # Green zone (D2)
            [1470, 1762, 1920, 2115]    # Blue zone (D3)
        ]
        
        # Drop zone management
        self.d_zone_occupied = [False, False, False]  # Is zone occupied by a crate?
        self.d_zone_assigned = [False, False, False]  # Is zone assigned to a bot?
        self.d_zone_bot = [None, None, None]  # Which bot is assigned to which zone
        
        # =============================================================================
        # DOCKING ZONE POSITIONS (mm, mm, deg)
        # =============================================================================
        # Final positions where bots return after completing all tasks
        self.docking_zone = [
            [1218.0, 205.0, 0.0],   # Crystal (bot 0)
            [1568.0, 202.0, 0.0],   # Frostbite (bot 1)
            [864.0, 204.0, 0.0]     # Glacio (bot 2)
        ]
        
        # =============================================================================
        # WAYPOINT NAVIGATION SETUP
        # =============================================================================
        # 12 waypoints forming a perimeter around the arena
        # Used to navigate around drop zones safely
        self.docking_waypoints = [
            [690, 690], [1040, 690], [1395, 690],     # Bottom row
            [1750, 690], [1750, 1045], [1750, 1400],  # Right side
            [1750, 1750], [1395, 1750], [1040, 1750], # Top row
            [690, 1750], [690, 1400], [690, 1045]     # Left side
        ]
        
        # Track waypoint navigation state for each bot
        self.current_waypoint_index = [None, None, None]  # Current waypoint in path
        self.waypoint_path = [[], [], []]  # List of waypoints for each bot
        
        # =============================================================================
        # NSEW ENTRY POINTS SYSTEM
        # =============================================================================
        # Distance from drop zone center to entry point (mm)
        self.entry_distance = 300.0
        
        # Entry point reservations: {(zone_id, direction): bot_id or None}
        # Prevents multiple bots from using same entry point
        self.entry_reservations = {}
        for zone_id in [0, 1, 2]:
            for direction in ['north', 'south', 'east', 'west']:
                self.entry_reservations[(zone_id, direction)] = None
        
        # Track which entry each bot is currently using
        self.bot_assigned_entry = [None, None, None]  # [(zone_id, direction), ...]
        
        # =============================================================================
        # STATE MACHINE & TASK TRACKING
        # =============================================================================
        self.STATE = ["IDLE", "IDLE", "IDLE"]  # Current state for each bot
        self.bot_pose = [[None]*3, [None]*3, [None]*3]  # [x, y, theta] for each bot
        self.bot_assigned_crate = [None, None, None]  # Which crate assigned to each bot
        
        # Bot names for ROS service calls
        self.bot_name = {
            0: "hb_crystal",
            1: "hb_frostbite", 
            2: "hb_glacio"
        }
        
        # Priority for conflict resolution (not currently used heavily)
        self.bot_priority = {0: 3, 1: 2, 2: 1}
        
        # Hardware control values
        self.arm_pos = [100.0, 100.0, 100.0]  # Arm position for each bot
        self.solenoid = [0.0, 0.0, 0.0]  # Solenoid state (0=off, 1=on)
        
        # =============================================================================
        # CRATE TRACKING
        # =============================================================================
        self.current_crates = set()  # All detected crates
        self.completed_crates = set()  # Crates successfully delivered
        self.crates = {}  # {crate_id: [x, y, theta]}
        
        # =============================================================================
        # TASK STATE FLAGS
        # =============================================================================
        # Track progress through pickup/drop sequences
        self.crate_reached = [False, False, False]  # Bot has reached crate position
        self.attach_success = [False, False, False]  # Crate successfully attached
        self.detach_success = [False, False, False]  # Crate successfully detached
        self.in_drop_zone = [False, False, False]  # Bot has reached drop zone
        self.in_docking_zone = [False, False, False]  # Bot has reached docking zone
        
        # =============================================================================
        # CRATE COLOR MAPPING
        # =============================================================================
        # Used for ROS service calls (attach/detach)
        self.crate_colour = ["red", "green", "blue"]
        
        # =============================================================================
        # TIMING CONTROL
        # =============================================================================
        # For pickup and drop sequences (arm movement timing)
        self.pick_start_time = [None, None, None]
        self.drop_start_time = [None, None, None]
        self.drop_wait_count = [0, 0, 0]  # Counter for drop verification delay
        self.wait_duration = 2.5  # Seconds to wait for arm movement
        
        # =============================================================================
        # HARDWARE-SPECIFIC PARAMETERS
        # =============================================================================
        # IR sensor state (not heavily used in this version)
        self.ir_state = [False, False, False]
        
        # Per-bot calibration adjustments (degrees)
        self.angle_correction = [2.0, -5.0, 2.0]
        
        # Distance threshold for crate approach (mm)
        self.dist_threshold = [210.0, 215.0, 210.0]
        
        # Offset for side-by-side crate placement (mm)
        self.offset = 50.0
        
        # =============================================================================
        # COLLISION AVOIDANCE
        # =============================================================================
        self.collision_threshold = 225.0  # mm - distance to trigger collision avoidance
        
        # =============================================================================
        # HOLONOMIC DRIVE KINEMATICS
        # =============================================================================
        # Pre-calculate inverse kinematics matrix for 3-wheel holonomic drive
        # Wheel angles: 30°, 150°, 270°
        alpha_deg = np.array([30, 150, 270])
        alpha_rad = np.radians(alpha_deg)
        
        M = np.array([
            [np.cos(alpha_rad[0] + np.pi/2), np.cos(alpha_rad[1] + np.pi/2), np.cos(alpha_rad[2] + np.pi/2)],
            [np.sin(alpha_rad[0] + np.pi/2), np.sin(alpha_rad[1] + np.pi/2), np.sin(alpha_rad[2] + np.pi/2)],
            [1, 1, 1]
        ])
        self.M_inv = np.linalg.inv(M)
        
        # =============================================================================
        # ROS SUBSCRIBERS
        # =============================================================================
        self.bot_sub = self.create_subscription(
            Poses2D, "/bot_pose", self.pose_cb, 10
        )
        self.crate_sub = self.create_subscription(
            Poses2D, "/crate_pose", self.crate_cb, 10
        )
        self.ir_sub = self.create_subscription(
            BotIrState, "/ir_sensor_state", self.ir_callback, 10
        )
        
        # =============================================================================
        # ROS PUBLISHER
        # =============================================================================
        self.vel_pub = self.create_publisher(BotCmdArray, '/bot_cmd', 10)
        
        # =============================================================================
        # CONTROL TIMER
        # =============================================================================
        # Main control loop runs at ~3 Hz (every 0.33 seconds)
        self.timer = self.create_timer(0.33, self.control_cb)
        
        print("=" * 60)
        print("MultiHolonomicController Initialized")
        print("✓ PID Controllers: Configured")
        print("✓ NSEW Entry Points: Enabled")
        print("✓ Path Collision Check: Enabled")
        print("✓ Waypoint Navigation: Ready")
        print("=" * 60)
    
    # =============================================================================
    # UTILITY METHODS
    # =============================================================================
    
    def is_point_in_drop_zone(self, x, y, margin=100):
        """
        Check if a point (x, y) is inside any drop zone
        
        Args:
            x, y: Point coordinates (mm)
            margin: Safety margin around drop zone (mm)
        
        Returns:
            True if point is in any drop zone, False otherwise
        """
        for zone in self.d_zone:
            if (zone[0] - margin < x < zone[1] + margin) and \
               (zone[2] - margin < y < zone[3] + margin):
                return True
        return False
    
    def distance(self, x1, y1, x2, y2):
        """Calculate Euclidean distance between two points"""
        return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    
    # =============================================================================
    # COLLISION AVOIDANCE METHODS
    # =============================================================================
    
    def check_collision_and_priority(self, bot_id, target_x, target_y):
        """
        Check if bot is too close to another bot and decide action
        
        Uses distance-based priority:
        - Bot farther from its target should yield (move backward)
        - Bot closer to target has priority
        
        Args:
            bot_id: ID of bot to check
            target_x, target_y: Bot's current target position
        
        Returns:
            (should_stop, should_reverse) tuple:
            - (False, False): Continue normally
            - (True, False): Stop in place
            - (False, True): Move backward
        """
        bot_x, bot_y = self.bot_pose[bot_id][0], self.bot_pose[bot_id][1]
        
        # Check distance to all other active bots
        for other_id in [0, 1, 2]:
            if other_id == bot_id:
                continue
            
            # Skip if other bot is idle or complete
            if self.STATE[other_id] in ["IDLE", "COMPLETE"]:
                continue
            
            other_x, other_y = self.bot_pose[other_id][0], self.bot_pose[other_id][1]
            bot_distance = self.distance(bot_x, bot_y, other_x, other_y)
            
            # Check if bots are too close
            if bot_distance < self.collision_threshold:
                # Calculate my distance to target
                my_dist_to_target = self.distance(bot_x, bot_y, target_x, target_y)
                
                # Get other bot's target
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
                
                # Compare distances to determine priority
                if other_target_x is not None:
                    other_dist_to_target = self.distance(other_x, other_y, other_target_x, other_target_y)
                    
                    # Bot farther from target should move backward
                    if my_dist_to_target > other_dist_to_target:
                        return False, True  # Move backward
        
        return False, False  # Continue normally
    
    def move_backward(self, bot_id):
        """
        Move bot backward to avoid collision
        
        Moves in robot's local -X direction (backward)
        """
        # Move backward in robot's local frame
        vx = -40.0  # Negative for backward (mm/s)
        vy = 0.0
        vtheta = 0.0
        
        # Convert to wheel velocities
        vel = np.array([[vx], [vy], [vtheta]])
        s = np.dot(self.M_inv, vel)
        
        wheel_vel = [bot_id, s[0][0], s[1][0], s[2][0], self.arm_pos[bot_id], self.solenoid[bot_id]]
        self.publish_wheel_velocities(wheel_vel)
    
    # =============================================================================
    # DROP TARGET CALCULATION
    # =============================================================================
    
    def get_drop_target(self, crate_id, zone):
        """
        Calculate drop target position for side-by-side placement
        
        Places crates with offset so they fit side-by-side in drop zone
        
        Args:
            crate_id: ID of crate to drop
            zone: Drop zone boundaries [x_min, x_max, y_min, y_max]
        
        Returns:
            (target_x, target_y) coordinates
        """
        center_x = (zone[0] + zone[1]) / 2
        center_y = (zone[2] + zone[3]) / 2
        
        # First crate of each color goes to left/bottom
        # Second crate goes to right/top
        if crate_id in [12, 13]:  # First crate
            return center_x - self.offset, center_y
        else:  # Second crate (30, 16, etc.)
            return center_x + self.offset, center_y
    
    # =============================================================================
    # NSEW ENTRY POINT METHODS
    # =============================================================================
    
    def calculate_entry_points(self, zone_id):
        """
        Calculate 4 entry points (North, South, East, West) for a drop zone
        
        Entry points are positioned at entry_distance (300mm) from zone center
        in each cardinal direction
        
        Args:
            zone_id: Drop zone ID (0=red, 1=green, 2=blue)
        
        Returns:
            dict: {'north': (x,y), 'south': (x,y), 'east': (x,y), 'west': (x,y)}
        """
        zone = self.d_zone[zone_id]
        center_x = (zone[0] + zone[1]) / 2
        center_y = (zone[2] + zone[3]) / 2
        
        entry_points = {
            'north': (center_x, center_y + self.entry_distance),
            'south': (center_x, center_y - self.entry_distance),
            'east': (center_x + self.entry_distance, center_y),
            'west': (center_x - self.entry_distance, center_y)
        }
        
        return entry_points
    
    def select_best_entry(self, bot_id, zone_id, target_x, target_y):
        """
        Select best entry point for approaching a drop zone
        
        Selection criteria:
        1. Entry must not be reserved by another bot
        2. Entry should not be inside another drop zone
        3. Prefer entry closest to bot's current position
        
        Args:
            bot_id: ID of bot selecting entry
            zone_id: Target drop zone ID
            target_x, target_y: Final drop target (for fallback)
        
        Returns:
            (direction, entry_x, entry_y) or (None, target_x, target_y) if all busy
        """
        entry_points = self.calculate_entry_points(zone_id)
        bot_x, bot_y = self.bot_pose[bot_id][0], self.bot_pose[bot_id][1]
        
        # Score each entry point
        entry_scores = []
        for direction, (entry_x, entry_y) in entry_points.items():
            # Check if entry is available (not reserved by another bot)
            reserved_by = self.entry_reservations.get((zone_id, direction))
            if reserved_by is not None and reserved_by != bot_id:
                continue  # Entry taken, skip it
            
            # Calculate distance from bot to entry point
            dist_to_entry = self.distance(bot_x, bot_y, entry_x, entry_y)
            
            # Penalize if entry point is inside another drop zone
            if self.is_point_in_drop_zone(entry_x, entry_y, margin=50):
                dist_to_entry += 10000  # Heavy penalty
            
            entry_scores.append((dist_to_entry, direction, entry_x, entry_y))
        
        # If no entries available, return None (go direct)
        if not entry_scores:
            print(f"⚠ Bot {bot_id}: All entries busy for zone {zone_id}, going direct")
            return None, target_x, target_y
        
        # Sort by distance (closest first)
        entry_scores.sort()
        best_dist, best_direction, best_x, best_y = entry_scores[0]
        
        print(f"✓ Bot {bot_id}: Selected {best_direction} entry for zone {zone_id}")
        return best_direction, best_x, best_y
    
    def reserve_entry(self, bot_id, zone_id, direction):
        """
        Reserve an entry point for a bot (like reserving a parking spot)
        
        Args:
            bot_id: Bot claiming the entry
            zone_id: Drop zone ID
            direction: Entry direction ('north', 'south', 'east', 'west')
        """
        if direction is None:
            return
        
        self.entry_reservations[(zone_id, direction)] = bot_id
        self.bot_assigned_entry[bot_id] = (zone_id, direction)
        print(f"✓ Bot {bot_id}: Reserved {direction} entry for zone {zone_id}")
    
    def release_entry(self, bot_id):
        """
        Release bot's reserved entry point (free the parking spot)
        
        Called after bot has finished dropping crate
        
        Args:
            bot_id: Bot releasing its entry
        """
        if self.bot_assigned_entry[bot_id] is not None:
            zone_id, direction = self.bot_assigned_entry[bot_id]
            self.entry_reservations[(zone_id, direction)] = None
            print(f"✓ Bot {bot_id}: Released {direction} entry for zone {zone_id}")
            self.bot_assigned_entry[bot_id] = None
    
    # =============================================================================
    # PATH COLLISION CHECK METHODS
    # =============================================================================
    
    def path_crosses_drop_zone(self, x1, y1, x2, y2, sample_points=8):
        """
        Check if straight line path crosses any drop zone
        
        Uses parametric line sampling to check points along the path
        
        Args:
            x1, y1: Start point
            x2, y2: End point
            sample_points: Number of points to sample along path
        
        Returns:
            True if path crosses a drop zone, False otherwise
        """
        for i in range(sample_points):
            # Parametric line: point = start + t * (end - start), t ∈ [0, 1]
            t = i / (sample_points - 1)
            check_x = x1 + t * (x2 - x1)
            check_y = y1 + t * (y2 - y1)
            
            # Check if this sample point is inside any drop zone
            if self.is_point_in_drop_zone(check_x, check_y, margin=100):
                return True
        
        return False
    
    def should_use_waypoints(self, bot_id, start_x, start_y, goal_x, goal_y):
        """
        Intelligently decide if waypoint navigation is needed
        
        Criteria for using waypoints:
        1. Path crosses drop zone (ALWAYS use waypoints)
        2. Very long distance (>1000mm) (use waypoints for efficiency)
        3. Other bots blocking direct path (use waypoints to avoid)
        
        Args:
            bot_id: Bot making the decision
            start_x, start_y: Starting position
            goal_x, goal_y: Goal position
        
        Returns:
            True if should use waypoints, False for direct navigation
        """
        # Check if direct path crosses any drop zones
        if self.path_crosses_drop_zone(start_x, start_y, goal_x, goal_y):
            print(f"✓ Bot {bot_id}: Path crosses drop zone → using waypoints")
            return True
        
        # Check if distance is very long
        dist = self.distance(start_x, start_y, goal_x, goal_y)
        if dist > 1000:
            print(f"✓ Bot {bot_id}: Long distance ({dist:.0f}mm) → using waypoints")
            return True
        
        # Check if other bots are blocking the direct path
        for other_id in [0, 1, 2]:
            if other_id == bot_id:
                continue
            
            # Skip idle/complete bots
            if self.STATE[other_id] in ["IDLE", "COMPLETE"]:
                continue
            
            other_x, other_y = self.bot_pose[other_id][0], self.bot_pose[other_id][1]
            
            # Check if other bot is within corridor of our path
            # Sample 5 points along our path
            for i in range(5):
                t = i / 4
                path_x = start_x + t * (goal_x - start_x)
                path_y = start_y + t * (goal_y - start_y)
                
                dist_to_path = self.distance(other_x, other_y, path_x, path_y)
                if dist_to_path < 300:  # Other bot is blocking
                    print(f"✓ Bot {bot_id}: Bot {other_id} blocking path → using waypoints")
                    return True
        
        # Path is clear, go direct
        print(f"✓ Bot {bot_id}: Clear path → going direct")
        return False
    
    # =============================================================================
    # WAYPOINT PATH FINDING
    # =============================================================================
    
    def find_waypoint_path(self, bot_id, start_x, start_y, goal_x, goal_y):
        """
        Find waypoint path along perimeter from start to goal
        
        Uses the 12 predefined waypoints forming a loop around the arena.
        Finds nearest waypoint to start and goal, then follows the loop
        in the shorter direction (clockwise or counter-clockwise).
        
        Args:
            bot_id: Bot ID (for logging)
            start_x, start_y: Starting position
            goal_x, goal_y: Goal position
        
        Returns:
            List of waypoint coordinates [(x,y), (x,y), ...]
        """
        # Find nearest waypoint to start position
        min_dist_start = float('inf')
        start_wp_idx = 0
        for i, wp in enumerate(self.docking_waypoints):
            # Skip waypoints that are inside drop zones
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
        
        # Create path along waypoints (shorter direction around loop)
        path = []
        num_waypoints = len(self.docking_waypoints)
        
        # Calculate distance going clockwise vs counter-clockwise
        if start_wp_idx <= goal_wp_idx:
            clockwise_dist = goal_wp_idx - start_wp_idx
            counter_clockwise_dist = num_waypoints - clockwise_dist
        else:
            counter_clockwise_dist = start_wp_idx - goal_wp_idx
            clockwise_dist = num_waypoints - counter_clockwise_dist
        
        # Choose shorter path direction
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
        
        # Add final waypoint if safe
        if not self.is_point_in_drop_zone(self.docking_waypoints[goal_wp_idx][0], 
                                           self.docking_waypoints[goal_wp_idx][1]):
            path.append(self.docking_waypoints[goal_wp_idx])
        
        return path
    
    # =============================================================================
    # CRATE ZONE VERIFICATION
    # =============================================================================
    
    def check_crate_in_zone(self, crate_id):
        """
        Verify if crate is correctly placed in its target drop zone
        
        Args:
            crate_id: ID of crate to check
        
        Returns:
            True if crate is within zone boundaries (with margin), False otherwise
        """
        if crate_id not in self.crates:
            return False
        
        x, y, w = self.crates[crate_id]
        zone_id = crate_id % 3  # Crate color determines zone
        zone = self.d_zone[zone_id]
        margin = 15  # Safety margin (mm)
        
        # Check if crate is within zone boundaries
        if (zone[0] + margin < x < zone[1] - margin) and \
           (zone[2] + margin < y < zone[3] - margin):
            self.d_zone_occupied[zone_id] = True
            return True
        else:
            return False
    
    # =============================================================================
    # CRATE ASSIGNMENT (HUNGARIAN ALGORITHM)
    # =============================================================================
    
    def assign_crates(self):
        """
        Assign crates to idle bots using Hungarian algorithm
        
        Process:
        1. Filter out completed and currently assigned crates
        2. Apply proximity filtering (avoid assigning close crates simultaneously)
        3. Calculate cost matrix (distance from each bot to each crate)
        4. Use Hungarian algorithm for optimal assignment
        5. Decide navigation strategy (waypoints vs direct)
        
        This runs every control cycle to assign new tasks as bots become idle
        """
        # Get currently assigned crates
        currently_assigned = [cid for cid in self.bot_assigned_crate if cid is not None]
        
        # Find unassigned crates (not completed, not currently being handled)
        unassigned_crates = [
           crate_id for crate_id in self.crates.keys() 
           if crate_id not in self.completed_crates 
           and crate_id not in currently_assigned
        ]
        
        if not unassigned_crates:
           return  # No crates to assign
        
        # =============================================================================
        # PROXIMITY FILTERING
        # Avoid assigning crates that are too close to each other
        # This prevents bots from colliding when picking up nearby crates
        # =============================================================================
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
                        # Keep higher ID crate (arbitrary tie-breaking)
                        if crate_id > other_id:
                            filtered_crates.remove(other_id)
                        else:
                            skip_this_crate = True
                        break
            
            if not too_close and not skip_this_crate:
                filtered_crates.append(crate_id)
        
        unassigned_crates = filtered_crates
        if not unassigned_crates:
            return  # All crates filtered out
        
        # Find idle bots
        idle_bots = [bot_id for bot_id in [0, 1, 2] if self.STATE[bot_id] == "IDLE"]
        
        if not idle_bots:
           return  # No bots available
        
        # =============================================================================
        # HUNGARIAN ALGORITHM - OPTIMAL ASSIGNMENT
        # Build cost matrix: cost[bot][crate] = distance from bot to crate
        # =============================================================================
        cost_matrix = []
        for bot_id in idle_bots:
            bot_costs = []
            for crate_id in unassigned_crates:
                bot_x, bot_y = self.bot_pose[bot_id][0], self.bot_pose[bot_id][1]
                crate_x, crate_y = self.crates[crate_id][0], self.crates[crate_id][1]
                
                distance = math.sqrt((crate_x - bot_x)**2 + (crate_y - bot_y)**2)
                bot_costs.append(distance)
            cost_matrix.append(bot_costs)
        
        # Solve assignment problem (minimize total distance)
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        # =============================================================================
        # APPLY ASSIGNMENTS & PLAN PATHS
        # =============================================================================
        for row, col in zip(row_ind, col_ind):
            bot_id = idle_bots[row]
            crate_id = unassigned_crates[col]
            self.bot_assigned_crate[bot_id] = crate_id
            
            # Get bot and crate positions
            bot_x, bot_y = self.bot_pose[bot_id][0], self.bot_pose[bot_id][1]
            crate_x, crate_y = self.crates[crate_id][0], self.crates[crate_id][1]
            
            # =============================================================================
            # SMART PATH DECISION (REPLACES ARBITRARY VISIT COUNT)
            # Analyze path and decide: waypoints or direct?
            # =============================================================================
            if self.should_use_waypoints(bot_id, bot_x, bot_y, crate_x, crate_y):
                # Path needs waypoints (crosses obstacles, long distance, etc.)
                self.waypoint_path[bot_id] = self.find_waypoint_path(
                    bot_id, bot_x, bot_y, crate_x, crate_y
                )
                self.current_waypoint_index[bot_id] = 0
            else:
                # Clear path - go direct
                self.waypoint_path[bot_id] = []
                self.current_waypoint_index[bot_id] = None
            
            # Transition to GO_TO_CRATE state
            self.STATE[bot_id] = "GO_TO_CRATE"
            print(f"✓ Assigned Crate {crate_id} to Bot {bot_id}")
    
    # =============================================================================
    # ROS CALLBACK METHODS
    # =============================================================================
    
    def pose_cb(self, msg):
        """
        Callback for /bot_pose topic
        Updates bot positions from perception system
        """
        for pose in msg.poses:
            bot_id = int(pose.id / 2)  # Convert pose ID to bot ID
            self.bot_pose[bot_id] = [pose.x, pose.y, pose.w]
    
    def crate_cb(self, msg):
        """
        Callback for /crate_pose topic
        Updates crate positions from perception system
        """
        for pose in msg.poses:
            # Only track specific crate IDs (12, 13, 30, 16)
            if pose.id in [12, 13, 30, 16]:
                self.crates[pose.id] = [pose.x, pose.y, pose.w]
    
    def ir_callback(self, msg):
        """
        Callback for IR sensor state
        Not heavily used in current implementation
        """
        bot_id = int(msg.id / 2)
        self.ir_state[bot_id] = msg.state
    
    def pid_reset(self, bot_id):
        """Reset all PID controllers for a bot (when switching targets)"""
        self.pid_controllers[bot_id]['x'].reset()
        self.pid_controllers[bot_id]['y'].reset()
        self.pid_controllers[bot_id]['theta'].reset()
    
    # =============================================================================
    # MAIN CONTROL LOOP
    # =============================================================================
    
    def control_cb(self):
        """
        Main control callback - runs at 3 Hz
        
        This is the heart of the controller. It:
        1. Updates dt for PID control
        2. Assigns crates to idle bots
        3. Executes state machine for each bot
        4. Publishes velocity commands
        """
        # Wait for all bot poses to be available
        if any(pose[0] is None for pose in self.bot_pose):
            return
        
        # Calculate time step for PID controllers
        now = self.get_clock().now()
        self.dt = (now - self.last_time).nanoseconds / 1e9
        if self.dt <= 0:
            return
        self.last_time = now
        
        # Try to assign crates to idle bots
        self.assign_crates()
        
        # =============================================================================
        # STATE MACHINE - Execute for each bot
        # =============================================================================
        for bot_id in [0, 1, 2]:
            state = self.STATE[bot_id]
            
            # =========================================================================
            # STATE: IDLE
            # Bot is waiting for task assignment
            # =========================================================================
            if state == "IDLE":
                print(f"Bot {bot_id} in IDLE")
                # Stop motors
                wheel_vel = [bot_id, 0.0, 0.0, 0.0, self.arm_pos[bot_id], self.solenoid[bot_id]]
                self.publish_wheel_velocities(wheel_vel)
            
            # =========================================================================
            # STATE: GO_TO_CRATE
            # Bot is navigating to assigned crate
            # =========================================================================
            elif state == "GO_TO_CRATE":
                print(f"Bot {bot_id} GO_TO_CRATE {self.bot_assigned_crate[bot_id]}")
                crate_id = self.bot_assigned_crate[bot_id]
                
                # Safety check - is crate still visible?
                if crate_id is None or crate_id not in self.crates:
                    self.STATE[bot_id] = "IDLE"
                    continue
                
                target_x, target_y, crate_theta = self.crates[crate_id]
                
                # --- Collision avoidance check ---
                should_stop, should_reverse = self.check_collision_and_priority(
                    bot_id, target_x, target_y
                )
                
                if should_reverse:
                    print(f"Bot {bot_id} REVERSING - collision avoidance")
                    self.move_backward(bot_id)
                    continue
                elif should_stop:
                    print(f"Bot {bot_id} STOPPED - collision avoidance")
                    wheel_vel = [bot_id, 0.0, 0.0, 0.0, self.arm_pos[bot_id], self.solenoid[bot_id]]
                    self.publish_wheel_velocities(wheel_vel)
                    continue
                
                # --- Navigate using waypoints or direct ---
                if self.waypoint_path[bot_id] and \
                   self.current_waypoint_index[bot_id] < len(self.waypoint_path[bot_id]):
                    
                    # WAYPOINT NAVIGATION
                    # Following waypoint path (avoiding obstacles)
                    current_wp = self.waypoint_path[bot_id][self.current_waypoint_index[bot_id]]
                    wp_x, wp_y = current_wp[0], current_wp[1]
                    
                    # Calculate target orientation toward waypoint
                    error_x = wp_x - self.bot_pose[bot_id][0]
                    error_y = wp_y - self.bot_pose[bot_id][1]
                    target_theta = math.atan2(error_y, error_x) - (math.pi/2)
                    
                    # Navigate to waypoint (no slowdown, loose thresholds)
                    dist, theta_error = self.go_to_target(
                        bot_id, wp_x, wp_y, target_theta,
                        error_threshold=80,
                        angle_threshold=0.3, 
                        use_target_theta=True,
                        slow_down=False  # Full speed through waypoints
                    )
                    
                    # Check if waypoint reached
                    if dist < 80:
                        self.current_waypoint_index[bot_id] += 1
                        print(f"✓ Bot {bot_id}: Waypoint {self.current_waypoint_index[bot_id]}/{len(self.waypoint_path[bot_id])}")
                        self.pid_reset(bot_id)
                
                else:
                    # DIRECT NAVIGATION
                    # Either no waypoints needed, or all waypoints completed
                    error_x = target_x - self.bot_pose[bot_id][0]
                    error_y = target_y - self.bot_pose[bot_id][1]
                    
                    # Apply per-bot angle correction (calibration)
                    target_theta = math.atan2(error_y, error_x) - (math.pi/2) - \
                                 np.radians(self.angle_correction[bot_id])
                    
                    # Final precision approach to crate
                    dist, theta = self.go_to_target(
                        bot_id, target_x, target_y, target_theta,
                        error_threshold=self.dist_threshold[bot_id],
                        angle_threshold=0.1,
                        use_target_theta=True,
                        slow_down=True  # Slow down for precision
                    )
                    
                    # Check if reached crate
                    if dist < self.dist_threshold[bot_id] and abs(theta) < 0.1:
                        self.STATE[bot_id] = "PICK_UP"
                        self.pid_reset(bot_id)
            
            # =========================================================================
            # STATE: PICK_UP
            # Bot is picking up crate (arm + solenoid sequence)
            # =========================================================================
            elif state == "PICK_UP":
                print(f"Bot {bot_id} PICK_UP")
                
                # Start pickup sequence
                if not self.crate_reached[bot_id]:
                    self.pick_crate(bot_id)
                    self.crate_reached[bot_id] = True
                else:
                    # Retry if pickup failed
                    if not self.attach_success[bot_id]:
                        self.pick_crate(bot_id)
                
                # Once pickup successful, transition to drop zone navigation
                if self.attach_success[bot_id]:
                    self.STATE[bot_id] = "GO_TO_DROP_ZONE"
                    self.pid_reset(bot_id)
            
            # =========================================================================
            # STATE: GO_TO_DROP_ZONE
            # Bot is navigating to drop zone with crate
            # Uses NSEW entry point system
            # =========================================================================
            elif state == "GO_TO_DROP_ZONE":
                crate_id = self.bot_assigned_crate[bot_id]
                drop_zone_id = crate_id % 3  # Crate color determines zone
                
                zone = self.d_zone[drop_zone_id]
                # Calculate final drop target (with offset for side-by-side)
                final_target_x, final_target_y = self.get_drop_target(crate_id, zone)
                
                # =================================================================
                # NSEW ENTRY POINT NAVIGATION
                # Two-phase approach: 1) Navigate to entry point, 2) Final approach
                # =================================================================
                
                # --- Phase 1: Select and navigate to entry point ---
                if self.bot_assigned_entry[bot_id] is None:
                    # Haven't selected entry yet - do it now
                    direction, entry_x, entry_y = self.select_best_entry(
                        bot_id, drop_zone_id, final_target_x, final_target_y
                    )
                    
                    if direction is not None:
                        # Reserve this entry point
                        self.reserve_entry(bot_id, drop_zone_id, direction)
                        
                        # Decide if we need waypoints to reach entry
                        bot_x, bot_y = self.bot_pose[bot_id][0], self.bot_pose[bot_id][1]
                        if self.should_use_waypoints(bot_id, bot_x, bot_y, entry_x, entry_y):
                            # Use waypoint path to entry
                            self.waypoint_path[bot_id] = self.find_waypoint_path(
                                bot_id, bot_x, bot_y, entry_x, entry_y
                            )
                            self.current_waypoint_index[bot_id] = 0
                        else:
                            # Direct path to entry
                            self.waypoint_path[bot_id] = []
                            self.current_waypoint_index[bot_id] = None
                
                # --- Determine current navigation target ---
                if self.bot_assigned_entry[bot_id] is not None:
                    # We have an entry assigned
                    zone_id_assigned, direction_assigned = self.bot_assigned_entry[bot_id]
                    entry_points = self.calculate_entry_points(zone_id_assigned)
                    entry_x, entry_y = entry_points[direction_assigned]
                    
                    # Check if we've reached the entry point
                    bot_x, bot_y = self.bot_pose[bot_id][0], self.bot_pose[bot_id][1]
                    dist_to_entry = self.distance(bot_x, bot_y, entry_x, entry_y)
                    
                    if dist_to_entry < 100:
                        # Phase 2: At entry, now approach final drop target
                        print(f"✓ Bot {bot_id}: Reached entry, approaching drop target")
                        target_x, target_y = final_target_x, final_target_y
                        # Clear waypoints for direct final approach
                        self.waypoint_path[bot_id] = []
                        self.current_waypoint_index[bot_id] = None
                    else:
                        # Still navigating to entry point
                        target_x, target_y = entry_x, entry_y
                else:
                    # No entry assigned (all busy) - go direct to drop target
                    target_x, target_y = final_target_x, final_target_y
                
                # --- Collision avoidance ---
                should_stop, should_reverse = self.check_collision_and_priority(
                    bot_id, target_x, target_y
                )
                
                if should_reverse:
                    print(f"Bot {bot_id} REVERSING - collision avoidance")
                    self.move_backward(bot_id)
                    continue
                elif should_stop:
                    print(f"Bot {bot_id} STOPPED - collision avoidance")
                    wheel_vel = [bot_id, 0.0, 0.0, 0.0, self.arm_pos[bot_id], self.solenoid[bot_id]]
                    self.publish_wheel_velocities(wheel_vel)
                    continue
                
                # --- Navigate using waypoints or direct ---
                if self.waypoint_path[bot_id] and \
                   self.current_waypoint_index[bot_id] < len(self.waypoint_path[bot_id]):
                    
                    # Following waypoint path
                    current_wp = self.waypoint_path[bot_id][self.current_waypoint_index[bot_id]]
                    wp_x, wp_y = current_wp[0], current_wp[1]
                    
                    error_x = wp_x - self.bot_pose[bot_id][0]
                    error_y = wp_y - self.bot_pose[bot_id][1]
                    target_theta = math.atan2(error_y, error_x) - (math.pi/2)
                    
                    dist, theta_error = self.go_to_target(
                        bot_id, wp_x, wp_y, target_theta,
                        error_threshold=80,
                        angle_threshold=0.3, 
                        use_target_theta=True,
                        slow_down=False
                    )
                    
                    if dist < 80:
                        self.current_waypoint_index[bot_id] += 1
                        print(f"✓ Bot {bot_id}: Waypoint {self.current_waypoint_index[bot_id]}/{len(self.waypoint_path[bot_id])}")
                        self.pid_reset(bot_id)
                
                else:
                    # Direct navigation to target (entry or final drop)
                    error_x = target_x - self.bot_pose[bot_id][0]
                    error_y = target_y - self.bot_pose[bot_id][1]
                    target_theta = math.atan2(error_y, error_x) - (math.pi/2)
                    
                    # Adjust threshold based on navigation phase
                    if self.bot_assigned_entry[bot_id] is not None:
                        zone_id_assigned, direction_assigned = self.bot_assigned_entry[bot_id]
                        entry_points = self.calculate_entry_points(zone_id_assigned)
                        entry_x, entry_y = entry_points[direction_assigned]
                        dist_to_entry = self.distance(
                            self.bot_pose[bot_id][0], 
                            self.bot_pose[bot_id][1], 
                            entry_x, entry_y
                        )
                        
                        if dist_to_entry < 100:
                            # At entry, doing final approach - tight threshold
                            threshold = 200
                            slow = True
                        else:
                            # Going to entry - loose threshold
                            threshold = 120
                            slow = False
                    else:
                        # Direct to drop (no entry) - tight threshold
                        threshold = 200
                        slow = True
                    
                    dist, theta = self.go_to_target(
                        bot_id, target_x, target_y, target_theta, 
                        error_threshold=threshold,
                        angle_threshold=0.2, 
                        use_target_theta=True,
                        slow_down=slow
                    )
                    
                    # Check if reached drop position
                    if dist < threshold and abs(theta) < 0.2:
                        self.STATE[bot_id] = "DROP"
                        self.pid_reset(bot_id)
            
            # =========================================================================
            # STATE: DROP
            # Bot is placing crate in drop zone
            # =========================================================================
            elif state == "DROP":
                print(f"Bot {bot_id} DROP")
                
                # Start drop sequence
                if not self.in_drop_zone[bot_id]:
                    self.place_crate(bot_id)
                    self.in_drop_zone[bot_id] = True
                else:
                    # Retry if drop failed
                    if not self.detach_success[bot_id]:
                        self.place_crate(bot_id)
                
                # After successful drop, wait and verify
                if self.detach_success[bot_id]:
                    self.drop_wait_count[bot_id] += 1
                    
                    # Wait 15 cycles (~5 seconds) before proceeding
                    if self.drop_wait_count[bot_id] >= 15:
                        crate_id = self.bot_assigned_crate[bot_id]
                        
                        # --- Release entry point reservation ---
                        self.release_entry(bot_id)
                        
                        # --- Verify crate placement ---
                        if self.check_crate_in_zone(crate_id):
                            print(f"✓ Bot {bot_id}: Crate {crate_id} placed correctly")
                            self.completed_crates.add(crate_id)
                        else:
                            print(f"✗ Bot {bot_id}: Crate {crate_id} NOT in zone")
                        
                        # --- Reset task state ---
                        self.bot_assigned_crate[bot_id] = None
                        self.crate_reached[bot_id] = False
                        self.attach_success[bot_id] = False
                        self.detach_success[bot_id] = False
                        self.in_drop_zone[bot_id] = False
                        self.drop_wait_count[bot_id] = 0
                        
                        # --- Check if all crates are done ---
                        currently_assigned = sum(1 for cid in self.bot_assigned_crate if cid is not None)
                        total_handled = len(self.completed_crates) + currently_assigned
                        all_crates_known = len(self.crates)
                        
                        if total_handled >= all_crates_known and all_crates_known > 0:
                            # All crates done - go to docking zone
                            self.STATE[bot_id] = "GO_TO_DOCKING_ZONE"
                            
                            # Plan path to docking zone
                            dock = self.docking_zone[bot_id]
                            bot_x, bot_y = self.bot_pose[bot_id][0], self.bot_pose[bot_id][1]
                            self.waypoint_path[bot_id] = self.find_waypoint_path(
                                bot_id, bot_x, bot_y, dock[0], dock[1]
                            )
                            self.current_waypoint_index[bot_id] = 0
                            print(f"✓ Bot {bot_id}: Path to docking, {len(self.waypoint_path[bot_id])} waypoints")
                        else:
                            # More crates to handle - return to IDLE
                            self.STATE[bot_id] = "IDLE"
                        
                        # --- Release drop zone ---
                        drop_zone_id = crate_id % 3
                        self.d_zone_assigned[drop_zone_id] = False
                        self.d_zone_bot[drop_zone_id] = None
                        self.pid_reset(bot_id)
            
            # =========================================================================
            # STATE: GO_TO_DOCKING_ZONE
            # Bot is returning to docking position after completing all tasks
            # =========================================================================
            elif state == "GO_TO_DOCKING_ZONE":
                print(f"Bot {bot_id} GO_TO_DOCKING_ZONE")
                dock = self.docking_zone[bot_id]
                
                target_x = dock[0]
                target_y = dock[1]
                
                # --- Collision avoidance ---
                should_stop, should_reverse = self.check_collision_and_priority(
                    bot_id, target_x, target_y
                )
                
                if should_reverse:
                    print(f"Bot {bot_id} REVERSING - collision avoidance")
                    self.move_backward(bot_id)
                    continue
                elif should_stop:
                    print(f"Bot {bot_id} STOPPED - collision avoidance")
                    wheel_vel = [bot_id, 0.0, 0.0, 0.0, self.arm_pos[bot_id], self.solenoid[bot_id]]
                    self.publish_wheel_velocities(wheel_vel)
                    continue
                
                # --- Navigate using waypoints or direct ---
                if self.waypoint_path[bot_id] and \
                   self.current_waypoint_index[bot_id] < len(self.waypoint_path[bot_id]):
                    
                    # Following waypoint path
                    current_wp = self.waypoint_path[bot_id][self.current_waypoint_index[bot_id]]
                    wp_x, wp_y = current_wp[0], current_wp[1]
                    
                    error_x = wp_x - self.bot_pose[bot_id][0]
                    error_y = wp_y - self.bot_pose[bot_id][1]
                    target_theta = math.atan2(error_y, error_x) - (math.pi/2)
                    
                    dist, theta_error = self.go_to_target(
                        bot_id, wp_x, wp_y, target_theta,
                        error_threshold=80,
                        angle_threshold=0.3, 
                        use_target_theta=True,
                        slow_down=False
                    )
                    
                    if dist < 80:
                        self.current_waypoint_index[bot_id] += 1
                        print(f"✓ Bot {bot_id}: Waypoint {self.current_waypoint_index[bot_id]}/{len(self.waypoint_path[bot_id])}")
                        self.pid_reset(bot_id)
                
                else:
                    # All waypoints done - final precision docking
                    dist, theta_error = self.go_to_target(
                        bot_id, dock[0], dock[1], dock[2],
                        error_threshold=50,
                        angle_threshold=0.25, 
                        use_target_theta=True,
                        slow_down=True
                    )
                    
                    # Check if docked
                    if dist < 50 and abs(theta_error) < 0.25:
                        self.in_docking_zone[bot_id] = True
                        self.STATE[bot_id] = "COMPLETE"
                        # Stop all motion
                        wheel_vel = [bot_id, 0.0, 0.0, 0.0, self.arm_pos[bot_id], self.solenoid[bot_id]]
                        self.publish_wheel_velocities(wheel_vel)
                        print(f"✓✓✓ Bot {bot_id} DOCKED! ✓✓✓")
    
    # =============================================================================
    # NAVIGATION CONTROL
    # =============================================================================
    
    def go_to_target(self, bot_id, target_x, target_y, target_theta, 
                     error_threshold, angle_threshold, use_target_theta, slow_down):
        """
        Low-level navigation control using PID
        
        Operates in two phases:
        1. Position control: Move to target (x, y)
        2. Orientation control: Rotate to target theta
        
        Args:
            bot_id: Bot to control
            target_x, target_y: Target position (mm)
            target_theta: Target orientation (radians)
            error_threshold: Distance threshold to consider position reached (mm)
            angle_threshold: Angle threshold to consider orientation reached (radians)
            use_target_theta: Whether to control orientation
            slow_down: Whether to reduce speed when close to target
        
        Returns:
            (distance, theta_error): Current error values
        """
        # Calculate position error
        error_x = target_x - self.bot_pose[bot_id][0]
        error_y = target_y - self.bot_pose[bot_id][1]
        
        # Calculate orientation error
        theta_robot = np.radians(self.bot_pose[bot_id][2])
        error_theta = target_theta - theta_robot
        # Normalize angle to [-pi, pi]
        error_theta = math.atan2(math.sin(error_theta), math.cos(error_theta))
        
        # Calculate distance to target
        dist = math.sqrt(error_x**2 + error_y**2)
        
        # === PHASE 1: POSITION CONTROL ===
        if dist > error_threshold:
            # Use PID to calculate velocities in global frame
            vx_global = self.pid_controllers[bot_id]['x'].compute(error_x, self.dt)
            vy_global = self.pid_controllers[bot_id]['y'].compute(error_y, self.dt)
            
            # Transform to robot local frame
            vx = vx_global * math.cos(theta_robot) + vy_global * math.sin(theta_robot)
            vy = -vx_global * math.sin(theta_robot) + vy_global * math.cos(theta_robot)
            vtheta = 0.0
            
            # Slow down when approaching target (if enabled)
            if slow_down and dist < (error_threshold + 50):
                vx = vx * 0.4
                vy = vy * 0.4
        
        # === PHASE 2: ORIENTATION CONTROL ===
        elif abs(error_theta) > angle_threshold and use_target_theta:
            # Stop translation, rotate in place
            vx = vy = 0.0
            
            # PID for rotation
            pid_output = self.pid_controllers[bot_id]['theta'].compute(error_theta, self.dt)
            
            # Enforce minimum rotation speed (prevent stalling)
            if pid_output >= 0:
                vtheta = max(pid_output, 25.0)
            else:
                vtheta = -max(abs(pid_output), 25.0)
        
        # === BOTH THRESHOLDS MET ===
        else:
            # Target reached - stop all motion
            vx = 0.0
            vy = 0.0
            vtheta = 0.0
        
        # === CONVERT TO WHEEL VELOCITIES ===
        # Use holonomic drive inverse kinematics
        vel = np.array([[vx], [vy], [vtheta]])
        s = np.dot(self.M_inv, vel)
        
        # Publish command
        wheel_vel = [bot_id, s[0][0], s[1][0], s[2][0], self.arm_pos[bot_id], self.solenoid[bot_id]]
        self.publish_wheel_velocities(wheel_vel)
        
        return dist, error_theta
    
    # =============================================================================
    # HARDWARE CONTROL - PICKUP & DROP
    # =============================================================================
    
    def pick_crate(self, bot_id):
        """
        Execute crate pickup sequence
        
        Hardware sequence:
        1. Lower arm (2.5 seconds)
        2. Activate solenoid
        3. Raise arm with crate
        """
        if self.pick_start_time[bot_id] is None:
            # Start pickup sequence
            self.pick_start_time[bot_id] = time.time()
            self.arm_pos[bot_id] = 80.0    # Lower arm
            self.solenoid[bot_id] = 1.0     # Activate solenoid
            wheel_vel = [bot_id, 0.0, 0.0, 0.0, self.arm_pos[bot_id], self.solenoid[bot_id]]
            self.publish_wheel_velocities(wheel_vel)
            return
        
        # Check if enough time has passed
        elapsed = time.time() - self.pick_start_time[bot_id]
        
        if elapsed >= self.wait_duration:
            # Pickup complete - raise arm
            self.arm_pos[bot_id] = 110.0
            self.publish_wheel_velocities([bot_id, 0.0, 0.0, 0.0, self.arm_pos[bot_id], self.solenoid[bot_id]])
            self.attach_success[bot_id] = True
            self.pick_start_time[bot_id] = None
    
    def place_crate(self, bot_id):
        """
        Execute crate placement sequence
        
        Hardware sequence:
        1. Lower arm (2.5 seconds)
        2. Deactivate solenoid
        3. Raise arm
        4. Back away slightly
        """
        if self.drop_start_time[bot_id] is None:
            # Start drop sequence
            self.drop_start_time[bot_id] = time.time()
            self.arm_pos[bot_id] = 80.0     # Lower arm
            self.solenoid[bot_id] = 0.0      # Release solenoid
            wheel_vel = [bot_id, 0.0, 0.0, 0.0, self.arm_pos[bot_id], self.solenoid[bot_id]]
            self.publish_wheel_velocities(wheel_vel)
            self.publish_wheel_velocities(wheel_vel)  # Send multiple times for reliability
            self.publish_wheel_velocities(wheel_vel)
            
            # Raise arm and back away
            self.arm_pos[bot_id] = 110.0
            self.solenoid[bot_id] = 0.0
            wheel_vel = [bot_id, -30.0, 30.0, 0.0, self.arm_pos[bot_id], self.solenoid[bot_id]]
            self.publish_wheel_velocities(wheel_vel)
            return
        
        # Check if enough time has passed
        elapsed = time.time() - self.drop_start_time[bot_id]
        
        if elapsed >= self.wait_duration:
            # Drop complete
            self.arm_pos[bot_id] = 110.0
            self.publish_wheel_velocities([bot_id, 0.0, 0.0, 0.0, self.arm_pos[bot_id], self.solenoid[bot_id]])
            self.detach_success[bot_id] = True
            self.drop_start_time[bot_id] = None
    
    # =============================================================================
    # ROS PUBLISHING
    # =============================================================================
    
    def publish_wheel_velocities(self, wheel_vel):
        """
        Publish velocity command to /bot_cmd topic
        
        Args:
            wheel_vel: [bot_id, m1, m2, m3, arm, solenoid]
        """
        msg = BotCmdArray()
        cmd = BotCmd()
        cmd.id = int(wheel_vel[0]) * 2  # Convert bot_id to ROS ID
        cmd.m1 = wheel_vel[1]
        cmd.m2 = wheel_vel[2]
        cmd.m3 = wheel_vel[3]
        cmd.base = wheel_vel[4]
        cmd.elbow = wheel_vel[5]
        msg.cmds.append(cmd)
        self.vel_pub.publish(msg)


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def main(args=None):
    """Initialize and run the controller node"""
    rclpy.init(args=args)
    controller = MultiHolonomicController()
    
    try:
        rclpy.spin(controller)
    except KeyboardInterrupt:
        print("\n" + "="*60)
        print("Controller shutting down...")
        print("="*60)
    finally:
        controller.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()