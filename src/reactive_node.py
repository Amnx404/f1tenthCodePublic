#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import numpy as np
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped


class ReactiveFollowGap(Node):

    def __init__(self):
        super().__init__('reactive_node')

        self.lidar_sub = self.create_subscription(
            LaserScan, '/scan', self.lidar_callback, 10)

        self.drive_pub = self.create_publisher(
            AckermannDriveStamped, '/drive', 10)

        # ============================
        # F1TENTH vehicle parameters
        # ============================
        self.car_width = 0.31          # meters
        self.car_length = 0.58         # meters
        self.wheelbase = 0.15875 + 0.17145  # front + rear axle distances
        self.max_speed = 18.0           # m/s (reduced from 20.0)
        self.min_speed = 1.8            # m/s (increased min speed for better control)
        self.a_lat_max = 9.51           # max lateral acceleration

        # Additional reactive parameters
        self.clearance = 0.20
        self.safety_radius = 0.35
        self.max_lidar_dist = 6.0
        self.disparity_threshold = 0.7

        # Safety distance thresholds
        self.critical_distance = 0.5     
        self.caution_distance = 1.8      
        self.slow_down_factor = 0.4      

        # Low-pass filter
        self.prev_steering = 0.0
        self.prev_speed = 0.0
        
        # State tracking
        self.turning_state = 0
        self.straight_count = 0

        self.get_logger().info("Reactive Gap Node - Fixed narrow leaks & head-on collisions")

    # =========================
    # LIDAR PREPROCESS
    # =========================
    def preprocess_lidar(self, ranges):
        ranges = np.array(ranges)
        total = len(ranges)
        start = int(total * 0.25)
        end = int(total * 0.75)
        ranges = ranges[start:end]

        ranges[np.isinf(ranges)] = self.max_lidar_dist
        ranges = np.clip(ranges, 0.0, self.max_lidar_dist)

        kernel = np.ones(3) / 3
        ranges = np.convolve(ranges, kernel, mode='same')

        return ranges

    # =========================
    # DISPARITY EXTENDER
    # =========================
    def apply_disparity_extender(self, ranges):
        new_ranges = np.copy(ranges)
        angle_increment = np.pi / len(ranges)

        for i in range(len(ranges) - 1):
            diff = ranges[i] - ranges[i + 1]
            if abs(diff) > self.disparity_threshold:
                closer = i if ranges[i] < ranges[i + 1] else i + 1
                d = ranges[closer]
                extend_angle = np.arctan((self.car_width / 2 + self.clearance) / max(d, 1e-3))
                extend_beams = int(extend_angle / angle_increment)
                for j in range(-extend_beams, extend_beams):
                    idx = closer + j
                    if 0 <= idx < len(ranges):
                        new_ranges[idx] = min(new_ranges[idx], d)
        return new_ranges

    # =========================
    # SAFETY BUBBLE
    # =========================
    def apply_safety_bubble(self, ranges):
        closest = np.argmin(ranges)
        d = ranges[closest]
        d = max(d, 1e-3)

        angle_increment = np.pi / len(ranges)
        bubble_angle = np.arctan(self.safety_radius / d)
        bubble = int(bubble_angle / angle_increment)

        start = max(0, closest - bubble)
        end = min(len(ranges) - 1, closest + bubble)
        ranges[start:end] = 0.0
        return ranges

    # =========================
    # FIND BEST GAP - FIXED FOR NARROW LEAKS
    # =========================
    def find_best_gap(self, ranges):
        angle_increment = np.pi / len(ranges)
        best_score = -1
        best_start = 0
        best_end = 0
        
        # Track closest gap for emergency
        closest_gap_dist = float('inf')
        closest_gap_center = len(ranges) // 2
        
        i = 0
        while i < len(ranges):
            if ranges[i] > 0:
                start = i
                while i < len(ranges) and ranges[i] > 0:
                    i += 1
                end = i
                gap_size = end - start
                
                if gap_size > 5:
                    min_dist = np.min(ranges[start:end])
                    
                    # Track closest gap
                    if min_dist < closest_gap_dist:
                        closest_gap_dist = min_dist
                        closest_gap_center = (start + end) // 2
                    
                    # Calculate width more accurately - FIX FOR NARROW LEAKS
                    delta_theta = gap_size * angle_increment
                    
                    # Check if gap is consistently wide enough, not just at min point
                    # This prevents narrow scan leaks
                    
                    # Sample multiple points in the gap to check width
                    sample_points = 3
                    width_sufficient = True
                    
                    for s in range(sample_points):
                        sample_idx = start + int((gap_size * s) / sample_points)
                        if sample_idx >= end:
                            break
                        d_sample = ranges[sample_idx]
                        if d_sample <= 0:
                            continue
                            
                        # Calculate width at this distance
                        width_at_point = 2 * d_sample * np.tan(delta_theta / 2)
                        required_width = self.car_width + self.clearance
                        
                        if width_at_point < required_width * 0.9:  # 10% margin
                            width_sufficient = False
                            break
                    
                    if width_sufficient:
                        # Calculate width for scoring using average of multiple points
                        avg_dist = np.mean(ranges[start:end])
                        width = 2 * avg_dist * np.tan(delta_theta / 2)
                        
                        # Center penalty
                        center_dist_penalty = abs((start + end)//2 - len(ranges)//2) / len(ranges)
                        
                        # Score formula
                        score = min_dist * width * (1.0 - 0.3 * center_dist_penalty)
                        
                        if score > best_score:
                            best_score = score
                            best_start = start
                            best_end = end
            else:
                i += 1
                
        # If no valid gap found, use closest gap
        if best_score < 0 and closest_gap_dist < float('inf'):
            self.get_logger().warn(f"No wide enough gap! Using closest at {closest_gap_dist:.2f}m")
            gap_width = int(0.5 / angle_increment)
            best_start = max(0, closest_gap_center - gap_width//2)
            best_end = min(len(ranges), closest_gap_center + gap_width//2)
            
        return best_start, best_end

    # =========================
    # TURN DETECTION
    # =========================
    def detect_turn(self, ranges, steering):
        left_dist = np.min(ranges[:40]) if len(ranges) > 40 else np.min(ranges)
        right_dist = np.min(ranges[-40:]) if len(ranges) > 40 else np.min(ranges)
        
        if left_dist < 1.5 or right_dist < 1.5:
            if left_dist < right_dist:
                return -1
            else:
                return 1
        
        if abs(steering) > 0.15:
            return 1 if steering > 0 else -1
            
        return 0

    # =========================
    # HEAD-ON COLLISION AVOIDANCE - NEW FUNCTION
    # =========================
    def avoid_head_on_collision(self, ranges, steering):
        """
        Check for obstacles directly ahead and adjust steering to avoid head-on collision
        """
        center_idx = len(ranges) // 2
        ahead_range = 20  # Check 20 beams ahead
        
        # Check directly ahead
        ahead_slice = ranges[center_idx - ahead_range//2:center_idx + ahead_range//2]
        min_ahead = np.min(ahead_slice) if len(ahead_slice) > 0 else self.max_lidar_dist
        
        # If obstacle detected directly ahead
        if min_ahead < 1.2:  # Less than 1.2 meters ahead
            self.get_logger().warn(f"Head-on obstacle at {min_ahead:.2f}m! Adjusting course")
            
            # Look for clear space on left and right
            left_slice = ranges[center_idx - ahead_range:center_idx]
            right_slice = ranges[center_idx:center_idx + ahead_range]
            
            left_min = np.min(left_slice) if len(left_slice) > 0 else 0
            right_min = np.min(right_slice) if len(right_slice) > 0 else 0
            
            # Steer towards clearer side
            if left_min > right_min and left_min > 1.0:
                return -0.25  # Steer left
            elif right_min > left_min and right_min > 1.0:
                return 0.25   # Steer right
            else:
                # Both sides tight - slow down and continue current steering
                return steering * 0.8
        
        return steering

    # =========================
    # SPEED CONTROL - SLIGHTLY SLOWER
    # =========================
    def calculate_speed(self, ranges, steering, is_turning):
        center_idx = len(ranges) // 2
        look_ahead = 30
        
        if steering > 0:
            path_slice = ranges[center_idx - look_ahead:center_idx]
        else:
            path_slice = ranges[center_idx:center_idx + look_ahead]
            
        path_dist = np.min(path_slice) if len(path_slice) > 0 else self.max_lidar_dist
        min_dist = np.min(ranges[ranges > 0]) if np.any(ranges > 0) else self.max_lidar_dist
        
        # REDUCED SPEEDS - SLIGHTLY SLOWER
        if is_turning == 0:  # STRAIGHT
            if min_dist > 2.0:
                base_speed = self.max_speed * 0.7  # Reduced from 0.8
            else:
                base_speed = self.max_speed * 0.5  # Reduced from 0.6
                
        elif abs(steering) < 0.2:  # GENTLE TURN
            base_speed = self.max_speed * 0.4  # Reduced from 0.5
            
        else:  # SHARP TURN
            turn_sharpness = min(1.0, abs(steering) * 2)
            base_speed = self.max_speed * (0.3 - 0.1 * turn_sharpness)  # Reduced
        
        # Distance-based factors
        if min_dist < 0.8:
            speed_factor = max(0.25, min_dist / 2.0)  # More conservative
        elif min_dist < 1.5:
            speed_factor = max(0.4, min_dist / 3.0)   # More conservative
        elif min_dist < 2.5:
            speed_factor = max(0.6, min_dist / 4.0)   # More conservative
        else:
            speed_factor = 0.9
            
        path_factor = min(1.0, path_dist / 2.0)  # Reduced from 1.5
        
        speed = base_speed * speed_factor * path_factor
        speed = max(speed, self.min_speed)
        
        return speed

    # =========================
    # LOOK-AHEAD STEERING
    # =========================
    def calculate_steering_with_lookahead(self, ranges, current_steering):
        center_idx = len(ranges) // 2
        look_ahead = 15
        
        if current_steering > 0:
            look_idx = max(0, center_idx - look_ahead)
            future_dist = np.min(ranges[look_idx:center_idx]) if look_idx < center_idx else ranges[center_idx]
        else:
            look_idx = min(len(ranges)-1, center_idx + look_ahead)
            future_dist = np.min(ranges[center_idx:look_idx]) if center_idx < look_idx else ranges[center_idx]
        
        if future_dist < 1.0:
            if current_steering > 0:
                return current_steering - 0.05
            else:
                return current_steering + 0.05
                
        return current_steering

    # =========================
    # MAIN CALLBACK
    # =========================
    def lidar_callback(self, data):
        # Process LiDAR data
        ranges = self.preprocess_lidar(data.ranges)
        ranges = self.apply_disparity_extender(ranges)
        ranges = self.apply_safety_bubble(ranges)

        # Find best gap (with narrow leak fix)
        start, end = self.find_best_gap(ranges)

        # Calculate steering
        if end <= start:
            best_index = len(ranges) // 2
        else:
            gap_distances = ranges[start:end]
            weights = gap_distances**2 / (np.sum(gap_distances**2) + 1e-3)
            best_index = int(np.sum(np.arange(start, end) * weights))

        # Convert to steering angle
        fov = np.pi * 0.5
        angle_offset = -fov / 2
        steering = (best_index / len(ranges)) * fov + angle_offset

        # Apply look-ahead steering adjustment
        steering = self.calculate_steering_with_lookahead(ranges, steering)
        
        # HEAD-ON COLLISION AVOIDANCE - OVERRIDE IF NEEDED
        steering = self.avoid_head_on_collision(ranges, steering)

        # Detect if we're in a turn
        is_turning = self.detect_turn(ranges, steering)
        
        # Update turning state
        alpha_turn = 0.1
        self.turning_state = alpha_turn * is_turning + (1 - alpha_turn) * self.turning_state

        # Calculate adaptive speed (slightly slower)
        speed = self.calculate_speed(ranges, steering, abs(self.turning_state) > 0.3)

        # Quick side check
        left_dist = np.min(ranges[:30]) if len(ranges) > 30 else np.min(ranges)
        right_dist = np.min(ranges[-30:]) if len(ranges) > 30 else np.min(ranges)
        
        # Emergency brake
        if abs(steering) > 0.1:
            if (steering > 0 and right_dist < 0.4) or (steering < 0 and left_dist < 0.4):
                speed = min(speed, self.min_speed)
                self.get_logger().warn(f"Emergency slow: L={left_dist:.2f}, R={right_dist:.2f}")

        # Apply low-pass filter
        if abs(steering) < 0.1:
            alpha_steer = 0.1
            alpha_speed = 0.05
        else:
            alpha_steer = 0.3
            alpha_speed = 0.1
        
        steering = alpha_steer * steering + (1 - alpha_steer) * self.prev_steering
        speed = alpha_speed * speed + (1 - alpha_speed) * self.prev_speed
        
        self.prev_steering = steering
        self.prev_speed = speed

        # Publish command
        msg = AckermannDriveStamped()
        msg.drive.steering_angle = steering
        msg.drive.speed = speed
        self.drive_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = ReactiveFollowGap()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()