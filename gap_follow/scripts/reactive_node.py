#!/usr/bin/env python3
import rclpy
from rclpy.node import Node

import math
import numpy as np
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped, AckermannDrive

class ReactiveFollowGap(Node):
    """ 
    Implement Wall Following on the car
    This is just a template, you are free to implement your own node!
    """
    def __init__(self):
        super().__init__('reactive_node')
        # Topics & Subs, Pubs
        lidarscan_topic = '/scan'
        drive_topic = '/drive'

        # TODO: Subscribe to LIDAR
        self.sub_scan = self.create_subscription(LaserScan, lidarscan_topic, self.lidar_callback, 10)
        # TODO: Publish to drive
        self.pub_drive = self.create_publisher(AckermannDriveStamped, drive_topic, 10)
        
        # Parameters (Tune ❤️❤️❤️)
        self.bubble_radius = 0.7  # Radius of safety bubble in meters    0.7 (critical for L shape trap)
        self.preprocess_conv_size = 3 # Moving average window
        self.max_lidar_dist = 3.5   # Max reliable distance to consider 3.5 
        self.max_speed = 3.0        # Max speed on straights  
        self.min_speed = 0.5        # Min speed in sharp corners
        self.fov_angle = np.radians(160) # Only use front 130 degrees
        self.prev_steering_angle = 0.0
        self.alpha = 0.2            # Smoothing factor (0.0 to 1.0). Lower = smoother but more lag.
        self.car_width = 0.35        # The width to extend the obstacle (0.5m ~ half car width)
        self.disparity_threshold = 0.3  # Minimum jump in distance to consider it a disparity 

    def preprocess_lidar(self, ranges):
        """ Preprocess the LiDAR scan array. Expert implementation includes:
            1.Setting each value to the mean over some window
            2.Rejecting high values (eg. > 3m)
        """
        # Convert to numpy array
        proc_ranges = np.array(ranges)
        
        # Replace 'inf' to 3m, 'nan' to 0. 
        proc_ranges = np.nan_to_num(proc_ranges, posinf=self.max_lidar_dist, nan=0.0)
        
        # 1. Clip high values to a max reliable distance
        proc_ranges = np.clip(proc_ranges, 0, self.max_lidar_dist)

        # 2. Smooth data using a averaging each point with its neighbors. This prevents a single "glitchy" zero-reading from mistaken
        proc_ranges = np.convolve(proc_ranges, np.ones(self.preprocess_conv_size), 'same') / self.preprocess_conv_size

        return proc_ranges

    def find_max_gap(self, free_space_ranges):
        """ Return the start index & end index of the max gap in free_space_ranges
        """
        # 1. Create a boolean mask: True where distance is non-zero , False where obstacles (0.0) are.
        # Treat very small values as 0 to filter out noise
        mask = free_space_ranges > 0.1  
        
        # 2. Convert mask to int (0/1) and take difference between neighbors.
        #    1->0 becomes -1 (gap ended). 0->1 becomes 1 (gap started).
        dmask = np.diff(mask.astype(int))

        # 3. Find indices where gaps start (value went from 0 to 1). [add +1 because diff shifts index by one.]
        run_starts = np.where(dmask == 1)[0] + 1
        # 4. Find indices where gaps end (value went from 1 to 0).
        run_ends = np.where(dmask == -1)[0] + 1

        # 5. Handle edge case: If first element is valid, the diff won't catch it. 
        #    Manually insert 0 as a start index.
        if mask[0]:
            run_starts = np.insert(run_starts, 0, 0)

        # 6. Handle edge case: If last element is valid, the diff won't catch the end.
        #    Manually append the array length as an end index.
        if mask[-1]:
            run_ends = np.append(run_ends, len(free_space_ranges))
            
        # 7. SAFETY FALLBACK: If the disparity extender closes ALL gaps
        if len(run_starts) == 0:
            # Panic mode: Find the single furthest point available and try to squeeze toward it
            best_idx = np.argmax(free_space_ranges)
            return best_idx, best_idx + 1

        gap_max_depths = []
        gap_widths = []

        # 8. Analyze every valid gap
        for s, e in zip(run_starts, run_ends):
            gap_max_depths.append(np.max(free_space_ranges[s:e]))
            gap_widths.append(e - s)
            
        gap_max_depths = np.array(gap_max_depths)
        gap_widths = np.array(gap_widths)
        
        # 1. Find the absolute maximum depth available across all gaps
        global_max_depth = np.max(gap_max_depths)
        
        # 2. Filter down to ONLY the gaps that reach this depth
        deepest_gap_indices = np.where(gap_max_depths == global_max_depth)[0]
        
        # 3. If there are multiple deep gaps, choose the widest one to prevent hitting walls
        best_gap_idx = deepest_gap_indices[np.argmax(gap_widths[deepest_gap_indices])]
        
        return run_starts[best_gap_idx], run_ends[best_gap_idx]
    
    def find_best_point(self, start_i, end_i, ranges):
        """Start_i & end_i are start and end indicies of max-gap range, respectively
        Return index of best point in ranges
	    Naive: Choose the furthest point within ranges and go there
        !we find the CENTER of the max points!
        """
        # ======================= Ver 1.0: Absolute Furthest Point (Naive) ==========================
        # ============= With this logic, the car gets stuck in L-shape traps and tight corners ======
        # ============= because it always picks the absolute furthest point,            =============
        # ============= which is often right at the wall vertex.                        =============

        # # 1. Slice the full array to get only data inside identified max gap.
        # gap = ranges[start_i:end_i]

        # # 2. Find the maximum distance in this gap (likely 3.0m due to clipping)
        # max_dist = np.max(gap)
        
        # # 3. Find ALL indices where the distance equals the max_dist
        # #    (e.g., if the gap is [2.9, 3.0, 3.0, 3.0, 2.8], this finds indices 1, 2, 3)
        # max_indices = np.where(gap == max_dist)[0]
        
        # # 4. Pick the middle index from these max points
        # #    (e.g., from indices [1, 2, 3], we pick 2) -> centers the steering trajectory in the open space.
        # current_max_idx = max_indices[len(max_indices) // 2]
        
        # # 5. Convert local gap index back to global ranges index
        # best_point_idx = start_i + current_max_idx
        # ====================================== Ver 1.0 END =========================================

        # To SWAP: Just comment out the other Version ❤️❤️❤️

        # ======================= Ver 2.0: Deepest Point with Center Bias (Expert) ===========================
        # =========== This logic reduces the L-shape trap & tight corner issues by blending two strategies ===
        # =========== But fails the 3 rectangluar obstacle and later ellipse obstacle ========================

        gap = ranges[start_i:end_i]
        
        # Safety catch
        if len(gap) == 0:
            return start_i

        # Find the max depth in this gap
        max_dist = np.max(gap)
        
        # Get all indices that are at (or very close to) the max depth
        # The -0.1 tolerance groups the deep region together
        max_indices_local = np.where(gap >= max_dist - 0.1)[0]
        
        # Convert local gap indices back to the slice indices
        max_indices_global = start_i + max_indices_local
        
        # --- 1. FIX THE L-SHAPE TRAP (CENTER BIAS) ---
        # Instead of picking an arbitrary deep point, find the deep point 
        # that requires the LEAST steering.
        # 'len(ranges) // 2' is exactly straight ahead of the car.
        straight_ahead_idx = len(ranges) // 2
        
        # Find which of our deep points is closest to straight ahead
        distances_from_center = np.abs(max_indices_global - straight_ahead_idx)
        best_deep_idx = max_indices_global[np.argmin(distances_from_center)]
        
        # --- 2. FIX THE TIGHT CORNERS (APEX REPULSION) ---
        # The spatial center of the gap is physically the furthest point from both walls.
        # By itself it causes wiggling, but blended, it's a great safety buffer.
        spatial_gap_center = start_i + (len(gap) // 2)
        
        # Blend them: 70% deep point (for stability), 30% spatial center (to push wide)
        # This acts like a magnet pushing the car away from the inner wall vertex.
        best_point_idx = int((best_deep_idx * 0.7) + (spatial_gap_center * 0.3))

        # ======================================== Ver 2.0 END ================================================


        return best_point_idx

    def lidar_callback(self, data):
        """ Process each LiDAR scan as per the Follow Gap algorithm & publish an AckermannDriveStamped Message
        """
        ranges = np.array(data.ranges) # 1. Convert ROS message data to a numpy array.
        
        # 2: FOV Slicing - Calculate indices for the Field of View
        # Assume 0 index is angle_min. We want to center around 0 angle.
        angle_increment = data.angle_increment
        angle_min = data.angle_min
        
        # Calculate index range for our FOV
        fov_min_idx = int(((-self.fov_angle / 2) - angle_min) / angle_increment)
        fov_max_idx = int(((self.fov_angle / 2) - angle_min) / angle_increment)
        
        # Clamp indices to array bounds
        fov_min_idx = max(0, fov_min_idx)
        fov_max_idx = min(len(ranges), fov_max_idx)
        
        # Preprocess only the sliced data
        sliced_ranges = ranges[fov_min_idx:fov_max_idx]
        proc_ranges = self.preprocess_lidar(sliced_ranges)
        # KEEP a clean copy for reading true depths BEFORE any zeroing happens
        proc_ranges_copy = proc_ranges.copy()

        # 3. A. Disparity Extender (Find the closest obstacle to the car. 
        
            # Vectorized detection: Find differences between adjacent elements
            # diffs[i] = proc_ranges[i+1] - proc_ranges[i]
        diffs = np.diff(proc_ranges_copy)

            # Get indices where the jump is larger than threshold, returns an array of indices [i, j, k...] where disparities exist
        disparity_indices = np.where(np.abs(diffs) > self.disparity_threshold)[0]

        # Iterate ONLY over the disparities (usually < 10 points), not the whole array (faster)
        for i in disparity_indices:
            depth_curr = proc_ranges_copy[i]
            depth_next = proc_ranges_copy[i+1]
            
            # Determine closer point to calculate extension angle
            min_depth = min(depth_curr, depth_next)
            
            # Avoid division by zero
            if min_depth < 0.05:
                min_depth = 0.05

            # Calculate how wide (in indices) to extend the safety zero-out
            angle_width = math.atan(self.car_width / (min_depth + 0.001)) # +0.001 to prevent div by zero
            idx_width = int(angle_width / angle_increment)

            # Extend zeros from the closer edge onto the further edge
            if depth_curr < depth_next:
                # Current is close (obstacle), Next is deep (gap).
                # We need to eat into the gap on the RIGHT (next indices)
                end_idx = min(len(proc_ranges), i + 1 + idx_width)
                proc_ranges[i+1 : end_idx] = 0.0
            else:
                # Current is deep (gap), Next is close (obstacle).
                # We need to eat into the gap on the LEFT (previous indices)
                start_idx = max(0, i - idx_width)
                proc_ranges[start_idx : i+1] = 0.0

        # B. --- SAFETY BUBBLE (Run this SECOND) ---
        # Essential for avoiding flat walls where no disparities exist
        closest_point_idx = np.argmin(proc_ranges_copy)
        min_dist = proc_ranges_copy[closest_point_idx]

        if min_dist < self.bubble_radius:
            bubble_angle = math.atan(self.bubble_radius / (min_dist + 0.001)) # +0.001 prevents div by 0
            bubble_idx_window = int(bubble_angle / angle_increment)
            
            start_bubble = max(0, closest_point_idx - bubble_idx_window)
            end_bubble = min(len(proc_ranges), closest_point_idx + bubble_idx_window)
            proc_ranges[start_bubble:end_bubble] = 0.0


        # 4. Find the start/end of the largest consecutive sequence of non-zero points.
        start_i, end_i = self.find_max_gap(proc_ranges)

        # 5. Pick the best target within that gap (the furthest point). Add back the offset from slicing to get global index in original ranges array.
        best_point_idx = self.find_best_point(start_i, end_i, proc_ranges) + fov_min_idx  

        # 6. Convert the target index into a steering angle in radians.
        steering_angle = angle_min + (best_point_idx * angle_increment)
        
        # Optional: Add Smoothing (Exponential Weighted Moving Average) to reduce wheel jitter
        self.prev_steering_angle = (self.alpha * steering_angle) + ((1.0 - self.alpha) * self.prev_steering_angle) 
        steering_angle = self.prev_steering_angle

        # 7. Calculate Speed based on steering angle.   (tune❤️❤️)
        # If steering < 10 degrees, go max speed. Only slow down for sharp turns.
        steering_abs = abs(steering_angle)
        if steering_abs < np.radians(10):
            speed = self.max_speed
        elif steering_abs > np.radians(20):
            speed = self.min_speed
        else:
            # Linear drop only between 10 and 20 degrees
            ratio = (steering_abs - np.radians(10)) / np.radians(10)
            speed = self.max_speed - (ratio * (self.max_speed - self.min_speed))

        # 10. Publish the drive command to the car.
        drive_msg = AckermannDriveStamped()
        drive_msg.drive.speed = float(speed)
        drive_msg.drive.steering_angle = float(steering_angle)
        self.pub_drive.publish(drive_msg)


def main(args=None):
    rclpy.init(args=args)
    print("WallFollow Initialized")
    reactive_node = ReactiveFollowGap()
    rclpy.spin(reactive_node)

    reactive_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()