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
        self.bubble_radius = 0.68  # Radius of safety bubble in meters 
        self.preprocess_conv_size = 3 # Moving average window
        self.max_lidar_dist = 3.0    # Max reliable distance to consider
        self.max_speed = 5.0        # Max speed on straights  
        self.min_speed = 2.0        # Min speed in sharp corners
        self.fov_angle = np.radians(130) # Only use front 130 degrees
        self.prev_steering_angle = 0.0
        self.alpha = 0.2  # Smoothing factor (0.0 to 1.0). Lower = smoother but more lag.

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
        mask = free_space_ranges > 0.0
        
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
            
        # 7. Safety check: If no gaps found, return the full range (panic).
        if len(run_starts) == 0:
            return 0, len(free_space_ranges) - 1

        # Calculate length of each gap
        lengths = run_ends - run_starts
        # Pick the longest gap
        longest_gap_idx = np.argmax(lengths)
        
        return run_starts[longest_gap_idx], run_ends[longest_gap_idx]
    
    def find_best_point(self, start_i, end_i, ranges):
        """Start_i & end_i are start and end indicies of max-gap range, respectively
        Return index of best point in ranges
	    Naive: Choose the furthest point within ranges and go there
        !we find the CENTER of the max points!
        """
        # 1. Slice the full array to get only data inside identified max gap.
        gap = ranges[start_i:end_i]

        # 2. Find the maximum distance in this gap (likely 3.0m due to clipping)
        max_dist = np.max(gap)
        
        # 3. Find ALL indices where the distance equals the max_dist
        #    (e.g., if the gap is [2.9, 3.0, 3.0, 3.0, 2.8], this finds indices 1, 2, 3)
        max_indices = np.where(gap == max_dist)[0]
        
        # 4. Pick the middle index from these max points
        #    (e.g., from indices [1, 2, 3], we pick 2) -> centers the steering trajectory in the open space.
        current_max_idx = max_indices[len(max_indices) // 2]
        
        # 5. Convert local gap index back to global ranges index
        best_point_idx = start_i + current_max_idx
        
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

        # 3. Disparity Extender (Find the closest obstacle to the car. 
        closest_point_idx = np.argmin(proc_ranges)
        min_dist = proc_ranges[closest_point_idx]
        
        # --- SAFETY BUBBLE ---
        # Essential for avoiding flat walls where no disparities exist
        if min_dist < self.bubble_radius:
            bubble_angle = math.atan(self.bubble_radius / (min_dist + 0.001)) # +0.001 prevents div by 0
            bubble_idx_window = int(bubble_angle / angle_increment)
            
            start_bubble = max(0, closest_point_idx - bubble_idx_window)
            end_bubble = min(len(proc_ranges), closest_point_idx + bubble_idx_window)
            proc_ranges[start_bubble:end_bubble] = 0.0

        # Tune parameters ❤️❤️
        car_width = 0.55 # The width to extend the obstacle (0.5m ~ half car width)
        disparity_threshold = 0.3  # Minimum jump in distance to consider it a disparity 

        # Vectorized detection: Find differences between adjacent elements
        # diffs[i] = proc_ranges[i+1] - proc_ranges[i]
        proc_ranges_copy = proc_ranges.copy()
        diffs = np.diff(proc_ranges_copy)

        # Get indices where the jump is larger than threshold, returns an array of indices [i, j, k...] where disparities exist
        disparity_indices = np.where(np.abs(diffs) > disparity_threshold)[0]

        # Iterate ONLY over the disparities (usually < 10 points), not the whole array (faster)
        for i in disparity_indices:
            depth_curr = proc_ranges_copy[i]
            depth_next = proc_ranges_copy[i+1]

            # If either point was already zeroed out by the safety bubble, skip it
            # (optional optimization, keeps logic clean)
            if depth_curr == 0.0 or depth_next == 0.0:
                 continue
            
            # Determine closer point to calculate extension angle
            min_depth = min(depth_curr, depth_next)
            
            # Avoid division by zero
            if min_depth < 0.05:
                min_depth = 0.05

            # Calculate how wide (in indices) to extend the safety zero-out
            angle_width = math.atan(car_width / (min_depth + 0.001)) # +0.001 to prevent div by zero
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