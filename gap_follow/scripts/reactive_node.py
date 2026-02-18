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
        self.bubble_radius = 0.5  # Radius of safety bubble in meters 
        self.preprocess_conv_size = 3 # Moving average window
        self.max_lidar_dist = 3.0    # Max reliable distance to consider
        self.max_speed = 4.0        # Max speed on straights
        self.min_speed = 1.5        # Min speed in sharp corners
        self.fov_angle = np.radians(135) # Only use front 135 degrees

    def preprocess_lidar(self, ranges):
        """ Preprocess the LiDAR scan array. Expert implementation includes:
            1.Setting each value to the mean over some window
            2.Rejecting high values (eg. > 3m)
        """
        # Convert to numpy array
        proc_ranges = np.array(ranges)

        # 1. Clip high values to a max reliable distance
        proc_ranges = np.clip(proc_ranges, 0, self.max_lidar_dist)

        # 2. Smooth data using a averaging each point with its neighbors. This prevents a single "glitchy" zero-reading from mistaken
        proc_ranges = np.convolve(proc_ranges, np.ones(self.preprocess_conv_size), 'same') / self.preprocess_conv_size

        # Replace 'inf' to 3m, 'nan' to 0. 
        proc_ranges = np.nan_to_num(proc_ranges, posinf=self.max_lidar_dist)

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
        ranges = np.array(data.ranges)              # 1. Convert ROS message data to a numpy array.
        proc_ranges = self.preprocess_lidar(ranges) # 2. Clean up data (smooth it, clip max distance).
        
        # 3. Find the closest obstacle to the car. 
        closest_point_idx = np.argmin(proc_ranges)
        min_dist = proc_ranges[closest_point_idx]
        
        # 4. Create the Safety Bubble.
        #    We calculate how wide (in array indices) the bubble needs to be to cover 'bubble_radius' meters at distance 'min_dist'.
        if min_dist > 0:
            angle_per_idx = data.angle_increment
            # Trigonometry: angle = arctan(radius / distance)
            bubble_angle = math.atan(self.bubble_radius / min_dist)
            bubble_idx_window = int(bubble_angle / angle_per_idx)
            
            # Determine start/end indices, ensuring we don't go out of bounds (0 or len).
            start_bubble = max(0, closest_point_idx - bubble_idx_window)
            end_bubble = min(len(proc_ranges), closest_point_idx + bubble_idx_window)
            
            # 5. "Zero out" the bubble.
            #    Any point inside the bubble is treated as an obstacle (distance 0), splitting the available space into separate gaps.
            proc_ranges[start_bubble:end_bubble] = 0.0

        # 6. Find the start/end of the largest consecutive sequence of non-zero points.
        start_i, end_i = self.find_max_gap(proc_ranges)

        # 7. Pick the best target within that gap (the furthest point).
        best_point_idx = self.find_best_point(start_i, end_i, proc_ranges)

        # 8. Convert the target index into a steering angle in radians.
        steering_angle = data.angle_min + (best_point_idx * data.angle_increment)

        # 9. Calculate Speed based on steering angle.   (tune❤️❤️)
        #    If steering is sharp (>20 deg), slow down. If straight, go fast.
        steering_abs = abs(steering_angle)
        if steering_abs > np.radians(20):
            speed = self.min_speed
        else:
            # Linearly increase speed as steering angle approaches 0.
            speed = self.max_speed - (steering_abs / np.radians(20)) * (self.max_speed - self.min_speed)
        
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