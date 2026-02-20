#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

import numpy as np
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped

class WallFollow(Node):
    """ 
    Implement Wall Following on the car
    """
    def __init__(self):
        super().__init__('wall_follow_node')

        lidarscan_topic = '/scan'
        drive_topic = '/drive'

        # TODO: create subscribers and publishers
        self.scan_sub = self.create_subscription(
            LaserScan,
            lidarscan_topic,
            self.scan_callback,
            10
	)
        self.drive_pub = self.create_publisher(
            AckermannDriveStamped,
            drive_topic,
            10
        )

        # TODO: set PID gains
        
        self.kp = 0.7
        self.kd = 0.2
        self.ki = 0.0

        # TODO: store history
        
        self.integral = 0.0
        self.prev_error = 0.0
        self.error = 0.0

        # TODO: store any necessary values you think you'll need
        self.desired_distance = 1.0
        self.lookahead_distance = 0.85
        self.theta = np.radians(52)
        self.prev_time = self.get_clock().now()
        

    def get_range(self, range_data, angle):
        """
        Simple helper to return the corresponding range measurement at a given angle. Make sure you take care of NaNs and infs.

        Args:
            range_data: single range array from the LiDAR
            angle: between angle_min and angle_max of the LiDAR

        Returns:
            range: range measurement in meters at the given angle

        """

        #TODO: implement
        angle_min = range_data.angle_min
        angle_increment = range_data.angle_increment
        index = int((angle - angle_min) / angle_increment)
        index = max(0, min(index, len(range_data.ranges) - 1))
        range_val = range_data.ranges[index]
        if np.isnan(range_val) or np.isinf(range_val):
            return 0.0
        return range_val


    def get_error(self, range_data, dist):
        """
        Calculates the error to the wall. Follow the wall to the left (going counter clockwise in the Levine loop). You potentially will need to use get_range()

        Args:
            range_data: single range array from the LiDAR
            dist: desired distance to the wall

        Returns:
            error: calculated error
        """

        #TODO:implement
        b = self.get_range(range_data, np.pi / 2)
        a = self.get_range(range_data, np.pi / 2 - self.theta)
        
        if a == 0.0 or b == 0.0:
            return self.prev_error
        
        numerator = a * np.cos(self.theta) - b
        denominator = a * np.sin(self.theta)
        
        if abs(denominator) < 0.0001:
            alpha = 0.0
        else:
            alpha = np.arctan2(numerator, denominator)
        
        D_t = b * np.cos(alpha)
        D_t_plus_1 = D_t + self.lookahead_distance * np.sin(alpha)

        ### ERROR Debugged 
        error = D_t_plus_1 - dist
        
        return error
        


    def pid_control(self, error, velocity):
        """
        Based on the calculated error, publish vehicle control

        Args:
            error: calculated error
            velocity: desired velocity

        Returns:
            None
        """
        
	# TODO: Use kp, ki & kd to implement a PID controller
        current_time = self.get_clock().now()
        dt = (current_time - self.prev_time).nanoseconds / 1e9
        
        if dt <= 0.0:
            dt = 0.01
        
        P = self.kp * error
        
        self.integral += error * dt
        self.integral = np.clip(self.integral, -100, 100)
        I = self.ki * self.integral
        
        derivative = (error - self.prev_error) / dt
        D = self.kd * derivative
        
        angle = P + I + D
        angle = np.clip(angle, -0.52, 0.52)
        
        self.prev_error = error
        self.prev_time = current_time
        
        drive_msg = AckermannDriveStamped()
        # TODO: fill in drive message and publish
        drive_msg.drive.steering_angle = angle
        drive_msg.drive.speed = velocity
        self.drive_pub.publish(drive_msg)
        

    def scan_callback(self, msg):
        """
        Callback function for LaserScan messages. Calculate the error and publish the drive message in this function.

        Args:
            msg: Incoming LaserScan message

        Returns:
            None
        """
        
        #error = 0.0 # TODO: replace with error calculated by get_error()
        #velocity = 0.0 # TODO: calculate desired car velocity based on error
        #self.pid_control(error, velocity) # TODO: actuate the car with PID
        
        #if not hasattr(self, '_lidar_printed'):
        #    print(f"LIDAR angle_min: {msg.angle_min:.3f} rad ({np.degrees(msg.angle_min):.1f}°)")
        #    print(f"LIDAR angle_max: {msg.angle_max:.3f} rad ({np.degrees(msg.angle_max):.1f}°)")
        #   print(f"LIDAR angle_increment: {msg.angle_increment:.5f} rad ({np.degrees(msg.angle_increment):.3f}°)")
        #   print(f"LIDAR num_ranges: {len(msg.ranges)}")
        #   self._lidar_printed = True

        
        # TODO: replace with error calculated by get_error()
        error = self.get_error(msg, self.desired_distance)
        
        #velocity = 0.0  TODO: calculate desired car velocity based on error
        steering_angle_estimate = abs(self.kp * error)
        if steering_angle_estimate < np.radians(10):
            velocity = 1.5
        elif steering_angle_estimate < np.radians(20):
            velocity = 1.0
        else:
            velocity = 0.5
        
        # DEBUG OUTPUT PRINTS
        if not hasattr(self, '_count'):
            self._count = 0
        self._count += 1
        if self._count % 50 == 0:
            print(f"Error: {error:.3f}m, Velocity: {velocity:.2f} m/s, Steering Est: {np.degrees(steering_angle_estimate):.1f}°")
        
        ###
        self.pid_control(error, velocity) # TODO: actuate the car with PID




def main(args=None):
    rclpy.init(args=args)
    print("WallFollow Initialized")
    wall_follow_node = WallFollow()
    rclpy.spin(wall_follow_node)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    wall_follow_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
