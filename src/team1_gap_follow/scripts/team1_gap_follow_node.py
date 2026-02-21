#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import numpy as np
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped
from visualization_msgs.msg import MarkerArray, Marker
from geometry_msgs.msg import Point


class ReactiveFollowGap(Node):

    def __init__(self):
        super().__init__('reactive_node')

        self.scan_sub = self.create_subscription(
            LaserScan,
            '/scan',
            self.lidar_callback,
            10
        )

        self.drive_pub = self.create_publisher(
            AckermannDriveStamped,
            '/drive',
            10
        )

        # Visualization publishers
        self.viz_scan_pub = self.create_publisher(
            LaserScan,
            '/gap_viz_scan',
            10
        )
        self.viz_gap_pub = self.create_publisher(
            MarkerArray,
            '/gap_viz',
            10
        )

        # Vehicle parameters
        self.max_speed = 4.0
        self.car_width = 0.38
        self.wheelbase = 0.33
        self.max_brake_accel = 3.0
        self.max_lat_accel = 4.0

        self.obstacle_threshold = 0.1  # treat anything closer than this as obstacle
        
        # Low-pass filter for smooth steering
        self.prev_steering = 0.2
        
        # Deadband parameters for straight-line smoothing
        self.deadband_threshold = 0.001  # radians (~5.73 degrees) - ignore small steering commands
        self.straight_confidence_counter = 0  # counter for how many consecutive small steering commands
        self.straight_confidence_threshold = 3  # need this many consecutive small steerings to engage deadband
        self.prev_raw_steering = 0.0  # store raw steering before deadband
        
        # Hysteresis for turning mode
        self.was_turning = False

    # ------------------------------------------------
    # Visualization helpers
    # ------------------------------------------------

    def _index_to_xy(self, index, range_val, angle_min, angle_increment):
        """Laser frame: angle 0 = forward (+x), left = +y."""
        angle = angle_min + index * angle_increment
        return (range_val * np.cos(angle), range_val * np.sin(angle))

    def _publish_scan_viz(self, data, ranges_full):
        """Publish a LaserScan with same metadata as data but ranges = ranges_full."""
        msg = LaserScan()
        msg.header = data.header
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.angle_min = data.angle_min
        msg.angle_max = data.angle_max
        msg.angle_increment = data.angle_increment
        msg.time_increment = data.time_increment
        msg.scan_time = data.scan_time
        msg.range_min = data.range_min
        msg.range_max = data.range_max
        msg.ranges = [float(r) for r in ranges_full]
        return msg

    def _publish_gap_markers(self, data, front, start, gap_start, gap_end, best, frame_id='laser', speed=None, steering_angle=None):
        """Publish gap segment (line), best point (sphere), target arrow, and optional speed/steering text."""
        angle_min = data.angle_min
        angle_inc = data.angle_increment
        stamp = self.get_clock().now().to_msg()

        # Indices and ranges in full scan
        idx_start = start + gap_start
        idx_end = start + gap_end
        idx_best = start + best
        r_start = float(front[gap_start]) if gap_start < len(front) else 0.0
        r_end = float(front[gap_end]) if gap_end < len(front) else 0.0
        r_best = float(front[best]) if best < len(front) else 0.0

        x0, y0 = self._index_to_xy(idx_start, r_start, angle_min, angle_inc)
        x1, y1 = self._index_to_xy(idx_end, r_end, angle_min, angle_inc)
        xb, yb = self._index_to_xy(idx_best, r_best, angle_min, angle_inc)

        ma = MarkerArray()
        # ----- Gap visualization -----
        # Gap wedge (filled triangle: origin -> gap_start -> gap_end) - semi-transparent green
        gap_wedge = Marker()
        gap_wedge.header.frame_id = frame_id
        gap_wedge.header.stamp = stamp
        gap_wedge.ns = 'gap'
        gap_wedge.id = 0
        gap_wedge.type = Marker.TRIANGLE_LIST
        gap_wedge.action = Marker.ADD
        gap_wedge.scale.x = 1.0
        gap_wedge.scale.y = 1.0
        gap_wedge.scale.z = 1.0
        gap_wedge.color.r = 0.0
        gap_wedge.color.g = 0.8
        gap_wedge.color.b = 0.2
        gap_wedge.color.a = 0.35
        # One triangle: origin, gap_start, gap_end
        gap_wedge.points = [
            Point(x=0.0, y=0.0, z=0.0),
            Point(x=x0, y=y0, z=0.0),
            Point(x=x1, y=y1, z=0.0),
        ]
        ma.markers.append(gap_wedge)
        # Gap boundary line (green)
        gap_line = Marker()
        gap_line.header.frame_id = frame_id
        gap_line.header.stamp = stamp
        gap_line.ns = 'gap'
        gap_line.id = 1
        gap_line.type = Marker.LINE_STRIP
        gap_line.action = Marker.ADD
        gap_line.scale.x = 0.12
        gap_line.color.r = 0.0
        gap_line.color.g = 1.0
        gap_line.color.b = 0.0
        gap_line.color.a = 1.0
        gap_line.points = [Point(x=x0, y=y0, z=0.0), Point(x=x1, y=y1, z=0.0)]
        ma.markers.append(gap_line)
        # Gap label at center of gap
        gap_center_x = (x0 + x1) / 2
        gap_center_y = (y0 + y1) / 2
        gap_text = Marker()
        gap_text.header.frame_id = frame_id
        gap_text.header.stamp = stamp
        gap_text.ns = 'gap'
        gap_text.id = 2
        gap_text.type = Marker.TEXT_VIEW_FACING
        gap_text.action = Marker.ADD
        gap_text.pose.position.x = gap_center_x
        gap_text.pose.position.y = gap_center_y
        gap_text.pose.position.z = 0.2
        gap_text.pose.orientation.w = 1.0
        gap_text.scale.z = 0.12
        gap_text.color.r = 0.0
        gap_text.color.g = 1.0
        gap_text.color.b = 0.0
        gap_text.color.a = 1.0
        gap_text.text = 'Gap'
        ma.markers.append(gap_text)
        # ----- Goal visualization -----
        # Goal point (red sphere)
        best_m = Marker()
        best_m.header.frame_id = frame_id
        best_m.header.stamp = stamp
        best_m.ns = 'goal'
        best_m.id = 0
        best_m.type = Marker.SPHERE
        best_m.action = Marker.ADD
        best_m.pose.position.x = xb
        best_m.pose.position.y = yb
        best_m.pose.position.z = 0.0
        best_m.pose.orientation.w = 1.0
        best_m.scale.x = best_m.scale.y = best_m.scale.z = 0.45
        best_m.color.r = 1.0
        best_m.color.g = 0.2
        best_m.color.b = 0.0
        best_m.color.a = 1.0
        ma.markers.append(best_m)
        # Goal label at best point
        goal_text = Marker()
        goal_text.header.frame_id = frame_id
        goal_text.header.stamp = stamp
        goal_text.ns = 'goal'
        goal_text.id = 1
        goal_text.type = Marker.TEXT_VIEW_FACING
        goal_text.action = Marker.ADD
        goal_text.pose.position.x = xb
        goal_text.pose.position.y = yb
        goal_text.pose.position.z = 0.35
        goal_text.pose.orientation.w = 1.0
        goal_text.scale.z = 0.14
        goal_text.color.r = 1.0
        goal_text.color.g = 0.9
        goal_text.color.b = 0.0
        goal_text.color.a = 1.0
        goal_text.text = 'Goal'
        ma.markers.append(goal_text)
        # Target direction arrow (origin -> best point)
        arrow_m = Marker()
        arrow_m.header.frame_id = frame_id
        arrow_m.header.stamp = stamp
        arrow_m.ns = 'hud'
        arrow_m.id = 0
        arrow_m.type = Marker.ARROW
        arrow_m.action = Marker.ADD
        arrow_m.points = [Point(x=0.0, y=0.0, z=0.0), Point(x=xb, y=yb, z=0.0)]
        arrow_m.scale.x = 0.08
        arrow_m.scale.y = 0.12
        arrow_m.color.r = 0.2
        arrow_m.color.g = 0.6
        arrow_m.color.b = 1.0
        arrow_m.color.a = 0.9
        ma.markers.append(arrow_m)
        # Speed/steering text (in front of robot for readability)
        if speed is not None and steering_angle is not None:
            text_m = Marker()
            text_m.header.frame_id = frame_id
            text_m.header.stamp = stamp
            text_m.ns = 'hud'
            text_m.id = 1
            text_m.type = Marker.TEXT_VIEW_FACING
            text_m.action = Marker.ADD
            text_m.pose.position.x = 0.5
            text_m.pose.position.y = 0.0
            text_m.pose.position.z = 0.3
            text_m.pose.orientation.w = 1.0
            text_m.scale.z = 0.15
            text_m.color.r = 1.0
            text_m.color.g = 1.0
            text_m.color.b = 1.0
            text_m.color.a = 1.0
            text_m.text = f'speed={speed:.2f} m/s  steer={np.degrees(steering_angle):.1f} deg'
            ma.markers.append(text_m)
        self.viz_gap_pub.publish(ma)

    # ------------------------------------------------

    def preprocess(self, ranges):
        ranges = np.array(ranges)
        ranges[np.isnan(ranges)] = 0.0
        ranges[np.isinf(ranges)] = 0.0
        ranges = np.clip(ranges, 0.0, 10.0)

        ranges = np.convolve(ranges, np.ones(5)/5, mode='same')

        return ranges

    # --------------------------------------------------
    
    def apply_steering_deadband(self, raw_steering):
        """
        Apply a deadband to steering for straight-line driving.
        Small steering angles get pulled toward zero to reduce oscillations.
        """
        abs_steering = abs(raw_steering)
        
        # Check if steering command is very small
        if abs_steering < self.deadband_threshold:
            self.straight_confidence_counter += 1
        else:
            self.straight_confidence_counter = 0
        
        # If we've had several consecutive small steering commands,
        # we're probably on a straight - pull to zero
        if self.straight_confidence_counter >= self.straight_confidence_threshold:
            # Progressive deadband: smooth transition to zero
            if abs_steering < self.deadband_threshold * 0.5:
                # Very small -> completely zero
                return 0.0
            else:
                # Scale down gradually
                scale_factor = (abs_steering - self.deadband_threshold * 0.5) / (self.deadband_threshold * 0.5)
                return np.sign(raw_steering) * abs_steering * max(0, scale_factor)
        else:
            # Not confident we're on a straight yet - pass through
            return raw_steering

    # --------------------------------------------------

    def inflate_obstacles(self, ranges, angle_increment):
        inflated = np.copy(ranges)

        for i in range(len(ranges)):

            d = ranges[i]

            if 0.05 < d < self.obstacle_threshold:

                # angular width needed for half car
                safety_angle = np.arctan(
                    (self.car_width / 2.0) / max(d, 0.01)
                )

                beams = int(safety_angle / angle_increment)

                start = max(0, i - beams)
                end = min(len(ranges)-1, i + beams)

                inflated[start:end+1] = 0.0

        return inflated

    # --------------------------------------------------

    def find_max_gap(self, ranges):

        max_len = 0
        max_start = 0
        max_end = 0

        curr_start = 0
        curr_len = 0

        for i in range(len(ranges)):
            if ranges[i] > 0.05:
                if curr_len == 0:
                    curr_start = i
                curr_len += 1
            else:
                if curr_len > max_len:
                    max_len = curr_len
                    max_start = curr_start
                    max_end = i - 1
                curr_len = 0

        if curr_len > max_len:
            max_start = curr_start
            max_end = len(ranges) - 1

        return max_start, max_end

    # --------------------------------------------------

    def lidar_callback(self, data):

        ranges = self.preprocess(data.ranges)

        # restrict to front 180°
        total = len(ranges)
        center = total // 2
        #fov = total * 3 // 16   # 135 degrees
        fov = total * 29 // 144   # 145 degrees

        start = center - fov
        end = center + fov

        front = ranges[start:end]

        # Inflate obstacles properly
        front = self.inflate_obstacles(front,
                                        data.angle_increment)

        # Find max gap
        gap_start, gap_end = self.find_max_gap(front)

        if gap_end <= gap_start:
            best = len(front)//2
        else:
            gap_ranges = front[gap_start:gap_end+1]

            indices = np.arange(gap_start, gap_end+1)

            # Add bias for farther gaps to encourage turning
            gap_center = (gap_start + gap_end) // 2
            center_idx = len(front) // 2
            
            # Calculate how far this gap is from center (0 to 1)
            off_center_factor = abs(gap_center - center_idx) / len(front)
            
            # Add turning bias - gaps far from center get extra weight
            turning_bias = 2.26 * off_center_factor
            
            weights = gap_ranges * turning_bias
            
            # ===== SAFETY CHECK FOR DIVISION BY ZERO =====
            weights_sum = np.sum(weights)
            if weights_sum > 1e-6:  # Small threshold to avoid division by zero
                weighted_index = np.sum(indices * weights) / weights_sum
            else:
                # Fallback to center of gap if all weights are near zero
                weighted_index = np.mean(indices)
            # ===== END OF SAFETY CHECK =====

            best = int(weighted_index)

        global_index = best + start

        # ----- Visualization: processed scan (gap markers published after speed/steering) -----
        ranges_full = np.array(self.preprocess(data.ranges), dtype=float)
        ranges_full[start:end] = front
        self.viz_scan_pub.publish(self._publish_scan_viz(data, ranges_full))

        raw_steering_angle = (
            data.angle_min +
            global_index * data.angle_increment
        )
        
        # Store raw steering before filtering
        self.prev_raw_steering = raw_steering_angle

        # ---------------- SPEED CALCULATION ----------------
        
        # Get minimum distance in front (for safety)
        min_dist_front = np.min(front[front > 0.05]) if np.any(front > 0.05) else 3.0
        
        # Check if we're approaching a corner
        center_idx = len(front) // 2
        gap_center = (gap_start + gap_end) // 2
        
        # Hysteresis for turning mode
        raw_turning = abs(gap_center - center_idx) > len(front) * 0.08
        if raw_turning:
            self.was_turning = True
        elif abs(gap_center - center_idx) < len(front) * 0.05:
            self.was_turning = False
        is_turning = self.was_turning
        
        # Look at the path we're about to take
        lookahead_idx = 20
        if raw_steering_angle > 0:  # Turning right
            path_slice = front[max(0, center_idx - lookahead_idx):center_idx]
        else:  # Turning left
            path_slice = front[center_idx:min(len(front), center_idx + lookahead_idx)]
        
        path_clearance = np.min(path_slice) if len(path_slice) > 0 and np.any(path_slice > 0.05) else min_dist_front
        
        # Progressive speed based on situation
        if is_turning and min_dist_front < 2.5:
            corner_speed = self.max_speed * 0.4
            turn_sharpness = abs(gap_center - center_idx) / len(front)
            corner_speed *= (1.0 - 0.3 * turn_sharpness)
            speed = max(corner_speed, 0.1)
        else:
            v_brake = min(self.max_speed, np.sqrt(2 * self.max_brake_accel * path_clearance) * 0.45)
            curvature = np.tan(raw_steering_angle) / self.wheelbase
            if abs(curvature) < 1e-4:
                v_curve = self.max_speed
            else:
                v_curve = np.sqrt(self.max_lat_accel / abs(curvature))
            speed = min(self.max_speed, max(v_brake, 0.1), v_curve)
        
        # Apply deadband and low-pass filter
        steering_with_deadband = self.apply_steering_deadband(raw_steering_angle)
        alpha = 0.87
        steering_angle = alpha * steering_with_deadband + (1 - alpha) * self.prev_steering
        self.prev_steering = steering_angle

        # Publish drive command
        msg = AckermannDriveStamped()
        msg.drive.speed = float(speed)
        msg.drive.steering_angle = float(steering_angle)
        self.drive_pub.publish(msg)

        # Visualization: gap + best point + arrow + speed/steering text
        frame_id = data.header.frame_id if data.header.frame_id else 'laser'
        self._publish_gap_markers(data, front, start, gap_start, gap_end, best, frame_id, speed=speed, steering_angle=steering_angle)


def main(args=None):
    rclpy.init(args=args)
    node = ReactiveFollowGap()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()