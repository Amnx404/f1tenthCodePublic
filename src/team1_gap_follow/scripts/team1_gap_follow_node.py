#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import numpy as np
from collections import deque
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped
from visualization_msgs.msg import MarkerArray, Marker
from geometry_msgs.msg import Point


class ReactiveFollowGap(Node):

    def __init__(self):
        super().__init__('reactive_node')

        # Declare and Load Parameters
        self._declare_params()
        
        self.scan_sub = self.create_subscription(LaserScan, '/scan', self.lidar_callback, 10)
        self.drive_pub = self.create_publisher(AckermannDriveStamped, '/drive', 10)
        self.viz_scan_pub = self.create_publisher(LaserScan, '/gap_viz_scan', 10)
        self.viz_gap_pub = self.create_publisher(MarkerArray, '/gap_viz', 10)

        # State Variables
        self.scan_history = deque(maxlen=self.get_parameter('history_size').value)
        self.prev_steer = 0.0
        self.prev_time = self.get_clock().now().nanoseconds / 1e9
        
        # PID State
        self.integral_error = 0.0
        self.prev_error = 0.0

    def _declare_params(self):
        """Declares tunable configuration numbers."""
        self.declare_parameter('fov_degrees', 220.0)
        self.declare_parameter('history_size', 5) 
        self.declare_parameter('max_range', 6.0)
        self.declare_parameter('car_width', 0.5)   
        self.declare_parameter('disparity_threshold', 0.3)
        self.declare_parameter('lookahead_distance', 3.0)
        
        # Speed params
        self.declare_parameter('max_speed', 4.0)
        self.declare_parameter('min_speed', 1.0)
        
        # PID Controller and Smoothing Params
        self.declare_parameter('kp', 1.0)
        self.declare_parameter('ki', 0.01)
        self.declare_parameter('kd', 0.05)
        self.declare_parameter('steer_smoothing', 0.2) 

        # Side Protection Params
        self.declare_parameter('side_safety_dist', 0.45) # Distance to trigger override
        self.declare_parameter('side_angle_window', 30.0) # Angular window width in degrees

    def lidar_callback(self, data):
        # 0. Fetch latest parameters dynamically
        fov = self.get_parameter('fov_degrees').value
        history_size = self.get_parameter('history_size').value
        max_rng = self.get_parameter('max_range').value
        car_width = self.get_parameter('car_width').value
        disp_thresh = self.get_parameter('disparity_threshold').value
        lookahead = self.get_parameter('lookahead_distance').value
        
        kp = self.get_parameter('kp').value
        ki = self.get_parameter('ki').value
        kd = self.get_parameter('kd').value
        alpha = self.get_parameter('steer_smoothing').value
        
        side_dist = self.get_parameter('side_safety_dist').value
        side_window = np.radians(self.get_parameter('side_angle_window').value)

        if self.scan_history.maxlen != history_size:
            self.scan_history = deque(maxlen=history_size)

        # 1. Preprocess & Temporal Rolling Mean
        ranges = np.array(data.ranges, dtype=float)
        ranges = np.nan_to_num(ranges, posinf=max_rng, nan=0.0)
        ranges = np.clip(ranges, 0.0, max_rng)
        
        self.scan_history.append(ranges)
        smoothed_ranges = np.mean(self.scan_history, axis=0)

        angle_min = data.angle_min
        angle_inc = data.angle_increment
        n = len(smoothed_ranges)
        center = n // 2

        # 2. Extract FOV
        half = int(np.radians(fov / 2.0) / angle_inc)
        lo = max(0, center - half)
        hi = min(n, center + half)
        scan = smoothed_ranges[lo:hi].copy()

        # 3. Disparity Extender
        for i in range(len(scan) - 1):
            diff = scan[i + 1] - scan[i]
            if abs(diff) > disp_thresh:
                closer = min(scan[i], scan[i + 1])
                if closer < 0.1: continue
                
                angle_to_cover = np.arctan2(car_width / 2.0, closer)
                w = int(np.ceil(angle_to_cover / angle_inc))
                
                if diff > 0:
                    scan[i + 1 : min(len(scan), i + 1 + w)] = 0.0
                else:
                    scan[max(0, i - w + 1) : i + 1] = 0.0

        # 4. Immediate Safety Bubble
        closest = int(np.argmin(scan))
        if scan[closest] < 0.5 and scan[closest] > 0.0:
            bw = int(np.ceil(np.arctan2(car_width / 2.0, scan[closest]) / angle_inc))
            scan[max(0, closest - bw): min(len(scan), closest + bw)] = 0.0

        # 5. Find Contiguous Gaps & Select Goal
        mask = scan > 0.1
        gaps = []
        start = None
        for i, val in enumerate(mask):
            if val and start is None:
                start = i
            elif not val and start is not None:
                gaps.append((start, i))
                start = None
        if start is not None:
            gaps.append((start, len(mask)))

        if gaps:
            best_gap = max(gaps, key=lambda g: float(np.max(scan[g[0]:g[1]])))
            gap_scan = scan[best_gap[0]:best_gap[1]]
            local_goal_idx = np.argmax(gap_scan)
            goal = best_gap[0] + local_goal_idx
        else:
            goal = len(scan) // 2
            best_gap = (goal, goal + 1)

        # 6. Lookahead Distance & Target Angle
        target_distance = min(scan[goal], lookahead)
        global_idx = goal + lo
        target_angle = angle_min + global_idx * angle_inc

        # 7. PID Controller for Steering
        current_time = self.get_clock().now().nanoseconds / 1e9
        dt = current_time - self.prev_time
        if dt <= 0.0: dt = 0.01
        
        error = target_angle
        self.integral_error += error * dt
        derivative = (error - self.prev_error) / dt
        
        pid_steer = (kp * error) + (ki * self.integral_error) + (kd * derivative)
        
        self.prev_error = error
        self.prev_time = current_time

        steering_angle = alpha * pid_steer + (1.0 - alpha) * self.prev_steer
        steering_angle = np.clip(steering_angle, -0.4, 0.4)

        # ---------------------------------------------------------
        # 8. SIDE STEERING PROTECTION
        # ---------------------------------------------------------
        # Calculate indices for exactly left (+90 deg) and right (-90 deg)
        left_idx = int((np.pi/2.0 - angle_min) / angle_inc)
        right_idx = int((-np.pi/2.0 - angle_min) / angle_inc)
        window_bins = int((side_window / 2.0) / angle_inc)

        # Safely extract side windows from the fully smoothed ranges
        left_window = smoothed_ranges[max(0, left_idx - window_bins) : min(n, left_idx + window_bins)]
        right_window = smoothed_ranges[max(0, right_idx - window_bins) : min(n, right_idx + window_bins)]

        min_left = np.min(left_window) if len(left_window) > 0 else max_rng
        min_right = np.min(right_window) if len(right_window) > 0 else max_rng

        protection_active = "NONE"

        # If PID wants to turn left, but left side is blocked
        if steering_angle > 0.0 and min_left < side_dist:
            steering_angle = 0.0  # Clamp to straight
            protection_active = "LEFT PROT"
            
        # If PID wants to turn right, but right side is blocked
        elif steering_angle < 0.0 and min_right < side_dist:
            steering_angle = 0.0  # Clamp to straight
            protection_active = "RIGHT PROT"

        self.prev_steer = steering_angle
        # ---------------------------------------------------------

        # 9. Speed Control
        max_spd = self.get_parameter('max_speed').value
        min_spd = self.get_parameter('min_speed').value
        abs_steer = abs(steering_angle)
        
        if abs_steer < np.radians(10):
            speed = max_spd
        elif abs_steer > np.radians(20):
            speed = min_spd
        else:
            t = (abs_steer - np.radians(10)) / np.radians(10)
            speed = max_spd - t * (max_spd - min_spd)

        # 10. Publish drive
        msg = AckermannDriveStamped()
        msg.drive.speed = float(speed)
        msg.drive.steering_angle = float(steering_angle)
        self.drive_pub.publish(msg)

        # 11. Visualization updates
        full = np.array(ranges, dtype=float)
        full[lo:hi] = scan
        self.viz_scan_pub.publish(self._make_scan(data, full))
        self._publish_markers(data, scan, lo, best_gap, goal, target_distance, steering_angle, target_angle, speed, protection_active)

    # ── Viz helpers ──────────────────────────────────────────────────────

    def _make_scan(self, data, ranges_full):
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

    def _publish_markers(self, data, scan, lo, gap, goal_idx, target_dist, steer, target_angle, speed, protection_status):
        a_min = data.angle_min
        a_inc = data.angle_increment
        stamp = self.get_clock().now().to_msg()
        fid = data.header.frame_id or 'laser'

        def to_xy(idx, r):
            a = a_min + idx * a_inc
            return float(r * np.cos(a)), float(r * np.sin(a))

        gs, ge = gap
        last = max(gs, ge - 1)
        x0, y0 = to_xy(lo + gs, float(scan[gs]) if gs < len(scan) else 0.0)
        x1, y1 = to_xy(lo + last, float(scan[last]) if last < len(scan) else 0.0)
        
        xg = float(target_dist * np.cos(target_angle))
        yg = float(target_dist * np.sin(target_angle))

        ma = MarkerArray()

        # 1. Gap line
        m = Marker()
        m.header.frame_id = fid
        m.header.stamp = stamp
        m.ns, m.id = 'gap', 0
        m.type = Marker.LINE_STRIP
        m.action = Marker.ADD
        m.scale.x = 0.1
        m.color.g, m.color.a = 1.0, 1.0
        m.points = [Point(x=x0, y=y0, z=0.0), Point(x=x1, y=y1, z=0.0)]
        ma.markers.append(m)

        # 2. Target Sphere
        m = Marker()
        m.header.frame_id = fid
        m.header.stamp = stamp
        m.ns, m.id = 'goal', 0
        m.type = Marker.SPHERE
        m.action = Marker.ADD
        m.pose.position.x, m.pose.position.y = xg, yg
        m.pose.orientation.w = 1.0
        m.scale.x = m.scale.y = m.scale.z = 0.4
        m.color.r, m.color.a = 1.0, 1.0
        ma.markers.append(m)

        # 3. Steering Arrow
        sl = 2.0
        sx, sy = sl * np.cos(steer), sl * np.sin(steer)
        m = Marker()
        m.header.frame_id = fid
        m.header.stamp = stamp
        m.ns, m.id = 'steer', 0
        m.type = Marker.ARROW
        m.action = Marker.ADD
        m.points = [Point(x=0.0, y=0.0, z=0.05), Point(x=sx, y=sy, z=0.05)]
        m.scale.x, m.scale.y = 0.1, 0.15
        
        # Color arrow RED if protection is overriding it, YELLOW if normal
        if protection_status != "NONE":
            m.color.r, m.color.g, m.color.b, m.color.a = 1.0, 0.0, 0.0, 1.0 
        else:
            m.color.r, m.color.g, m.color.b, m.color.a = 1.0, 0.85, 0.0, 1.0
        ma.markers.append(m)

        # 4. Text HUD
        m = Marker()
        m.header.frame_id = fid
        m.header.stamp = stamp
        m.ns, m.id = 'hud', 0
        m.type = Marker.TEXT_VIEW_FACING
        m.action = Marker.ADD
        m.pose.position.x, m.pose.position.z = -1.0, 0.5
        m.pose.orientation.w = 1.0
        m.scale.z = 0.25
        
        if protection_status != "NONE":
            m.color.r, m.color.g, m.color.b, m.color.a = 1.0, 0.2, 0.2, 1.0
            hud_text = f'Spd: {speed:.1f} m/s | Steer: {np.degrees(steer):.0f} deg\n[ {protection_status} ACTIVE ]'
        else:
            m.color.r = m.color.g = m.color.b = m.color.a = 1.0
            hud_text = f'Spd: {speed:.1f} m/s | Steer: {np.degrees(steer):.0f} deg'
            
        m.text = hud_text
        ma.markers.append(m)

        self.viz_gap_pub.publish(ma)


def main(args=None):
    rclpy.init(args=args)
    node = ReactiveFollowGap()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()