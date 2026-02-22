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

        self._declare_params()
        
        self.scan_sub = self.create_subscription(LaserScan, '/scan', self.lidar_callback, 10)
        self.drive_pub = self.create_publisher(AckermannDriveStamped, '/drive', 10)
        self.viz_scan_pub = self.create_publisher(LaserScan, '/gap_viz_scan', 10)
        self.viz_gap_pub = self.create_publisher(MarkerArray, '/gap_viz', 10)

        # State Variables
        self.scan_history = deque(maxlen=self.get_parameter('history_size').value)
        self.goal_history = deque(maxlen=self.get_parameter('goal_history_size').value)
        self.prev_steer = 0.0
        self.prev_target_angle = 0.0  # Added for deadband tracking
        self.prev_time = self.get_clock().now().nanoseconds / 1e9
        
        # PID State
        self.integral_error = 0.0
        self.prev_error = 0.0

    def _declare_params(self):
        self.declare_parameter('fov_degrees', 180.0)
        self.declare_parameter('history_size', 5) 
        self.declare_parameter('max_range', 6.0)
        self.declare_parameter('car_width', 0.5)   
        self.declare_parameter('disparity_threshold', 0.3)
        self.declare_parameter('lookahead_distance', 3.0)
        self.declare_parameter('bubble_radius', 0.5) 
        self.declare_parameter('goal_history_size', 25) 
        
        # New Param: Goal Cutoff (Deadband)
        self.declare_parameter('goal_deadband_deg', 1.5) # Minimum degree change required to update goal
        
        # Speed Params
        self.declare_parameter('max_speed', 4.0)
        self.declare_parameter('min_speed', 1.0)
        
        # PID & Smoothing
        self.declare_parameter('kp', 1.0)
        self.declare_parameter('ki', 0.01)
        self.declare_parameter('kd', 0.05)
        self.declare_parameter('steer_smoothing', 0.2) 

        # Wall Smoothener / Side Protection Params
        self.declare_parameter('wall_clearance', 0.9)      
        self.declare_parameter('repulsion_gain', 1.5)      
        self.declare_parameter('side_safety_dist', 0.45)   
        self.declare_parameter('side_angle_window', 30.0)  

    def _apply_wall_smoothener(self, smoothed_ranges, angle_min, angle_inc, n, base_steer):
        wall_clearance = self.get_parameter('wall_clearance').value
        repulsion_gain = self.get_parameter('repulsion_gain').value
        side_dist = self.get_parameter('side_safety_dist').value
        side_window = np.radians(self.get_parameter('side_angle_window').value)
        max_rng = self.get_parameter('max_range').value

        # 1. Calculate Proactive Repulsion (using 60 to 120 degree windows)
        left_start = max(0, int((np.radians(60) - angle_min) / angle_inc))
        left_end = min(n, int((np.radians(120) - angle_min) / angle_inc))
        right_start = max(0, int((np.radians(-120) - angle_min) / angle_inc))
        right_end = min(n, int((np.radians(-60) - angle_min) / angle_inc))

        left_window_repulse = smoothed_ranges[left_start:left_end]
        right_window_repulse = smoothed_ranges[right_start:right_end]

        min_left_repulse = np.min(left_window_repulse) if len(left_window_repulse) > 0 else max_rng
        min_right_repulse = np.min(right_window_repulse) if len(right_window_repulse) > 0 else max_rng

        repulsion_steer = 0.0
        status = "NONE"

        if min_left_repulse < wall_clearance:
            repulsion_steer -= repulsion_gain * (wall_clearance - min_left_repulse)
            status = "PUSHING RIGHT"
        if min_right_repulse < wall_clearance:
            repulsion_steer += repulsion_gain * (wall_clearance - min_right_repulse)
            status = "PUSHING LEFT" if status == "NONE" else "SQUEEZED"

        combined_steer = base_steer + repulsion_steer

        # 2. Hard Clamp Safety Check (Exactly 90 degrees left/right)
        left_idx = int((np.pi/2.0 - angle_min) / angle_inc)
        right_idx = int((-np.pi/2.0 - angle_min) / angle_inc)
        window_bins = int((side_window / 2.0) / angle_inc)

        left_window_clamp = smoothed_ranges[max(0, left_idx - window_bins) : min(n, left_idx + window_bins)]
        right_window_clamp = smoothed_ranges[max(0, right_idx - window_bins) : min(n, right_idx + window_bins)]

        min_left_clamp = np.min(left_window_clamp) if len(left_window_clamp) > 0 else max_rng
        min_right_clamp = np.min(right_window_clamp) if len(right_window_clamp) > 0 else max_rng

        if combined_steer > 0.0 and min_left_clamp < side_dist:
            combined_steer = 0.0
            status = "LEFT PROT"
        elif combined_steer < 0.0 and min_right_clamp < side_dist:
            combined_steer = 0.0
            status = "RIGHT PROT"

        return combined_steer, repulsion_steer, status

    def lidar_callback(self, data):
        # 0. Fetch params
        fov = self.get_parameter('fov_degrees').value
        history_size = self.get_parameter('history_size').value
        max_rng = self.get_parameter('max_range').value
        car_width = self.get_parameter('car_width').value
        disp_thresh = self.get_parameter('disparity_threshold').value
        lookahead = self.get_parameter('lookahead_distance').value
        bubble_radius = self.get_parameter('bubble_radius').value
        goal_hist_size = self.get_parameter('goal_history_size').value
        deadband = np.radians(self.get_parameter('goal_deadband_deg').value)
        
        kp = self.get_parameter('kp').value
        ki = self.get_parameter('ki').value
        kd = self.get_parameter('kd').value
        alpha = self.get_parameter('steer_smoothing').value

        # Dynamic queue resizing
        if self.scan_history.maxlen != history_size:
            self.scan_history = deque(maxlen=history_size)
        if self.goal_history.maxlen != goal_hist_size:
            self.goal_history = deque(maxlen=goal_hist_size)

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

        # 4. Obstacle Bubbling
        closest = int(np.argmin(scan))
        closest_dist = scan[closest]
        if closest_dist > 0.0:
            ratio = bubble_radius / closest_dist
            if ratio >= 1.0:
                bw = len(scan) 
            else:
                bw = int(np.ceil(np.arcsin(ratio) / angle_inc))
            scan[max(0, closest - bw): min(len(scan), closest + bw + 1)] = 0.0

        # 5. Find Gaps & Goal
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

        # 6. Cartesian Goal Smoothing
        raw_target_dist = min(scan[goal], lookahead)
        global_idx = goal + lo
        raw_target_angle = angle_min + global_idx * angle_inc

        raw_x = raw_target_dist * np.cos(raw_target_angle)
        raw_y = raw_target_dist * np.sin(raw_target_angle)
        self.goal_history.append((raw_x, raw_y))

        avg_x = float(np.mean([p[0] for p in self.goal_history]))
        avg_y = float(np.mean([p[1] for p in self.goal_history]))
        smoothed_target_angle = float(np.arctan2(avg_y, avg_x))

        # --- GOAL CUTOFF (DEADBAND) ---
        # Only update the target angle if it exceeds the deadband threshold
        if abs(smoothed_target_angle - self.prev_target_angle) > deadband:
            self.prev_target_angle = smoothed_target_angle
        else:
            smoothed_target_angle = self.prev_target_angle
        # ------------------------------

        # 7. PID Controller for base steering
        current_time = self.get_clock().now().nanoseconds / 1e9
        dt = current_time - self.prev_time
        if dt <= 0.0: dt = 0.01
        
        error = smoothed_target_angle
        self.integral_error += error * dt
        derivative = (error - self.prev_error) / dt
        
        pid_steer = (kp * error) + (ki * self.integral_error) + (kd * derivative)
        self.prev_error = error
        self.prev_time = current_time

        # 8. WALL SMOOTHENER (Repulsion + Clamps)
        target_steer, repulsion_val, protection_status = self._apply_wall_smoothener(
            smoothed_ranges, angle_min, angle_inc, n, pid_steer
        )

        # Apply EMA Filter for mechanical smoothness
        steering_angle = alpha * target_steer + (1.0 - alpha) * self.prev_steer
        steering_angle = np.clip(steering_angle, -0.4, 0.4)
        self.prev_steer = steering_angle

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

        # 10. Publish
        msg = AckermannDriveStamped()
        msg.drive.speed = float(speed)
        msg.drive.steering_angle = float(steering_angle)
        self.drive_pub.publish(msg)

        # 11. Visualization updates
        full = np.array(ranges, dtype=float)
        full[lo:hi] = scan
        self.viz_scan_pub.publish(self._make_scan(data, full))
        
        self._publish_markers(
            data, scan, lo, best_gap, 
            raw_x, raw_y, avg_x, avg_y, 
            steering_angle, repulsion_val, speed, protection_status
        )

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

    def _publish_markers(self, data, scan, lo, gap, rx, ry, sx, sy, steer, repulse, speed, protection_status):
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

        # 2. Raw Target Sphere
        m = Marker()
        m.header.frame_id = fid
        m.header.stamp = stamp
        m.ns, m.id = 'raw_goal', 0
        m.type = Marker.SPHERE
        m.action = Marker.ADD
        m.pose.position.x, m.pose.position.y = float(rx), float(ry)
        m.pose.orientation.w = 1.0
        m.scale.x = m.scale.y = m.scale.z = 0.2
        m.color.r, m.color.g, m.color.b, m.color.a = 0.0, 1.0, 1.0, 0.7
        ma.markers.append(m)

        # 3. Smoothed Target Sphere
        m = Marker()
        m.header.frame_id = fid
        m.header.stamp = stamp
        m.ns, m.id = 'smooth_goal', 0
        m.type = Marker.SPHERE
        m.action = Marker.ADD
        m.pose.position.x, m.pose.position.y = float(sx), float(sy)
        m.pose.orientation.w = 1.0
        m.scale.x = m.scale.y = m.scale.z = 0.4
        m.color.r, m.color.a = 1.0, 1.0
        ma.markers.append(m)

        # 4. Final Steering Arrow
        sl = 2.0
        arr_x, arr_y = sl * np.cos(steer), sl * np.sin(steer)
        m = Marker()
        m.header.frame_id = fid
        m.header.stamp = stamp
        m.ns, m.id = 'steer', 0
        m.type = Marker.ARROW
        m.action = Marker.ADD
        m.points = [Point(x=0.0, y=0.0, z=0.05), Point(x=arr_x, y=arr_y, z=0.05)]
        m.scale.x, m.scale.y = 0.1, 0.15
        
        if "PROT" in protection_status:
            m.color.r, m.color.g, m.color.b, m.color.a = 1.0, 0.0, 0.0, 1.0 
        else:
            m.color.r, m.color.g, m.color.b, m.color.a = 1.0, 0.85, 0.0, 1.0
        ma.markers.append(m)

        # 5. Repulsion Force Arrow
        if abs(repulse) > 0.01:
            rl = 2.0 * abs(repulse) * 2.0 
            rep_angle = np.pi/2.0 if repulse > 0 else -np.pi/2.0
            rx_force, ry_force = rl * np.cos(rep_angle), rl * np.sin(rep_angle)
            
            m = Marker()
            m.header.frame_id = fid
            m.header.stamp = stamp
            m.ns, m.id = 'repulsion_force', 0
            m.type = Marker.ARROW
            m.action = Marker.ADD
            m.points = [Point(x=0.0, y=0.0, z=0.1), Point(x=rx_force, y=ry_force, z=0.1)]
            m.scale.x, m.scale.y = 0.15, 0.2
            m.color.r, m.color.g, m.color.b, m.color.a = 1.0, 0.0, 1.0, 1.0
            ma.markers.append(m)
        else:
            m = Marker()
            m.header.frame_id = fid
            m.header.stamp = stamp
            m.ns, m.id = 'repulsion_force', 0
            m.action = Marker.DELETE
            ma.markers.append(m)

        # 6. Text HUD
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
            m.color.r, m.color.g, m.color.b, m.color.a = 1.0, 0.2, 1.0, 1.0
            hud_text = f'Spd: {speed:.1f} m/s | Steer: {np.degrees(steer):.0f} deg\n[ {protection_status} ]'
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