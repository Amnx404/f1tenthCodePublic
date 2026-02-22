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
        self.smoothed_goal_x = None  
        self.smoothed_goal_y = None  
        
        self.prev_steer = 0.0
        self.prev_target_angle = 0.0  
        self.prev_time = self.get_clock().now().nanoseconds / 1e9
        
        # PID State (integral is time-windowed via integral_history)
        self.integral_history = deque()  # (timestamp, error*dt) for windowed integral
        self.prev_error = 0.0
        # Input-to-output latency logging (throttle to ~1 Hz)
        self._last_latency_log_time = 0.0
        # Closure-rate brake: previous forward clearance and time for d(clearance)/dt
        self._prev_forward_clearance = None
        self._prev_closure_time = None

    def _declare_params(self):
        self.declare_parameter('fov_degrees', 220.0)
        self.declare_parameter('history_size', 2) 
        self.declare_parameter('max_range', 5)
        self.declare_parameter('car_width', 0.4)   
        self.declare_parameter('disparity_threshold', 0.4)
        self.declare_parameter('lookahead_distance', 2)
        self.declare_parameter('min_lookahead', 0.8)
        # Ratio: adaptive_lookahead = forward_clearance * this, clamped to [min_lookahead, lookahead_distance]
        self.declare_parameter('lookahead_ratio', 0.6)
        self.declare_parameter('bubble_radius', 0.4) 
        
        # NEW: Fixed Targeting Window to prevent "Rearview Mirror" effect
        self.declare_parameter('aim_window_degrees', 190.0) 
        
        self.declare_parameter('goal_smoothing_alpha', 0.5) 
        self.declare_parameter('goal_deadband_deg', 3)
        # Blend between gap center (0) and deepest point (1); lower = wider arcs on turns
        self.declare_parameter('gap_target_bias', 0.75)
        # Max rate the goal angle can change (deg/s) – prevents snapping into turns too early
        self.declare_parameter('max_goal_rate_deg_s', 360.0)
        
        self.declare_parameter('max_speed', 4)
        self.declare_parameter('min_speed', 0.8)
        

        self.declare_parameter('kp', 0.7)
        self.declare_parameter('ki', 0.0003)
        self.declare_parameter('kd', 0.15)
        self.declare_parameter('steer_smoothing', 0.97) 

        self.declare_parameter('wall_clearance', 0.4)      
        self.declare_parameter('repulsion_gain', 2.5)      
        self.declare_parameter('side_safety_dist', 0.13)   
        self.declare_parameter('side_angle_window', 2.0)
        # Integral: time window (only sum errors over last N sec) and windup clamp
        self.declare_parameter('integral_window_sec', 2.0)
        self.declare_parameter('integral_clamp', 0.5)
        # Brake when approach rate (closure rate) exceeds this (m/s); negative d(forward_clearance)/dt
        self.declare_parameter('closure_rate_brake_threshold', 2.0)
        # Brake when time-to-collision (forward_clearance / |closure_rate|) is below this (seconds)
        self.declare_parameter('ttc_brake_threshold', 0.5)
        # Brake if forward clearance (in emergency-brake cone) is below this (m), regardless of TTC
        self.declare_parameter('min_clearance_brake', 0.25)
        # Half-angle (deg) for forward clearance / emergency brake: ±this around straight ahead (total = 2× this)
        self.declare_parameter('forward_clearance_half_deg', 5.0)

    def _apply_wall_smoothener(self, smoothed_ranges, angle_min, angle_inc, n, base_steer, lookahead):
        wall_clearance = self.get_parameter('wall_clearance').value
        repulsion_gain = self.get_parameter('repulsion_gain').value
        side_dist = self.get_parameter('side_safety_dist').value
        side_window = np.radians(self.get_parameter('side_angle_window').value)
        max_rng = self.get_parameter('max_range').value

        # 1. Proactive Repulsion (only consider obstacles within lookahead distance)
        left_start = max(0, int((np.radians(60) - angle_min) / angle_inc))
        left_end = min(n, int((np.radians(120) - angle_min) / angle_inc))
        right_start = max(0, int((np.radians(-120) - angle_min) / angle_inc))
        right_end = min(n, int((np.radians(-60) - angle_min) / angle_inc))

        left_window_repulse = smoothed_ranges[left_start:left_end]
        right_window_repulse = smoothed_ranges[right_start:right_end]

        # Min distance within lookahead, ignoring zeros/noise (avoid spurious repulsion when no wall)
        min_valid = 0.05  # ignore readings below this (m) - zeros and noise trigger false "wall"
        left_valid = left_window_repulse[(left_window_repulse >= min_valid) & (left_window_repulse <= lookahead)]
        right_valid = right_window_repulse[(right_window_repulse >= min_valid) & (right_window_repulse <= lookahead)]
        min_left_repulse = float(np.min(left_valid)) if len(left_valid) > 0 else max_rng
        min_right_repulse = float(np.min(right_valid)) if len(right_valid) > 0 else max_rng

        repulsion_steer = 0.0
        status = "NONE"

        if min_left_repulse < wall_clearance:
            repulsion_steer -= repulsion_gain * (wall_clearance - min_left_repulse)
            status = "PUSHING RIGHT"
        if min_right_repulse < wall_clearance:
            repulsion_steer += repulsion_gain * (wall_clearance - min_right_repulse)
            status = "PUSHING LEFT" if status == "NONE" else "SQUEEZED"

        combined_steer = base_steer + repulsion_steer

        # 2. Hard Clamp Safety Check
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
        t_start = self.get_clock().now().nanoseconds / 1e9
        fov = self.get_parameter('fov_degrees').value
        history_size = self.get_parameter('history_size').value
        max_rng = self.get_parameter('max_range').value
        car_width = self.get_parameter('car_width').value
        disp_thresh = self.get_parameter('disparity_threshold').value
        lookahead = self.get_parameter('lookahead_distance').value
        bubble_radius = self.get_parameter('bubble_radius').value
        
        aim_window = self.get_parameter('aim_window_degrees').value
        goal_alpha = self.get_parameter('goal_smoothing_alpha').value
        deadband = np.radians(self.get_parameter('goal_deadband_deg').value)
        
        kp = self.get_parameter('kp').value
        ki = self.get_parameter('ki').value
        kd = self.get_parameter('kd').value
        steer_alpha = self.get_parameter('steer_smoothing').value

        if self.scan_history.maxlen != history_size:
            self.scan_history = deque(maxlen=history_size)

        # 1. Preprocess & Temporal Rolling Mean
        ranges = np.array(data.ranges, dtype=np.float32)
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

        # 3. HIGH-SPEED Vectorized Disparity Extender
        diffs = np.diff(scan)
        jumps = np.where(np.abs(diffs) > disp_thresh)[0]
        
        for i in jumps:
            closer = min(scan[i], scan[i + 1])
            if closer < 0.1: continue
            w = int(np.ceil(np.arctan2(car_width / 2.0, closer) / angle_inc))
            
            if diffs[i] > 0:
                scan[i + 1 : min(len(scan), i + 1 + w)] = 0.0
            else:
                scan[max(0, i - w + 1) : i + 1] = 0.0

        # 4. Obstacle Bubbling
        closest = int(np.argmin(scan))
        closest_dist = scan[closest]
        if closest_dist > 0.0:
            ratio = bubble_radius / closest_dist
            bw = len(scan) if ratio >= 1.0 else int(np.ceil(np.arcsin(ratio) / angle_inc))
            scan[max(0, closest - bw): min(len(scan), closest + bw + 1)] = 0.0

        # 5. Gap Finding with FIXED AIM WINDOW
        mask = scan > 0.1
        padded_mask = np.concatenate(([False], mask, [False]))
        diffs_mask = np.diff(padded_mask.astype(int))
        
        starts = np.where(diffs_mask == 1)[0]
        ends = np.where(diffs_mask == -1)[0]
        
        # Create a mask that is only 1.0 inside the aiming window
        aim_half = int(np.radians(aim_window) / 2.0 / angle_inc)
        aim_start = max(0, len(scan) // 2 - aim_half)
        aim_end = min(len(scan), len(scan) // 2 + aim_half)
        
        aim_mask = np.zeros(len(scan), dtype=np.float32)
        aim_mask[aim_start:aim_end] = 1.0

        if len(starts) > 0:
            max_ranges = []
            for s, e in zip(starts, ends):
                # Zeros out depth values outside the aim window to prevent looking backward
                gap_vals = scan[s:e] * aim_mask[s:e]
                max_ranges.append(np.max(gap_vals) if len(gap_vals) > 0 else 0.0)
                
            best_idx = np.argmax(max_ranges)
            best_gap = (starts[best_idx], ends[best_idx])
            
            # Deepest Point (Restricted strictly to the aim window)
            gap_aim_scan = scan[best_gap[0]:best_gap[1]] * aim_mask[best_gap[0]:best_gap[1]]
            deepest_idx = best_gap[0] + np.argmax(gap_aim_scan)
            gap_center_idx = (best_gap[0] + best_gap[1]) // 2
        else:
            best_gap = (len(scan)//2, len(scan)//2 + 1)
            deepest_idx = len(scan) // 2
            gap_center_idx = deepest_idx

        # Blend gap center with deepest point
        gap_bias = self.get_parameter('gap_target_bias').value
        target_idx = int(round((1.0 - gap_bias) * gap_center_idx + gap_bias * deepest_idx))
        target_idx = np.clip(target_idx, best_gap[0], best_gap[1] - 1)

        # Adaptive lookahead: shrinks when a wall is close ahead, expands on straights
        min_la = self.get_parameter('min_lookahead').value
        fwd_half_deg = self.get_parameter('forward_clearance_half_deg').value
        fwd_half = int(np.radians(fwd_half_deg) / angle_inc)
        fwd_center = len(scan) // 2
        fwd_slice = scan[max(0, fwd_center - fwd_half) : min(len(scan), fwd_center + fwd_half)]
        fwd_valid = fwd_slice[fwd_slice > 0.01]  # ignore zeroed-out rays (bubbled/disparity)
        forward_clearance = float(np.min(fwd_valid)) if len(fwd_valid) > 0 else lookahead
        la_ratio = self.get_parameter('lookahead_ratio').value
        adaptive_lookahead = float(np.clip(forward_clearance * la_ratio, min_la, lookahead))

        # Convert the blended target to XY coordinates (Cyan Sphere)
        deep_dist = min(scan[target_idx], adaptive_lookahead)
        deep_angle = angle_min + (target_idx + lo) * angle_inc
        deep_x = deep_dist * np.cos(deep_angle)
        deep_y = deep_dist * np.sin(deep_angle)

        # -------------------------------------------------------------
        # 6. CARTESIAN EMA GOAL FILTERING + RATE LIMITER
        # EMA smooths the target, rate limiter caps how fast the goal
        # angle can slew (deg/s) so the car follows a gentle arc
        # instead of snapping into turns.
        # -------------------------------------------------------------
        if self.smoothed_goal_x is None:
            self.smoothed_goal_x = deep_x
            self.smoothed_goal_y = deep_y
        else:
            self.smoothed_goal_x = (goal_alpha * deep_x) + ((1.0 - goal_alpha) * self.smoothed_goal_x)
            self.smoothed_goal_y = (goal_alpha * deep_y) + ((1.0 - goal_alpha) * self.smoothed_goal_y)

        # Clamp smoothed goal distance to adaptive lookahead so the red dot stays within range
        ema_dist = np.hypot(self.smoothed_goal_x, self.smoothed_goal_y)
        if ema_dist > adaptive_lookahead and ema_dist > 1e-6:
            scale = adaptive_lookahead / ema_dist
            self.smoothed_goal_x *= scale
            self.smoothed_goal_y *= scale

        ema_angle = float(np.arctan2(self.smoothed_goal_y, self.smoothed_goal_x))

        # Rate-limit the goal angle (deg/s)
        max_goal_rate = np.radians(self.get_parameter('max_goal_rate_deg_s').value)
        angle_delta = ema_angle - self.prev_target_angle
        # Wrap to [-pi, pi]
        angle_delta = (angle_delta + np.pi) % (2 * np.pi) - np.pi
        dt_goal = (self.get_clock().now().nanoseconds / 1e9) - self.prev_time
        if dt_goal <= 0:
            dt_goal = 0.01
        max_change = max_goal_rate * dt_goal
        angle_delta = np.clip(angle_delta, -max_change, max_change)
        smoothed_target_angle = float(self.prev_target_angle + angle_delta)

        # Deadband: ignore tiny corrections
        if abs(smoothed_target_angle - self.prev_target_angle) > deadband:
            self.prev_target_angle = smoothed_target_angle
        else:
            smoothed_target_angle = self.prev_target_angle

        # 7. PID Controller (time-windowed integral + clamp)
        current_time = self.get_clock().now().nanoseconds / 1e9
        dt = current_time - self.prev_time
        if dt <= 0.0:
            dt = 0.01

        # Closure rate: speed of approach toward forward obstacle (negative = closing)
        closure_rate = None
        if self._prev_closure_time is not None:
            closure_dt = current_time - self._prev_closure_time
            if closure_dt > 1e-6:
                closure_rate = (forward_clearance - self._prev_forward_clearance) / closure_dt
        self._prev_forward_clearance = forward_clearance
        self._prev_closure_time = current_time

        error = smoothed_target_angle
        integral_window_sec = self.get_parameter('integral_window_sec').value
        integral_clamp = self.get_parameter('integral_clamp').value

        # Time-based integral window: keep only (t, error*dt) within last integral_window_sec
        self.integral_history.append((current_time, error * dt))
        cutoff = current_time - integral_window_sec
        while self.integral_history and self.integral_history[0][0] < cutoff:
            self.integral_history.popleft()
        integral_error = sum(delta for _, delta in self.integral_history)
        integral_error = np.clip(integral_error, -integral_clamp, integral_clamp)

        derivative = (error - self.prev_error) / dt
        pid_steer = (kp * error) + (ki * integral_error) + (kd * derivative)
        self.prev_error = error
        self.prev_time = current_time

        # 8. WALL SMOOTHENER (use adaptive lookahead so repulsion only considers obstacles within lookahead)
        target_steer, repulsion_val, protection_status = self._apply_wall_smoothener(
            smoothed_ranges, angle_min, angle_inc, n, pid_steer, adaptive_lookahead
        )

        steering_angle = steer_alpha * target_steer + (1.0 - steer_alpha) * self.prev_steer
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

        # Emergency brake: min clearance, closure-rate threshold, and/or time-to-collision
        min_clearance_brake = self.get_parameter('min_clearance_brake').value
        closure_threshold = self.get_parameter('closure_rate_brake_threshold').value
        ttc_threshold = self.get_parameter('ttc_brake_threshold').value
        do_brake = False
        if forward_clearance < min_clearance_brake:  # too close regardless of TTC
            do_brake = True
        elif closure_rate is not None and closure_rate < 0:  # closing
            if closure_rate < -closure_threshold:
                do_brake = True
            # TTC = distance / approach_speed (only when closing fast enough to be meaningful)
            elif abs(closure_rate) > 0.1 and forward_clearance > 0.01:
                ttc = forward_clearance / abs(closure_rate)
                if ttc < ttc_threshold:
                    do_brake = True
        if do_brake:
            speed = min_spd

        # TTC for logging (when closing)
        ttc_s = (forward_clearance / abs(closure_rate)) if (closure_rate is not None and closure_rate < 0 and abs(closure_rate) > 0.1 and forward_clearance > 0.01) else float('nan')

        # Console: print everything
        cr_str = f'{closure_rate:.2f}' if closure_rate is not None else 'None'
        target_angle_deg = np.degrees(angle_min + (target_idx + lo) * angle_inc)
        self.get_logger().info(
            f'fwd_clear={forward_clearance:.2f} adapt_la={adaptive_lookahead:.2f} '
            f'gap_target_deg={target_angle_deg:.1f} '
            f'closure_rate={cr_str} ttc={ttc_s:.2f}s do_brake={do_brake} speed={speed:.2f} '
            f'steer_deg={np.degrees(steering_angle):.1f} error_deg={np.degrees(error):.2f} '
            f'integral={integral_error:.4f} repulsion={repulsion_val:.3f} protection={protection_status}'
        )

        # 10. Publish
        msg = AckermannDriveStamped()
        msg.drive.speed = float(speed)
        msg.drive.steering_angle = float(steering_angle)
        self.drive_pub.publish(msg)

        # Input-to-output latency (throttled to ~1 Hz)
        t_end = self.get_clock().now().nanoseconds / 1e9
        latency_ms = (t_end - t_start) * 1000.0
        if t_end - self._last_latency_log_time >= 0.1:
            self.get_logger().info(f'input→output latency: {latency_ms:.2f} ms')
            self._last_latency_log_time = t_end

        # 11. Visualization
        full = np.array(ranges, dtype=np.float32)
        full[lo:hi] = scan
        self.viz_scan_pub.publish(self._make_scan(data, full))
        
        self._publish_markers(
            data, scan, lo, best_gap, 
            self.smoothed_goal_x, self.smoothed_goal_y, deep_x, deep_y, 
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
        msg.ranges = ranges_full.tolist() 
        return msg

    def _publish_markers(self, data, scan, lo, gap, sx, sy, dx, dy, steer, repulse, speed, protection_status):
        a_min = data.angle_min
        a_inc = data.angle_increment
        stamp = self.get_clock().now().to_msg()
        fid = data.header.frame_id or 'laser'

        def to_xy(idx, r):
            a = a_min + idx * a_inc
            return float(r * np.cos(a)), float(r * np.sin(a))

        gs, ge = gap
        ma = MarkerArray()

        # 1. ACTUAL CONTOUR GAP VISUALIZATION
        m = Marker()
        m.header.frame_id = fid
        m.header.stamp = stamp
        m.ns, m.id = 'gap', 0
        m.type = Marker.LINE_STRIP
        m.action = Marker.ADD
        m.scale.x = 0.08
        m.color.g, m.color.a = 1.0, 1.0
        
        contour_points = []
        for i in range(gs, ge):
            if scan[i] > 0.1:
                px, py = to_xy(lo + i, float(scan[i]))
                contour_points.append(Point(x=px, y=py, z=0.0))
        m.points = contour_points
        ma.markers.append(m)

        # 2. Deepest Point inside Aim Window (Cyan Sphere)
        m = Marker()
        m.header.frame_id = fid
        m.header.stamp = stamp
        m.ns, m.id = 'deepest_point', 0
        m.type = Marker.SPHERE
        m.action = Marker.ADD
        m.pose.position.x, m.pose.position.y = float(dx), float(dy)
        m.pose.orientation.w = 1.0
        m.scale.x = m.scale.y = m.scale.z = 0.25
        m.color.r, m.color.g, m.color.b, m.color.a = 0.0, 1.0, 1.0, 0.7
        ma.markers.append(m)

        # 3. EMA Filtered Target Goal (Red Sphere) - Now chases the Cyan Sphere!
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