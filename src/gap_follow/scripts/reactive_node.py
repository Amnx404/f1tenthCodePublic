#!/usr/bin/env python3
import rclpy
from rclpy.node import Node

import math
import numpy as np
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped, AckermannDrive
from visualization_msgs.msg import MarkerArray, Marker
from geometry_msgs.msg import Point

class ReactiveFollowGap(Node):
    """
    Follow the Gap
    """
    def __init__(self):
        super().__init__('reactive_node')
        lidarscan_topic = '/scan'
        drive_topic = '/drive'

        # print its running
        self.get_logger().info('ReactiveFollowGap is running')

        self.sub_scan  = self.create_subscription(LaserScan, lidarscan_topic, self.lidar_callback, 10)
        self.pub_drive = self.create_publisher(AckermannDriveStamped, drive_topic, 10)
        # Visualization: pre-process (smoothed) and post-process (disparity + bubble) scans, plus gap/best point
        self.pub_scan_before = self.create_publisher(LaserScan, '/gap_viz_scan_before', 10)
        self.pub_scan_proc  = self.create_publisher(LaserScan, '/gap_viz_scan_proc', 10)
        self.pub_gap_viz    = self.create_publisher(MarkerArray, '/gap_viz', 10)

        # ── Tunable Parameters  ──────────────────────────────────────────
        self.bubble_radius        = 0.15              # Safety bubble radius (m)
        self.preprocess_conv_size = 10               # Smoothing window size
        self.max_lidar_dist       = 5             # Max LiDAR distance (m)
        self.max_speed            = 5              # Max speed on straights (m/s)
        self.min_speed            = 1.5               # Min speed at sharp turns (m/s)
        self.fov_angle            = np.radians(270)  # Total field of view (rad)
        self.car_width            = 0.35             # Car width for disparity extension (m)
        self.disparity_threshold  = 0.5              # Min distance jump to trigger extension (m)
        self.far_guide_gain       = 0.25             # Far-field balance gain (keep small)
        self.alpha                = 0.35             # EMA smoothing (0=slowest, 1=fastest)
        self.max_steer_step       = np.radians(25.0) # Max steering change per callback (rad)
        self.side_clearance_min   = 0.4             # Min side clearance to allow turning (m)
        self.steer_momentum       = 0.4              # How much to favour gaps in the current steer direction (0=none, 1=strong)
        # ──────────────────────────────────────────────────────────────────

        self.prev_steering_angle = 0.0

    def preprocess_lidar(self, ranges):
        proc = np.array(ranges, dtype=float)
        proc = np.nan_to_num(proc, posinf=self.max_lidar_dist, nan=0.0)
        proc = np.clip(proc, 0.0, self.max_lidar_dist)
        kernel = np.ones(self.preprocess_conv_size) / self.preprocess_conv_size
        proc = np.convolve(proc, kernel, 'same')
        return proc

    def sector_indices(self, angle_min, angle_increment, total_len, start_angle, end_angle):
        """Convert angle range to clamped array indices."""
        s = int((start_angle - angle_min) / angle_increment)
        e = int((end_angle   - angle_min) / angle_increment)
        s = max(0, min(total_len, s))
        e = max(0, min(total_len, e))
        return (s, e) if s <= e else (e, s)

    def find_max_gap(self, free_space_ranges, preferred_idx=None):
        """Return (start, end) indices of the best-scored gap.
        Score = depth + width - heading_penalty + momentum_bonus.
        preferred_idx biases toward gaps aligned with current steering.
        """
        mask  = free_space_ranges > 0.1
        dmask = np.diff(mask.astype(int))
        run_starts = np.where(dmask ==  1)[0] + 1
        run_ends   = np.where(dmask == -1)[0] + 1
        if mask[0]:  run_starts = np.insert(run_starts, 0, 0)
        if mask[-1]: run_ends   = np.append(run_ends, len(free_space_ranges))

        if len(run_starts) == 0:
            best = int(np.argmax(free_space_ranges))
            return best, best + 1

        center_idx = len(free_space_ranges) // 2
        if preferred_idx is None:
            preferred_idx = center_idx
        best_gap, best_score = (run_starts[0], run_ends[0]), -np.inf

        for s, e in zip(run_starts, run_ends):
            if e <= s:
                continue
            gap_mid = (s + e) // 2
            gap_slice = free_space_ranges[s:e]
            gap_depth       = float(np.max(gap_slice)) / max(self.max_lidar_dist, 1e-6)
            gap_width       = float(e - s) / max(len(free_space_ranges), 1)
            heading_penalty = float(abs(gap_mid - center_idx)) / max(center_idx, 1)
            momentum_bonus  = 1.0 - float(abs(gap_mid - preferred_idx)) / max(len(free_space_ranges), 1)
            score = (1.2 * gap_depth + 0.9 * gap_width
                     - 0.25 * heading_penalty
                     + self.steer_momentum * momentum_bonus)
            if score > best_score:
                best_score, best_gap = score, (s, e)

        return best_gap

    def find_best_point(self, start_i, end_i, ranges, preferred_idx=None):
        """Return index of best target within the gap.
        Blends deepest point, spatial center, and steering momentum.
        """
        gap = ranges[start_i:end_i]
        if len(gap) == 0:
            return start_i

        straight_ahead = len(ranges) // 2
        if preferred_idx is None:
            preferred_idx = straight_ahead

        max_dist    = np.max(gap)
        deep_local  = np.where(gap >= max_dist - 0.1)[0]
        deep_global = start_i + deep_local

        best_deep      = deep_global[np.argmin(np.abs(deep_global - preferred_idx))]
        spatial_center = start_i + len(gap) // 2

        return int(0.5 * best_deep + 0.2 * spatial_center + 0.3 * np.clip(preferred_idx, start_i, end_i - 1))

    def _index_to_xy(self, index, range_val, angle_min, angle_increment):
        """Laser frame: angle 0 = forward (+x), left = +y."""
        angle = angle_min + index * angle_increment
        return (range_val * math.cos(angle), range_val * math.sin(angle))

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

    def _publish_gap_markers(self, data, proc, start_i, end_i, best_idx_slice, fov_min_idx):
        """Publish gap segment (line) and best point (sphere) in scan frame."""
        angle_min = data.angle_min
        angle_inc = data.angle_increment
        frame_id = data.header.frame_id or 'base_link'

        # Indices in full frame for angles; ranges from slice
        start_full = start_i + fov_min_idx
        end_full   = end_i - 1 + fov_min_idx  # end_i is exclusive, last gap index is end_i-1
        r0 = float(proc[start_i])
        r1 = float(proc[end_i - 1]) if end_i > start_i else r0
        best_local = best_idx_slice - fov_min_idx
        r_best = float(proc[best_local]) if 0 <= best_local < len(proc) else 0.0

        x0, y0 = self._index_to_xy(start_full, r0, angle_min, angle_inc)
        x1, y1 = self._index_to_xy(end_full, r1, angle_min, angle_inc)
        xb, yb = self._index_to_xy(best_idx_slice, r_best, angle_min, angle_inc)

        ma = MarkerArray()
        # Gap line
        gap_m = Marker()
        gap_m.header.frame_id = frame_id
        gap_m.header.stamp = self.get_clock().now().to_msg()
        gap_m.ns = 'gap'
        gap_m.id = 0
        gap_m.type = Marker.LINE_STRIP
        gap_m.action = Marker.ADD
        gap_m.scale.x = 0.15
        gap_m.color.r = 0.0
        gap_m.color.g = 1.0
        gap_m.color.b = 0.0
        gap_m.color.a = 1.0
        gap_m.points = [Point(x=x0, y=y0, z=0.0), Point(x=x1, y=y1, z=0.0)]
        ma.markers.append(gap_m)
        # Best point
        best_m = Marker()
        best_m.header.frame_id = frame_id
        best_m.header.stamp = self.get_clock().now().to_msg()
        best_m.ns = 'gap'
        best_m.id = 1
        best_m.type = Marker.SPHERE
        best_m.action = Marker.ADD
        best_m.pose.position.x = xb
        best_m.pose.position.y = yb
        best_m.pose.position.z = 0.0
        best_m.pose.orientation.w = 1.0
        best_m.scale.x = best_m.scale.y = best_m.scale.z = 0.5
        best_m.color.r = 1.0
        best_m.color.g = 0.0
        best_m.color.b = 0.0
        best_m.color.a = 1.0
        ma.markers.append(best_m)
        self.pub_gap_viz.publish(ma)

    def lidar_callback(self, data):
        ranges          = np.array(data.ranges)
        angle_increment = data.angle_increment
        angle_min       = data.angle_min
        n               = len(ranges)

        # ── 1. FOV Slice ──────────────────────────────────────────────────
        fov_min_idx = max(0, int((-self.fov_angle / 2 - angle_min) / angle_increment))
        fov_max_idx = min(n, int(( self.fov_angle / 2 - angle_min) / angle_increment))

        proc      = self.preprocess_lidar(ranges[fov_min_idx:fov_max_idx])
        proc_copy = proc.copy()  # clean reference for bubble & disparity

        # ── 2. Disparity Extension ────────────────────────────────────────
        # At edges between close obstacles and open space, extend the close
        # value into the gap to prevent the car from clipping corners.
        for i in np.where(np.abs(np.diff(proc_copy)) > self.disparity_threshold)[0]:
            d0, d1 = proc_copy[i], proc_copy[i + 1]
            md = max(min(d0, d1), 0.05)
            w  = int(math.atan(self.car_width / (md + 0.001)) / angle_increment)
            if d0 < d1:   # obstacle on left side → extend rightward into gap
                proc[i + 1 : min(len(proc), i + 1 + w)] = 0.0
            else:          # obstacle on right side → extend leftward into gap
                proc[max(0, i - w + 1) : i + 1] = 0.0

        # ── 3. Safety Bubble ──────────────────────────────────────────────
        # Zero out an angular region around the single closest point.
        closest_idx = int(np.argmin(proc_copy))
        min_dist    = proc_copy[closest_idx]
        if min_dist < self.bubble_radius:
            bw = int(math.atan(self.bubble_radius / (min_dist + 0.001)) / angle_increment)
            proc[max(0, closest_idx - bw) : min(len(proc), closest_idx + bw)] = 0.0

        # ── 4. Find Gap & Best Point ──────────────────────────────────────
        steer_offset   = int(self.prev_steering_angle / angle_increment)
        preferred_idx  = len(proc) // 2 + steer_offset
        preferred_idx  = max(0, min(len(proc) - 1, preferred_idx))
        start_i, end_i = self.find_max_gap(proc, preferred_idx)
        best_idx       = self.find_best_point(start_i, end_i, proc, preferred_idx) + fov_min_idx

        # ── 4b. Visualization: scan before (smoothed only) and after (disparity + bubble), gap/best point
        full_before = np.array(ranges, dtype=float)
        full_before[fov_min_idx:fov_max_idx] = proc_copy
        full_proc = np.array(ranges, dtype=float)
        full_proc[fov_min_idx:fov_max_idx] = proc
        self.pub_scan_before.publish(self._publish_scan_viz(data, full_before))
        self.pub_scan_proc.publish(self._publish_scan_viz(data, full_proc))
        self._publish_gap_markers(data, proc, start_i, end_i, best_idx, fov_min_idx)

        # ── 5. Raw Steering Angle ─────────────────────────────────────────
        steering_angle = angle_min + best_idx * angle_increment
        raw_gap_steer  = steering_angle  # before far-field bias

        # ── 6. Mild Far-Field Balance ─────────────────────────────────────
        proc_full = self.preprocess_lidar(ranges)
        front_s, front_e = self.sector_indices(angle_min, angle_increment, n, -np.radians(10),  np.radians(10))
        far_l_s, far_l_e = self.sector_indices(angle_min, angle_increment, n,  np.radians(22),  np.radians(75))
        far_r_s, far_r_e = self.sector_indices(angle_min, angle_increment, n, -np.radians(75), -np.radians(22))

        far_left    = float(np.percentile(proc_full[far_l_s:far_l_e], 80)) if far_l_e > far_l_s else self.max_lidar_dist
        far_right   = float(np.percentile(proc_full[far_r_s:far_r_e], 80)) if far_r_e > far_r_s else self.max_lidar_dist
        forward_cl  = float(np.min(proc_full[front_s:front_e]))             if front_e > front_s else self.max_lidar_dist
        far_balance = np.clip((far_left - far_right) / max(self.max_lidar_dist, 1e-6), -1.0, 1.0)
        danger      = np.clip((1.5 - forward_cl) / 1.5, 0.0, 1.0)
        steering_angle += self.far_guide_gain * (1.0 + danger) * far_balance

        # ── 7. Rate Limit + EMA Smoothing ────────────────────────────────
        limited = self.prev_steering_angle + np.clip(
            steering_angle - self.prev_steering_angle,
            -self.max_steer_step, self.max_steer_step
        )
        self.prev_steering_angle = self.alpha * limited + (1.0 - self.alpha) * self.prev_steering_angle
        steering_angle = self.prev_steering_angle

        # ── 7b. Side-Clearance Gate ─────────────────────────────────────
        left_s, left_e   = self.sector_indices(angle_min, angle_increment, n, np.radians(80), np.radians(100))
        right_s, right_e = self.sector_indices(angle_min, angle_increment, n, -np.radians(100), -np.radians(80))
        left_dist  = float(np.min(proc_full[left_s:left_e]))  if left_e  > left_s  else self.max_lidar_dist
        right_dist = float(np.min(proc_full[right_s:right_e])) if right_e > right_s else self.max_lidar_dist

        if steering_angle > 0 and left_dist < self.side_clearance_min:
            steering_angle = -abs(steering_angle)
        elif steering_angle < 0 and right_dist < self.side_clearance_min:
            steering_angle = abs(steering_angle)

        # ── 8. Speed Control ──────────────────────────────────────────────
        abs_steer = abs(steering_angle)
        if abs_steer < np.radians(10):
            speed = self.max_speed
        elif abs_steer > np.radians(20):
            speed = self.min_speed
        else:
            ratio = (abs_steer - np.radians(10)) / np.radians(10)
            speed = self.max_speed - ratio * (self.max_speed - self.min_speed)

        # ── DEBUG LOG ─────────────────────────────────────────────────────
        self.get_logger().info(
            f"close={min_dist:.2f}m | "
            f"gap=[{start_i},{end_i}](w={end_i-start_i}) | "
            f"gap_steer={np.degrees(raw_gap_steer):.1f}° | "
            f"far_balance={far_balance:+.2f}(L={far_left:.1f} R={far_right:.1f}) fwd={forward_cl:.2f}m | "
            f"side(L={left_dist:.2f} R={right_dist:.2f}) | "
            f"final={np.degrees(steering_angle):.1f}° | "
            f"speed={speed:.2f}m/s"
        )
        # ── DEBUG LOG END ─────────────────────────────────────────────────

        # ── 9. Publish ────────────────────────────────────────────────────
        msg = AckermannDriveStamped()
        msg.header.stamp         = self.get_clock().now().to_msg()
        msg.drive.speed          = float(speed) #speed)
        msg.drive.steering_angle = float(steering_angle)
        self.pub_drive.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    print("ReactiveFollowGap Initialized")
    reactive_node = ReactiveFollowGap()
    rclpy.spin(reactive_node)
    reactive_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()