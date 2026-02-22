// reactive_node.cpp — Single-file reactive follow-gap for F1/10th
// 3 fixes over Python original:
//   1. PID integral windup clamp
//   2. Params cached into struct once per callback (not 20+ get_parameter calls)
//   3. Pre-allocated buffers reused across callbacks

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/laser_scan.hpp>
#include <ackermann_msgs/msg/ackermann_drive_stamped.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <geometry_msgs/msg/point.hpp>

#include <vector>
#include <deque>
#include <cmath>
#include <algorithm>
#include <string>
#include <utility>
#include <cstdio>

class ReactiveFollowGap : public rclcpp::Node {
public:
    ReactiveFollowGap() : Node("reactive_node") {
        declare_parameter("fov_degrees",          220.0);
        declare_parameter("history_size",         3);
        declare_parameter("max_range",            4.0);
        declare_parameter("car_width",            0.35);
        declare_parameter("disparity_threshold",  0.2);
        declare_parameter("lookahead_distance",   1.5);
        declare_parameter("bubble_radius",        0.4);
        declare_parameter("aim_window_degrees",   180.0);
        declare_parameter("goal_smoothing_alpha", 0.3);
        declare_parameter("goal_deadband_deg",    3.0);
        declare_parameter("max_speed",            0.0);
        declare_parameter("min_speed",            0.0);
        declare_parameter("kp",                   0.9);
        declare_parameter("ki",                   0.003);
        declare_parameter("kd",                   0.1);
        declare_parameter("steer_smoothing",      0.3);
        declare_parameter("wall_clearance",       0.4);
        declare_parameter("repulsion_gain",       1.9);
        declare_parameter("side_safety_dist",     0.2);
        declare_parameter("side_angle_window",    10.0);
        declare_parameter("integral_clamp",       1.0);   // NEW: windup limit

        scan_sub_     = create_subscription<sensor_msgs::msg::LaserScan>(
            "/scan", 10, std::bind(&ReactiveFollowGap::lidarCallback, this, std::placeholders::_1));
        drive_pub_    = create_publisher<ackermann_msgs::msg::AckermannDriveStamped>("/drive", 10);
        viz_scan_pub_ = create_publisher<sensor_msgs::msg::LaserScan>("/gap_viz_scan", 10);
        viz_gap_pub_  = create_publisher<visualization_msgs::msg::MarkerArray>("/gap_viz", 10);

        prev_time_ = now().nanoseconds() / 1e9;

        double max_s = get_parameter("max_speed").as_double();
        double min_s = get_parameter("min_speed").as_double();
        RCLCPP_INFO(this->get_logger(), "reactive_node running (drive speed params: max=%.2f min=%.2f m/s)", max_s, min_s);
    }

private:
    // ── Cached param struct (FIX #2) ──
    struct P {
        double fov_degrees, max_range, car_width, disparity_threshold;
        double lookahead_distance, bubble_radius, aim_window_degrees;
        double goal_smoothing_alpha, goal_deadband_rad;
        double max_speed, min_speed;
        double kp, ki, kd, steer_smoothing;
        double wall_clearance, repulsion_gain, side_safety_dist, side_angle_window_rad;
        double integral_clamp;
        int history_size;
    };

    void refreshParams(P& p) {
        p.fov_degrees          = get_parameter("fov_degrees").as_double();
        p.history_size         = get_parameter("history_size").as_int();
        p.max_range            = get_parameter("max_range").as_double();
        p.car_width            = get_parameter("car_width").as_double();
        p.disparity_threshold  = get_parameter("disparity_threshold").as_double();
        p.lookahead_distance   = get_parameter("lookahead_distance").as_double();
        p.bubble_radius        = get_parameter("bubble_radius").as_double();
        p.aim_window_degrees   = get_parameter("aim_window_degrees").as_double();
        p.goal_smoothing_alpha = get_parameter("goal_smoothing_alpha").as_double();
        p.goal_deadband_rad    = get_parameter("goal_deadband_deg").as_double() * M_PI / 180.0;
        p.max_speed            = get_parameter("max_speed").as_double();
        p.min_speed            = get_parameter("min_speed").as_double();
        p.kp                   = get_parameter("kp").as_double();
        p.ki                   = get_parameter("ki").as_double();
        p.kd                   = get_parameter("kd").as_double();
        p.steer_smoothing      = get_parameter("steer_smoothing").as_double();
        p.wall_clearance       = get_parameter("wall_clearance").as_double();
        p.repulsion_gain       = get_parameter("repulsion_gain").as_double();
        p.side_safety_dist     = get_parameter("side_safety_dist").as_double();
        p.side_angle_window_rad= get_parameter("side_angle_window").as_double() * M_PI / 180.0;
        p.integral_clamp       = get_parameter("integral_clamp").as_double();
    }

    // ── Wall Smoothener ──
    struct WallResult { double combined, repulsion; std::string status; };

    WallResult applyWallSmoothener(const P& p, double angle_min, double angle_inc, int n, double base_steer) {
        auto min_in = [&](int lo, int hi) -> double {
            lo = std::max(lo, 0); hi = std::min(hi, n);
            if (lo >= hi) return p.max_range;
            return *std::min_element(smoothed_.begin() + lo, smoothed_.begin() + hi);
        };

        int ls = std::max(0, (int)((M_PI / 3.0    - angle_min) / angle_inc));
        int le = std::min(n, (int)((2*M_PI / 3.0  - angle_min) / angle_inc));
        int rs = std::max(0, (int)((-2*M_PI / 3.0 - angle_min) / angle_inc));
        int re = std::min(n, (int)((-M_PI / 3.0   - angle_min) / angle_inc));

        double ml = min_in(ls, le), mr = min_in(rs, re);

        double rep = 0.0;
        std::string status = "NONE";
        if (ml < p.wall_clearance) { rep -= p.repulsion_gain * (p.wall_clearance - ml); status = "PUSHING RIGHT"; }
        if (mr < p.wall_clearance) { rep += p.repulsion_gain * (p.wall_clearance - mr); status = (status == "NONE") ? "PUSHING LEFT" : "SQUEEZED"; }

        double combined = base_steer + rep;

        int li = (int)((M_PI/2.0  - angle_min) / angle_inc);
        int ri = (int)((-M_PI/2.0 - angle_min) / angle_inc);
        int wb = (int)(p.side_angle_window_rad / 2.0 / angle_inc);

        double cml = min_in(li - wb, li + wb);
        double cmr = min_in(ri - wb, ri + wb);

        if (combined > 0.0 && cml < p.side_safety_dist)      { combined = 0.0; status = "LEFT PROT"; }
        else if (combined < 0.0 && cmr < p.side_safety_dist)  { combined = 0.0; status = "RIGHT PROT"; }

        return {combined, rep, status};
    }

    // ── Main Callback ──
    void lidarCallback(const sensor_msgs::msg::LaserScan::SharedPtr data) {
        P p; refreshParams(p);

        double angle_min = data->angle_min;
        double angle_inc = data->angle_increment;
        int n = (int)data->ranges.size();
        int center = n / 2;

        // Trim history
        while ((int)scan_history_.size() >= p.history_size) scan_history_.pop_front();

        // ── 1. Preprocess (FIX #3: reuse clean_ buffer) ──
        clean_.resize(n);
        for (int i = 0; i < n; ++i) {
            double v = data->ranges[i];
            if (std::isnan(v) || std::isinf(v)) v = p.max_range;
            clean_[i] = std::clamp(v, 0.0, p.max_range);
        }
        scan_history_.push_back(clean_);

        // Temporal mean (FIX #3: reuse smoothed_)
        smoothed_.assign(n, 0.0);
        for (const auto& h : scan_history_)
            for (int i = 0; i < n; ++i) smoothed_[i] += h[i];
        double inv = 1.0 / (double)scan_history_.size();
        for (int i = 0; i < n; ++i) smoothed_[i] *= inv;

        // ── 2. Extract FOV (FIX #3: reuse scan_) ──
        int half_fov = (int)(p.fov_degrees / 2.0 * M_PI / 180.0 / angle_inc);
        int lo = std::max(0, center - half_fov);
        int hi = std::min(n, center + half_fov);
        int scan_len = hi - lo;

        scan_.assign(smoothed_.begin() + lo, smoothed_.begin() + hi);

        // ── 3. Disparity Extender ──
        for (int i = 0; i + 1 < scan_len; ++i) {
            double diff = scan_[i + 1] - scan_[i];
            if (std::abs(diff) <= p.disparity_threshold) continue;
            double closer = std::min(scan_[i], scan_[i + 1]);
            if (closer < 0.1) continue;
            int w = (int)std::ceil(std::atan2(p.car_width / 2.0, closer) / angle_inc);
            if (diff > 0) {
                for (int j = i + 1; j < std::min(scan_len, i + 1 + w); ++j) scan_[j] = 0.0;
            } else {
                for (int j = std::max(0, i - w + 1); j <= i; ++j) scan_[j] = 0.0;
            }
        }

        // ── 4. Obstacle Bubbling ──
        int closest_idx = 0;
        double closest_dist = scan_[0];
        for (int i = 1; i < scan_len; ++i)
            if (scan_[i] < closest_dist) { closest_dist = scan_[i]; closest_idx = i; }

        if (closest_dist > 0.0) {
            double ratio = p.bubble_radius / closest_dist;
            int bw = (ratio >= 1.0) ? scan_len : (int)std::ceil(std::asin(ratio) / angle_inc);
            for (int i = std::max(0, closest_idx - bw); i < std::min(scan_len, closest_idx + bw + 1); ++i)
                scan_[i] = 0.0;
        }

        // ── 5. Gap Finding with Aim Window (FIX #3: reuse aim_mask_) ──
        aim_mask_.assign(scan_len, 0.0);
        int aim_half  = (int)(p.aim_window_degrees * M_PI / 180.0 / 2.0 / angle_inc);
        int aim_start = std::max(0, scan_len / 2 - aim_half);
        int aim_end   = std::min(scan_len, scan_len / 2 + aim_half);
        for (int i = aim_start; i < aim_end; ++i) aim_mask_[i] = 1.0;

        // Find gap runs
        struct Gap { int s, e; };
        gaps_.clear();
        bool in_gap = false; int gs = 0;
        for (int i = 0; i <= scan_len; ++i) {
            bool valid = (i < scan_len) && (scan_[i] > 0.1);
            if (valid && !in_gap)  { gs = i; in_gap = true; }
            if (!valid && in_gap)  { gaps_.push_back({gs, i}); in_gap = false; }
        }

        std::pair<int,int> best_gap;
        int deepest_idx;

        if (!gaps_.empty()) {
            double best_score = -1.0; int best_gi = 0;
            for (int gi = 0; gi < (int)gaps_.size(); ++gi) {
                double mx = 0.0;
                for (int j = gaps_[gi].s; j < gaps_[gi].e; ++j)
                    mx = std::max(mx, scan_[j] * aim_mask_[j]);
                if (mx > best_score) { best_score = mx; best_gi = gi; }
            }
            best_gap = {gaps_[best_gi].s, gaps_[best_gi].e};

            double deep_best = -1.0; deepest_idx = best_gap.first;
            for (int j = best_gap.first; j < best_gap.second; ++j) {
                double v = scan_[j] * aim_mask_[j];
                if (v > deep_best) { deep_best = v; deepest_idx = j; }
            }
        } else {
            best_gap = {scan_len / 2, scan_len / 2 + 1};
            deepest_idx = scan_len / 2;
        }

        double deep_dist  = std::min(scan_[deepest_idx], p.lookahead_distance);
        double deep_angle = angle_min + (double)(deepest_idx + lo) * angle_inc;
        double deep_x = deep_dist * std::cos(deep_angle);
        double deep_y = deep_dist * std::sin(deep_angle);

        // ── 6. Cartesian EMA Goal Filtering ──
        if (!goal_initialized_) {
            sgx_ = deep_x; sgy_ = deep_y; goal_initialized_ = true;
        } else {
            sgx_ = p.goal_smoothing_alpha * deep_x + (1.0 - p.goal_smoothing_alpha) * sgx_;
            sgy_ = p.goal_smoothing_alpha * deep_y + (1.0 - p.goal_smoothing_alpha) * sgy_;
        }
        double smoothed_target_angle = std::atan2(sgy_, sgx_);

        // Deadband
        if (std::abs(smoothed_target_angle - prev_target_angle_) > p.goal_deadband_rad)
            prev_target_angle_ = smoothed_target_angle;
        else
            smoothed_target_angle = prev_target_angle_;

        // ── 7. PID (FIX #1: integral windup clamp) ──
        double t = now().nanoseconds() / 1e9;
        double dt = t - prev_time_;
        if (dt <= 0.0) dt = 0.01;

        double error = smoothed_target_angle;
        integral_error_ += error * dt;
        integral_error_ = std::clamp(integral_error_, -p.integral_clamp, p.integral_clamp);  // ← WINDUP FIX
        double derivative = (error - prev_error_) / dt;

        double pid_steer = p.kp * error + p.ki * integral_error_ + p.kd * derivative;
        prev_error_ = error;
        prev_time_  = t;

        // ── 8. Wall Smoothener ──
        auto [target_steer, repulsion_val, protection_status] =
            applyWallSmoothener(p, angle_min, angle_inc, n, pid_steer);

        double steering_angle = p.steer_smoothing * target_steer + (1.0 - p.steer_smoothing) * prev_steer_;
        steering_angle = std::clamp(steering_angle, -0.4, 0.4);
        prev_steer_ = steering_angle;

        // ── 9. Speed Control ──
        double abs_steer = std::abs(steering_angle);
        double speed;
        constexpr double lo_t = 10.0 * M_PI / 180.0, hi_t = 20.0 * M_PI / 180.0;
        if      (abs_steer < lo_t) speed = p.max_speed;
        else if (abs_steer > hi_t) speed = p.min_speed;
        else { double f = (abs_steer - lo_t) / (hi_t - lo_t); speed = p.max_speed - f * (p.max_speed - p.min_speed); }

        // ── 10. Publish Drive ──
        auto msg = ackermann_msgs::msg::AckermannDriveStamped();
        msg.drive.speed          = (float)speed;
        msg.drive.steering_angle = (float)steering_angle;
        drive_pub_->publish(msg);

        RCLCPP_INFO_THROTTLE(this->get_logger(), *get_clock(), 1000, "drive speed=%.2f m/s steer=%.3f rad", speed, steering_angle);

        // ── 11. Visualization ──
        viz_full_.resize(n);
        for (int i = 0; i < n; ++i) viz_full_[i] = (float)data->ranges[i];
        for (int i = 0; i < scan_len; ++i) viz_full_[lo + i] = (float)scan_[i];

        auto scan_msg = sensor_msgs::msg::LaserScan();
        scan_msg.header = data->header; scan_msg.header.stamp = now();
        scan_msg.angle_min = data->angle_min; scan_msg.angle_max = data->angle_max;
        scan_msg.angle_increment = data->angle_increment;
        scan_msg.time_increment = data->time_increment;
        scan_msg.scan_time = data->scan_time;
        scan_msg.range_min = data->range_min; scan_msg.range_max = data->range_max;
        scan_msg.ranges = viz_full_;
        viz_scan_pub_->publish(scan_msg);

        publishMarkers(data, lo, best_gap, sgx_, sgy_, deep_x, deep_y,
                       steering_angle, repulsion_val, speed, protection_status);
    }

    // ── Marker Visualization ──
    void publishMarkers(
            const sensor_msgs::msg::LaserScan::SharedPtr& data, int lo,
            std::pair<int,int> gap, double sx, double sy, double dx, double dy,
            double steer, double repulse, double speed, const std::string& prot) {

        double a_min = data->angle_min, a_inc = data->angle_increment;
        auto stamp = now();
        std::string fid = data->header.frame_id.empty() ? "laser" : data->header.frame_id;

        struct XY { double x, y; };
        auto to_xy = [&](int idx, double r) -> XY {
            double a = a_min + (double)idx * a_inc;
            return {r * std::cos(a), r * std::sin(a)};
        };

        visualization_msgs::msg::MarkerArray ma;
        auto hdr = [&](visualization_msgs::msg::Marker& m, const std::string& ns, int id) {
            m.header.frame_id = fid; m.header.stamp = stamp; m.ns = ns; m.id = id;
        };

        // 1. Gap contour
        { visualization_msgs::msg::Marker m; hdr(m, "gap", 0);
          m.type = visualization_msgs::msg::Marker::LINE_STRIP;
          m.action = visualization_msgs::msg::Marker::ADD;
          m.scale.x = 0.08; m.color.g = 1.0f; m.color.a = 1.0f;
          for (int i = gap.first; i < gap.second; ++i) {
              if (scan_[i] > 0.1) {
                  XY xy = to_xy(lo + i, scan_[i]);
                  geometry_msgs::msg::Point pt; pt.x = xy.x; pt.y = xy.y; m.points.push_back(pt);
              }
          }
          ma.markers.push_back(m);
        }

        // 2. Deepest point (cyan)
        { visualization_msgs::msg::Marker m; hdr(m, "deepest_point", 0);
          m.type = visualization_msgs::msg::Marker::SPHERE;
          m.action = visualization_msgs::msg::Marker::ADD;
          m.pose.position.x = dx; m.pose.position.y = dy; m.pose.orientation.w = 1.0;
          m.scale.x = m.scale.y = m.scale.z = 0.25;
          m.color.g = 1.0f; m.color.b = 1.0f; m.color.a = 0.7f;
          ma.markers.push_back(m);
        }

        // 3. EMA goal (red)
        { visualization_msgs::msg::Marker m; hdr(m, "smooth_goal", 0);
          m.type = visualization_msgs::msg::Marker::SPHERE;
          m.action = visualization_msgs::msg::Marker::ADD;
          m.pose.position.x = sx; m.pose.position.y = sy; m.pose.orientation.w = 1.0;
          m.scale.x = m.scale.y = m.scale.z = 0.4;
          m.color.r = 1.0f; m.color.a = 1.0f;
          ma.markers.push_back(m);
        }

        // 4. Steering arrow
        { visualization_msgs::msg::Marker m; hdr(m, "steer", 0);
          m.type = visualization_msgs::msg::Marker::ARROW;
          m.action = visualization_msgs::msg::Marker::ADD;
          geometry_msgs::msg::Point p0, p1;
          p0.z = 0.05; p1.x = 2.0*std::cos(steer); p1.y = 2.0*std::sin(steer); p1.z = 0.05;
          m.points = {p0, p1}; m.scale.x = 0.1; m.scale.y = 0.15;
          if (prot.find("PROT") != std::string::npos) { m.color.r = 1.0f; m.color.a = 1.0f; }
          else { m.color.r = 1.0f; m.color.g = 0.85f; m.color.a = 1.0f; }
          ma.markers.push_back(m);
        }

        // 5. Repulsion arrow
        { visualization_msgs::msg::Marker m; hdr(m, "repulsion_force", 0);
          if (std::abs(repulse) > 0.01) {
              m.type = visualization_msgs::msg::Marker::ARROW;
              m.action = visualization_msgs::msg::Marker::ADD;
              double rl = 4.0 * std::abs(repulse);
              double ang = (repulse > 0) ? M_PI/2.0 : -M_PI/2.0;
              geometry_msgs::msg::Point p0, p1;
              p0.z = 0.1; p1.x = rl*std::cos(ang); p1.y = rl*std::sin(ang); p1.z = 0.1;
              m.points = {p0, p1}; m.scale.x = 0.15; m.scale.y = 0.2;
              m.color.r = 1.0f; m.color.b = 1.0f; m.color.a = 1.0f;
          } else { m.action = visualization_msgs::msg::Marker::DELETE; }
          ma.markers.push_back(m);
        }

        // 6. HUD text
        { visualization_msgs::msg::Marker m; hdr(m, "hud", 0);
          m.type = visualization_msgs::msg::Marker::TEXT_VIEW_FACING;
          m.action = visualization_msgs::msg::Marker::ADD;
          m.pose.position.x = -1.0; m.pose.position.z = 0.5; m.pose.orientation.w = 1.0;
          m.scale.z = 0.25;
          char buf[128];
          if (prot != "NONE") {
              m.color.r = 1.0f; m.color.g = 0.2f; m.color.b = 1.0f; m.color.a = 1.0f;
              std::snprintf(buf, sizeof(buf), "Spd: %.1f m/s | Steer: %.0f deg\n[ %s ]",
                            speed, steer*180.0/M_PI, prot.c_str());
          } else {
              m.color.r = m.color.g = m.color.b = m.color.a = 1.0f;
              std::snprintf(buf, sizeof(buf), "Spd: %.1f m/s | Steer: %.0f deg", speed, steer*180.0/M_PI);
          }
          m.text = buf; ma.markers.push_back(m);
        }

        viz_gap_pub_->publish(ma);
    }

    // ── ROS I/O ──
    rclcpp::Subscription<sensor_msgs::msg::LaserScan>::SharedPtr scan_sub_;
    rclcpp::Publisher<ackermann_msgs::msg::AckermannDriveStamped>::SharedPtr drive_pub_;
    rclcpp::Publisher<sensor_msgs::msg::LaserScan>::SharedPtr viz_scan_pub_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr viz_gap_pub_;

    // ── Pre-allocated buffers (FIX #3) ──
    std::vector<double> clean_;        // sanitized input
    std::vector<double> smoothed_;     // temporal mean
    std::vector<double> scan_;         // FOV working copy
    std::vector<double> aim_mask_;     // aim window
    std::vector<float>  viz_full_;     // viz scan output
    struct GapRun { int s, e; };
    std::vector<GapRun> gaps_;         // gap run buffer

    // ── State ──
    std::deque<std::vector<double>> scan_history_;
    double sgx_ = 0.0, sgy_ = 0.0;
    bool   goal_initialized_ = false;
    double prev_steer_ = 0.0, prev_target_angle_ = 0.0, prev_time_ = 0.0;
    double integral_error_ = 0.0, prev_error_ = 0.0;
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<ReactiveFollowGap>());
    rclcpp::shutdown();
    return 0;
}