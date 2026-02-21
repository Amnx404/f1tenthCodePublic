#include "rclcpp/rclcpp.hpp"
#include <string>
#include <vector>
#include <deque>
#include <algorithm>
#include <cmath>
#include <limits>
#include "sensor_msgs/msg/laser_scan.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "ackermann_msgs/msg/ackermann_drive_stamped.hpp"
#include "visualization_msgs/msg/marker_array.hpp"
#include "geometry_msgs/msg/point.hpp"
#include "geometry_msgs/msg/point_stamped.hpp"
#include "tf2_ros/transform_listener.h"
#include "tf2_ros/buffer.h"
#include "tf2_geometry_msgs/tf2_geometry_msgs.hpp"
#include <tf2/exceptions.h>

using std::placeholders::_1;

class ReactiveFollowGap : public rclcpp::Node {

public:
    ReactiveFollowGap() : Node("reactive_node")
    {
        this->declare_parameter<int>("window_size", 5);
        window_size_n_ = this->get_parameter("window_size").get_value<int>();
        this->declare_parameter<double>("speed_straight", 2.0);
        speed_straight_ = static_cast<float>(this->get_parameter("speed_straight").get_value<double>());
        this->declare_parameter<double>("speed_turning", 1.0);
        speed_turning_ = static_cast<float>(this->get_parameter("speed_turning").get_value<double>());
        this->declare_parameter<double>("steering_threshold_frac", 0.8);
        steering_threshold_frac_ = static_cast<float>(this->get_parameter("steering_threshold_frac").get_value<double>());
        this->declare_parameter<double>("steering_kp", 2.5);
        steering_kp_ = static_cast<float>(this->get_parameter("steering_kp").get_value<double>());
        this->declare_parameter<double>("max_steering", 0.4);
        max_steering_ = static_cast<float>(this->get_parameter("max_steering").get_value<double>());
        this->declare_parameter<double>("disparity_threshold", 0.3);
        disparity_threshold_ = static_cast<float>(this->get_parameter("disparity_threshold").get_value<double>());
        this->declare_parameter<double>("car_width", 0.3);
        car_width_ = static_cast<float>(this->get_parameter("car_width").get_value<double>());
        this->declare_parameter<double>("bubble_distance_threshold", 0.5);
        bubble_distance_threshold_ = static_cast<float>(this->get_parameter("bubble_distance_threshold").get_value<double>());
        this->declare_parameter<int>("bubble_radius_indices", 2);
        bubble_radius_indices_ = this->get_parameter("bubble_radius_indices").get_value<int>();
        this->declare_parameter<double>("fov_deg", 140.0);
        fov_half_rad_ = static_cast<float>(this->get_parameter("fov_deg").get_value<double>() * 0.5 * M_PI / 180.0);
        this->declare_parameter<int>("min_gap_size", 10);
        min_gap_size_ = this->get_parameter("min_gap_size").get_value<int>();

        lidar_sub_ = this->create_subscription<sensor_msgs::msg::LaserScan>(
            lidar_topic_, 10, std::bind(&ReactiveFollowGap::lidar_callback, this, _1));
        drive_pub_ = this->create_publisher<ackermann_msgs::msg::AckermannDriveStamped>(drive_topic_, 10);
        odom_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
            odom_topic_, 10, std::bind(&ReactiveFollowGap::odom_callback, this, _1));
        viz_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>("/gap_viz", 10);
        scan_before_pub_ = this->create_publisher<sensor_msgs::msg::LaserScan>("/gap_viz_scan_before", 10);
        scan_proc_pub_ = this->create_publisher<sensor_msgs::msg::LaserScan>("/gap_viz_scan_proc", 10);
        tf_buffer_ = std::make_shared<tf2_ros::Buffer>(this->get_clock());
        tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);
    }

    std::vector<float> get_averaged_ranges() const
    {
        const size_t count = lidar_deque_.size();
        if (count == 0 || lidar_sum_.empty()) return {};
        std::vector<float> out(lidar_sum_.size());
        const float inv_count = 1.0f / static_cast<float>(count);
        for (size_t i = 0; i < lidar_sum_.size(); ++i)
            out[i] = lidar_sum_[i] * inv_count;
        return out;
    }

private:
    std::string lidar_topic_ = "/scan";
    std::string drive_topic_ = "/drive";
    std::string odom_topic_ = "/odom";

    rclcpp::Subscription<sensor_msgs::msg::LaserScan>::SharedPtr lidar_sub_;
    rclcpp::Publisher<ackermann_msgs::msg::AckermannDriveStamped>::SharedPtr drive_pub_;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr viz_pub_;
    rclcpp::Publisher<sensor_msgs::msg::LaserScan>::SharedPtr scan_before_pub_;
    rclcpp::Publisher<sensor_msgs::msg::LaserScan>::SharedPtr scan_proc_pub_;
    std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
    std::shared_ptr<tf2_ros::TransformListener> tf_listener_;

    int window_size_n_;
    float speed_straight_;
    float speed_turning_;
    float steering_threshold_frac_;
    float steering_kp_;
    float max_steering_;
    float disparity_threshold_;
    float car_width_;
    float bubble_distance_threshold_;
    int bubble_radius_indices_;
    float fov_half_rad_;
    int min_gap_size_;

    std::deque<std::vector<float>> lidar_deque_;
    std::vector<float> lidar_sum_;

    // ── Preprocessing ──────────────────────────────────────────────────

    std::vector<float> preprocess_lidar(const std::vector<float>& ranges, float angle_increment)
    {
        std::vector<float> proc(ranges);
        for (size_t i = 1; i < proc.size(); i++) {
            float diff = std::abs(proc[i] - proc[i - 1]);
            if (diff > disparity_threshold_) {
                float closer = std::min(proc[i], proc[i - 1]);
                float inc = (angle_increment > 1e-6f) ? angle_increment : 0.00436f;
                int num_pts = static_cast<int>(std::atan2(car_width_, closer) / inc);
                int start = (proc[i] > proc[i - 1]) ? static_cast<int>(i) : static_cast<int>(i) - 1;
                int dir   = (proc[i] > proc[i - 1]) ? 1 : -1;
                for (int j = 0; j < num_pts; j++) {
                    int idx = start + j * dir;
                    if (idx >= 0 && idx < static_cast<int>(proc.size()))
                        proc[idx] = closer;
                }
            }
        }
        return proc;
    }

    void crop_fov(std::vector<float>& ranges, float angle_min, float angle_inc)
    {
        for (size_t i = 0; i < ranges.size(); ++i) {
            float angle = angle_min + static_cast<float>(i) * angle_inc;
            if (std::abs(angle) > fov_half_rad_)
                ranges[i] = 0.0f;
        }
    }

    void apply_bubble(std::vector<float>& ranges)
    {
        const int n = static_cast<int>(ranges.size());
        if (n == 0) return;
        int closest_i = -1;
        float min_r = std::numeric_limits<float>::max();
        for (int i = 0; i < n; ++i) {
            if (ranges[i] > 0.01f && ranges[i] < min_r) {
                min_r = ranges[i];
                closest_i = i;
            }
        }
        if (closest_i < 0 || min_r > bubble_distance_threshold_) return;
        int lo = std::max(0, closest_i - bubble_radius_indices_);
        int hi = std::min(n - 1, closest_i + bubble_radius_indices_);
        for (int i = lo; i <= hi; ++i) ranges[i] = 0.0f;
    }

    // ── Gap finding ────────────────────────────────────────────────────

    struct Gap {
        int start;
        int end;
        int deepest_i;      // index of the farthest point in this gap
        float deepest_r;    // range at that index
    };

    // Find all gaps. For each gap, record which point is deepest.
    std::vector<Gap> find_all_gaps(const std::vector<float>& ranges, float scan_range_max)
    {
        const float min_range = 0.1f;
        const float max_range = scan_range_max;
        const int n = static_cast<int>(ranges.size());
        std::vector<Gap> gaps;
        int i = 0;
        while (i < n) {
            if (ranges[i] >= min_range && ranges[i] <= max_range) {
                int seg_start = i;
                int best_i = i;
                float best_r = ranges[i];
                while (i < n && ranges[i] >= min_range && ranges[i] <= max_range) {
                    if (ranges[i] > best_r) {
                        best_r = ranges[i];
                        best_i = i;
                    }
                    ++i;
                }
                int seg_end = i - 1;
                // Only keep gaps wide enough for the car
                if ((seg_end - seg_start + 1) >= min_gap_size_)
                    gaps.push_back({seg_start, seg_end, best_i, best_r});
            } else {
                ++i;
            }
        }
        return gaps;
    }

    // Pick the gap whose deepest point is farthest. That's it.
    const Gap* pick_deepest_gap(const std::vector<Gap>& gaps)
    {
        if (gaps.empty()) return nullptr;
        const Gap* best = &gaps[0];
        for (size_t i = 1; i < gaps.size(); ++i) {
            if (gaps[i].deepest_r > best->deepest_r)
                best = &gaps[i];
        }
        return best;
    }

    // ── Sliding window average ─────────────────────────────────────────

    void update_lidar_deque_and_sum(const std::vector<float>& ranges)
    {
        const size_t num_beams = ranges.size();
        if (num_beams == 0) return;
        if (lidar_sum_.size() != num_beams)
            lidar_sum_.assign(num_beams, 0.0f);
        if (static_cast<int>(lidar_deque_.size()) < window_size_n_) {
            lidar_deque_.push_back(ranges);
            for (size_t i = 0; i < num_beams; ++i)
                lidar_sum_[i] += ranges[i];
        } else {
            const std::vector<float>& old = lidar_deque_.front();
            for (size_t i = 0; i < num_beams; ++i) {
                lidar_sum_[i] -= old[i];
                lidar_sum_[i] += ranges[i];
            }
            lidar_deque_.pop_front();
            lidar_deque_.push_back(ranges);
        }
    }

    // ── Visualization ──────────────────────────────────────────────────

    void publish_scan_viz(const sensor_msgs::msg::LaserScan::ConstSharedPtr& scan_msg,
                          const std::vector<float>& ranges,
                          rclcpp::Publisher<sensor_msgs::msg::LaserScan>::SharedPtr& pub)
    {
        if (ranges.size() != scan_msg->ranges.size()) return;
        sensor_msgs::msg::LaserScan out;
        out.header = scan_msg->header;
        out.header.stamp = this->now();
        out.angle_min = scan_msg->angle_min;
        out.angle_max = scan_msg->angle_max;
        out.angle_increment = scan_msg->angle_increment;
        out.time_increment = scan_msg->time_increment;
        out.scan_time = scan_msg->scan_time;
        out.range_min = scan_msg->range_min;
        out.range_max = scan_msg->range_max;
        out.ranges.assign(ranges.begin(), ranges.end());
        pub->publish(out);
    }

    static void index_to_xy(int index, float range, float angle_min,
                            float angle_inc, float& x, float& y)
    {
        float angle = angle_min + static_cast<float>(index) * angle_inc;
        x = range * std::cos(angle);
        y = range * std::sin(angle);
    }

    bool transform_point(const std::string& src, const std::string& tgt,
                          double x, double y, double z, rclcpp::Time stamp,
                          double& ox, double& oy, double& oz)
    {
        geometry_msgs::msg::PointStamped in_pt, out_pt;
        in_pt.header.frame_id = src;
        in_pt.header.stamp = stamp;
        in_pt.point.x = x; in_pt.point.y = y; in_pt.point.z = z;
        try {
            out_pt = tf_buffer_->transform(in_pt, tgt);
            ox = out_pt.point.x; oy = out_pt.point.y; oz = out_pt.point.z;
            return true;
        } catch (const tf2::TransformException&) { return false; }
    }

    void publish_gap_viz(const sensor_msgs::msg::LaserScan::ConstSharedPtr& scan_msg,
                          const Gap& gap)
    {
        const std::string scan_frame = scan_msg->header.frame_id.empty()
                                           ? "base_link" : scan_msg->header.frame_id;
        const float amin = scan_msg->angle_min;
        const float ainc = scan_msg->angle_increment;
        rclcpp::Time stamp = (scan_msg->header.stamp.sec == 0 &&
                              scan_msg->header.stamp.nanosec == 0)
                                 ? this->now() : rclcpp::Time(scan_msg->header.stamp);

        // Gap endpoints at the deepest range so the line is visible out in the world
        float x0, y0, x1, y1, gx, gy;
        index_to_xy(gap.start, gap.deepest_r, amin, ainc, x0, y0);
        index_to_xy(gap.end,   gap.deepest_r, amin, ainc, x1, y1);
        // Goal marker halfway to the deepest point
        index_to_xy(gap.deepest_i, gap.deepest_r * 0.5f, amin, ainc, gx, gy);

        double mx0=x0, my0=y0, mz0=0, mx1=x1, my1=y1, mz1=0, mgx=gx, mgy=gy, mgz=0;
        std::string frame_id = scan_frame;
        for (const std::string tgt : {"map", "odom"}) {
            if (transform_point(scan_frame, tgt, x0, y0, 0, stamp, mx0, my0, mz0) &&
                transform_point(scan_frame, tgt, x1, y1, 0, stamp, mx1, my1, mz1) &&
                transform_point(scan_frame, tgt, gx, gy, 0, stamp, mgx, mgy, mgz)) {
                frame_id = tgt;
                break;
            }
        }

        visualization_msgs::msg::MarkerArray ma;

        // Green line showing the gap opening at the depth of the deepest point
        visualization_msgs::msg::Marker line;
        line.header.stamp = this->now();
        line.header.frame_id = frame_id;
        line.ns = "gap"; line.id = 0;
        line.type = visualization_msgs::msg::Marker::LINE_STRIP;
        line.action = visualization_msgs::msg::Marker::ADD;
        line.scale.x = 0.15f;
        line.color.g = 1.0f; line.color.a = 1.0f;
        line.lifetime = rclcpp::Duration(0, 500000000);
        geometry_msgs::msg::Point p0, p1;
        p0.x = mx0; p0.y = my0; p0.z = mz0;
        p1.x = mx1; p1.y = my1; p1.z = mz1;
        line.points.push_back(p0);
        line.points.push_back(p1);
        ma.markers.push_back(line);

        // Red sphere = goal (halfway to deepest point)
        visualization_msgs::msg::Marker goal;
        goal.header.stamp = this->now();
        goal.header.frame_id = frame_id;
        goal.ns = "gap"; goal.id = 1;
        goal.type = visualization_msgs::msg::Marker::SPHERE;
        goal.action = visualization_msgs::msg::Marker::ADD;
        goal.pose.position.x = mgx;
        goal.pose.position.y = mgy;
        goal.pose.position.z = mgz;
        goal.pose.orientation.w = 1.0;
        goal.scale.x = goal.scale.y = goal.scale.z = 0.5f;
        goal.color.r = 1.0f; goal.color.a = 1.0f;
        goal.lifetime = rclcpp::Duration(0, 500000000);
        ma.markers.push_back(goal);

        viz_pub_->publish(ma);
    }

    // ── Main callback ──────────────────────────────────────────────────

    void lidar_callback(const sensor_msgs::msg::LaserScan::ConstSharedPtr scan_msg)
    {
        std::vector<float> ranges(scan_msg->ranges.begin(), scan_msg->ranges.end());
        update_lidar_deque_and_sum(ranges);

        auto avg = get_averaged_ranges();
        if (avg.empty()) return;
        publish_scan_viz(scan_msg, avg, scan_before_pub_);

        const float angle_min = scan_msg->angle_min;
        const float angle_inc = scan_msg->angle_increment;

        // Pipeline: disparity extend → crop FOV → safety bubble
        auto proc = preprocess_lidar(avg, angle_inc);
        crop_fov(proc, angle_min, angle_inc);
        apply_bubble(proc);
        publish_scan_viz(scan_msg, proc, scan_proc_pub_);

        // Find all gaps, pick the one with the farthest point
        auto gaps = find_all_gaps(proc, scan_msg->range_max);
        const Gap* best = pick_deepest_gap(gaps);
        if (!best) return;

        publish_gap_viz(scan_msg, *best);

        // Steer toward the deepest point
        float angle = angle_min + static_cast<float>(best->deepest_i) * angle_inc;
        float steering = steering_kp_ * angle;
        steering = std::max(-max_steering_, std::min(max_steering_, steering));

        float threshold = steering_threshold_frac_ * max_steering_;
        float speed = (std::abs(steering) >= threshold) ? speed_turning_ : speed_straight_;

        ackermann_msgs::msg::AckermannDriveStamped msg;
        msg.header.stamp = this->now();
        msg.header.frame_id = "base_link";
        msg.drive.steering_angle = steering;
        msg.drive.speed = speed;
        drive_pub_->publish(msg);
    }

    void odom_callback(const nav_msgs::msg::Odometry::ConstSharedPtr) {}
};

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<ReactiveFollowGap>());
    rclcpp::shutdown();
    return 0;
}