#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/twist.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <nav_msgs/msg/path.hpp>
#include <tf2/utils.h>

#include "pid_controller.hpp"
#include "pure_pursuit.hpp"

class ControlNode : public rclcpp::Node {
public:
    ControlNode() : Node("control_node") {
        declare_parameters();
        init_controllers();
        init_ros();
        
        RCLCPP_INFO(get_logger(), "Control Node initialized");
        RCLCPP_INFO(get_logger(), "Waiting for path on /path topic...");
    }

private:
    void declare_parameters() {
        declare_parameter("lookahead_min", 1.0);
        declare_parameter("k_lookahead", 0.5);
        declare_parameter("wheelbase", 0.3);
        declare_parameter("max_velocity", 10.0);
        declare_parameter("goal_tolerance", 0.5);
        declare_parameter("control_rate", 30.0);
        
        declare_parameter("pid.kp", 1.0);
        declare_parameter("pid.ki", 0.1);
        declare_parameter("pid.kd", 0.05);
        declare_parameter("pid.kff", 0.8);
    }
    
    void init_controllers() {
        double lookahead_min = get_parameter("lookahead_min").as_double();
        double k_lookahead = get_parameter("k_lookahead").as_double();
        double wheelbase = get_parameter("wheelbase").as_double();
        
        pure_pursuit_ = std::make_unique<PurePursuit>(lookahead_min, k_lookahead, wheelbase);
        
        double kp  = get_parameter("pid.kp").as_double();
        double ki  = get_parameter("pid.ki").as_double();
        double kd  = get_parameter("pid.kd").as_double();
        double kff = get_parameter("pid.kff").as_double();
        
        pidVel = std::make_unique<PIDController>(kp, ki, kd, kff);
        pidVel->setLimits(-5.0, 5.0);
        
        max_velocity_ = get_parameter("max_velocity").as_double();
        goal_tolerance_ = get_parameter("goal_tolerance").as_double();
    }
    
    void init_ros() {
        cmd_vel_pub_ = create_publisher<geometry_msgs::msg::Twist>("/cmd_vel", 10);
        
        odom_sub_ = create_subscription<nav_msgs::msg::Odometry>(
            "/odom", 10,
            std::bind(&ControlNode::odom_callback, this, std::placeholders::_1));
        
        path_sub_ = create_subscription<nav_msgs::msg::Path>(
            "/path", 10,
            std::bind(&ControlNode::path_callback, this, std::placeholders::_1));
        
        goal_sub_ = create_subscription<geometry_msgs::msg::PoseStamped>(
            "/goal_pose", 10,
            std::bind(&ControlNode::goal_callback, this, std::placeholders::_1));
        
        double rate = get_parameter("control_rate").as_double();
        timer_ = create_wall_timer(
            std::chrono::milliseconds(static_cast<int>(1000.0 / rate)),
            std::bind(&ControlNode::control_loop, this));
        
        last_time_ = now();
    }
    
    void odom_callback(const nav_msgs::msg::Odometry::SharedPtr msg) {
        current_pose_.x = msg->pose.pose.position.x;
        current_pose_.y = msg->pose.pose.position.y;
        
        tf2::Quaternion q(
            msg->pose.pose.orientation.x,
            msg->pose.pose.orientation.y,
            msg->pose.pose.orientation.z,
            msg->pose.pose.orientation.w);
        current_pose_.yaw = tf2::getYaw(q);
        
        currentVel = msg->twist.twist.linear.x;
        has_odom_ = true;
    }
    
    void path_callback(const nav_msgs::msg::Path::SharedPtr msg) {
        std::vector<Waypoint> path;
        for (const auto& pose : msg->poses) {
            path.push_back({pose.pose.position.x, pose.pose.position.y});
        }
        
        if (!path.empty()) {
            pure_pursuit_->setPath(path);
            pidVel->reset();
            has_path_ = true;
            RCLCPP_INFO(get_logger(), "Received path with %zu waypoints", path.size());
        }
    }
    
    void goal_callback(const geometry_msgs::msg::PoseStamped::SharedPtr msg) {
        // Simple straight-line path from current position to goal
        std::vector<Waypoint> path;
        
        if (has_odom_) {
            double goal_x = msg->pose.position.x;
            double goal_y = msg->pose.position.y;
            
            // Create simple path with intermediate points
            int num_points = 20;
            for (int i = 0; i <= num_points; ++i) {
                double t = static_cast<double>(i) / num_points;
                path.push_back({
                    current_pose_.x + t * (goal_x - current_pose_.x),
                    current_pose_.y + t * (goal_y - current_pose_.y)
                });
            }
            
            pure_pursuit_->setPath(path);
            pidVel->reset();
            has_path_ = true;
            RCLCPP_INFO(get_logger(), "Goal received: (%.2f, %.2f)", goal_x, goal_y);
        }
    }
    
    void control_loop() {
        if (!has_odom_ || !has_path_) {
            return;
        }
        
        auto current_time = now();
        double dt = (current_time - last_time_).seconds();
        last_time_ = current_time;
        
        if (dt <= 0.0 || dt > 1.0) {
            return;
        }
        
        geometry_msgs::msg::Twist cmd;
        
        // Check if reached goal
        if (pure_pursuit_->reachedGoal(current_pose_, goal_tolerance_)) {
            cmd.linear.x = 0.0;
            cmd.angular.z = 0.0;
            cmd_vel_pub_->publish(cmd);
            
            if (has_path_) {
                RCLCPP_INFO(get_logger(), "Goal reached!");
                has_path_ = false;
            }
            return;
        }
        
        // Compute steering with Pure Pursuit (velocity-dependent lookahead)
        double steeringAngle = 0.0;
        if (!pure_pursuit_->computeSteering(current_pose_, currentVel, steeringAngle)) {
            cmd.linear.x = 0.0;
            cmd.angular.z = 0.0;
            cmd_vel_pub_->publish(cmd);
            return;
        }
        
        // Compute velocity with PID + FF
        double tgtVel = max_velocity_;
        
        // Slow down near goal
        double dist_to_goal = pure_pursuit_->distanceToGoal(current_pose_);
        if (dist_to_goal < 2.0) {
            tgtVel = max_velocity_ * (dist_to_goal / 2.0);
            tgtVel = std::max(tgtVel, 0.3);
        }
        
        // Slow down for sharp turns
        if (std::abs(steeringAngle) > 0.3) {
            tgtVel *= 0.5;
        }
        
        double velocity_cmd = pidVel->compute(currentVel, dt, tgtVel);
        
        cmd.linear.x = velocity_cmd;
        cmd.angular.z = steeringAngle;
        
        cmd_vel_pub_->publish(cmd);
    }

    // Controllers
    std::unique_ptr<PurePursuit> pure_pursuit_;
    std::unique_ptr<PIDController> pidVel;
    
    // ROS
    rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr cmd_vel_pub_;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
    rclcpp::Subscription<nav_msgs::msg::Path>::SharedPtr path_sub_;
    rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr goal_sub_;
    rclcpp::TimerBase::SharedPtr timer_;
    
    // State
    Pose2D current_pose_{0.0, 0.0, 0.0};
    double currentVel{0.0};
    bool has_odom_{false};
    bool has_path_{false};
    
    // Parameters
    double max_velocity_;
    double goal_tolerance_;
    
    // Timing
    rclcpp::Time last_time_;
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<ControlNode>());
    rclcpp::shutdown();
    return 0;
}
