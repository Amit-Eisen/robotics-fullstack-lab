#pragma once

#include <vector>
#include <cmath>

struct Pose2D {
    double x;
    double y;
    double yaw;
};

struct Waypoint {
    double x;
    double y;
};

class PurePursuit {
public:
    // Constructor with velocity-dependent lookahead: ld = ld_min + k_ld * velocity
    PurePursuit(double lookahead_min, double k_lookahead, double wheelbase);
    
    // Set the path to follow (list of waypoints)
    void setPath(const std::vector<Waypoint>& path);
    
    // Compute steering angle given current pose and velocity. Returns true if path is valid.
    bool computeSteering(const Pose2D& currentPos, double velocity, double& steeringAngle);
    
    // Check if we reached the goal (last waypoint)
    bool reachedGoal(const Pose2D& currentPos, double tolerance = 0.5);
    
    // Get distance to goal
    double distanceToGoal(const Pose2D& currentPos);
    
    // Set lookahead parameters
    void setLookaheadParams(double lookahead_min, double k_lookahead);
    
    // Get current lookahead distance (for debugging/visualization)
    double getCurrentLookahead() const { return currentLookahead; }
    
    // Get current target point (for visualization)
    Waypoint getLookaheadPoint() const { return lookaheadPoint; }
    
    // Get current path index
    size_t getCurrentIndex() const { return currentIndex; }

private:
    // Calculate velocity-dependent lookahead distance
    double calcLookaheadDist(double velocity);
    
    // Find the lookahead point on the path
    bool findLookaheadPoint(const Pose2D& currentPos, double lookaheadDist);
    
    // Calculate distance between two points
    double distance(double x1, double y1, double x2, double y2);
    
    // Normalize angle to [-pi, pi]
    double normalizeAngle(double angle);

    double ldMin;
    double kLd;
    double wheelBase;
    double currentLookahead;
    std::vector<Waypoint> path;
    size_t currentIndex;
    Waypoint lookaheadPoint;
};
