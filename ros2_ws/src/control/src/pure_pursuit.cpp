#include "pure_pursuit.hpp"
#include <algorithm>

PurePursuit::PurePursuit(double lookahead_min, double k_lookahead, double wheelbase)
    : ldMin(lookahead_min),
      kLd(k_lookahead),
      wheelBase(wheelbase),
      currentLookahead(lookahead_min),
      currentIndex(0) {
    lookaheadPoint = {0.0, 0.0};
}

void PurePursuit::setPath(const std::vector<Waypoint>& newPath)
{
    this->path = newPath;
    currentIndex = 0;
}

double PurePursuit::calcLookaheadDist(double velocity)
{
    return ldMin + kLd * std::abs(velocity);
}

bool PurePursuit::computeSteering(const Pose2D& currentPos, double velocity, double& steeringAngle)
{
    currentLookahead = calcLookaheadDist(velocity);
    
    if (path.empty() || !findLookaheadPoint(currentPos, currentLookahead))
    {
        steeringAngle = 0.0;
        return false;
    }

    double dx = lookaheadPoint.x - currentPos.x;
    double dy = lookaheadPoint.y - currentPos.y;

    double local_x = dx * cos(-currentPos.yaw) - dy * sin(-currentPos.yaw);
    double local_y = dx * sin(-currentPos.yaw) + dy * cos(-currentPos.yaw);

    double ld_squared = local_x * local_x + local_y * local_y;

    if (ld_squared < 0.01)
    {
        steeringAngle = 0.0;
        return true;
    }

    double curvature = 2.0 * local_y / ld_squared;
    steeringAngle = atan(wheelBase * curvature);
    steeringAngle = std::clamp(steeringAngle, -0.5, 0.5);

    return true;
}

bool PurePursuit::findLookaheadPoint(const Pose2D& currentPos, double lookaheadDist)
{
    if (path.empty()) return false;

    double min_dist = 1e9;
    size_t closest_idx = currentIndex;

    for (size_t i = currentIndex; i < path.size(); ++i)
    {
        double d = distance(currentPos.x, currentPos.y, path[i].x, path[i].y);
        if (d < min_dist)
        {
            min_dist = d;
            closest_idx = i;
        }
    }

    currentIndex = closest_idx;

    for (size_t i = currentIndex; i < path.size(); ++i)
    {
        double d = distance(currentPos.x, currentPos.y, path[i].x, path[i].y);
        if (d >= lookaheadDist)
        {
            lookaheadPoint = path[i];
            return true;
        }
    }

    if (!path.empty())
    {
        lookaheadPoint = path.back();
        return true;
    }

    return false;
}

bool PurePursuit::reachedGoal(const Pose2D& currentPos, double tolerance)
{
    if (path.empty()) return true;
    return distanceToGoal(currentPos) < tolerance;
}

double PurePursuit::distanceToGoal(const Pose2D& currentPos)
{
    if (path.empty()) return 0.0;
    return distance(currentPos.x, currentPos.y, path.back().x, path.back().y);
}

void PurePursuit::setLookaheadParams(double lookahead_min, double k_lookahead)
{
    ldMin = lookahead_min;
    kLd = k_lookahead;
}

double PurePursuit::distance(double x1, double y1, double x2, double y2)
{
    return sqrt((x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1));
}

double PurePursuit::normalizeAngle(double angle)
{
    while (angle > M_PI) angle -= 2.0 * M_PI;
    while (angle < -M_PI) angle += 2.0 * M_PI;
    return angle;
}
