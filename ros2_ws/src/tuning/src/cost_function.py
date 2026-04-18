"""
Cost function for controller tuning optimization.
Computes a scalar cost from trajectory data for Optuna to minimize.
"""

import math
import numpy as np
from dataclasses import dataclass
from typing import List, Optional
from controllers import Waypoint, Pose2D


@dataclass
class TrajectoryPoint:
    pose: Pose2D
    velocity: float
    steering: float
    timestamp: float


@dataclass
class EpisodeResult:
    trajectory: List[TrajectoryPoint]
    path: List[Waypoint]
    collision: bool
    completed: bool
    total_time: float


@dataclass
class CostWeights:
    cross_track: float = 10.0
    heading: float = 2.0
    control_effort: float = 0.5
    control_smoothness: float = 1.0
    collision: float = 1000.0
    timeout: float = 100.0
    completion_bonus: float = -50.0


def compute_cross_track_error(trajectory: List[TrajectoryPoint], 
                               path: List[Waypoint]) -> float:
    """Compute mean cross-track error over the trajectory."""
    if not trajectory or not path:
        return float('inf')
    
    total_error = 0.0
    
    for traj_point in trajectory:
        min_dist = float('inf')
        for wp in path:
            dist = math.sqrt((traj_point.pose.x - wp.x)**2 + 
                           (traj_point.pose.y - wp.y)**2)
            min_dist = min(min_dist, dist)
        total_error += min_dist
    
    return total_error / len(trajectory)


def compute_heading_error(trajectory: List[TrajectoryPoint],
                          path: List[Waypoint]) -> float:
    """Compute mean heading error relative to path direction."""
    if len(trajectory) < 2 or len(path) < 2:
        return 0.0
    
    total_error = 0.0
    count = 0
    
    for traj_point in trajectory:
        closest_idx = 0
        min_dist = float('inf')
        
        for i, wp in enumerate(path):
            dist = math.sqrt((traj_point.pose.x - wp.x)**2 + 
                           (traj_point.pose.y - wp.y)**2)
            if dist < min_dist:
                min_dist = dist
                closest_idx = i
        
        if closest_idx < len(path) - 1:
            dx = path[closest_idx + 1].x - path[closest_idx].x
            dy = path[closest_idx + 1].y - path[closest_idx].y
            path_heading = math.atan2(dy, dx)
            
            heading_error = abs(_normalize_angle(traj_point.pose.yaw - path_heading))
            total_error += heading_error
            count += 1
    
    return total_error / count if count > 0 else 0.0


def compute_control_effort(trajectory: List[TrajectoryPoint]) -> float:
    """Compute total control effort (sum of absolute steering angles)."""
    if not trajectory:
        return 0.0
    
    return sum(abs(tp.steering) for tp in trajectory) / len(trajectory)


def compute_control_smoothness(trajectory: List[TrajectoryPoint]) -> float:
    """Compute steering rate of change (jerkiness penalty)."""
    if len(trajectory) < 2:
        return 0.0
    
    total_change = 0.0
    
    for i in range(1, len(trajectory)):
        dt = trajectory[i].timestamp - trajectory[i-1].timestamp
        if dt > 0:
            steering_rate = abs(trajectory[i].steering - trajectory[i-1].steering) / dt
            total_change += steering_rate
    
    return total_change / (len(trajectory) - 1)


def compute_cost(result: EpisodeResult, weights: Optional[CostWeights] = None) -> float:
    """
    Compute total cost for an episode. Lower is better.
    
    Components:
    - Cross-track error: How far from the path
    - Heading error: How aligned with path direction
    - Control effort: Penalize excessive steering
    - Control smoothness: Penalize jerky steering
    - Collision penalty: Large penalty for hitting obstacles
    - Timeout penalty: If didn't complete in time
    - Completion bonus: Reward for reaching the goal
    """
    if weights is None:
        weights = CostWeights()
    
    if not result.trajectory:
        return weights.collision + weights.timeout
    
    cte = compute_cross_track_error(result.trajectory, result.path)
    heading_err = compute_heading_error(result.trajectory, result.path)
    effort = compute_control_effort(result.trajectory)
    smoothness = compute_control_smoothness(result.trajectory)
    
    cost = (weights.cross_track * cte +
            weights.heading * heading_err +
            weights.control_effort * effort +
            weights.control_smoothness * smoothness)
    
    if result.collision:
        cost += weights.collision
    
    if not result.completed:
        cost += weights.timeout
    else:
        cost += weights.completion_bonus
    
    return cost


def _normalize_angle(angle: float) -> float:
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle
