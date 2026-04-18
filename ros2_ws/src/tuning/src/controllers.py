"""
Pure Pursuit and PID controllers for autonomous vehicle control.
Python implementation mirroring the C++ control package for fast tuning iterations.
"""

import math
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional


@dataclass
class Pose2D:
    x: float
    y: float
    yaw: float


@dataclass
class Waypoint:
    x: float
    y: float


class PIDController:
    """PID controller with feedforward term for velocity control."""
    
    def __init__(self, kp: float, ki: float, kd: float, kff: float = 0.0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.kff = kff
        
        self.integral = 0.0
        self.prev_error = 0.0
        self.has_prev_error = False
        
        self.min_output = -1000.0
        self.max_output = 1000.0
    
    def compute(self, current: float, dt: float, target: float) -> float:
        """Compute PID output given current value, timestep, and target."""
        if dt <= 0.0:
            return 0.0
        
        error = target - current
        self.integral += error * dt
        
        derivative = 0.0
        if self.has_prev_error:
            derivative = (error - self.prev_error) / dt
        
        self.prev_error = error
        self.has_prev_error = True
        
        output = (self.kp * error + 
                  self.ki * self.integral + 
                  self.kd * derivative + 
                  self.kff * target)
        
        return np.clip(output, self.min_output, self.max_output)
    
    def reset(self):
        self.integral = 0.0
        self.prev_error = 0.0
        self.has_prev_error = False
    
    def set_limits(self, min_output: float, max_output: float):
        self.min_output = min_output
        self.max_output = max_output
    
    def set_gains(self, kp: float, ki: float, kd: float, kff: float):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.kff = kff


class PurePursuit:
    """
    Pure Pursuit path tracking controller with velocity-dependent lookahead.
    Lookahead distance: ld = ld_min + k_ld * velocity
    """
    
    def __init__(self, lookahead_min: float, k_lookahead: float, wheelbase: float):
        self.ld_min = lookahead_min
        self.k_ld = k_lookahead
        self.wheelbase = wheelbase
        
        self.path: List[Waypoint] = []
        self.current_index = 0
        self.lookahead_point: Optional[Waypoint] = None
        
        self.max_steering = 0.5
    
    def set_path(self, path: List[Waypoint]):
        self.path = path
        self.current_index = 0
    
    def get_lookahead_distance(self, velocity: float) -> float:
        """Calculate velocity-dependent lookahead distance."""
        return self.ld_min + self.k_ld * abs(velocity)
    
    def compute_steering(self, current_pos: Pose2D, velocity: float) -> Tuple[float, bool]:
        """
        Compute steering angle using Pure Pursuit algorithm.
        Returns (steering_angle, success).
        """
        if not self.path:
            return 0.0, False
        
        lookahead_dist = self.get_lookahead_distance(velocity)
        
        if not self._find_lookahead_point(current_pos, lookahead_dist):
            return 0.0, False
        
        dx = self.lookahead_point.x - current_pos.x
        dy = self.lookahead_point.y - current_pos.y
        
        local_x = dx * math.cos(-current_pos.yaw) - dy * math.sin(-current_pos.yaw)
        local_y = dx * math.sin(-current_pos.yaw) + dy * math.cos(-current_pos.yaw)
        
        ld_squared = local_x * local_x + local_y * local_y
        
        if ld_squared < 0.01:
            return 0.0, True
        
        curvature = 2.0 * local_y / ld_squared
        steering_angle = math.atan(self.wheelbase * curvature)
        steering_angle = np.clip(steering_angle, -self.max_steering, self.max_steering)
        
        return steering_angle, True
    
    def _find_lookahead_point(self, current_pos: Pose2D, lookahead_dist: float) -> bool:
        """Find the lookahead point on the path."""
        if not self.path:
            return False
        
        min_dist = float('inf')
        closest_idx = self.current_index
        
        for i in range(self.current_index, len(self.path)):
            d = self._distance(current_pos.x, current_pos.y, 
                              self.path[i].x, self.path[i].y)
            if d < min_dist:
                min_dist = d
                closest_idx = i
        
        self.current_index = closest_idx
        
        for i in range(self.current_index, len(self.path)):
            d = self._distance(current_pos.x, current_pos.y,
                              self.path[i].x, self.path[i].y)
            if d >= lookahead_dist:
                self.lookahead_point = self.path[i]
                return True
        
        if self.path:
            self.lookahead_point = self.path[-1]
            return True
        
        return False
    
    def reached_goal(self, current_pos: Pose2D, tolerance: float) -> bool:
        if not self.path:
            return True
        return self.distance_to_goal(current_pos) < tolerance
    
    def distance_to_goal(self, current_pos: Pose2D) -> float:
        if not self.path:
            return 0.0
        return self._distance(current_pos.x, current_pos.y,
                             self.path[-1].x, self.path[-1].y)
    
    @staticmethod
    def _distance(x1: float, y1: float, x2: float, y2: float) -> float:
        return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    
    @staticmethod
    def _normalize_angle(angle: float) -> float:
        while angle > math.pi:
            angle -= 2.0 * math.pi
        while angle < -math.pi:
            angle += 2.0 * math.pi
        return angle
