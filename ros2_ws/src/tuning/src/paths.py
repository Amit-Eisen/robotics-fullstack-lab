"""
Test path generators for controller tuning.
Each generator returns a list of Waypoint objects.
"""

import math
import numpy as np
from typing import List
from controllers import Waypoint


def straight_path(length: float = 20.0, spacing: float = 0.5) -> List[Waypoint]:
    """Generate a straight path along the x-axis."""
    path = []
    x = 0.0
    while x <= length:
        path.append(Waypoint(x=x, y=0.0))
        x += spacing
    return path


def turn_90(radius: float = 5.0, spacing: float = 0.5, direction: str = 'left') -> List[Waypoint]:
    """Generate a 90-degree turn path. Starts straight, then curves."""
    path = []
    
    straight_length = 5.0
    x = 0.0
    while x < straight_length:
        path.append(Waypoint(x=x, y=0.0))
        x += spacing
    
    sign = 1 if direction == 'left' else -1
    center_x = straight_length
    center_y = sign * radius
    
    num_arc_points = int((math.pi / 2) * radius / spacing)
    for i in range(num_arc_points + 1):
        angle = -sign * math.pi / 2 + sign * (i / num_arc_points) * (math.pi / 2)
        px = center_x + radius * math.cos(angle)
        py = center_y + radius * math.sin(angle)
        path.append(Waypoint(x=px, y=py))
    
    end_x = center_x + radius
    end_y = center_y + sign * radius
    for i in range(1, int(straight_length / spacing) + 1):
        path.append(Waypoint(x=end_x, y=end_y + sign * i * spacing))
    
    return path


def s_curve(amplitude: float = 3.0, wavelength: float = 15.0, 
            length: float = 30.0, spacing: float = 0.5) -> List[Waypoint]:
    """Generate an S-curve path using sine wave."""
    path = []
    x = 0.0
    while x <= length:
        y = amplitude * math.sin(2 * math.pi * x / wavelength)
        path.append(Waypoint(x=x, y=y))
        x += spacing
    return path


def circuit(size: float = 15.0, corner_radius: float = 3.0, 
            spacing: float = 0.5) -> List[Waypoint]:
    """Generate a rectangular circuit with rounded corners."""
    path = []
    
    half = size / 2
    r = corner_radius
    
    segments = [
        ('straight', (-half + r, -half), (half - r, -half)),
        ('arc', (half - r, -half + r), r, -math.pi/2, 0),
        ('straight', (half, -half + r), (half, half - r)),
        ('arc', (half - r, half - r), r, 0, math.pi/2),
        ('straight', (half - r, half), (-half + r, half)),
        ('arc', (-half + r, half - r), r, math.pi/2, math.pi),
        ('straight', (-half, half - r), (-half, -half + r)),
        ('arc', (-half + r, -half + r), r, math.pi, 3*math.pi/2),
    ]
    
    for seg in segments:
        if seg[0] == 'straight':
            start, end = seg[1], seg[2]
            dist = math.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
            num_points = max(int(dist / spacing), 1)
            for i in range(num_points):
                t = i / num_points
                x = start[0] + t * (end[0] - start[0])
                y = start[1] + t * (end[1] - start[1])
                path.append(Waypoint(x=x, y=y))
        
        elif seg[0] == 'arc':
            center, radius, start_angle, end_angle = seg[1], seg[2], seg[3], seg[4]
            arc_length = abs(end_angle - start_angle) * radius
            num_points = max(int(arc_length / spacing), 1)
            for i in range(num_points):
                t = i / num_points
                angle = start_angle + t * (end_angle - start_angle)
                x = center[0] + radius * math.cos(angle)
                y = center[1] + radius * math.sin(angle)
                path.append(Waypoint(x=x, y=y))
    
    return path


def slalom(num_cones: int = 6, cone_spacing: float = 5.0, 
           lateral_offset: float = 2.0, spacing: float = 0.5) -> List[Waypoint]:
    """Generate a slalom path weaving between cones."""
    path = []
    
    total_length = (num_cones + 1) * cone_spacing
    x = 0.0
    
    while x <= total_length:
        cone_index = x / cone_spacing
        y = lateral_offset * math.sin(math.pi * cone_index)
        path.append(Waypoint(x=x, y=y))
        x += spacing
    
    return path


def get_all_test_paths() -> dict:
    """Return a dictionary of all test paths for comprehensive tuning."""
    return {
        'straight': straight_path(length=20.0),
        'turn_left': turn_90(radius=5.0, direction='left'),
        'turn_right': turn_90(radius=5.0, direction='right'),
        's_curve': s_curve(amplitude=3.0, wavelength=15.0, length=30.0),
        'circuit': circuit(size=15.0, corner_radius=3.0),
        'slalom': slalom(num_cones=6, cone_spacing=5.0),
    }
