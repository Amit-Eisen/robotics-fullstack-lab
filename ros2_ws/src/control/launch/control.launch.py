#!/usr/bin/env python3

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    config_file = PathJoinSubstitution([
        FindPackageShare('control'),
        'config',
        'control_params.yaml'
    ])
    
    control_node = Node(
        package='control',
        executable='control_node',
        name='control_node',
        output='screen',
        parameters=[config_file]
    )
    
    return LaunchDescription([
        control_node,
    ])
