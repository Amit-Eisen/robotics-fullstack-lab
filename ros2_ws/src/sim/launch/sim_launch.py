#!/usr/bin/env python3
"""Launch file for simulation with RViz"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch.conditions import IfCondition
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    use_rviz_arg = DeclareLaunchArgument(
        'use_rviz',
        default_value='true',
        description='Launch RViz2'
    )
    
    sim_node = Node(
        package='sim',
        executable='ros2_sim_node',
        name='ros2_sim_node',
        output='screen'
    )
    
    rviz_config_file = PathJoinSubstitution([
        FindPackageShare('sim'),
        'config',
        'sim.rviz'
    ])
    
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        arguments=['-d', rviz_config_file],
        condition=IfCondition(LaunchConfiguration('use_rviz'))
    )
    
    return LaunchDescription([
        use_rviz_arg,
        sim_node,
        rviz_node,
    ])
