#!/usr/bin/env python3
import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg_share = get_package_share_directory('team1_gap_follow')
    config_path = os.path.join(pkg_share, 'config', 'team1_gap_follow.yaml')

    return LaunchDescription([
        DeclareLaunchArgument('params_file', default_value=config_path,
                             description='Path to team1_gap_follow params YAML'),
        Node(
            package='team1_gap_follow',
            executable='team1_gap_follow_cpp',
            name='reactive_node',
            output='screen',
            parameters=[LaunchConfiguration('params_file')],
        ),
    ])
