"""Launch the standalone Hybrid-A* planner node."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg_share = get_package_share_directory('smac_planner_node')
    default_params = os.path.join(pkg_share, 'config', 'smac_planner_node.yaml')

    params_file = LaunchConfiguration('params_file')

    return LaunchDescription([
        DeclareLaunchArgument(
            'params_file',
            default_value=default_params,
            description='Full path to the parameter file for smac_planner_node',
        ),
        Node(
            package='smac_planner_node',
            executable='smac_planner_node',
            name='smac_planner_node',
            output='screen',
            parameters=[params_file],
        ),
    ])
