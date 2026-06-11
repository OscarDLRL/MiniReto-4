"""
traffic_waypoints.launch.py

Execute the traffic light detection as well as the 
waypoint odometry detection.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo, TimerAction
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    params_file = PathJoinSubstitution([
        FindPackageShare('waypoints_puzzlebot'),
        'config',
        'params.yaml'
    ])

    return LaunchDescription([
        DeclareLaunchArgument('detect_area', default_value='left'),
        DeclareLaunchArgument('debug_image', default_value='true'),

        LogInfo(msg='Launching trayectory with traffic light detection'),

        Node(
            package='micro_ros_agent',
            executable='micro_ros_agent',
            name='micro_ros_agent',
            output='screen',
            arguments=['serial', '--dev', '/dev/ttyUSB0'],
        ),

        Node(
            package='vision_puzzlebot',
            executable='traffic_detect',
            name='traffic_light_detector',
            output='screen',
        ),

        Node(
            package='waypoints_puzzlebot',
            executable='waypoints',
            name='trayectory_generator',
            output='screen',
        ),
    ])
