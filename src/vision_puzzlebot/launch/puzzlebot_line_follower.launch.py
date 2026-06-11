"""
Puzzlebot Line Follower Launcher
By Arthwwr 
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo, TimerAction
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    params_file = PathJoinSubstitution([
        FindPackageShare('vision_puzzlebot'),
        'config',
        'params.yaml'
    ])

    return LaunchDescription([
        DeclareLaunchArgument('mode', default_value='hsv'),
        DeclareLaunchArgument('debug', default_value='true'),

        LogInfo(msg='------- Inicializando seguidor de linea (sin deteccion de señales) -------'),

        Node(
            package='vision_puzzlebot',
            executable='camera',
            name='vision_camera',
            output='screen',
        ),

        Node(
            package='vision_puzzlebot',
            executable='traffic_detect',
            name='vision_traffic',
            output='screen',
            parameters=[params_file],
        ),

        Node(
            package='vision_puzzlebot',
            executable='line_follower',
            name='vision_line_follower',
            output='screen',
            parameters=[params_file],
        ),

        Node(
            package='waypoints_puzzlebot',
            executable='line_follow',
            name='control_line_follower',
            output='screen',
            parameters=[params_file],
        ),
    ])
