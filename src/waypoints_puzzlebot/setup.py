from setuptools import find_packages, setup
import os
from glob import glob


package_name = 'waypoints_puzzlebot'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name), glob('launch/*.launch.py'))
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='puzzlebot',
    maintainer_email='oscardelarosalopez05@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            "waypoints = waypoints_puzzlebot.8_waypoints:main",
            "test = waypoints_puzzlebot.test_odometry:main",
            "pid_id = waypoints_puzzlebot.pid_identification:main",
            "line_follow = waypoints_puzzlebot.line_follow:main"

        ],
    },
)
