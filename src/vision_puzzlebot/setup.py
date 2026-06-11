from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'vision_puzzlebot'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),

    # Archivos que van dentro del paquete Python instalado
    package_data={
        package_name: [
            'best.onnx',
            'best.pt',
            'yolov4-tiny-signs.cfg',
            'yolov4-tiny-signs_good.weights',
            'obj.names',
        ],
    },
    include_package_data=True,

    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),

        ('share/' + package_name, ['package.xml']),

        (os.path.join('share', package_name, 'launch'),
            glob('launch/*.py')),

        (os.path.join('share', package_name, 'config'),
            glob('config/*.yaml')),

        (os.path.join('share', package_name, 'data'),
            glob('data/*.onnx') +
            glob('data/*.npz') +
            glob('data/*.cfg') +
            glob('data/*.weights') +
            glob('data/*.names')),

        # Templates de señales viales
        (os.path.join('share', package_name, 'data', 'signs'),
            glob('data/signs/*.jpg') +
            glob('data/signs/*.jpeg') +
            glob('data/signs/*.png')),
    ],

    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='puzzlebot',
    maintainer_email='oscardelarosalopez05@gmail.com',
    description='Vision nodes for PuzzleBot: line follower, traffic light, road signs',
    license='TODO: License declaration',
    tests_require=['pytest'],

    entry_points={
        'console_scripts': [
            'traffic_detect = vision_puzzlebot.trafficlight_detect:main',
            'line_follower  = vision_puzzlebot.line_follower_camera:main',
            'camera         = vision_puzzlebot.cam_publish:main',
        ],
    },
)
