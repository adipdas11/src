from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'hand_teleop'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'),
            glob(os.path.join('launch', '*launch.[pxy][yma]*'))),
        (os.path.join('share', package_name, 'config'),
            glob(os.path.join('config', '*.yaml'))
            + glob(os.path.join('config', '*.task'))),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='adip',
    maintainer_email='adip@todo.todo',
    description='Hand tracking teleoperation for dual-arm robot using MediaPipe',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'hand_tracker = hand_teleop.hand_tracker_node:main',
            'hand_robot_bridge = hand_teleop.hand_robot_bridge:main',
        ],
    },
)
