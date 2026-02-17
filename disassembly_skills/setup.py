from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'disassembly_skills'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # Add these two lines:
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='adip',
    maintainer_email='adipdas11@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [         
            'detect_aruco = disassembly_skills.detect_aruco:main',
            'aruco_navigator = disassembly_skills.aruco_navigation_node:main',
            'master_unscrew_agent = disassembly_skills.master_unscrew_agent:main',
            'object_hold_skill = disassembly_skills.object_hold_skill:main',
            'unscrew_skill = disassembly_skills.unscrew_skill:main',
            'aruco_tool_camera_calib = disassembly_skills.aruco_tool_camera_calib:main',
        ],
    },
)
