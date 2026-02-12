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
            # Grasp Test            
            'test_grasp_uf850 = disassembly_skills.test_grasp_uf850:main',
            'detect_aruco = disassembly_skills.detect_aruco:main',
            'aruco_navigator = disassembly_skills.aruco_navigation_node:main',
            'screw_zone_targeter = disassembly_skills.screw_zone_targeter:main',
            'force_guarded_descent = disassembly_skills.force_guarded_descent:main',
            'visual_servo = disassembly_skills.visual_servo:main',
            'velocity_backend = disassembly_skills.velocity_backend:main',
        ],
    },
)
