from setuptools import setup
import os
from glob import glob

package_name = 'sim'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    package_dir={package_name: 'src'},
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.rviz')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    description='Autonomous vehicle simulation',
    license='MIT',
    entry_points={
        'console_scripts': [
            'ros2_sim_node = sim.ros2_sim_node:main',
        ],
    },
)
