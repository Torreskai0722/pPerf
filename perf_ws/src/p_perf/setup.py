from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'p_perf'

setup(
    name=package_name,
    version='0.1.3',
    packages=find_packages(exclude=['test']),       # this just find all the packages, excluding test
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='root',
    maintainer_email='myuguo@udel.edu',
    description='Depends on nsys to do the kernel level profiling',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            f'sensor_publish_node = {package_name}.sensor_publisher:main',
            f'inference_node = {package_name}.inferencer:main',
            f"sensor_replay_node = {package_name}.sensor_replayer:main",
            f'inferencer_ms_node = {package_name}.inferencer_ms:main',
        ],
    },
)
