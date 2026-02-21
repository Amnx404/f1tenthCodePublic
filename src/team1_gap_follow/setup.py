from setuptools import setup

package_name = 'team1_gap_follow'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='zzangupenn, Hongrui Zheng',
    maintainer_email='zzang@seas.upenn.edu, billyzheng.bz@gmail.com',
    description='Team1 F1TENTH gap follow (no name conflicts)',
    license='MIT',
    tests_require=['pytest'],
    # Node is run via CMake-installed script: ros2 run team1_gap_follow team1_gap_follow_node.py
)
