from pathlib import Path
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

def generate_launch_description():
    folder=Path(get_package_share_directory('arm_description'))
    return LaunchDescription([
        Node(package='robot_state_publisher', executable='robot_state_publisher',
             parameters=[{'robot_description':ParameterValue((folder/'urdf/arm.urdf').read_text(),value_type=str)}]),
        Node(package='arm_description', executable='initial_pose_publisher.py',output='screen'),
        Node(package='rviz2', executable='rviz2',arguments=['-d',str(folder/'config/display.rviz')]),
    ])
