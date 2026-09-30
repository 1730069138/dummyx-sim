"""Seven GUI sliders -> nonlinear gripper coupling -> full URDF / RViz.

The controls-only URDF supplies limits to the GUI, never to robot_state_publisher.
GUI 'Center' restores the accepted CAD initial pose via the zeros parameters.
"""
import json
from pathlib import Path
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    folder = Path(get_package_share_directory('arm_description'))
    initial = json.loads((folder / 'config/initial_pose.json').read_text())['joint_positions']
    gui_parameters = {'rate': 30}
    gui_parameters.update({f'zeros.joint{i}': float(initial[f'joint{i}']) for i in range(1, 8)})
    return LaunchDescription([
        DeclareLaunchArgument('rviz', default_value='true', description='Open RViz alongside sliders'),
        Node(
            package='robot_state_publisher', executable='robot_state_publisher',
            parameters=[{
                'robot_description': ParameterValue((folder / 'urdf/arm.urdf').read_text(), value_type=str),
                'publish_frequency': 30.0,
            }], output='screen'),
        Node(
            package='joint_state_publisher_gui', executable='joint_state_publisher_gui',
            name='arm_joint_sliders',
            arguments=[str(folder / 'config/gui_controls.urdf')],
            parameters=[gui_parameters],
            remappings=[('joint_states', 'arm_command')], output='screen'),
        # This is the only /joint_states publisher; it expands joint7 into passive joint values.
        Node(
            package='arm_description', executable='initial_pose_publisher.py',
            name='arm_gripper_coupling', output='screen'),
        Node(
            package='rviz2', executable='rviz2',
            arguments=['-d', str(folder / 'config/display.rviz')],
            condition=IfCondition(LaunchConfiguration('rviz'))),
    ])
