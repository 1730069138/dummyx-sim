#!/usr/bin/env python3
"""Publish accepted CAD pose and nonlinear gripper coupling. Never sends motor commands."""
import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from ament_index_python.packages import get_package_share_directory
from gripper_kinematics import coupled_positions

class PosePublisher(Node):
    def __init__(self):
        super().__init__('arm_initial_pose')
        folder=Path(get_package_share_directory('arm_description'))
        self.positions=json.loads((folder/'config/initial_pose.json').read_text())['joint_positions']
        self.limits={}
        for j in ET.parse(folder/'urdf/arm.urdf').getroot().findall('joint'):
            lim=j.find('limit')
            if lim is not None:self.limits[j.get('name')]=(float(lim.get('lower')),float(lim.get('upper')))
        self.pub=self.create_publisher(JointState,'joint_states',10)
        self.sub=self.create_subscription(JointState,'arm_command',self.command,10)
        self.timer=self.create_timer(0.1,self.publish)
    def command(self,msg):
        if len(msg.name)!=len(msg.position):
            self.get_logger().error('Rejected command: name/position length mismatch');return
        candidate=self.positions.copy()
        for name,value in zip(msg.name,msg.position):
            if name not in {f'joint{i}' for i in range(1,8)}:
                self.get_logger().error('Only joint1..joint7 accept commands');return
            lo,hi=self.limits[name]
            if not math.isfinite(value) or not lo-1e-9<=value<=hi+1e-9:
                self.get_logger().error('Rejected out-of-range command: '+name);return
            candidate[name]=value
        candidate.update(coupled_positions(candidate['joint7']))
        self.positions=candidate
    def publish(self):
        msg=JointState();msg.header.stamp=self.get_clock().now().to_msg()
        msg.name=list(self.positions);msg.position=list(self.positions.values());self.pub.publish(msg)

def main():
    rclpy.init();node=PosePublisher()
    try:rclpy.spin(node)
    finally:node.destroy_node();rclpy.shutdown()
if __name__=='__main__':main()
