#!/usr/bin/env python3
"""ROS2 Simulation Node - Single threaded, ROS2 clock"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
import numpy as np
import cv2

from sensor_msgs.msg import Image, PointCloud2, PointField
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist, TransformStamped
from std_msgs.msg import Header
from tf2_ros import TransformBroadcaster
import struct

from sim.simulation import BasicSimulation


class SimulationNode(Node):
    """ROS2 node wrapping PyBullet simulation - all in main thread"""
    
    def __init__(self):
        super().__init__('simulation_node')
        self.get_logger().info('Initializing Simulation Node...')
        
        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )
        
        self.camera_pub = self.create_publisher(Image, '/camera/image_raw', qos)
        self.depth_pub = self.create_publisher(Image, '/camera/depth', qos)
        self.lidar_pub = self.create_publisher(PointCloud2, '/lidar/points', qos)
        self.odom_pub = self.create_publisher(Odometry, '/odom', qos)
        
        self.cmd_vel_sub = self.create_subscription(
            Twist, '/cmd_vel', self.cmd_vel_callback, qos
        )
        
        self.tf_broadcaster = TransformBroadcaster(self)
        
        self.sim = BasicSimulation()
        
        self.sim_hz = 30.0
        self.physics_steps_per_update = 8
        self.timer = self.create_timer(1.0 / self.sim_hz, self.simulation_step)
        
        self.show_visualization = True
        self.setup_visualization()
        
        self.get_logger().info('Simulation Node ready!')
        self.get_logger().info('Subscribe to /cmd_vel to control the car')
        self.get_logger().info('Publishing: /camera/image_raw, /lidar/points, /odom, /tf')
    
    def setup_visualization(self):
        """Setup OpenCV windows"""
        if self.show_visualization:
            cv2.namedWindow("Car Camera", cv2.WINDOW_NORMAL)
            cv2.namedWindow("External View", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Car Camera", 640, 480)
            cv2.resizeWindow("External View", 640, 480)
            cv2.moveWindow("Car Camera", 50, 50)
            cv2.moveWindow("External View", 700, 50)
    
    def cmd_vel_callback(self, msg: Twist):
        """Receive velocity commands"""
        self.sim.tgtVel = msg.linear.x
        self.sim.target_steering = msg.angular.z
    
    def simulation_step(self):
        """Main simulation loop - called by ROS2 timer"""

        for _ in range(self.physics_steps_per_update):
            self.sim.step()

        now = self.get_clock().now().to_msg()
        state = self.sim.get_car_state()

        self.publish_odometry(state, now)
        self.publish_tf(state, now)
        self.publish_camera(now)
        self.publish_lidar(state, now)

        if self.show_visualization:
            self.update_visualization(state)

    def publish_odometry(self, state, stamp):
        """Publish odometry message"""
        msg = Odometry()
        msg.header.stamp = stamp
        msg.header.frame_id = 'odom'
        msg.child_frame_id = 'base_link'

        msg.pose.pose.position.x = state['position'][0]
        msg.pose.pose.position.y = state['position'][1]
        msg.pose.pose.position.z = state['position'][2]

        msg.pose.pose.orientation.x = state['orientation'][0]
        msg.pose.pose.orientation.y = state['orientation'][1]
        msg.pose.pose.orientation.z = state['orientation'][2]
        msg.pose.pose.orientation.w = state['orientation'][3]

        msg.twist.twist.linear.x = state['linear_velocity'][0]
        msg.twist.twist.linear.y = state['linear_velocity'][1]
        msg.twist.twist.linear.z = state['linear_velocity'][2]

        msg.twist.twist.angular.x = state['angular_velocity'][0]
        msg.twist.twist.angular.y = state['angular_velocity'][1]
        msg.twist.twist.angular.z = state['angular_velocity'][2]

        self.odom_pub.publish(msg)

    def publish_tf(self, state, stamp):

        """Publish TF transform odom -> base_link"""

        t = TransformStamped()
        t.header.stamp = stamp
        t.header.frame_id = 'odom'
        t.child_frame_id = 'base_link'

        t.transform.translation.x = state['position'][0]
        t.transform.translation.y = state['position'][1]
        t.transform.translation.z = state['position'][2]

        t.transform.rotation.x = state['orientation'][0]
        t.transform.rotation.y = state['orientation'][1]
        t.transform.rotation.z = state['orientation'][2]
        t.transform.rotation.w = state['orientation'][3]

        self.tf_broadcaster.sendTransform(t)

    def publish_camera(self, stamp):

        """Publish camera image"""

        rgb_img, depth_img = self.sim.get_camera_image()
        self.last_camera_img = rgb_img

        msg = Image()
        msg.header.stamp = stamp
        msg.header.frame_id = 'camera_link'
        msg.height = rgb_img.shape[0]
        msg.width = rgb_img.shape[1]
        msg.encoding = 'bgr8'
        msg.is_bigendian = False
        msg.step = rgb_img.shape[1] * 3
        msg.data = rgb_img.tobytes()

        self.camera_pub.publish(msg)

    def update_visualization(self, state):

        """Update OpenCV windows with camera feeds"""

        car_img = self.last_camera_img if hasattr(self, 'last_camera_img') else None
        ext_img = self.sim.get_external_camera_image()

        if car_img is not None:
            speed = state['speed']
            pos = state['position']

            display_img = car_img.copy()
            cv2.putText(display_img, f"Speed: {speed:.1f} m/s", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(display_img, f"Pos: ({pos[0]:.1f}, {pos[1]:.1f})", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(display_img, f"Vel cmd: {self.sim.tgtVel:.1f}", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow("Car Camera", display_img)

        if ext_img is not None:
            cv2.imshow("External View", ext_img)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            self.get_logger().info('Quit requested')
            raise KeyboardInterrupt

    def publish_lidar(self, state, stamp):
        """Publish LiDAR as PointCloud2"""
        distances = self.sim.get_lidar_scan()
        yaw = state['euler'][2]
        pos = state['position']

        points = []
        for i, dist in enumerate(distances):
            if dist < 9.9:
                angle = yaw + (i / len(distances)) * 2 * np.pi
                x = pos[0] + dist * np.cos(angle)
                y = pos[1] + dist * np.sin(angle)
                z = pos[2] + 0.3
                points.append([x, y, z])

        if not points:
            return

        msg = PointCloud2()
        msg.header.stamp = stamp
        msg.header.frame_id = 'odom'
        msg.height = 1
        msg.width = len(points)
        msg.is_dense = True
        msg.is_bigendian = False

        msg.fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
        ]
        msg.point_step = 12
        msg.row_step = msg.point_step * len(points)

        data = b''
        for pt in points:
            data += struct.pack('fff', pt[0], pt[1], pt[2])
        msg.data = data

        self.lidar_pub.publish(msg)

    def destroy_node(self):
        """Clean shutdown"""
        if self.show_visualization:
            cv2.destroyAllWindows()
        self.sim.disconnect()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = SimulationNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
