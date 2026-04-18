#!/usr/bin/env python3
"""Simple test controller - sends cmd_vel commands to move the car"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
import time


class TestController(Node):
    def __init__(self):
        super().__init__('test_controller')
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.get_logger().info('Test Controller Started')
    
    def send_command(self, linear_x, angular_z, duration):
        """Send velocity command for specified duration"""
        twist = Twist()
        twist.linear.x = float(linear_x)
        twist.angular.z = float(angular_z)
        
        start_time = time.time()
        while time.time() - start_time < duration:
            self.cmd_vel_pub.publish(twist)
            time.sleep(0.1)
        
        self.get_logger().info(f'Sent: linear={linear_x:.2f}, angular={angular_z:.2f} for {duration}s')
    
    def stop(self):
        """Stop the car"""
        twist = Twist()
        for _ in range(5):
            self.cmd_vel_pub.publish(twist)
            time.sleep(0.1)
        self.get_logger().info('STOP')
    
    def run_test_sequence(self):
        """Run a test driving sequence"""
        self.get_logger().info('=== Starting Test Sequence ===')
        time.sleep(1)
        
        self.get_logger().info('1. Forward')
        self.send_command(2.0, 0.0, 3.0)
        self.stop()
        time.sleep(0.5)
        
        self.get_logger().info('2. Turn left')
        self.send_command(1.0, 0.5, 2.0)
        self.stop()
        time.sleep(0.5)
        
        self.get_logger().info('3. Forward')
        self.send_command(2.0, 0.0, 3.0)
        self.stop()
        time.sleep(0.5)
        
        self.get_logger().info('4. Turn right')
        self.send_command(1.0, -0.5, 2.0)
        self.stop()
        time.sleep(0.5)
        
        self.get_logger().info('5. Backward')
        self.send_command(-1.0, 0.0, 2.0)
        self.stop()
        
        self.get_logger().info('=== Test Sequence Complete ===')


def main(args=None):
    rclpy.init(args=args)
    controller = TestController()
    
    try:
        controller.run_test_sequence()
    except KeyboardInterrupt:
        controller.get_logger().info('Interrupted')
    finally:
        controller.stop()
        controller.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
