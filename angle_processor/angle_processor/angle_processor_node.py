#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32
from std_srvs.srv import SetBool

class AngleProcessorNode(Node):
    """
    Node that subscribes to IMU and image angles, combines them,
    and calls the rotate_base service with the combined angle.
    """
    
    def __init__(self):
        super().__init__('angle_processor_node')
        
        # Initialize angles
        self.imu_angle = 0.0
        self.image_angle = 0.0
        
        # Track service calls
        self.service_call_count = 0
        
        # Create subscribers
        self.imu_subscription = self.create_subscription(
            Float32,
            '/imu',
            self.imu_callback,
            10)
            
        self.image_subscription = self.create_subscription(
            Float32,
            '/image_raw',
            self.image_callback,
            10)
        
        # Create real service client
        self.rotate_base_client = self.create_client(SetBool, '/rotate_base')
        
        # Wait for service to be available
        self.get_logger().info('Waiting for /rotate_base service...')
        self.service_check_timer = self.create_timer(1.0, self.check_service)
    
    def check_service(self):
        """Check if the service is available"""
        if self.rotate_base_client.service_is_ready():
            self.get_logger().info('/rotate_base service is now available')
            self.service_check_timer.cancel()
    
    def imu_callback(self, msg):
        """Callback for IMU angle data."""
        self.imu_angle = msg.data
        self.get_logger().info(f'Received IMU angle: {self.imu_angle}')
        self.process_angles()
    
    def image_callback(self, msg):
        """Callback for image angle data."""
        self.image_angle = msg.data
        self.get_logger().info(f'Received image angle: {self.image_angle}')
        self.process_angles()
    
    def process_angles(self):
        """Process the angles and call the rotate service."""
        combined_angle = self.imu_angle + self.image_angle
        self.get_logger().info(f'Combined angle: {combined_angle}')
        self.call_rotate_service(combined_angle)
    
    def call_rotate_service(self, angle):
        """Call the rotate_base service with the given angle."""
        if not self.rotate_base_client.service_is_ready():
            self.get_logger().warning('Service not ready yet. Cannot send rotation request.')
            return
            
        request = SetBool.Request()
        # For a real implementation, we would need a custom service message type
        # that supports float values for angles, but for testing we'll use SetBool
        request.data = bool(angle > 0)  # Convert angle to bool (True if positive, False if negative)
        
        self.service_call_count += 1
        self.get_logger().info(f'======== ROTATION REQUEST #{self.service_call_count} ========')
        self.get_logger().info(f'Sending rotation request with angle value: {angle} (converted to bool: {request.data})')
        
        future = self.rotate_base_client.call_async(request)
        future.add_done_callback(self.rotate_service_callback)
    
    
    def rotate_service_callback(self, future):
        """Callback for the rotate service response."""
        try:
            response = future.result()
            if response.success:
                self.get_logger().info('======== ROTATION RESPONSE ========')
                self.get_logger().info(f'  {response.message}')
                self.get_logger().info('==================================')
            else:
                self.get_logger().warning(f'Rotation failed: {response.message}')
            self.get_logger().info(f'--------------------------------')
        except Exception as e:
            self.get_logger().error(f'Service call failed: {str(e)}')


def main(args=None):
    rclpy.init(args=args)
    node = AngleProcessorNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main() 


