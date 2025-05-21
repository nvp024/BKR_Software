#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_srvs.srv import SetBool
import time

class RotateBaseServer(Node):
    """
    Service server that handles rotate_base service requests using std_srvs.srv.SetBool
    """
    def __init__(self):
        super().__init__('rotate_base_server')
        
        # Create the service server
        self.srv = self.create_service(
            SetBool, 
            '/rotate_base', 
            self.rotate_base_callback
        )
        
        # Track received requests
        self.received_angles = []
        self.request_count = 0
        
        self.get_logger().info('Rotate Base Service Server started')
        
    def rotate_base_callback(self, request, response):
        """Callback for the rotate_base service"""
        # Convert boolean to angle representation
        angle_value = 1.0 if request.data else -1.0
        self.request_count += 1
        
        self.received_angles.append(angle_value)
        self.get_logger().info(f'Received rotate request #{self.request_count} with data: {request.data} (interpreted as angle: {angle_value})')
        
        # Simulate processing
        self.get_logger().info('Processing rotation...')
        # Simulate some processing time
        time.sleep(0.2)
        
        # Set response with detailed information
        response.success = True
        response.message = f"ROTATION_SUCCESS: Request #{self.request_count} - Rotated base to angle: {angle_value}. Command executed successfully!"
        
        self.get_logger().info(f'Sending success response: {response.message}')
        
        return response


def main(args=None):
    rclpy.init(args=args)
    server = RotateBaseServer()
    
    try:
        rclpy.spin(server)
    except KeyboardInterrupt:
        pass
    finally:
        server.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main() 