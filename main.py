#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, Imu
from cv_bridge import CvBridge
import numpy as np
from basket_detect import BasketDetector
from robot_interfaces.srv import RotateBase

class BasketTrackingNode(Node):
    def __init__(self):
        super().__init__('basket_tracking_node')
        
        # Khởi tạo detector
        self.detector = BasketDetector()
        self.cv_bridge = CvBridge()
        
        # Biến lưu góc từ IMU
        self.imu_angle = 0.0
        
        # Tạo subscriber cho image và imu
        self.image_sub = self.create_subscription(
            Image,
            '/image_raw',
            self.image_callback,
            10
        )
        
        self.imu_sub = self.create_subscription(
            Imu,
            '/imu',
            self.imu_callback,
            10
        )
        
        # Tạo service client cho /rotate_base
        self.rotate_client = self.create_client(
            RotateBase,
            '/rotate_base'
        )
        while not self.rotate_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Service /rotate_base chưa sẵn sàng, đang đợi...')
        
        self.get_logger().info('Basket tracking node đã khởi tạo')
        
    def imu_callback(self, msg):
        """Callback xử lý dữ liệu IMU"""
        # Chỉ lưu góc từ IMU, không xử lý
        self.imu_angle = msg.orientation.z  # Giả sử góc z là góc yaw
        
    def rotate_response_callback(self, future):
        """Callback xử lý response từ service rotate_base"""
        try:
            response = future.result()
            if response.success:
                self.get_logger().info('Xoay robot thành công')
            else:
                self.get_logger().warn('Xoay robot không thành công')
        except Exception as e:
            self.get_logger().error(f'Lỗi khi xoay robot: {str(e)}')
        
    def send_rotate_request(self, angle):
        """Gửi request tới service /rotate_base"""
        request = RotateBase.Request()
        request.angle = float(angle)
        
        # Gửi request và đăng ký callback xử lý response
        future = self.rotate_client.call_async(request)
        future.add_done_callback(self.rotate_response_callback)
        
    def image_callback(self, msg):
        """Callback xử lý ảnh từ camera"""
        try:
            # Chuyển đổi ROS Image sang OpenCV format
            cv_image = self.cv_bridge.imgmsg_to_cv2(msg, "bgr8")
            
            # Xử lý frame để phát hiện bảng rổ
            _ = self.detector.process_frame(cv_image)
            
            # Lấy góc phát hiện được từ detector
            detected_angle = self.detector.get_last_angle()
            
            if detected_angle is not None:
                # Tính tổng góc cần xoay (góc phát hiện + góc hiện tại của robot)
                total_angle = detected_angle + self.imu_angle
                
                # Gửi request tới service /rotate_base với góc tổng
                self.send_rotate_request(total_angle)
                
                # Log thông tin
                self.get_logger().info(f'Góc phát hiện: {detected_angle:.2f}°, ' +
                                     f'Góc IMU: {self.imu_angle:.2f}°, ' +
                                     f'Tổng góc: {total_angle:.2f}°')
                
        except Exception as e:
            self.get_logger().error(f'Lỗi xử lý ảnh: {str(e)}')

def main(args=None):
    rclpy.init(args=args)
    node = BasketTrackingNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main() 