#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import cv2
from .models.detector import BasketDetector
from .config.settings import CONFIG
from .utils.robot_control import RobotController

class BasketDetectionNode(Node):
    def __init__(self):
        super().__init__('basket_detection_node')
        
        # Khởi tạo detector và controller
        self.detector = BasketDetector()
        self.controller = RobotController()
        
        # Tạo timer cho việc xử lý video
        self.timer = self.create_timer(1.0/30.0, self.timer_callback)  # 30 FPS
        
        # Khởi tạo video capture
        self.cap = None
        self.is_processing = False
        
    def start_processing(self, source):
        """Bắt đầu xử lý video từ camera hoặc file"""
        if isinstance(source, str):
            self.cap = cv2.VideoCapture(source)  # File video
        else:
            self.cap = cv2.VideoCapture(source)  # Camera (source = 0, 1, ...)
            
        if not self.cap.isOpened():
            self.get_logger().error('Không thể mở video source')
            return False
            
        # Lấy kích thước video
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Tính toán kích thước hiển thị
        max_height, max_width = CONFIG['display_size']
        scale = min(max_height/height, max_width/width)
        self.display_size = (int(width * scale), int(height * scale))
        
        self.is_processing = True
        return True
        
    def timer_callback(self):
        """Callback được gọi theo timer để xử lý frame"""
        if not self.is_processing or self.cap is None:
            return
            
        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().info('Kết thúc video')
            self.stop_processing()
            return
            
        # Xử lý frame để phát hiện bảng rổ
        result = self.detector.process_frame(frame)
        
        # Điều khiển robot nếu phát hiện được bảng rổ
        if self.detector.get_last_angle() is not None:
            self.controller.process_frame(self.detector)
        
        # Hiển thị kết quả
        display_frame = cv2.resize(result, self.display_size)
        cv2.imshow("Kết quả", display_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            self.stop_processing()
            
    def stop_processing(self):
        """Dừng xử lý video"""
        self.is_processing = False
        if self.cap is not None:
            self.cap.release()
        cv2.destroyAllWindows()

def main(args=None):
    rclpy.init(args=args)
    
    node = BasketDetectionNode()
    
    # Menu chọn chế độ
    print("Chọn chế độ:")
    print("1 - Camera")
    print("2 - Video file")
    mode = input("Nhập lựa chọn (1 hoặc 2): ")
    
    try:
        if mode == '1':
            # Sử dụng camera (có thể thay đổi index nếu có nhiều camera)
            if node.start_processing(0):
                rclpy.spin(node)
            
        elif mode == '2':
            video_path = input("Nhập đường dẫn video: ")
            if node.start_processing(video_path):
                rclpy.spin(node)
            
        else:
            print("❌ Lựa chọn không hợp lệ")
            
    except KeyboardInterrupt:
        print("\n⚠️ Đã dừng chương trình")
        
    finally:
        # Cleanup
        node.stop_processing()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main() 