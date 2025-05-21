import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32
from config.settings import CONFIG

class PIDController:
    def __init__(self, kp=0.01, ki=0.001, kd=0.005):
        self.kp = kp  # Hệ số tỷ lệ
        self.ki = ki  # Hệ số tích phân
        self.kd = kd  # Hệ số đạo hàm
        
        self.previous_error = 0
        self.integral = 0
        
    def compute(self, error, dt):
        """Tính toán tín hiệu điều khiển PID"""
        # Cập nhật integral
        self.integral += error * dt
        
        # Giới hạn integral để tránh windup
        self.integral = np.clip(self.integral, -CONFIG['pid']['max_integral'], CONFIG['pid']['max_integral'])
        
        # Tính đạo hàm
        derivative = (error - self.previous_error) / dt
        
        # Tính tín hiệu điều khiển
        output = (self.kp * error +  # Phần tỷ lệ
                 self.ki * self.integral +  # Phần tích phân
                 self.kd * derivative)  # Phần đạo hàm
        
        # Giới hạn output
        output = np.clip(output, CONFIG['pid']['output_limits'][0], CONFIG['pid']['output_limits'][1])
        
        # Lưu error cho lần sau
        self.previous_error = error
        
        return output

class RobotController(Node):
    def __init__(self):
        super().__init__('robot_controller')
        
        # Khởi tạo publisher cho topic /rotate_base
        self.rotation_publisher = self.create_publisher(Float32, '/rotate_base', 10)
        
        # Khởi tạo PID controller với tham số từ config
        self.pid = PIDController(
            kp=CONFIG['pid']['kp'],
            ki=CONFIG['pid']['ki'],
            kd=CONFIG['pid']['kd']
        )
        
        # Lưu thời gian cho PID
        self.last_time = self.get_clock().now()
        
        # Lấy kích thước frame từ config
        self.frame_width = CONFIG['frame_size'][0]
        self.frame_height = CONFIG['frame_size'][1]
        self.frame_center_x = self.frame_width // 2
        self.frame_center_y = self.frame_height // 2
        
    def print_debug_info(self, angle, control_signal):
        """In thông tin debug ra terminal"""
        print("\n" * 2)  # Thay thế bằng newlines trên Windows
        
        # In thông tin
        print("="*50)
        print("THÔNG TIN ĐIỀU KHIỂN ROBOT")
        print("="*50)
        
        # Hiển thị góc với dấu (giống như trên imshow)
        angle_text = f"+{angle:.1f}°" if angle > 0 else f"{angle:.1f}°"
        print(f"Góc lệch với trục y:   {angle_text}")
        
        # Hiển thị hướng quay
        direction = "PHẢI" if angle > 0 else "TRÁI"
        if abs(angle) < 0.5:
            direction = "ĐÚNG TÂM"
            
        print(f"Hướng cần quay:      {direction}")
        print(f"Tốc độ quay:         {abs(control_signal):>6.3f}")
        
        # Hiển thị trạng thái robot
        status = "🔄" if abs(angle) >= 0.5 else "✅"
        print(f"Trạng thái:          {status}")
        
        # Mô phỏng gửi lệnh tới robot
        arrow = "→" if angle > 0 else "←"
        if abs(angle) < 0.5:
            arrow = "•"
        print(f"Gửi lệnh quay:       {arrow * min(int(abs(control_signal * 20)), 10)}")
        print("-"*50)
        
        # Hiển thị thanh tiến trình
        bar_width = 40
        center = bar_width // 2
        # Chuyển góc sang vị trí trên thanh tiến trình (-180° đến +180°)
        position = center + int((angle / 180) * center)  # Scale góc từ ±180° sang vị trí trên thanh
        position = np.clip(position, 0, bar_width-1)
        
        progress_bar = ["-"] * bar_width
        progress_bar[center] = "|"  # Đánh dấu tâm (0°)
        progress_bar[position] = "O"  # Vị trí hiện tại
        print("".join(progress_bar))
        print("-"*50)
        
    def process_frame(self, detector):
        """Xử lý một frame và điều khiển robot"""
        # Lấy góc từ detector
        angle = detector.get_last_angle()
        
        if angle is not None:
            # Tính thời gian từ lần xử lý trước
            current_time = self.get_clock().now()
            dt = (current_time - self.last_time).nanoseconds / 1e9  # Chuyển sang giây
            self.last_time = current_time
            
            # Tính tín hiệu điều khiển từ PID
            control_signal = self.pid.compute(angle, dt)
            
            # Đảo dấu control_signal vì góc dương cần quay ngược chiều
            control_signal = -control_signal
            
            # Tạo và gửi message
            msg = Float32()
            msg.data = float(control_signal)
            self.rotation_publisher.publish(msg)
            
            # In thông tin debug
            self.print_debug_info(angle, control_signal)
            
            return control_signal
        else:
            print("\n❌ Không tìm thấy bảng rổ!")
            return None

def init_robot_control():
    """Khởi tạo ROS2 và robot controller"""
    rclpy.init()
    controller = RobotController()
    return controller

def shutdown_robot_control(controller):
    """Đóng ROS2 và giải phóng tài nguyên"""
    controller.destroy_node()
    rclpy.shutdown()