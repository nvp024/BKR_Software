import cv2
import numpy as np
import time
from config.settings import CONFIG
from models.detector import BasketDetector

class SimplePIDController:
    def __init__(self, kp=0.01, ki=0.001, kd=0.005):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.previous_error = 0
        self.integral = 0
        
    def compute(self, error, dt):
        # Cập nhật integral với giới hạn
        self.integral += error * dt
        self.integral = np.clip(self.integral, -CONFIG['pid']['max_integral'], CONFIG['pid']['max_integral'])
        
        # Tính đạo hàm
        derivative = (error - self.previous_error) / dt
        
        # Tính output với giới hạn
        output = (self.kp * error + 
                 self.ki * self.integral + 
                 self.kd * derivative)
        output = np.clip(output, CONFIG['pid']['output_limits'][0], CONFIG['pid']['output_limits'][1])
        
        self.previous_error = error
        return output

class SimpleRobotController:
    def __init__(self):
        # Khởi tạo PID với tham số từ config
        self.pid = SimplePIDController(
            kp=CONFIG['pid']['kp'],
            ki=CONFIG['pid']['ki'],
            kd=CONFIG['pid']['kd']
        )
        
        # Lấy thông số camera từ config
        self.frame_width = CONFIG['frame_size'][0]
        self.frame_height = CONFIG['frame_size'][1]
        self.frame_center_x = self.frame_width // 2
        self.frame_center_y = self.frame_height // 2
        
        # Thời gian cho PID
        self.last_time = time.time()
        
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
            current_time = time.time()
            dt = current_time - self.last_time
            self.last_time = current_time
            
            # Tính tín hiệu điều khiển từ PID
            control_signal = self.pid.compute(angle, dt)
            
            # Mô phỏng gửi lệnh và in debug
            self.print_debug_info(angle, control_signal)
            
            return control_signal
        else:
            print("\n❌ Không tìm thấy bảng rổ!")
            return None

def process_video_with_control(video_path):
    """Xử lý video thực tế với điều khiển robot"""
    # Khởi tạo detector và controller
    detector = BasketDetector()
    controller = SimpleRobotController()
    
    # Mở video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("❌ Không thể mở video")
        return
        
    # Lấy kích thước video
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Tính toán kích thước hiển thị
    max_height, max_width = CONFIG['display_size']
    scale = min(max_height/height, max_width/width)
    display_size = (int(width * scale), int(height * scale))
    
    print("🎥 Đang xử lý video...")
    print("📏 Kích thước frame:", width, "x", height)
    print("🎯 Nhấn 'q' để thoát")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Phát hiện bảng rổ và lấy centroid
        result_frame = detector.process_frame(frame)
        centroid_x = detector.get_last_centroid_x()
        centroid_y = detector.get_last_centroid_y()
        
        # Điều khiển robot dựa trên centroid
        if centroid_x is not None and centroid_y is not None:
            controller.process_frame(detector)
        
        # Hiển thị frame
        display_frame = cv2.resize(result_frame, display_size)
        cv2.imshow("Video Detection", display_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

def process_image_with_control(image_path):
    """Xử lý ảnh thực tế với điều khiển robot"""
    # Khởi tạo detector và controller
    detector = BasketDetector()
    controller = SimpleRobotController()
    
    # Đọc ảnh
    image = cv2.imread(image_path)
    if image is None:
        print("❌ Không thể đọc ảnh")
        return
        
    # Phát hiện bảng rổ và lấy centroid
    result_image = detector.process_frame(image)
    centroid_x = detector.get_last_centroid_x()
    centroid_y = detector.get_last_centroid_y()
    
    # Điều khiển robot dựa trên centroid
    if centroid_x is not None and centroid_y is not None:
        controller.process_frame(detector)
    
    # Hiển thị ảnh
    cv2.imshow("Image Detection", result_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    print("Chọn chế độ test:")
    print("1 - Xử lý ảnh thực tế")
    print("2 - Xử lý video thực tế")
    mode = input("Nhập lựa chọn (1 hoặc 2): ")
    
    try:
        if mode == '1':
            image_path = input("Nhập đường dẫn ảnh: ")
            process_image_with_control(image_path)
        elif mode == '2':
            video_path = input("Nhập đường dẫn video: ")
            process_video_with_control(video_path)
        else:
            print("❌ Lựa chọn không hợp lệ")
    except Exception as e:
        print(f"❌ Lỗi: {str(e)}") 