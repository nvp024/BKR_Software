import cv2
import numpy as np
from ultralytics import YOLO
from config.settings import CONFIG
from utils.image_processing import process_roi

class BasketDetector:
    def __init__(self):
        """Khởi tạo detector với model đã train"""
        self.model = YOLO(CONFIG['model_path'])
        self.last_centroid_x = None
        self.last_centroid_y = None
        self.last_angle = None
    
    def get_last_centroid_x(self):
        """Trả về tọa độ x của centroid cuối cùng"""
        return self.last_centroid_x
        
    def get_last_centroid_y(self):
        """Trả về tọa độ y của centroid cuối cùng"""
        return self.last_centroid_y
        
    def get_last_angle(self):
        """Trả về góc cuối cùng đã tính"""
        return self.last_angle
    
    def process_frame(self, frame):
        """Xử lý một frame để phát hiện bảng rổ và rổ"""
        # Reset centroid và góc
        self.last_centroid_x = None
        self.last_centroid_y = None
        self.last_angle = None
        
        # Chạy model để phát hiện đối tượng
        results = self.model.predict(frame, conf=CONFIG['confidence'], verbose=False)
        
        # Sao chép frame để vẽ
        annotated_frame = results[0].plot()
        
        # Vẽ tâm camera và đường dọc
        center_x = frame.shape[1] // 2
        center_y = frame.shape[0] // 2
        
        # Vẽ trục y đi qua tâm ảnh (kéo dài toàn bộ chiều cao)
        cv2.line(annotated_frame, (center_x, 0), (center_x, frame.shape[0]), (255, 255, 255), 1, cv2.LINE_AA)
        
        # Vẽ đường chữ thập ở tâm camera
        cv2.line(annotated_frame, (center_x - 20, center_y), (center_x + 20, center_y), (0, 0, 255), 2)  # Đường ngang
        cv2.line(annotated_frame, (center_x, center_y - 20), (center_x, center_y + 20), (0, 0, 255), 2)  # Đường dọc
        
        # Vẽ điểm tâm với 2 vòng tròn
        cv2.circle(annotated_frame, (center_x, center_y), 8, (0, 0, 255), -1)  # Điểm tâm đỏ
        cv2.circle(annotated_frame, (center_x, center_y), 12, (255, 255, 255), 2)  # Viền trắng
        
        # Duyệt qua tất cả các box được phát hiện
        for box in results[0].boxes:
            # Lấy tọa độ, class_id, và độ tin cậy
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            class_id = int(box.cls[0].item())
            
            # Xử lý bảng rổ
            if class_id == 0:  # Bảng rổ
                # Cắt và xử lý vùng bảng rổ
                roi = frame[y1:y2, x1:x2]
                contour, centroid, _ = process_roi(roi)
                
                if centroid is not None:
                    # Chuyển tọa độ về frame gốc
                    global_cx = x1 + centroid[0]
                    global_cy = y1 + centroid[1]
                    
                    # Lưu tọa độ centroid
                    self.last_centroid_x = global_cx
                    self.last_centroid_y = global_cy
                    
                    # Vẽ centroid và thông tin
                    cv2.circle(annotated_frame, (global_cx, global_cy), 5, CONFIG['colors']['centroid'], -1)
                    cv2.circle(annotated_frame, (global_cx, global_cy), 8, CONFIG['colors']['centroid_outline'], 2)
                    cv2.putText(annotated_frame, "Backboard Center", 
                              (global_cx + 10, global_cy), CONFIG['font'], CONFIG['font_scale'], 
                              CONFIG['colors']['centroid'], CONFIG['font_thickness'])
                    
                    # Vẽ đường nối từ tâm camera đến centroid
                    # Đường thẳng chính
                    cv2.line(annotated_frame, (center_x, center_y), (global_cx, global_cy), (0, 255, 255), 2, cv2.LINE_AA)
                    
                    # Vẽ đường đứt nét bằng cách vẽ nhiều đoạn nhỏ
                    # Đường ngang từ centroid
                    dash_length = 10
                    gap_length = 5
                    x_start = min(global_cx, center_x)
                    x_end = max(global_cx, center_x)
                    y_start = min(global_cy, center_y)
                    y_end = max(global_cy, center_y)
                    
                    # Vẽ đường ngang đứt nét
                    x = x_start
                    while x < x_end:
                        x_next = min(x + dash_length, x_end)
                        cv2.line(annotated_frame, (int(x), global_cy), (int(x_next), global_cy), 
                               (0, 255, 255), 1, cv2.LINE_AA)
                        x = x_next + gap_length
                    
                    # Vẽ đường dọc đứt nét
                    y = y_start
                    while y < y_end:
                        y_next = min(y + dash_length, y_end)
                        cv2.line(annotated_frame, (global_cx, int(y)), (global_cx, int(y_next)), 
                               (0, 255, 255), 1, cv2.LINE_AA)
                        y = y_next + gap_length
                    
                    # Tính góc với trục y (0° đến ±180°)
                    dx = global_cx - center_x
                    dy = global_cy - center_y
                    angle = np.arctan2(dx, -dy) * 180 / np.pi  # Đổi dấu dy để 0° hướng lên
                    
                    # Lưu góc đã tính
                    self.last_angle = angle
                    
                    # Vẽ vòng cung thể hiện góc với trục y
                    radius = 50  # Bán kính vòng cung
                    # Góc bắt đầu là 0 (trục y hướng lên)
                    start_angle = 0
                    end_angle = angle
                    
                    # Đảm bảo vẽ theo đường ngắn nhất
                    if abs(end_angle) > 180:
                        end_angle = -np.sign(end_angle) * (360 - abs(end_angle))
                    
                    # Vẽ vòng cung với nhiều đoạn nhỏ để tạo đường cong mịn
                    num_segments = 100
                    angles = np.linspace(start_angle, end_angle, num_segments) * np.pi / 180
                    arc_points = np.array([[center_x + int(radius * np.sin(angle)), 
                                          center_y - int(radius * np.cos(angle))] 
                                         for angle in angles])
                    
                    # Vẽ vòng cung
                    for i in range(len(arc_points) - 1):
                        cv2.line(annotated_frame, 
                               tuple(arc_points[i]), 
                               tuple(arc_points[i + 1]), 
                               (0, 255, 255), 2, cv2.LINE_AA)
                    
                    # Vẽ text góc với nền đen để dễ đọc
                    # Thêm dấu + cho góc dương để dễ phân biệt
                    angle_text = f"+{angle:.1f}°" if angle > 0 else f"{angle:.1f}°"
                    text_size = cv2.getTextSize(angle_text, CONFIG['font'], CONFIG['font_scale'], CONFIG['font_thickness'])[0]
                    cv2.rectangle(annotated_frame, 
                                (center_x + 20, center_y - text_size[1] - 5),
                                (center_x + 20 + text_size[0] + 10, center_y + 5),
                                (0, 0, 0), -1)
                    cv2.putText(annotated_frame, angle_text,
                              (center_x + 25, center_y), CONFIG['font'],
                              CONFIG['font_scale'], (0, 255, 255), CONFIG['font_thickness'])
                    
                    # Vẽ contour
                    if contour is not None:
                        shifted_contour = contour.copy()
                        shifted_contour[:,:,0] += x1
                        shifted_contour[:,:,1] += y1
                        cv2.drawContours(annotated_frame, [shifted_contour], -1, CONFIG['colors']['contour'], 2)
            
            # Xử lý rổ
            elif class_id == 1:  # Rổ
                top_center_x = (x1 + x2) // 2
                top_center_y = y1
                cv2.circle(annotated_frame, (top_center_x, top_center_y), 5, CONFIG['colors']['hoop'], -1)
                cv2.putText(annotated_frame, "Hoop Top",
                          (top_center_x + 10, top_center_y), CONFIG['font'], CONFIG['font_scale'],
                          CONFIG['colors']['hoop'], CONFIG['font_thickness'])
        
        return annotated_frame
    
    def process_image(self, image_path):
        """Xử lý một ảnh"""
        # Đọc ảnh
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError("Không thể đọc ảnh từ đường dẫn đã cho")
            
        return self.process_frame(image)
    
    def process_video(self, video_path):
        """Xử lý video"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError("Không thể mở video từ đường dẫn đã cho")
        
        # Lấy thông tin video
        orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Tính toán kích thước hiển thị
        display_scale = min(MAX_DISPLAY_HEIGHT / orig_height, 
                          MAX_DISPLAY_WIDTH / orig_width)
        display_size = (int(orig_width * display_scale), 
                       int(orig_height * display_scale))
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Xử lý frame
            result_frame = self.process_frame(frame)
            
            # Resize để hiển thị
            display_frame = cv2.resize(result_frame, display_size, 
                                     interpolation=cv2.INTER_AREA)
            
            # Hiển thị kết quả
            cv2.imshow("Kết quả", display_frame)
            
            # Thoát nếu nhấn 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows() 