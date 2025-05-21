import cv2

# Cấu hình cho việc phát hiện và xử lý bảng rổ
CONFIG = {
    # Model
    'model_path': 'v2.pt',
    'confidence': 0.25,
    
    # Camera
    'frame_size': (640, 480),  # width, height
    'fov_horizontal': 60,  # độ
    
    # Hiển thị
    'display_size': (800, 1200),
    'debug_size': (300, 200),
    
    # HSV thresholds
    'hsv_white': ([0, 0, 160], [180, 45, 255]),  # (lower, upper)
    'hsv_black': ([0, 0, 0], [180, 255, 50]),
    
    # Contour filtering
    'min_area': 200,
    'aspect_ratio': (1.2, 2.0),
    'max_area_ratio': 0.9,
    
    # Morphology
    'kernel_sizes': {
        'small': (3, 3),
        'medium': (5, 5)
    },
    'morph_iters': {
        'open': 1,
        'close': 2
    },
    
    # PID Control
    'pid': {
        'kp': 0.01,  # Proportional gain
        'ki': 0.001, # Integral gain
        'kd': 0.005, # Derivative gain
        'max_integral': 100.0,  # Anti-windup
        'output_limits': (-1.0, 1.0)  # Min/max output
    },
    
    # Colors (BGR)
    'colors': {
        'centroid': (0, 0, 255),      # Đỏ
        'centroid_outline': (255, 255, 255),  # Trắng
        'contour': (0, 255, 0),       # Xanh lá
        'backboard': (0, 255, 255),   # Vàng
        'hoop': (0, 255, 255)         # Vàng
    },
    
    # Text
    'font': cv2.FONT_HERSHEY_SIMPLEX,
    'font_scale': 0.6,
    'font_thickness': 2
} 