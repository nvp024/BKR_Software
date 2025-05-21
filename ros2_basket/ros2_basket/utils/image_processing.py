import cv2
import numpy as np
from config.settings import CONFIG

def create_kernel(size):
    """Tạo kernel cho morphology operations"""
    return np.ones(size, np.uint8)

def process_roi(roi):
    """Xử lý một vùng ROI để phát hiện bảng rổ"""
    # Chuyển sang HSV
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    
    # Tạo mask cho màu trắng và đen
    white_lower, white_upper = CONFIG['hsv_white']
    black_lower, black_upper = CONFIG['hsv_black']
    
    white_mask = cv2.inRange(hsv, np.array(white_lower), np.array(white_upper))
    black_mask = cv2.inRange(hsv, np.array(black_lower), np.array(black_upper))
    
    # Kết hợp masks
    hsv_mask = cv2.bitwise_or(white_mask, black_mask)
    
    # Áp dụng morphology
    kernel_small = create_kernel(CONFIG['kernel_sizes']['small'])
    kernel_medium = create_kernel(CONFIG['kernel_sizes']['medium'])
    
    # Opening để loại bỏ nhiễu
    processed_mask = cv2.morphologyEx(hsv_mask, cv2.MORPH_OPEN, kernel_small, 
                                    iterations=CONFIG['morph_iters']['open'])
    # Closing để kết nối các vùng
    processed_mask = cv2.morphologyEx(processed_mask, cv2.MORPH_CLOSE, kernel_medium, 
                                    iterations=CONFIG['morph_iters']['close'])
    
    # Tìm contours
    contours, _ = cv2.findContours(processed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None, None, roi.copy()
    
    # Lọc contours
    valid_contours = []
    roi_area = roi.shape[0] * roi.shape[1]
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < CONFIG['min_area']:
            continue
            
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = float(w)/h
        min_ratio, max_ratio = CONFIG['aspect_ratio']
        
        if (min_ratio <= aspect_ratio <= max_ratio and 
            area / roi_area <= CONFIG['max_area_ratio']):
            valid_contours.append(cnt)
    
    if not valid_contours:
        return None, None, roi.copy()
    
    # Lấy contour lớn nhất và tạo hull
    largest_contour = max(valid_contours, key=cv2.contourArea)
    hull = cv2.convexHull(largest_contour)
    
    # Tính centroid
    M = cv2.moments(hull)
    if M["m00"] == 0:
        return None, None, roi.copy()
        
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    centroid = (cx, cy)
    
    # Vẽ kết quả
    result = roi.copy()
    # Vẽ vùng bảng rổ với màu bán trong suốt
    overlay = result.copy()
    cv2.drawContours(overlay, [hull], -1, CONFIG['colors']['backboard'], -1)
    cv2.addWeighted(overlay, 0.5, result, 0.5, 0, result)
    # Vẽ viền
    cv2.drawContours(result, [hull], -1, CONFIG['colors']['contour'], 2)
    # Vẽ centroid
    cv2.circle(result, centroid, 5, CONFIG['colors']['centroid'], -1)
    cv2.circle(result, centroid, 8, CONFIG['colors']['centroid_outline'], 2)
    
    # Debug windows
    debug_images = {
        '1. Original ROI': roi,
        '2. HSV Mask': cv2.cvtColor(hsv_mask, cv2.COLOR_GRAY2BGR),
        '3. Processed Mask': cv2.cvtColor(processed_mask, cv2.COLOR_GRAY2BGR),
        '4. Final Result': result
    }
    
    for name, img in debug_images.items():
        cv2.imshow(name, cv2.resize(img, CONFIG['debug_size']))
    
    return hull, centroid, result 