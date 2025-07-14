import cv2
import numpy as np
import os
from collections import OrderedDict

FACIAL_LANDMARKS_IDXS = OrderedDict([
    ("mouth", (48, 68)), #48-67是嘴巴关键点
    ("right_eyebrow", (17, 22)), #17-21是右眉毛关键点
    ("left_eyebrow", (22, 27)), #22-26是左眉毛关键点
    ("right_eye", (36, 42)), #36-41是右眼关键点
    ("left_eye", (42, 48)), #42-47是左眼关键点
    ("nose", (27, 36)), #27-35是鼻子关键点
    ("jaw", (0, 17)) #0-16是下巴关键点
])

def get_reference_points(landmarks):
    """
    获取面部参考点（鼻子和下巴的稳定点）
    """
    # 使用鼻尖(30)和下巴中点(8)作为参考点
    nose_tip = landmarks[30]
    chin = landmarks[8]
    return np.array([nose_tip, chin])

def calculate_relative_motion(prev_landmarks, current_landmarks, region="right_eye"):
    """
    计算眼睛区域相对于面部参考点的运动
    """
    if prev_landmarks is None or current_landmarks is None:
        return np.array([0, 0])
    
    # 获取参考点
    prev_ref = get_reference_points(prev_landmarks)
    curr_ref = get_reference_points(current_landmarks)
    
    # 计算参考点的平均移动
    ref_motion = np.mean(curr_ref - prev_ref, axis=0)
    
    # 获取眼睛区域的点
    (start, end) = FACIAL_LANDMARKS_IDXS[region]
    prev_eye = prev_landmarks[start:end]
    curr_eye = current_landmarks[start:end]
    
    # 计算眼睛区域的平均移动
    eye_motion = np.mean(curr_eye - prev_eye, axis=0)
    
    # 计算相对运动（眼睛相对于参考点的运动）
    relative_motion = eye_motion - ref_motion
    
    return relative_motion

def extract_eye_region(frame, landmarks, region="right_eye"):
    """
    提取眼睛区域的图像
    """
    if region not in FACIAL_LANDMARKS_IDXS:
        return None
    
    (start, end) = FACIAL_LANDMARKS_IDXS[region]
    points = landmarks[start:end]
    
    # 获取区域的边界框
    x = int(np.min(points[:, 0]))
    y = int(np.min(points[:, 1]))
    w = int(np.max(points[:, 0]) - x)
    h = int(np.max(points[:, 1]) - y)
    
    # 确保边界框在图像内
    x = max(0, x)
    y = max(0, y)
    w = min(w, frame.shape[1] - x)
    h = min(h, frame.shape[0] - y)
    
    if w <= 0 or h <= 0:
        return None
    
    # 提取区域图像
    roi = frame[y:y+h, x:x+w]
    if roi.size == 0:
        return None
    
    return roi

def analyze_texture(frame, landmarks, region="right_eye"):
    """
    分析特定区域的纹理特征
    """
    roi = extract_eye_region(frame, landmarks, region)
    if roi is None:
        return 0
    
    # 计算区域的方差作为纹理特征
    return np.var(roi)

def detect_occlusion_by_motion(frame, prev_landmarks, current_landmarks, base_texture_score, region="right_eye", threshold=15.0):
    """
    改进的遮挡检测函数，使用相对运动和基准帧纹理比较
    """
    if region not in FACIAL_LANDMARKS_IDXS or prev_landmarks is None:
        return False
    
    # 1. 计算眼睛的相对运动
    relative_motion = calculate_relative_motion(prev_landmarks, current_landmarks, region)
    relative_motion_magnitude = np.linalg.norm(relative_motion)
    
    # 2. 分析纹理特征（与基准帧比较）
    current_texture_score = analyze_texture(frame, current_landmarks, region)
    texture_change = abs(current_texture_score - base_texture_score) / (base_texture_score + 1e-6)
    
    # 3. 综合判断
    motion_threshold = threshold
    texture_threshold = 0.5  # 纹理变化阈值
    
    is_occluded = (relative_motion_magnitude > motion_threshold or texture_change > texture_threshold)
    
    return is_occluded

def main():
    # 初始化摄像头
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("Unable to open the camara")
        return
    
    # 加载Haar级联分类器（人脸检测）
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    if face_cascade.empty():
        print("无法加载人脸检测模型")
        return
    
    # 修改模型加载部分
    # 获取当前脚本目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, "lbfmodel.yaml")

    print(f"模型文件路径: {model_path}")

    # 初始化LBF人脸标记检测器
    facemark = cv2.face.createFacemarkLBF() #获取68个面部关键点
    facemark.loadModel(model_path)
    
    # 设置窗口
    cv2.namedWindow('Face Detection', cv2.WINDOW_NORMAL)
    
    prev_landmarks = None
    prev_frame = None
    base_frame = None
    base_landmarks = None
    base_texture_score = None
    is_base_frame_set = False

    count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            print("无法获取帧")
            break
        
        # 转换为灰度图像（人脸检测需要）
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 人脸检测
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            
            try:
                # 检测人脸关键点
                face_rects = np.array([(x, y, w, h)], dtype=np.int32)
                success, landmarks = facemark.fit(gray, face_rects)
                
                if success and len(landmarks) > 0:
                    # 绘制关键点
                    for landmark in landmarks:
                        current_landmark = landmark[0]
                        
                        # 设置基准帧
                        if not is_base_frame_set:
                            base_frame = gray.copy()
                            base_landmarks = current_landmark.copy()
                            base_texture_score = analyze_texture(base_frame, base_landmarks, region="right_eye")
                            is_base_frame_set = True
                            print("基准帧已设置")
                            
                        if prev_landmarks is not None and prev_frame is not None and base_texture_score is not None:
                            if detect_occlusion_by_motion(gray, prev_landmarks, current_landmark, base_texture_score, region="right_eye", threshold=5.0):
                                count += 1
                                cv2.putText(frame, "Right Eye Possibly Occluded", (x, y - 60),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                                print("右眼可能被遮挡" + str(count))
                        
                        # 更新前一帧的数据
                        prev_landmarks = current_landmark.copy()
                        prev_frame = gray.copy()

                        # 绘制关键点和参考点
                        for i, point in enumerate(landmark[0]):
                            x_point, y_point = int(point[0]), int(point[1])
                            # 特别标记参考点（鼻尖和下巴中点）
                            if i in [8, 30]:  # 参考点用蓝色标记
                                cv2.circle(frame, (x_point, y_point), 3, (255, 0, 0), -1)
                            else:  # 其他点用绿色标记
                                cv2.circle(frame, (x_point, y_point), 2, (0, 255, 0), -1)
                            
            except Exception as e:
                print(f"关键点检测错误: {e}")
        
        # 显示检测到的人脸数量
        cv2.putText(frame, f'Faces: {len(faces)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # 显示结果
        cv2.imshow('Face Detection', frame)
        
        # 退出条件
        key = cv2.waitKey(20) & 0xFF
        if key == 27 or cv2.getWindowProperty('Face Detection', cv2.WND_PROP_VISIBLE) < 1:
            break
    
    # 释放资源
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()


