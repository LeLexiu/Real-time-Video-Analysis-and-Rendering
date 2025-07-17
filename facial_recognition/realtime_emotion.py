import cv2
import numpy as np
import joblib
import os
import sys

# 添加项目根目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(project_root)

from .process_dataset import init_detectors, normalize_landmarks
from .special_effects import apply_emotion_effects, init_emojis

import time
from collections import deque, Counter
from PIL import Image
import random

class EmotionHistory:
    def __init__(self, window_seconds=2):
        self.window_seconds = window_seconds
        self.history = deque()  # 存储(timestamp, emotion)元组
    
    def add_emotion(self, emotion):
        current_time = time.time()
        self.history.append((current_time, emotion))
        
        # 移除超过时间窗口的记录
        while self.history and current_time - self.history[0][0] > self.window_seconds:
            self.history.popleft()
    
    def get_dominant_emotion(self):
        if not self.history:
            return None, 0
        
        # 统计情绪频率
        emotions = [e for _, e in self.history]
        emotion_counts = Counter(emotions)
        
        # 获取最常见的情绪及其出现次数
        dominant_emotion, count = emotion_counts.most_common(1)[0]
        percentage = count / len(self.history)
        
        return dominant_emotion, percentage

class EmotionGame:
    def __init__(self):
        self.emotions = ['happy', 'angry', 'sad', 'surprise', 'fear','neutral']
        self.current_target = None
        self.start_time = None
        self.success_time = 0  # 成功保持表情的时间
        self.required_time = 2.0  # 需要保持表情2秒
        self.score = 0
        self.generate_new_target()
    
    def generate_new_target(self):
        """生成新的目标表情"""
        if self.current_target is None:
            self.current_target = random.choice(self.emotions)
        else:
            # 确保新的目标表情与当前的不同
            new_target = random.choice([e for e in self.emotions if e != self.current_target])
            self.current_target = new_target
        self.start_time = None
        self.success_time = 0
    
    def check_emotion(self, dominant_emotion):
        """检查当前表情是否匹配目标表情"""
        current_time = time.time()
        
        if dominant_emotion == self.current_target:
            if self.start_time is None:
                self.start_time = current_time
            self.success_time = current_time - self.start_time
            
            if self.success_time >= self.required_time:
                self.score += 1
                self.generate_new_target()
                return True, "太棒了！成功模仿表情！"
            return False, f"保持表情 {self.success_time:.1f}/{self.required_time}秒"
        else:
            self.start_time = None
            self.success_time = 0
            return False, "请模仿目标表情"

def adjust_probabilities(probabilities, weights=None):
    """
    调整情绪预测的概率权重
    """
    if weights is None:
        
        weights = {
            'angry': 0.8,
            'disgust': 1.3,
            'fear': 1.4,
            'happy': 0.7,
            'neutral': 1.1,  
            'sad': 1.3,
            'surprise': 0.5  
        }
    
    emotions = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
    adjusted_probs = np.array([p * weights[e] for p, e in zip(probabilities, emotions)])
    # 重新归一化概率
    return adjusted_probs / adjusted_probs.sum()

def process_frame(frame, face_cascade, facemark, model):
    """
    处理单帧图像
    """
    # 转换为灰度图
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    
    # 检测人脸
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(100, 100),
        maxSize=(800, 800)
    )
    
    results = []
    for (x, y, w, h) in faces:
        # 检测关键点
        face_roi = np.array([[x, y, w, h]], dtype=np.int32)
        success, landmarks = facemark.fit(gray, face_roi)
        
        if success:
            landmarks = landmarks[0][0]  # 获取关键点坐标
            
            # 归一化关键点
            normalized_landmarks = normalize_landmarks(landmarks)
            
            if normalized_landmarks is not None:
                # 预测情绪
                probabilities = model.predict_proba([normalized_landmarks])[0]
                # 调整概率权重
                adjusted_probabilities = adjust_probabilities(probabilities)
                # 根据调整后的概率选择情绪
                emotion = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise'][np.argmax(adjusted_probabilities)]
                
                results.append({
                    'bbox': (x, y, w, h),
                    'landmarks': landmarks,
                    'emotion': emotion,
                    'probabilities': adjusted_probabilities
                })
    
    return results


def draw_results(frame, results, dominant_emotion, emojis, extra_emojis):
    """
    在图像上绘制检测结果
    """
    emotions = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
    emotion_colors = {
        'angry': (0, 0, 255),      # 红色
        'disgust': (0, 255, 0),    # 绿色
        'fear': (255, 0, 0),       # 蓝色
        'happy': (0, 255, 255),    # 黄色
        'neutral': (128, 128, 128), # 灰色
        'sad': (255, 0, 255),      # 紫色
        'surprise': (255, 255, 0)   # 青色
    }
    
    frame_with_effects = frame.copy()
    
    for result in results:
        x, y, w, h = result['bbox']
        emotion = dominant_emotion #不显示实时表情 显示当前domain表情
        landmarks = result['landmarks']
        probabilities = result['probabilities']
        
        # 获取对应的emoji
        animated_emoji = emojis.get(emotion)
        extra_emoji = extra_emojis.get(emotion)

        # 应用情绪特效和emoji
        frame_with_effects = apply_emotion_effects(
            frame_with_effects, 
            emotion, 
            (x, y, w, h), 
            animated_emoji, 
            extra_emoji,
            landmarks  # 传入landmarks参数
        )
        
        # 绘制人脸框
        color = emotion_colors[emotion]
        cv2.rectangle(frame_with_effects, (x, y), (x+w, y+h), color, 2)
        
        # 绘制情绪标签和概率
        label = f"{emotion}: {probabilities[emotions.index(emotion)]:.2f}"
        cv2.putText(frame_with_effects, label, (x, y-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        
        # 绘制关键点
        for (x, y) in landmarks:
            cv2.circle(frame_with_effects, (int(x), int(y)), 2, (0, 255, 0), -1)
        
        # 在右侧绘制概率条形图
        bar_width = 100
        bar_height = 15
        bar_gap = 5
        start_x = frame_with_effects.shape[1] - bar_width - 10
        start_y = 30
        
        for i, (emotion, prob) in enumerate(zip(emotions, probabilities)):
            y = start_y + i * (bar_height + bar_gap)
            # 绘制情绪标签
            cv2.putText(frame_with_effects, f"{emotion}", (start_x - 70, y + bar_height), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            # 绘制概率条
            bar_length = int(prob * bar_width)
            cv2.rectangle(frame_with_effects, (start_x, y), 
                         (start_x + bar_length, y + bar_height), 
                         emotion_colors[emotion], -1)
            # 绘制概率值
            cv2.putText(frame_with_effects, f"{prob:.2f}", 
                       (start_x + bar_width + 5, y + bar_height), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return frame_with_effects

def draw_game_ui(frame, game, dominant_emotion):
    """绘制游戏界面"""
    # 添加半透明的顶部信息栏
    info_bar = frame[:100].copy()
    overlay = info_bar.copy()
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], 100), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, info_bar, 0.5, 0, info_bar)
    frame[:100] = info_bar
    
    # 显示目标表情
    cv2.putText(frame, f"target emotion: {game.current_target}", 
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    # 显示当前表情
    if dominant_emotion:
        cv2.putText(frame, f"current emotion: {dominant_emotion}", 
                    (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    # 显示得分
    cv2.putText(frame, f"score: {game.score}", 
                (frame.shape[1] - 150, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    # 如果正在保持正确的表情，显示进度条
    if game.success_time > 0:
        progress = min(game.success_time / game.required_time, 1.0)
        bar_width = 200
        filled_width = int(bar_width * progress)
        cv2.rectangle(frame, (frame.shape[1]//2 - bar_width//2, 80), 
                     (frame.shape[1]//2 + bar_width//2, 90), (255, 255, 255), 2)
        cv2.rectangle(frame, (frame.shape[1]//2 - bar_width//2, 80),
                     (frame.shape[1]//2 - bar_width//2 + filled_width, 90), (0, 255, 0), -1)

def main():
    print("正在加载模型和初始化检测器...")
    
    # 加载模型和初始化检测器
    model_path = r"facial_recognition/random_forest_model.joblib"
    try:
        model = joblib.load(model_path)
        print(f"成功加载模型: {model_path}")
    except Exception as e:
        print(f"加载模型失败: {str(e)}")
        return
        
    face_cascade, facemark = init_detectors()
    
    # 初始化情绪历史记录器
    emotion_history = EmotionHistory(window_seconds=3)
    
    # 初始化emoji
    emojis, extra_emojis = init_emojis()
    
    # 初始化游戏
    game = EmotionGame()
    
    # 初始化摄像头
    print("正在启动摄像头...")
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # 使用默认摄像头
    
    # 设置摄像头分辨率
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    if not cap.isOpened():
        print("无法打开摄像头！")
        return
    
    print("开始表情模仿游戏！请模仿屏幕上显示的表情。")
    print("按'q'键退出或点击窗口右上角关闭按钮退出...")
    
    # 创建窗口并设置为可调整大小
    window_name = 'Emotion Game'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    
    running = True
    while running:
        # 读取一帧
        ret, frame = cap.read()
        if not ret:
            print("无法读取视频帧！")
            break
            
        # 水平翻转图像（使其像镜子）
        frame = cv2.flip(frame, 1)
        
        # 处理当前帧
        results = process_frame(frame, face_cascade, facemark, model)
        
        # 更新情绪历史
        dominant_emotion = None
        if results:
            # 只取第一个检测到的人脸的情绪
            emotion_history.add_emotion(results[0]['emotion'])
            
            # 获取主导情绪
            dominant_emotion, percentage = emotion_history.get_dominant_emotion()
            if dominant_emotion:
                # 检查游戏状态
                success, message = game.check_emotion(dominant_emotion)
                if success:
                    # 播放成功动画或声音
                    pass
        
        # 绘制结果
        if results:
            frame = draw_results(frame, results, dominant_emotion, emojis, extra_emojis)
        
        # 绘制游戏界面
        draw_game_ui(frame, game, dominant_emotion)
        
        # 显示结果
        cv2.imshow(window_name, frame)
        
        # 检查是否按下'q'键或点击了窗口的关闭按钮
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            running = False
            break
    
    # 释放资源
    cap.release()
    cv2.destroyAllWindows()
    print(f"\n游戏结束！最终得分：{game.score}")

if __name__ == "__main__":
    main() 