# Import necessary libraries
import sys
import os
import cv2
import numpy as np
import threading
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk, ImageDraw, ImageFont
from ultralytics import YOLO
import random
import time
from collections import deque
import joblib

# 添加项目根目录到Python路径以导入情绪识别模块
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(project_root)

from .process_dataset import init_detectors, normalize_landmarks
from .special_effects import apply_emotion_effects, init_emojis
from .pose_detect import SKELETON, get_detected_poses
from .realtime_emotion import EmotionHistory
from models.pose_model import load_pose_model

# Load YOLOv8 pose model
model = load_pose_model()

def adjust_probabilities(probabilities, weights=None):
    """调整情绪预测的概率权重"""
    if weights is None:
        weights = {
            'angry': 0.9,
            'disgust': 1.3,
            'fear': 1.4,
            'happy': 0.7,
            'neutral': 0.9,  
            'sad': 1.2,
            'surprise': 0.5  
        }
    
    emotions = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
    adjusted_probs = np.array([p * weights[e] for p, e in zip(probabilities, emotions)])
    # 重新归一化概率
    return adjusted_probs / adjusted_probs.sum()

class PoseEmotionGame:
    def __init__(self):
        # Available poses and emotions
        self.poses = ['Hands Up', 'T-Pose', 'Hands Overhead', 'Salute']
        self.emotions = ['happy', 'angry', 'sad', 'surprise', 'fear', 'neutral']
        
        # Current targets
        self.current_target_pose = None
        self.current_target_emotion = None
        
        # Game state
        self.start_time = None
        self.success_time = 0
        self.required_time = 2.0  # Need to hold for 2 seconds
        self.timeout_time = 20.0  # Timeout after 20 seconds
        self.game_duration = 90.0  # Total game time: 1 minute 30 seconds
        self.game_start_time = time.time()  # When the game started
        self.game_over = False
        self.score = 0
        self.attempt_start_time = None  # Track when we start attempting a new pose/emotion
        
        # Generate first target
        self.generate_new_target()
    
    def generate_new_target(self):
        """Generate new target pose and emotion combination"""
        if self.current_target_pose is None:
            self.current_target_pose = random.choice(self.poses)
        else:
            # Ensure new target is different from current
            new_target = random.choice([p for p in self.poses if p != self.current_target_pose])
            self.current_target_pose = new_target
            
        if self.current_target_emotion is None:
            self.current_target_emotion = random.choice(self.emotions)
        else:
            new_emotion = random.choice([e for e in self.emotions if e != self.current_target_emotion])
            self.current_target_emotion = new_emotion
            
        self.start_time = None
        self.success_time = 0
        self.attempt_start_time = time.time()  # Reset attempt timer
    
    def check_game_over(self):
        """Check if the game time is up"""
        if not self.game_over and time.time() - self.game_start_time >= self.game_duration:
            self.game_over = True
            return True
        return False
    
    def get_remaining_game_time(self):
        """Get remaining game time in seconds"""
        return max(0, self.game_duration - (time.time() - self.game_start_time))
    
    def check_pose_and_emotion(self, detected_poses, dominant_emotion):
        """Check if current pose and emotion match targets"""
        current_time = time.time()
        
        # Check if game is over
        if self.check_game_over():
            return False, f"Game Over! Final Score: {self.score}"
        
        # Initialize attempt_start_time if not set
        if self.attempt_start_time is None:
            self.attempt_start_time = current_time
        
        # Check for timeout
        if current_time - self.attempt_start_time >= self.timeout_time:
            self.generate_new_target()
            return False, "Time's up! Moving to next pose and emotion..."
        
        # Check if both pose and emotion match
        pose_match = self.current_target_pose in detected_poses if detected_poses else False
        emotion_match = dominant_emotion == self.current_target_emotion if dominant_emotion else False
        
        if pose_match and emotion_match:
            if self.start_time is None:
                self.start_time = current_time
            self.success_time = current_time - self.start_time
            
            if self.success_time >= self.required_time:
                self.score += 1
                self.generate_new_target()
                return True, "Great! Successfully completed pose and emotion!"
            return False, f"Hold pose and emotion {self.success_time:.1f}/{self.required_time}s"
        else:
            self.start_time = None
            self.success_time = 0
            message = []
            if not pose_match:
                message.append(f"Make the {self.current_target_pose} pose")
            if not emotion_match:
                message.append(f"Show {self.current_target_emotion} emotion")
            # Add remaining time to message
            time_left = self.timeout_time - (current_time - self.attempt_start_time)
            message.append(f"Time left: {time_left:.1f}s")
            return False, " and ".join(message)

def draw_game_ui(frame, game, detected_poses, dominant_emotion):
    """Draw game interface"""
    height, width = frame.shape[:2]
    
    if game.game_over:
        # Display game over screen
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (width, height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Display final score
        text = "GAME OVER!"
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 2, 3)[0]
        text_x = (width - text_size[0]) // 2
        text_y = (height - text_size[1]) // 2
        cv2.putText(frame, text, (text_x, text_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
        
        score_text = f"Final Score: {game.score}"
        score_size = cv2.getTextSize(score_text, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 2)[0]
        score_x = (width - score_size[0]) // 2
        cv2.putText(frame, score_text, (score_x, text_y + 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)
        
        return frame
    
    # Add semi-transparent top info bar
    info_bar_height = 100
    info_bar = frame[:info_bar_height].copy()
    overlay = info_bar.copy()
    cv2.rectangle(overlay, (0, 0), (width, info_bar_height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, info_bar, 0.5, 0, info_bar)
    frame[:info_bar_height] = info_bar
    
    # Left side: Target information (smaller font)
    cv2.putText(frame, "Target:", 
                (20, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(frame, f"Pose: {game.current_target_pose}", 
                (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(frame, f"Emotion: {game.current_target_emotion}", 
                (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Center: Current status (smaller font)
    center_x = width // 2 - 100
    if detected_poses:
        poses_text = " | ".join(detected_poses)
        cv2.putText(frame, f"Current Pose: {poses_text}", 
                    (center_x, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    if dominant_emotion:
        cv2.putText(frame, f"Current Emotion: {dominant_emotion}", 
                    (center_x, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Right side: Score (keep original size for visibility)
    score_x = width - 150
    cv2.putText(frame, f"Score: {game.score}", 
                (score_x, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    # Bottom left: Time information
    time_x = 20
    time_y = height - 20  # Base position for time display
    
    # Game time (bottom left)
    game_time_left = game.get_remaining_game_time()
    minutes = int(game_time_left // 60)
    seconds = int(game_time_left % 60)
    time_color = (255, 255, 255) if game_time_left > 10 else (0, 0, 255)
    cv2.putText(frame, f"Game Time: {minutes}:{seconds:02d}", 
                (time_x, time_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, time_color, 2)
    
    # Attempt time (above game time)
    if game.attempt_start_time is not None:
        time_left = max(0, game.timeout_time - (time.time() - game.attempt_start_time))
        time_color = (255, 255, 255) if time_left > 3 else (0, 0, 255)
        cv2.putText(frame, f"Attempt Time: {time_left:.1f}s", 
                    (time_x, time_y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, time_color, 2)
    
    # Progress bar (center bottom)
    if game.success_time > 0:
        progress = min(game.success_time / game.required_time, 1.0)
        bar_width = 200
        bar_height = 10
        bar_x = (width - bar_width) // 2
        bar_y = height - 40
        
        # Draw background bar
        cv2.rectangle(frame, (bar_x, bar_y), 
                     (bar_x + bar_width, bar_y + bar_height), 
                     (255, 255, 255), 2)
        
        # Draw filled portion
        filled_width = int(bar_width * progress)
        cv2.rectangle(frame, (bar_x, bar_y),
                     (bar_x + filled_width, bar_y + bar_height), 
                     (0, 255, 0), -1)
    
    return frame

class PoseApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Pose and Emotion Game")
        # 设置窗口最小尺寸
        self.root.minsize(800, 600)
        # 使窗口最大化
        self.root.state('zoomed')

        self.running_file = False
        self.running_cam = False
        self.video_path = ""
        self.cap_file = None
        self.cap_cam = None
        self.show_video_frame = True
        self.show_skeleton = False  # 新增：控制是否显示骨架
        
        # 初始化情绪检测器
        self.face_cascade, self.facemark = init_detectors()
        model_path = os.path.join(project_root, "final_project/facial_expression_dataset/facial_expression_dataset/train/landmarks_data/classifier_results/evaluation_results_20250710_202704/random_forest_model.joblib")
        self.emotion_model = joblib.load(model_path)
        self.emotion_history = EmotionHistory(window_seconds=3)  # 使用3秒时间窗口
        
        # 初始化游戏
        self.game = PoseEmotionGame()
        
        # 初始化emoji
        self.emojis, self.extra_emojis = init_emojis()

        # 创建主框架并设置权重
        self.main_frame = tk.Frame(self.root)
        self.main_frame.pack(expand=True, fill=tk.BOTH)
        self.main_frame.grid_rowconfigure(0, weight=1)
        self.main_frame.grid_columnconfigure(0, weight=1)

        # 视频显示区域使用Grid布局
        self.video_label = tk.Label(self.main_frame)
        self.video_label.grid(row=0, column=0, sticky='nsew', padx=10, pady=10)

        # 控制按钮区域
        self.controls_frame = tk.Frame(self.main_frame)
        self.controls_frame.grid(row=1, column=0, sticky='ew', padx=10, pady=5)
        self.controls_frame.grid_columnconfigure(0, weight=1)
        self.controls_frame.grid_columnconfigure(1, weight=1)
        
        # 文件控制按钮
        self.file_controls = tk.Frame(self.controls_frame)
        self.file_controls.grid(row=0, column=0, sticky='w')
        tk.Button(self.file_controls, text="Open Video", command=self.load_video).pack(side=tk.LEFT, padx=5)
        tk.Button(self.file_controls, text="Start Video", command=self.start_video).pack(side=tk.LEFT, padx=5)
        tk.Button(self.file_controls, text="Stop Video", command=self.stop_video).pack(side=tk.LEFT, padx=5)
        tk.Button(self.file_controls, text="Show/Hide Video", command=self.toggle_video_display).pack(side=tk.LEFT, padx=5)

        # 摄像头控制按钮
        self.cam_controls = tk.Frame(self.controls_frame)
        self.cam_controls.grid(row=0, column=1, sticky='e')
        tk.Button(self.cam_controls, text="Start Game", command=self.start_cam).pack(side=tk.LEFT, padx=5)
        tk.Button(self.cam_controls, text="Stop Game", command=self.stop_cam).pack(side=tk.LEFT, padx=5)

        # 绑定窗口大小改变事件
        self.root.bind('<Configure>', self.on_window_resize)
        self.last_width = self.root.winfo_width()
        self.last_height = self.root.winfo_height()

    def load_video(self):
        path = filedialog.askopenfilename(filetypes=[("MP4 files", "*.mp4"), ("All files", "*.*")])
        if path:
            self.video_path = path
            messagebox.showinfo("Video Selected", os.path.basename(path))

    def start_video(self):
        if not self.video_path:
            messagebox.showwarning("No Video", "Please select a video first.")
            return
        if not self.running_file:
            self.running_file = True
            threading.Thread(target=self.process_video_file, daemon=True).start()

    def stop_video(self):
        self.running_file = False
        if self.cap_file:
            self.cap_file.release()
    
    def toggle_video_display(self):
        self.show_video_frame = not self.show_video_frame

    def start_cam(self):
        if not self.running_cam:
            self.cap_cam = cv2.VideoCapture(0)
            self.running_cam = True
            self.update_webcam_frame()

    def stop_cam(self):
        self.running_cam = False
        if self.cap_cam:
            self.cap_cam.release()

    def process_video_file(self):
        self.cap_file = cv2.VideoCapture(self.video_path)
        while self.cap_file.isOpened() and self.running_file:
            ret, frame = self.cap_file.read()
            if not ret:
                break
            frame = self.process_pose(frame)
            self.update_label(self.video_label, frame)  # 使用单一的video_label
        self.cap_file.release()

    def update_webcam_frame(self):
        if self.running_cam and self.cap_cam.isOpened():
            ret, frame = self.cap_cam.read()
            if ret:
                frame = self.process_pose(frame)
                self.update_label(self.video_label, frame)  # 使用单一的video_label
        # 每10毫秒刷新一次
        self.root.after(10, self.update_webcam_frame)

    def get_dominant_emotion(self):
        """获取主导情绪"""
        dominant_emotion, percentage = self.emotion_history.get_dominant_emotion()
        return dominant_emotion  # 只返回情绪，忽略百分比

    def process_pose(self, frame):
        results = model(frame, conf=0.3)
        height, width = frame.shape[:2]
        detected_poses = []

        if self.show_video_frame:
            overlay = frame.copy()
        else:
            overlay = np.ones_like(frame) * 255

        # Process pose detection
        for result in results:
            if result.keypoints is not None:
                keypoints_xyn = result.keypoints.xyn.cpu().numpy()
                for person_kpts in keypoints_xyn:
                    keypoints = [
                        (int(x * width), int(y * height)) for x, y in person_kpts
                    ]
                    # 助教建议在游戏时不显示骨架，避免影响用户体验
                    # 只在开启显示骨架时绘制
                    # if self.show_skeleton:
                    #     # Draw skeleton
                    #     for pt1, pt2 in skeleton:
                    #         if pt1 < len(keypoints) and pt2 < len(keypoints):
                    #             x1, y1 = keypoints[pt1]
                    #             x2, y2 = keypoints[pt2]
                    #             cv2.line(overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    # # Draw keypoints
                    # for x, y in keypoints:
                    #     cv2.circle(overlay, (x, y), 5, (0, 0, 255), -1)

                    # Detect poses
                    detected_poses = get_detected_poses(keypoints)
                    

        # Process emotion detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))
        
        current_probabilities = None
        
        for (x, y, w, h) in faces:
            face_roi = np.array([[x, y, w, h]], dtype=np.int32)
            success, landmarks = self.facemark.fit(gray, face_roi)
            
            if success:
                landmarks = landmarks[0][0]
                normalized_landmarks = normalize_landmarks(landmarks)
                
                if normalized_landmarks is not None:
                    # 将关键点数组展平为一维数组
                    flattened_landmarks = normalized_landmarks.flatten()
                    # 预测情绪概率
                    probabilities = self.emotion_model.predict_proba([flattened_landmarks])[0]
                    # 调整概率权重
                    adjusted_probabilities = adjust_probabilities(probabilities)
                    current_probabilities = adjusted_probabilities
                    
                    # 获取当前帧的情绪
                    emotion = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise'][np.argmax(adjusted_probabilities)]
                    self.emotion_history.add_emotion(emotion)
                    
                    # 在右上角显示情绪概率
                    emotions = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
                    bar_width = 100
                    bar_height = 15
                    bar_gap = 5
                    start_x = overlay.shape[1] - bar_width - 10
                    start_y = 30
                    
                    for i, (emotion, prob) in enumerate(zip(emotions, adjusted_probabilities)):
                        y = start_y + i * (bar_height + bar_gap)
                        # 绘制情绪标签
                        cv2.putText(overlay, f"{emotion}", (start_x - 70, y + bar_height), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                        # 绘制概率条
                        bar_length = int(prob * bar_width)
                        cv2.rectangle(overlay, (start_x, y), 
                                     (start_x + bar_length, y + bar_height), 
                                     (0, 255, 0), -1)
                        # 绘制概率值
                        cv2.putText(overlay, f"{prob:.2f}", 
                                   (start_x + bar_width + 5, y + bar_height), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        # 获取主导情绪（使用历史统计，避免情绪显示突变）
        dominant_emotion = self.get_dominant_emotion()

        # Update game state
        success, message = self.game.check_pose_and_emotion(detected_poses, dominant_emotion)
        
        # Draw game UI
        draw_game_ui(overlay, self.game, detected_poses, dominant_emotion)

        # Add emotion effects if emotion detected
        if dominant_emotion and len(faces) > 0:
            animated_emoji = self.emojis.get(dominant_emotion)
            extra_emoji = self.extra_emojis.get(dominant_emotion)
            x, y, w, h = faces[0]  # Use first detected face
            overlay = apply_emotion_effects(overlay, dominant_emotion, (x, y, w, h), 
                                         animated_emoji, extra_emoji, landmarks)

        return overlay

    def update_label(self, label, frame):
        """更新视频标签显示"""
        self.current_frame = frame  # 保存当前帧用于调整大小
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        
        # 获取窗口大小
        window_width = label.winfo_width()
        window_height = label.winfo_height()
        
        if window_width > 1 and window_height > 1:  # 确保窗口尺寸有效
            # 计算保持宽高比的新尺寸
            frame_height, frame_width = frame.shape[:2]
            aspect_ratio = frame_width / frame_height
            
            # 计算适合窗口的最大尺寸
            if window_width / window_height > aspect_ratio:
                # 窗口较宽，以高度为准
                new_height = window_height
                new_width = int(window_height * aspect_ratio)
            else:
                # 窗口较高，以宽度为准
                new_width = window_width
                new_height = int(window_width / aspect_ratio)
            
            size = (new_width, new_height)
        else:
            size = (1280, 720)  # 默认尺寸
        
        img = img.resize(size, Image.Resampling.LANCZOS)
        imgtk = ImageTk.PhotoImage(image=img)
        label.imgtk = imgtk
        label.configure(image=imgtk)

    def on_window_resize(self, event):
        """处理窗口大小改变事件"""
        # 忽略标签大小改变事件，只处理窗口大小改变
        if event.widget == self.root and (self.last_width != event.width or self.last_height != event.height):
            self.last_width = event.width
            self.last_height = event.height
            # 更新视频显示大小
            if hasattr(self, 'current_frame'):
                self.update_video_display()

    def update_video_display(self):
        """根据窗口大小更新视频显示"""
        if hasattr(self, 'current_frame'):
            # 获取窗口大小
            window_width = self.video_label.winfo_width()
            window_height = self.video_label.winfo_height()
            
            if window_width > 1 and window_height > 1:  # 确保窗口尺寸有效
                # 计算保持宽高比的新尺寸
                frame_height, frame_width = self.current_frame.shape[:2]
                aspect_ratio = frame_width / frame_height
                
                # 计算适合窗口的最大尺寸
                if window_width / window_height > aspect_ratio:
                    # 窗口较宽，以高度为准
                    new_height = window_height
                    new_width = int(window_height * aspect_ratio)
                else:
                    # 窗口较高，以宽度为准
                    new_width = window_width
                    new_height = int(window_width / aspect_ratio)
                
                # 更新显示
                self.update_label(self.video_label, self.current_frame, (new_width, new_height))

if __name__ == "__main__":
    root = tk.Tk()
    app = PoseApp(root)
    root.mainloop()