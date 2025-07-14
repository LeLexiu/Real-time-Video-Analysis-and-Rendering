import sys
import os
import cv2
import numpy as np
import threading
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
from PIL import Image, ImageTk
from ultralytics import YOLO
from collections import defaultdict

# Load YOLOv8 pose model
# This will be the starting model to extract poses from videos.
# However, you can replace it with any other pose model compatible with YOLOv8.
model = YOLO("yolov8n-pose.pt")

# COCO skeleton connections (keypoint pairs)
# This defines the connections between keypoints for drawing the skeleton.
# For better drawing of the skeleton, you can modify the pairs.
skeleton = [
    (0, 5), (0, 6),     # noise to shoulders
    (5, 6),             # shoulders
    (5, 7), (7, 9),     # left arm
    (6, 8), (8, 10),    # right arm
    (5, 11), (6, 12),   # torso sides
    (11, 12),           # hips
    (11, 13), (13, 15), # left leg
    (12, 14), (14, 16)  # right leg
]

# 定义骨骼索引的常见关节角度三元组，用于评分
# 格式: (关节名称, p1_idx, p2_idx, p3_idx)
JOINT_TRIPLETS = {
    "left_elbow": (5, 7, 9),  # 左肩-左肘-左手腕
    "right_elbow": (6, 8, 10),  # 右肩-右肘-右手腕
    "left_knee": (11, 13, 15),  # 左髋-左膝-左脚踝
    "right_knee": (12, 14, 16),  # 右髋-右膝-右脚踝
    "left_shoulder": (7, 5, 11),  # 左肘-左肩-左髋
    "right_shoulder": (8, 6, 12),  # 右肘-右肩-右髋
    "left_hip": (5, 11, 13),  # 左肩-左髋-左膝
    "right_hip": (6, 12, 14),  # 右肩-右髋-右膝
}

class PoseApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Stickman Dance GUI")
        self.root.geometry("1200x600") # You can adjust the window size as needed
        s = ttk.Style()
        s.theme_use('vista')  # 或者尝试 'vista''winnative'，其他字体（要换）

        # Font
        default_font = ("Segoe UI", 10)  # Windows 11 常用字体
        self.root.option_add("*Font", default_font)  # 设置所有控件的默认字体

        self.running_file = False
        self.running_cam = False
        self.video_path = ""
        self.cap_file = None
        self.cap_cam = None
        self.show_video_frame = True

        # Set up frames
        self.left_frame = ttk.Frame(self.root)
        self.left_frame.pack(side=tk.LEFT, padx=10)
        self.right_frame = ttk.Frame(self.root)
        self.right_frame.pack(side=tk.RIGHT, padx=10)

        # Properties of score
        self.video_angles = defaultdict(list)  # 存储参考视频的角度数据
        self.camera_angles = defaultdict(list)  # 存储摄像头捕获的用户角度数据
        self.score_threshold = 20  # 角度差异阈值（度），小于此值认为动作准确
        self.current_score = 0  # 当前实时分数
        self.video_frame_count = 0  # 记录视频已处理的帧数
        self.camera_frame_count = 0  # 记录摄像头已处理的帧数

        # Video File Window
        self.label_file = ttk.Label(self.left_frame)
        self.label_file.pack()
        self.controls_file = ttk.Frame(self.left_frame)
        self.controls_file.pack()
        ttk.Button(self.controls_file, text="Open Video", command=self.load_video).pack(side=tk.LEFT, padx=5)
        ttk.Button(self.controls_file, text="Start Video", command=self.start_video).pack(side=tk.LEFT, padx=5)
        ttk.Button(self.controls_file, text="Stop Video", command=self.stop_video).pack(side=tk.LEFT, padx=5)
        ttk.Button(self.controls_file, text="Show/Hide Video", command=self.toggle_video_display).pack(side=tk.LEFT, padx=5)

        # Webcam Window
        self.label_cam = ttk.Label(self.right_frame)
        self.label_cam.pack()
        self.controls_cam = ttk.Frame(self.right_frame)
        self.controls_cam.pack()
        ttk.Button(self.controls_cam, text="Start Webcam", command=self.start_cam).pack(side=tk.LEFT, padx=5)
        ttk.Button(self.controls_cam, text="Stop Webcam", command=self.stop_cam).pack(side=tk.LEFT, padx=5)

        # Scores showing
        self.score_label = ttk.Label(self.root, text=f"Scores: {self.current_score}", font=("Segoe UI", 24, "bold"))
        self.score_label.pack(side=tk.BOTTOM, pady=10)

    @staticmethod
    def calculate_angle(keypoints_pixels, p1_idx, p2_idx, p3_idx):
        """
        使用像素坐标计算三个关键点 (p1-p2-p3) 之间的角度。
        p2 是中间点（角度的顶点）。
        返回角度（度），如果关键点无效或退化则返回 None。
        """
        # 确保所有三个关键点都存在且有效
        if not (0 <= p1_idx < len(keypoints_pixels) and
                0 <= p2_idx < len(keypoints_pixels) and
                0 <= p3_idx < len(keypoints_pixels)):
            return None

        # 获取关键点坐标
        p1 = np.array(keypoints_pixels[p1_idx])
        p2 = np.array(keypoints_pixels[p2_idx])
        p3 = np.array(keypoints_pixels[p3_idx])

        # 从中间点发出的向量
        v1 = p1 - p2
        v2 = p3 - p2

        # 计算向量的模长
        magnitude_v1 = np.linalg.norm(v1)
        magnitude_v2 = np.linalg.norm(v2)

        # 避免除以零（如果点重合）
        if magnitude_v1 == 0 or magnitude_v2 == 0:
            return None  # 无法形成有效角度

        # 计算角度的余弦值（使用点积）
        dot_product = np.dot(v1, v2)
        cosine_angle = dot_product / (magnitude_v1 * magnitude_v2)

        # 将余弦值限制在 [-1, 1] 范围内，以避免浮点误差
        cosine_angle = np.clip(cosine_angle, -1.0, 1.0)

        # 计算弧度，然后转换为度数
        angle_radians = np.arccos(cosine_angle)
        angle_degrees = np.degrees(angle_radians)

        # 确保角度在 [0, 180] 之间（关节角度的常见范围）
        if angle_degrees > 180.0:
            angle_degrees = 360 - angle_degrees

        return angle_degrees

    def load_video(self):
        path = filedialog.askopenfilename(filetypes=[("MP4 files", "*.mp4"), ("All files", "*.*")])
        if path:
            self.video_path = path
            messagebox.showinfo("Video Selected", os.path.basename(path))

    def start_video(self):
        if not self.video_path:
            messagebox.showwarning("No videos", "Please select a video file first")
            return
        if not self.running_file:
            # 重置所有与评分相关的变量
            self.video_angles = defaultdict(list)
            self.camera_angles = defaultdict(list)
            self.video_frame_count = 0
            self.camera_frame_count = 0
            self.current_score = 0
            self.update_score_display()  # 更新显示为0

            self.running_file = True
            threading.Thread(target=self.process_video_file, daemon=True).start()
            # 确保摄像头在视频开始后也启动 -> 同步收集数据
            if not self.running_cam:
                self.start_cam()  # 自动启动摄像头

    def stop_video(self):
        self.running_file = False
        if self.cap_file:
            self.cap_file.release()
        # 视频停止后，计算最终分数
        # self.calculate_final_score()
    
    def toggle_video_display(self):
        self.show_video_frame = not self.show_video_frame

    def start_cam(self):
        if not self.running_cam:
            self.running_cam = True
            threading.Thread(target=self.process_webcam, daemon=True).start()

    def stop_cam(self):
        self.running_cam = False
        if self.cap_cam:
            self.cap_cam.release()
            self.cap_cam = None  # 确保在停止时释放并清空

    def process_video_file(self):
        self.cap_file = cv2.VideoCapture(self.video_path)
        # 每次开始新视频时重置角度数据和帧计数器
        self.video_angles = defaultdict(list)
        self.camera_angles = defaultdict(list)  # 确保摄像头角度也重置，以便新一轮游戏
        self.video_frame_count = 0
        self.camera_frame_count = 0  # 重置摄像头帧计数器
        self.current_score = 0
        self.update_score_display()  # 更新显示为0

        while self.cap_file.isOpened() and self.running_file:
            ret_file, frame_file = self.cap_file.read()
            if not ret_file:
                break

            # 处理视频帧的姿态
            display_frame_file, video_keypoints = self.process_pose(frame_file)

            # 存储视频帧的角度
            if video_keypoints:
                for joint_name, (p1, p2, p3) in JOINT_TRIPLETS.items():
                    angle = self.calculate_angle(video_keypoints, p1, p2, p3)
                    self.video_angles[joint_name].append(angle)  # 即使是 None 也存储，保持列表长度一致
            else:
                # 如果视频帧未检测到姿态，也记录 None，保持与摄像头帧的潜在同步
                for joint_name in JOINT_TRIPLETS:
                    self.video_angles[joint_name].append(None)

            self.video_frame_count += 1  # 视频帧计数器递增

            self.update_label(self.label_file, display_frame_file)

        self.cap_file.release()
        # 视频播放结束后，计算最终分数
        self.calculate_final_score()

    def process_webcam(self):
        self.cap_cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        # 摄像头启动时不需要重置角度，因为视频启动时已经重置

        while self.cap_cam.isOpened() and self.running_cam:
            ret_cam, frame_cam = self.cap_cam.read()
            if not ret_cam:
                break

            # 翻转摄像头帧以获得镜像效果
            frame_cam = cv2.flip(frame_cam, 1)

            # 处理摄像头帧的姿态
            display_frame_cam, camera_keypoints = self.process_pose(frame_cam)

            # 存储摄像头帧的角度
            if camera_keypoints:
                for joint_name, (p1, p2, p3) in JOINT_TRIPLETS.items():
                    angle = self.calculate_angle(camera_keypoints, p1, p2, p3)
                    self.camera_angles[joint_name].append(angle)
            else:
                for joint_name in JOINT_TRIPLETS:
                    self.camera_angles[joint_name].append(None)

            self.update_live_score()

            # 获取当前分数并将其转换为字符串
            score_text = f"Score: {int(self.current_score)}"

            # 在摄像头帧上绘制分数
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1
            font_thickness = 2
            text_size = cv2.getTextSize(score_text, font, font_scale, font_thickness)[0]
            text_x = 10
            text_y = frame_cam.shape[:2][0] - 30  # 靠近底部显示

            cv2.putText(display_frame_cam, score_text, (text_x, text_y), font, font_scale, (0, 255, 0), font_thickness,
                        cv2.LINE_AA)

            self.camera_frame_count += 1  # 摄像头帧计数器递增

            # 实时计算分数
            self.update_live_score()

            self.update_label(self.label_cam, display_frame_cam)

        self.cap_cam.release()
        # 如何实现摄像头停止不立即计算总分，因为总分是在视频结束后计算的

    def process_pose(self, frame):
        results = model(frame, conf=0.3)
        height, width = frame.shape[:2]

        overflow = frame.copy()
        if not self.show_video_frame:
            overflow = np.ones_like(frame) * 255  # 白色背景

        person_keypoints_pixels = None  # 初始化为 None

        # 遍历检测到的人物（YOLOv8 可以检测多个人）
        # 这里我们取第一个检测到的人物（通常是置信度最高的）
        if results:
            if first_result := results[0]:  # 使用海象运算符 (Python 3.8+)
                if first_result.keypoints is not None and len(first_result.keypoints.xyn) > 0:
                    # keypoints_xyn 是归一化坐标 (0-1)，转换为像素坐标
                    # 注意：xyn 包含 x, y 和 confidence，所以我们只取前两个作为坐标
                    person_kpts_xyn = first_result.keypoints.xyn[0].cpu().numpy()
                    person_keypoints_pixels = [
                        (int(kp[0] * width), int(kp[1] * height)) for kp in person_kpts_xyn
                    ]

                    # 在 overflow 上绘制骨骼和关键点
                    for pt1, pt2 in skeleton:
                        # 确保关键点索引在范围内
                        if pt1 < len(person_keypoints_pixels) and pt2 < len(person_keypoints_pixels):
                            x1, y1 = person_keypoints_pixels[pt1]
                            x2, y2 = person_keypoints_pixels[pt2]
                            cv2.line(overflow, (x1, y1), (x2, y2), (0, 255, 0), 2)  # 绿色线条

                    for kp_x, kp_y in person_keypoints_pixels:
                        cv2.circle(overflow, (kp_x, kp_y), 5, (0, 0, 255), -1)  # 红色圆点

        return overflow, person_keypoints_pixels  # 返回用于显示和评分的帧和关键点

    def update_label(self, label, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        img = img.resize((640, 384))  # Resize for better fit
        imgtk = ImageTk.PhotoImage(image=img)
        label.imgtk = imgtk
        label.configure(image=imgtk)

    def update_live_score(self):
        """
        实时更新分数。在每一帧处理后调用。
        """
        # 确保两个列表都有数据，并且可以找到对应的帧
        current_cam_idx = self.camera_frame_count - 1
        if current_cam_idx < 0:
            return

        # 尝试获取对应视频帧的索引，如果视频帧少于摄像头帧，则跳过
        if current_cam_idx >= self.video_frame_count:
            return

        frame_scores = []
        for joint_name in JOINT_TRIPLETS:
            video_angle = self.video_angles[joint_name][current_cam_idx] if current_cam_idx < len(
                self.video_angles[joint_name]) else None
            camera_angle = self.camera_angles[joint_name][current_cam_idx] if current_cam_idx < len(
                self.camera_angles[joint_name]) else None

            if video_angle is not None and camera_angle is not None:
                difference = abs(video_angle - camera_angle)
                if difference < self.score_threshold:
                    frame_scores.append(1)  # 1 表示该关节在该帧是“准确”的
                else:
                    frame_scores.append(0)
            else:
                frame_scores.append(0)  # 如果任一角度为 None，则该关节该帧不得分

        if frame_scores:
            # 计算当前帧的平均命中率
            frame_accuracy = sum(frame_scores) / len(frame_scores)
            # 将其加到总分中，可以根据需要调整权重
            # 这里我们简单地将每帧的准确率累加
            # 注意：采用累加方式，非平均分（如何实现平均分？）
            self.current_score += frame_accuracy * 100 / len(JOINT_TRIPLETS)  # 简单平均，可以调整权重

        self.update_score_display()

    def calculate_final_score(self):
        """
        在视频播放结束后计算最终分数。
        """
        # 取视频和摄像头帧数中最短的那个作为比较长度（解决分差问题）
        min_len = min(self.video_frame_count, self.camera_frame_count)
        if min_len == 0:
            messagebox.showinfo("Game over", "There is not enough pose data to calculate a score. Please make sure that the video and camera are working properly and detecting poses.")
            self.update_score_display(final=True)
            return

        all_joint_scores = []
        for joint_name in JOINT_TRIPLETS:
            video_angles_joint = self.video_angles[joint_name][:min_len]
            camera_angles_joint = self.camera_angles[joint_name][:min_len]

            accuracy_count = 0
            for i in range(min_len):
                video_angle = video_angles_joint[i]
                camera_angle = camera_angles_joint[i]

                if video_angle is not None and camera_angle is not None:
                    difference = abs(video_angle - camera_angle)
                    if difference < self.score_threshold:
                        accuracy_count += 1

            if min_len > 0:  # ！！！避免除以零
                joint_score = (accuracy_count / min_len) * 100
                all_joint_scores.append(joint_score)

        final_score = 0
        if all_joint_scores:
            final_score = np.mean(all_joint_scores)
            final_score += 20  # 奖励分

        if final_score > 100:
            final_score = 100

        self.current_score = int(final_score)  # 更新最终分数
        self.update_score_display(final=True)  # 更新显示为最终分数

        self.current_score = int(final_score)  # 更新最终分数
        self.update_score_display(final=True)  # 更新显示为最终分数
        # messagebox.showinfo("Game over", f"Your final score is: {self.current_score}")

    def update_score_display(self, final=False):
        """
        更新 GUI 上的分数显示。
        """
        if final:
            self.score_label.config(text=f"Final score: {self.current_score}")
        else:
            self.score_label.config(text=f"Score: {int(self.current_score)}")



if __name__ == "__main__":
    root = tk.Tk()
    app = PoseApp(root)
    root.mainloop()