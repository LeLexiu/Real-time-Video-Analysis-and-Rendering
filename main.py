import tkinter as tk
import os
from tkinter import messagebox, filedialog
import cv2
import threading
from collections import defaultdict
import time
from gui.gui import MainWindow
from gui.vedio_score_page import ScoreDisplay
from models.pose_model import load_pose_model, process_pose
from just_dance.vedio_mode.scoring import calculate_angle, JOINT_TRIPLETS, calculate_final_score, calculate_frame_score

class PoseApp:
    def __init__(self, root):
        self.root = root
        self.main_window = MainWindow(
            root,
            self._load_video_callback,
            self.start_video,
            self.stop_video,
            self.toggle_video_display,
            self.start_cam,
            self.stop_cam
        )
        self.score_display = ScoreDisplay(root)
        self.model = load_pose_model()

        self.running_file = False
        self.running_cam = False
        self.video_path = ""
        self.cap_file = None
        self.cap_cam = None
        self.show_video_frame = True

        self.video_angles = defaultdict(list)
        self.camera_angles = defaultdict(list)
        self.score_threshold = 20
        self.current_score = 0
        self.video_frame_count = 0
        self.camera_frame_count = 0

        self.target_fps = 30  # frame rate for video and camara processing
        self.frame_duration = 1.0 / self.target_fps # duration of each frame in seconds
        self.reaction_time_frames = int(0.8 * self.target_fps)  # 0.8 seconds reaction time in frames

    # Callback function to load video file
    def _load_video_callback(self):
        path = filedialog.askopenfilename(filetypes=[("MP4 files", "*.mp4"), ("All files", "*.*")])
        if path:
            self.video_path = path
            messagebox.showinfo("Video Selected", os.path.basename(path))
            return self.video_path
        return None

    def start_video(self):
        if not self.video_path:
            messagebox.showwarning("No videos", "Please select a video file first")
            return
        if not self.running_file:
            self.reset_game_state() # 重置所有与评分和捕获相关的状态
            self.running_file = True
            threading.Thread(target=self.process_video_file, daemon=True).start()
            # 确保摄像头在视频开始后也启动 -> 同步收集数据
            if not self.running_cam:
                self.start_cam() # automatically start webcam if not already running

    def stop_video(self):
        self.running_file = False
        if self.cap_file:
            self.cap_file.release()
            self.cap_file = None # 释放资源
        self.calculate_final_score_wrapper()

    def toggle_video_display(self):
        self.show_video_frame = not self.show_video_frame

    def start_cam(self):
        if not self.running_cam:
            self.cap_cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            if not self.cap_cam.isOpened():
                messagebox.showerror("Error", "Failed to open webcam, please check your camera settings.")
                self.cap_cam = None
                return

            self.cap_cam.set(cv2.CAP_PROP_XI_FRAMERATE, 50)  # Set the frame rate to 50 FPS
            self.cap_cam.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Set buffer size to 1 to reduce latency

            self.running_cam = True
            threading.Thread(target=self.process_webcam, daemon=True).start()

    def stop_cam(self):
        self.running_cam = False
        if self.cap_cam:
            self.cap_cam.release()
            self.cap_cam = None

    def process_video_file(self):
        self.cap_file = cv2.VideoCapture(self.video_path)
        if not self.cap_file.isOpened():
            messagebox.showerror("Error", "Could not open video file.")
            self.stop_video()
            return

        last_frame_time = time.time()  # record the time of the starting processed frame

        while self.cap_file.isOpened() and self.running_file:
            '''Fixed frame rate processing'''
            current_time = time.time()
            elapsed_time = current_time - last_frame_time

            # Wait if the current frame is processed too quickly
            if elapsed_time < self.frame_duration:
                time.sleep(self.frame_duration - elapsed_time)

            last_frame_time = time.time()

            ret_file, frame_file = self.cap_file.read()
            if not ret_file:
                break

            display_frame_file, video_keypoints = process_pose(self.model, frame_file, self.show_video_frame)

            # 存储视频帧的角度 Store angles for video frames
            if video_keypoints:
                self.store_angles(video_keypoints, self.video_angles)
            else:
                for joint_name in JOINT_TRIPLETS:
                    self.video_angles[joint_name].append(None)
            self.video_frame_count += 1

            self.main_window.update_label(self.main_window.label_file, display_frame_file)

        if self.cap_file:
            self.cap_file.release()
            self.cap_file = None
        self.calculate_final_score_wrapper() # Calculate final score after video processing

    def process_webcam(self):
        # self.cap_cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)

        if not self.cap_cam.isOpened():
            messagebox.showerror("Error", "Could not open webcam.")
            self.stop_cam()
            return

        last_frame_time = time.time()

        while self.cap_cam.isOpened() and self.running_cam:
            '''Fixed frame rate processing for webcam'''
            current_time = time.time()
            elapsed_time = current_time - last_frame_time

            # Wait if the current frame is processed too quickly
            if elapsed_time < self.frame_duration:
                time.sleep(self.frame_duration - elapsed_time)
            last_frame_time = time.time()  # Update the last frame time

            ret_cam, frame_cam = self.cap_cam.read()
            if not ret_cam:
                break

            frame_cam = cv2.flip(frame_cam, 1)

            display_frame_cam, camera_keypoints = process_pose(self.model, frame_cam, True)

            # 存储摄像头帧的角度
            if camera_keypoints:
                self.store_angles(camera_keypoints, self.camera_angles)
            else:
                for joint_name in JOINT_TRIPLETS:
                    self.camera_angles[joint_name].append(None)
            self.camera_frame_count += 1

            self.update_live_score() # 实时计算分数

            # Display the current score on the webcam frame
            score_text = f"Score: {int(self.current_score)}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1
            font_thickness = 2
            text_y = frame_cam.shape[0] - 30 if frame_cam.shape[0] > 30 else 10 # 靠近底部显示
            cv2.putText(display_frame_cam, score_text, (10, text_y), font, font_scale, (0, 255, 0), font_thickness, cv2.LINE_AA)

            self.main_window.update_label(self.main_window.label_cam, display_frame_cam)

        if self.cap_cam:
            self.cap_cam.release()
            self.cap_cam = None

    # store angles for each joint triplet in the provided dictionary
    def store_angles(self, keypoints, angle_dict):
        for joint_name, (p1, p2, p3) in JOINT_TRIPLETS.items():
            angle = calculate_angle(keypoints, p1, p2, p3)
            angle_dict[joint_name].append(angle)

    # Update the live score based on the current video and camera angles
    def update_live_score(self):
        # current_cam_idx = self.camera_frame_count - 1 # 当前摄像头帧的索引
        current_video_idx = self.video_frame_count - 1  # 当前视频帧的索引
        corresponding_cam_idx = current_video_idx - self.reaction_time_frames

        if current_video_idx >= 0 and \
                corresponding_cam_idx >= 0 and \
                corresponding_cam_idx < self.camera_frame_count:  # 确保延迟索引不超出摄像头已捕获的范围

            frame_accuracy = calculate_frame_score(self.video_angles, self.camera_angles, current_video_idx,
                                                   self.score_threshold)

            # 提取当前视频帧的关节角度
            video_angles_current_frame = {
                joint_name: self.video_angles[joint_name][current_video_idx]
                for joint_name in JOINT_TRIPLETS if current_video_idx < len(self.video_angles[joint_name])
            }

            # 提取对应的摄像头延迟帧的关节角度
            camera_angles_corresponding_frame = {
                joint_name: self.camera_angles[joint_name][corresponding_cam_idx]
                for joint_name in JOINT_TRIPLETS if corresponding_cam_idx < len(self.camera_angles[joint_name])
            }

            # 然后调用一个修改过的 calculate_frame_score，它接受字典形式的单帧角度
            frame_accuracy = self._calculate_single_frame_accuracy(
                video_angles_current_frame,
                camera_angles_corresponding_frame,
                self.score_threshold
            )

            if frame_accuracy is not None:
                self.current_score += (frame_accuracy * 100) / len(JOINT_TRIPLETS)
            self.score_display.update_score(int(self.current_score))
        else:
            # 如果帧数不够进行延迟对比，不进行实时更新，或者可以显示一个等待状态
            self.score_display.update_score(int(self.current_score))  # 仍然更新显示当前分数

        '''# Make sure the current index is valid
        if current_cam_idx >= 0 and current_cam_idx < self.video_frame_count:
            frame_accuracy = calculate_frame_score(self.video_angles, self.camera_angles, current_cam_idx, self.score_threshold)
            if frame_accuracy is not None:
                # 累加分数，每个关节的贡献是( 100 / 关节总数 * 准确率)
                self.current_score += (frame_accuracy * 100) / len(JOINT_TRIPLETS)
            self.score_display.update_score(self.current_score)
        else:
            # 如果视频或摄像头帧数不匹配，不进行实时更新，或者可以显示一个等待状态
            self.score_display.update_score(self.current_score) # 仍然更新显示当前分数'''

    '''Calculate the accuracy for a single frame based on video and camera angles'''
    def _calculate_single_frame_accuracy(self, video_angles_dict, camera_angles_dict, score_threshold):
        frame_scores = []
        for joint_name in JOINT_TRIPLETS:
            v_angle = video_angles_dict.get(joint_name)
            c_angle = camera_angles_dict.get(joint_name)

            if v_angle is not None and c_angle is not None:
                difference = abs(v_angle - c_angle)
                if difference < score_threshold:
                    frame_scores.append(1)
                else:
                    frame_scores.append(0)
            else:
                frame_scores.append(0)  # 角度缺失，该关节得 0 分

        if frame_scores:
            return sum(frame_scores) / len(frame_scores)
        return 0  # 如果没有可比较的关节

    # Calculate the final score after processing all frames
    def calculate_final_score_wrapper(self):
        #final_score = calculate_final_score(self.video_angles, self.camera_angles, self.score_threshold)
        final_score = calculate_final_score(
            self.video_angles,
            self.camera_angles,
            self.score_threshold,
            reaction_time_frames=self.reaction_time_frames  # 传递延迟帧数
        )
        self.current_score = final_score
        self.score_display.update_final_score(self.current_score)
        if self.video_frame_count == 0 or self.camera_frame_count == 0:
            messagebox.showinfo("Game Over", "No sufficient pose data captured. Please ensure video and camera are working.")
        else:
            messagebox.showinfo("Game Over", f"Your final score is: {self.current_score}")
        self.reset_game_state() # 游戏结束后重置状态

    # Reset the game state to initial values
    def reset_game_state(self):
        self.video_angles = defaultdict(list)
        self.camera_angles = defaultdict(list)
        self.video_frame_count = 0
        self.camera_frame_count = 0
        self.current_score = 0
        self.score_display.update_score(0) # Reset the score display
        # 确保停止视频和摄像头线程，并释放资源 Make sure to stop both video and camera threads, and release resources
        self.running_file = False
        self.running_cam = False
        if self.cap_file:
            self.cap_file.release()
            self.cap_file = None
        if self.cap_cam:
            self.cap_cam.release()
            self.cap_cam = None

if __name__ == "__main__":
    root = tk.Tk()
    app = PoseApp(root)
    root.mainloop()