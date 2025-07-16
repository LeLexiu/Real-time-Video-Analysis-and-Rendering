import cv2
import threading
import time
import numpy as np
import tkinter as tk
from .rank import show_score_ranking
from models.pose_model import load_pose_model
from .utils import *

model = load_pose_model()
skeleton = [(0,5),(0,6),(5,6),(5,7),(7,9),(6,8),(8,10),
            (5,11),(6,12),(11,12),(11,13),(13,15),(12,14),(14,16)]
angle_triplets = [
    (5, 7, 9),   # Right arm
    (6, 8, 10),  # Left arm
    (11, 13, 15),# Right leg
    (12, 14, 16),# Left leg
    (5, 11, 13), # Right body side
    (6, 12, 14)  # Left body side
]

class VideoMode:
    def __init__(self, root, go_back_callback):
        self.root = root
        self.go_back_callback = go_back_callback
        self.root.title("Dance Match Mode")
        self.root.geometry("1280x720")

        self.ref_video_path = None
        self.cap_ref = None
        self.cap_webcam = None
        self.running = False

        self.frame_count = 0
        self.reference_keypoints = None

        self.person_scores = {}
        self.time_limit = 30
        self.start_time = None
        self.scoring_active = False

        self.label_ref = tk.Label(root)
        self.label_ref.pack(side=tk.LEFT, padx=10)

        self.label_webcam = tk.Label(root)
        self.label_webcam.pack(side=tk.RIGHT, padx=10)

        self.timer_label = tk.Label(root, text="Time Left: 30s", font=("Arial", 16), fg="red")
        self.timer_label.pack(pady=10)

        self.control_frame = tk.Frame(root)
        self.control_frame.pack(pady=10)

        tk.Button(self.control_frame, text="Select Reference Video", command=self.load_video).pack(side=tk.LEFT, padx=5)
        tk.Button(self.control_frame, text="Start Game", command=self.start_game).pack(side=tk.LEFT, padx=5)
        tk.Button(self.control_frame, text="Go Back to Title", command=self.go_back).pack(side=tk.LEFT, padx=5)

    def go_back(self):
        self.running = False
        if self.cap_ref: self.cap_ref.release()
        if self.cap_webcam: self.cap_webcam.release()
        time.sleep(0.1)
        for widget in self.root.winfo_children():
            widget.destroy()
        self.go_back_callback()

    def load_video(self):
        from tkinter import filedialog
        path = filedialog.askopenfilename(filetypes=[("MP4 files", "*.mp4")])
        if path:
            self.ref_video_path = path

    def start_game(self):
        if not self.ref_video_path:
            from tkinter import messagebox
            messagebox.showwarning("Missing", "Please select a reference video first.")
            return
        self.person_scores = {}
        self.running = True
        self.start_time = time.time()
        self.scoring_active = True
        self.cap_ref = cv2.VideoCapture(self.ref_video_path)
        self.cap_webcam = cv2.VideoCapture(0, cv2.CAP_DSHOW)

        self.reference_keypoints = None
        threading.Thread(target=self.run, daemon=True).start()

    def run(self):
        while self.running:
            ref_ok, ref_frame = self.cap_ref.read()
            cam_ok, cam_frame = self.cap_webcam.read()

            if not ref_ok or not cam_ok:
                break

            ref_frame = self.draw_reference_pose(ref_frame)
            cam_frame = self.draw_pose_webcam(cam_frame)

            update_tkinter_label(self.label_ref, ref_frame, size=(480, 360))
            update_tkinter_label(self.label_webcam, cam_frame, size=(480, 360))

            elapsed = time.time() - self.start_time
            if elapsed >= self.time_limit:
                self.running = False
                self.scoring_active = False
                self.timer_label.config(text="Time's up!")
                show_score_ranking(self.root, self.person_scores)
                break
            else:
                self.timer_label.config(text=f"Time Left: {int(self.time_limit - elapsed)}s")

        if self.cap_ref:
            self.cap_ref.release()
        if self.cap_webcam:
            self.cap_webcam.release()
        time.sleep(0.1)

    def draw_reference_pose(self, frame):
        self.frame_count += 1
        height, width = frame.shape[:2]
        overlay = frame.copy()

        if self.frame_count % 3 == 0:
            results = model(frame, conf=0.3)
            best_person = None
            min_distance = float('inf')

            for result in results:
                if result.keypoints is not None:
                    keypoints_xyn = result.keypoints.xyn.cpu().numpy()
                    for person_kpts in keypoints_xyn:
                        keypoints = [(int(x * width), int(y * height)) for x, y in person_kpts]
                        nose_x, nose_y = keypoints[0]
                        distance = abs(nose_x - width // 2) + nose_y
                        if distance < min_distance:
                            min_distance = distance
                            best_person = keypoints

            if best_person:
                self.reference_keypoints = best_person

        if self.reference_keypoints:
            overlay = draw_skeleton(overlay, self.reference_keypoints)

        return overlay

    def draw_pose_webcam(self, frame):
        results = model(frame, conf=0.3)
        height, width = frame.shape[:2]
        overlay = frame.copy()

        for result in results:
            if result.keypoints is not None:
                keypoints_xyn = result.keypoints.xyn.cpu().numpy()
                for person_id, person_kpts in enumerate(keypoints_xyn):
                    keypoints = [(int(x * width), int(y * height)) for x, y in person_kpts]
                    overlay = draw_skeleton(overlay, keypoints)
                    if self.scoring_active and self.reference_keypoints:
                        score = self.compute_angle_similarity_score(keypoints, self.reference_keypoints, angle_triplets)
                        self.person_scores[person_id] = self.person_scores.get(person_id, 0) + score
                        nose_x, nose_y = keypoints[0]
                        cv2.putText(overlay, f"Player{person_id + 1}: {int(self.person_scores[person_id])}",
                                    (nose_x, nose_y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        return overlay

    def compute_angle(self, p1, p2, p3):
        v1 = np.array(p1) - np.array(p2)
        v2 = np.array(p3) - np.array(p2)
        cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
        return np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0)))

    def compute_angle_similarity_score(self, player_keypoints, ref_keypoints, triplets):
        diffs = []
        for a, b, c in triplets:
            if all(player_keypoints[i] and ref_keypoints[i] for i in (a, b, c)):
                angle_player = self.compute_angle(player_keypoints[a], player_keypoints[b], player_keypoints[c])
                angle_ref = self.compute_angle(ref_keypoints[a], ref_keypoints[b], ref_keypoints[c])
                diffs.append(abs(angle_player - angle_ref))
        if not diffs:
            return 0
        return max(0, 100 - np.mean(diffs))
