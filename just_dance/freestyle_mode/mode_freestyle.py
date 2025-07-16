import time
import tkinter as tk
from tkinter import messagebox

import cv2

from .rank import show_score_ranking
from models.pose_model import load_pose_model
from .utils import *

model = load_pose_model()

class FreestyleMode:
    def __init__(self, root, go_back_callback):
        self.root = root
        self.go_back_callback = go_back_callback
        self.root.title("Stickman Dance - Freestyle Mode")
        self.root.geometry("1200x600")

        self.cap_cam = None
        self.running_cam = False

        self.person_scores = {}
        self.person_prev_keypoints = {}

        self.time_limit = 20
        self.start_time = None
        self.scoring_active = False

        # Timer Label
        self.timer_label = tk.Label(self.root, text=f"Time Left: {self.time_limit}s", font=("Arial", 14), fg="red")
        self.timer_label.pack(pady=10)

        # Control Buttons just below timer label
        self.controls_cam = tk.Frame(self.root)
        self.controls_cam.pack(pady=10)
        tk.Button(self.controls_cam, text="Start Webcam", command=self.start_countdown).pack(side=tk.LEFT, padx=5)
        tk.Button(self.controls_cam, text="Stop Webcam", command=self.stop_cam).pack(side=tk.LEFT, padx=5)
        tk.Button(self.controls_cam, text="Go Back to Title", command=self.go_back).pack(side=tk.LEFT, padx=5)

        # Webcam Frame and Label below buttons
        self.cam_frame = tk.Frame(self.root)
        self.cam_frame.pack(pady=20, fill=tk.BOTH, expand=True)
        self.label_cam = tk.Label(self.cam_frame)
        self.label_cam.pack(fill=tk.BOTH, expand=True)

    def start_countdown(self):
        # Disable Start button during countdown
        for btn in self.controls_cam.winfo_children():
            if btn.cget("text") == "Start Webcam":
                btn.config(state=tk.DISABLED)

        def countdown(i):
            if i > 0:
                self.timer_label.config(text=f"Starting in {i}...")
                self.root.after(1000, countdown, i - 1)
            else:
                self.timer_label.config(text="Go!")
                self.start_cam()
        countdown(3)

    def start_cam(self):
        if not self.running_cam:
            self.cap_cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            if not self.cap_cam.isOpened():
                tk.messagebox.showerror("Error", "Unable to access the webcam.")
                # Re-enable Start button
                for btn in self.controls_cam.winfo_children():
                    if btn.cget("text") == "Start Webcam":
                        btn.config(state=tk.NORMAL)
                return
            self.running_cam = True
            self.scoring_active = True
            self.start_time = time.time()
            self.update_frame()

    def stop_cam(self):
        self.running_cam = False
        self.scoring_active = False
        if self.cap_cam:
            self.cap_cam.release()
            self.cap_cam = None
        self.timer_label.config(text=f"Time Left: {self.time_limit}s")

        # Re-enable Start button
        for btn in self.controls_cam.winfo_children():
            if btn.cget("text") == "Start Webcam":
                btn.config(state=tk.NORMAL)

    def go_back(self):
        self.stop_cam()
        time.sleep(0.1)
        for widget in self.root.winfo_children():
            widget.destroy()
        self.go_back_callback()

    def update_frame(self):
        if self.running_cam and self.cap_cam:
            ret, frame = self.cap_cam.read()
            if ret:
                processed_frame = self.process_pose(frame)
                update_tkinter_label(self.label_cam, processed_frame)
            else:
                # If failed to read frame, stop camera
                self.stop_cam()
                return

            # Update timer text
            elapsed = time.time() - self.start_time
            if elapsed >= self.time_limit:
                self.scoring_active = False
                self.timer_label.config(text="Time's up!")
                show_score_ranking(self.root, self.person_scores)
                self.stop_cam()
                return
            else:
                self.timer_label.config(text=f"Time Left: {int(self.time_limit - elapsed)}s")

            # Schedule next frame update (~30fps)
            self.root.after(33, self.update_frame)

    def process_pose(self, frame):
        results = model(frame, conf=0.3)
        height, width = frame.shape[:2]
        overlay = frame.copy()

        for result in results:
            if result.keypoints is not None:
                keypoints_xyn = result.keypoints.xyn.cpu().numpy()
                for person_id, person_kpts in enumerate(keypoints_xyn):
                    keypoints = [(int(x * width), int(y * height)) for x, y in person_kpts]
                    overlay = draw_skeleton(overlay, keypoints)
                    nose_x, nose_y = keypoints[0] if keypoints else (None, None)
                    if nose_x and nose_y:
                        cv2.putText(overlay, f"({person_id+1}): {int(self.person_scores.get(person_id, 0))}",
                                    (nose_x, nose_y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    if self.scoring_active and self.start_time:
                        score = self.compute_movement_score(keypoints, self.person_prev_keypoints.get(person_id))
                        self.person_scores[person_id] = self.person_scores.get(person_id, 0) + score
                        self.person_prev_keypoints[person_id] = keypoints
        return overlay

    def compute_movement_score(self, keypoints, prev_keypoints):
        movement_score = 0
        expressiveness_score = 0
        if prev_keypoints and len(keypoints) == len(prev_keypoints):
            for (x1, y1), (x2, y2) in zip(keypoints, prev_keypoints):
                movement_score += np.hypot(x2 - x1, y2 - y1)
        if len(keypoints) > 12:
            cx = (keypoints[11][0] + keypoints[12][0]) / 2
            cy = (keypoints[11][1] + keypoints[12][1]) / 2
            expressiveness_score = np.mean([np.hypot(x - cx, y - cy) for x, y in keypoints])
        return movement_score + 0.5 * expressiveness_score


