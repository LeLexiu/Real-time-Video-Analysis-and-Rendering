import cv2
import numpy as np
import time
import tkinter as tk
from tkinter import messagebox
from models.pose_model import load_pose_model
from .utils import draw_skeleton, update_tkinter_label
from .player_tracker import FaceBasedPlayerTracker

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

        # Default time limit - will be updated by user selection
        self.time_limit = 20
        self.start_time = None
        self.scoring_active = False

        # Initialize face-based player tracker
        self.face_tracker = FaceBasedPlayerTracker(
            max_distance_threshold=100,
            max_missing_frames=15,
            debug_mode=False  # Set to True to see face detection boxes
        )

        # Time limit selection
        self.time_selection_frame = tk.Frame(self.root)
        self.time_selection_frame.pack(pady=10)

        tk.Label(self.time_selection_frame, text="Select Time Limit:", font=("Arial", 12, "bold")).pack(side=tk.LEFT,
                                                                                                        padx=5)

        self.time_var = tk.StringVar(value="20 sec")
        time_options = ["20 sec", "30 sec", "40 sec", "1 min"]
        self.time_dropdown = tk.OptionMenu(self.time_selection_frame, self.time_var, *time_options,
                                           command=self.update_time_limit)
        self.time_dropdown.config(font=("Arial", 11))
        self.time_dropdown.pack(side=tk.LEFT, padx=5)

        # Timer Label
        self.timer_label = tk.Label(self.root, text=f"Time Left: {self.time_limit}s", font=("Arial", 14), fg="red")
        self.timer_label.pack(pady=10)

        # Control Buttons just below timer label
        self.controls_cam = tk.Frame(self.root)
        self.controls_cam.pack(pady=10)
        tk.Button(self.controls_cam, text="Start Webcam", command=self.start_countdown).pack(side=tk.LEFT, padx=5)
        tk.Button(self.controls_cam, text="Stop Webcam", command=self.stop_cam).pack(side=tk.LEFT, padx=5)
        tk.Button(self.controls_cam, text="Go Back to Title", command=self.go_back).pack(side=tk.LEFT, padx=5)

        # Debug toggle button (optional)
        self.debug_var = tk.BooleanVar()
        tk.Checkbutton(self.controls_cam, text="Debug Mode", variable=self.debug_var,
                       command=self.toggle_debug).pack(side=tk.LEFT, padx=5)

        # Webcam Frame and Label below buttons
        self.cam_frame = tk.Frame(self.root)
        self.cam_frame.pack(pady=20, fill=tk.BOTH, expand=True)
        self.label_cam = tk.Label(self.cam_frame)
        self.label_cam.pack(fill=tk.BOTH, expand=True)

    def update_time_limit(self, selected_time):
        """Update time limit based on user selection"""
        time_mapping = {
            "20 sec": 20,
            "30 sec": 30,
            "40 sec": 40,
            "1 min": 60
        }

        self.time_limit = time_mapping.get(selected_time, 20)
        self.timer_label.config(text=f"Time Left: {self.time_limit}s")

        # Update the scoring normalization for different time limits
        self.update_scoring_parameters()

    def toggle_debug(self):
        """Toggle debug mode for face tracking visualization."""
        self.face_tracker.debug_mode = self.debug_var.get()

    def start_countdown(self):
        # Reset tracker when starting new game
        self.face_tracker.reset()

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
                messagebox.showerror("Error", "Unable to access the webcam.")
                # Re-enable Start button
                for btn in self.controls_cam.winfo_children():
                    if btn.cget("text") == "Start Webcam":
                        btn.config(state=tk.NORMAL)
                return

            # Optimize webcam settings
            self.cap_cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap_cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap_cam.set(cv2.CAP_PROP_FPS, 30)
            self.cap_cam.set(cv2.CAP_PROP_BUFFERSIZE, 1)

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
                # Flip frame horizontally for mirror effect
                frame = cv2.flip(frame, 1)
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
                self.show_freestyle_ranking()
                self.stop_cam()
                return
            else:
                self.timer_label.config(text=f"Time Left: {int(self.time_limit - elapsed)}s")

            # Schedule next frame update (~30fps)
            self.root.after(33, self.update_frame)

    def process_pose(self, frame):
        """Process pose detection with face-based player tracking."""
        try:
            results = model(frame, conf=0.3, verbose=False)
            height, width = frame.shape[:2]
            overlay = frame.copy()

            # Collect all detected keypoints
            detected_keypoints = []
            for result in results:
                if result.keypoints is not None:
                    keypoints_xyn = result.keypoints.xyn.cpu().numpy()
                    for person_kpts in keypoints_xyn:
                        keypoints = [(int(x * width), int(y * height)) for x, y in person_kpts]
                        detected_keypoints.append(keypoints)

            # Get stable player mapping using face tracking
            stable_players = self.face_tracker.get_stable_player_mapping(frame, detected_keypoints)

            # Process each stable player
            for person_id, keypoints in stable_players.items():
                if keypoints:
                    # Draw skeleton
                    overlay = draw_skeleton(overlay, keypoints)

                    # Get nose position for score display
                    nose_x, nose_y = keypoints[0] if keypoints and len(keypoints) > 0 else (None, None)

                    if nose_x and nose_y and nose_x > 0 and nose_y > 0:
                        # Display score
                        score = int(self.person_scores.get(person_id, 0))
                        cv2.putText(overlay, f"Player {person_id + 1}: {score}",
                                    (nose_x, nose_y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

                    # Calculate and add score if game is active
                    if self.scoring_active and self.start_time:
                        score = self.compute_movement_score(keypoints, self.person_prev_keypoints.get(person_id))
                        self.person_scores[person_id] = self.person_scores.get(person_id, 0) + score
                        self.person_prev_keypoints[person_id] = keypoints

            # Add debug information if enabled
            if self.debug_var.get():
                overlay = self.face_tracker.draw_debug_info(overlay, show_faces=True, show_player_ids=True)

            return overlay

        except Exception as e:
            print(f"Error in process_pose: {e}")
            return frame

    def update_scoring_parameters(self):
        """Update scoring parameters based on selected time limit"""
        # Calculate total frames for the selected time limit
        fps = 30  # Assuming 30 FPS
        self.total_frames = self.time_limit * fps

        # Update the scoring system to maintain 100 points maximum regardless of time limit
        self.points_per_frame_target = 100.0 / self.total_frames

    def compute_movement_score(self, keypoints, prev_keypoints):
        """Compute normalized movement score - bounded to 100 points maximum"""
        movement_score = 0
        expressiveness_score = 0

        # Calculate movement score (how much the person moved)
        if prev_keypoints and len(keypoints) == len(prev_keypoints):
            for (x1, y1), (x2, y2) in zip(keypoints, prev_keypoints):
                if x1 > 0 and y1 > 0 and x2 > 0 and y2 > 0:
                    movement_score += np.hypot(x2 - x1, y2 - y1)

        # Calculate expressiveness score (how spread out the pose is)
        if len(keypoints) > 12:
            # Use hip center as reference
            left_hip = keypoints[11] if len(keypoints) > 11 else (0, 0)
            right_hip = keypoints[12] if len(keypoints) > 12 else (0, 0)

            if left_hip[0] > 0 and right_hip[0] > 0:
                cx = (left_hip[0] + right_hip[0]) / 2
                cy = (left_hip[1] + right_hip[1]) / 2

                valid_points = [(x, y) for x, y in keypoints if x > 0 and y > 0]
                if valid_points:
                    expressiveness_score = np.mean([np.hypot(x - cx, y - cy) for x, y in valid_points])

        # Combine scores with weighting
        combined_score = movement_score + 0.3 * expressiveness_score

        # Normalize to 100 points total over selected time duration
        # Get the target points per frame based on time limit
        target_points_per_frame = getattr(self, 'points_per_frame_target', 100.0 / (self.time_limit * 30))

        # Calculate points per frame (targeting 100 points maximum total)
        if combined_score > 0:
            # Scale so that consistent good dancing gives 100 points total
            target_combined_score_for_100_points = 200  # Calibrated value

            # Points per frame calculation
            raw_points_per_frame = (combined_score / target_combined_score_for_100_points)

            # Apply diminishing returns to prevent exceeding 100 points
            # Use a logarithmic scale to cap the maximum
            points_per_frame = target_points_per_frame * np.log(1 + raw_points_per_frame * 5)

            # Apply bonus for significant movement
            if movement_score > 30:  # Bonus for good movement
                points_per_frame *= 1.2

            # Hard cap to ensure we never exceed 100 points total
            max_points_per_frame = target_points_per_frame * 1.2  # 20% buffer
            points_per_frame = min(max_points_per_frame, points_per_frame)
        else:
            points_per_frame = 0

        return points_per_frame

    def get_performance_label(self, score):
        """Get performance label with emoji based on score"""
        if score >= 100:
            return "Perfect! 🎯"
        elif score >= 90:
            return "Excellent! ⭐"
        elif score >= 80:
            return "Great! 👍"
        elif score >= 70:
            return "Good! 👌"
        elif score >= 60:
            return "Nice! 😊"
        elif score >= 50:
            return "Okay 😐"
        elif score >= 40:
            return "Fair 😕"
        elif score >= 30:
            return "Poor 😞"
        else:
            return "Try Again 🔄"

    def show_freestyle_ranking(self):
        """Show freestyle ranking with performance labels and emojis"""
        if not self.person_scores:
            messagebox.showinfo("Results", "No scores to rank.")
            return

        # Sort by scores for proper ranking
        sorted_scores = sorted(self.person_scores.items(), key=lambda x: x[1], reverse=True)

        ranking_win = tk.Toplevel(self.root)
        ranking_win.title("Freestyle Results")
        ranking_win.geometry("500x400")

        tk.Label(ranking_win, text="🕺 Freestyle Rankings:", font=("Arial", 20, "bold")).pack(pady=10)

        for i, (pid, score) in enumerate(sorted_scores, start=1):
            performance_label = self.get_performance_label(score)

            # Add rank emoji for top 3
            rank_emoji = ""
            if i == 1:
                rank_emoji = "🥇 "
            elif i == 2:
                rank_emoji = "🥈 "
            elif i == 3:
                rank_emoji = "🥉 "

            tk.Label(
                ranking_win,
                text=f"{rank_emoji}{i}. Player {pid + 1}: {int(score)} - {performance_label}",
                font=("Arial", 16)
            ).pack(pady=5)

        # Show tracking statistics
        stats = self.face_tracker.get_statistics()
        stats_text = f"Tracking Stats: {stats['active_players']} players, {stats['avg_faces_per_frame']:.1f} faces/frame"
        tk.Label(ranking_win, text=stats_text, font=("Arial", 10), fg="gray").pack(pady=5)

        tk.Button(ranking_win, text="Close", font=("Arial", 14), command=ranking_win.destroy).pack(pady=20)