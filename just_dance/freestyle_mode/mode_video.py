import cv2
import threading
import time
import numpy as np
import tkinter as tk
from tkinter import messagebox, filedialog
from models.pose_model import load_pose_model
from .utils import draw_skeleton, update_tkinter_label
from .player_tracker import FaceBasedPlayerTracker

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
        self.game_started = False

        self.reference_keypoints = None
        self.person_scores = {}
        # Default time limit - will be updated by user selection
        self.time_limit = 30
        self.start_time = None
        self.scoring_active = False

        # Frame storage for display
        self.current_ref_frame = None
        self.current_cam_frame = None
        self.current_ref_keypoints = None
        self.current_cam_keypoints = []

        # Thread locks for frame access
        self.ref_frame_lock = threading.Lock()
        self.cam_frame_lock = threading.Lock()
        self.keypoints_lock = threading.Lock()

        # Video timing
        self.ref_fps = 30
        self.ref_frame_time = 1.0 / 30

        # Initialize face-based player tracker
        self.face_tracker = FaceBasedPlayerTracker(
            max_distance_threshold=100,
            max_missing_frames=15,
            debug_mode=False  # Set to True to see face detection boxes
        )

        self.setup_ui()

    def setup_ui(self):
        # Main container
        main_frame = tk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Time limit selection at the top
        time_selection_frame = tk.Frame(main_frame)
        time_selection_frame.pack(pady=10)

        tk.Label(time_selection_frame, text="Select Time Limit:", font=("Arial", 12, "bold")).pack(side=tk.LEFT, padx=5)

        self.time_var = tk.StringVar(value="30 sec")
        time_options = ["20 sec", "30 sec", "40 sec", "1 min"]
        self.time_dropdown = tk.OptionMenu(time_selection_frame, self.time_var, *time_options,
                                           command=self.update_time_limit)
        self.time_dropdown.config(font=("Arial", 11))
        self.time_dropdown.pack(side=tk.LEFT, padx=5)

        # Video frames container
        video_frame = tk.Frame(main_frame)
        video_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Reference video (left)
        ref_container = tk.Frame(video_frame)
        ref_container.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        tk.Label(ref_container, text="Reference Video", font=("Arial", 14)).pack()
        self.label_ref = tk.Label(ref_container, bg="black", width=60, height=30)
        self.label_ref.pack(fill=tk.BOTH, expand=True)

        # Webcam (right)
        cam_container = tk.Frame(video_frame)
        cam_container.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5)
        tk.Label(cam_container, text="Your Performance", font=("Arial", 14)).pack()
        self.label_webcam = tk.Label(cam_container, bg="black", width=60, height=30)
        self.label_webcam.pack(fill=tk.BOTH, expand=True)

        # Timer
        self.timer_label = tk.Label(main_frame, text="Select a video to start",
                                    font=("Arial", 16), fg="red")
        self.timer_label.pack(pady=10)

        # Controls
        self.control_frame = tk.Frame(main_frame)
        self.control_frame.pack(pady=10)

        self.select_btn = tk.Button(self.control_frame, text="Select Reference Video",
                                    command=self.load_video, font=("Arial", 12))
        self.select_btn.pack(side=tk.LEFT, padx=5)

        self.start_btn = tk.Button(self.control_frame, text="Start Game",
                                   command=self.start_game, font=("Arial", 12),
                                   state=tk.DISABLED)
        self.start_btn.pack(side=tk.LEFT, padx=5)

        self.back_btn = tk.Button(self.control_frame, text="Go Back to Title",
                                  command=self.go_back, font=("Arial", 12))
        self.back_btn.pack(side=tk.LEFT, padx=5)

        # Debug toggle button (optional)
        self.debug_var = tk.BooleanVar()
        tk.Checkbutton(self.control_frame, text="Debug Mode", variable=self.debug_var,
                       command=self.toggle_debug).pack(side=tk.LEFT, padx=5)

    def update_time_limit(self, selected_time):
        """Update time limit based on user selection"""
        time_mapping = {
            "20 sec": 20,
            "30 sec": 30,
            "40 sec": 40,
            "1 min": 60
        }

        self.time_limit = time_mapping.get(selected_time, 30)
        if not self.game_started:
            self.timer_label.config(text=f"Selected: {selected_time} duration")

        # Update the scoring normalization for different time limits
        self.update_scoring_parameters()

    def update_scoring_parameters(self):
        """Update scoring parameters based on selected time limit"""
        # Calculate expected number of detections based on time limit
        # Detection happens every 0.5 seconds
        self.expected_detections = self.time_limit // 0.5

        # Update scoring to maintain 100 points maximum regardless of time limit
        self.max_score_per_detection = 100.0 / self.expected_detections

    def toggle_debug(self):
        """Toggle debug mode for face tracking visualization."""
        self.face_tracker.debug_mode = self.debug_var.get()

    def load_video(self):
        path = filedialog.askopenfilename(
            title="Select Reference Video",
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv"), ("All files", "*.*")]
        )
        if path:
            self.ref_video_path = path
            self.start_btn.config(state=tk.NORMAL)
            self.timer_label.config(text=f"Video selected: {path.split('/')[-1]}")

    def start_game(self):
        if not self.ref_video_path:
            messagebox.showwarning("Warning", "Please select a reference video first.")
            return

        # Initialize captures
        self.cap_ref = cv2.VideoCapture(self.ref_video_path)
        self.cap_webcam = cv2.VideoCapture(0, cv2.CAP_DSHOW)

        if not self.cap_ref.isOpened():
            messagebox.showerror("Error", "Could not open reference video.")
            return

        if not self.cap_webcam.isOpened():
            messagebox.showerror("Error", "Could not open webcam.")
            return

        # Get actual FPS from reference video
        self.ref_fps = self.cap_ref.get(cv2.CAP_PROP_FPS)
        if self.ref_fps <= 0 or self.ref_fps > 60:
            self.ref_fps = 30
        self.ref_frame_time = 1.0 / self.ref_fps

        # Optimize webcam settings
        self.cap_webcam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap_webcam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap_webcam.set(cv2.CAP_PROP_FPS, 30)
        self.cap_webcam.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        # Reset game state
        self.running = True
        self.game_started = True
        self.start_time = time.time()
        self.scoring_active = True
        self.person_scores = {}

        # Reset frame storage
        self.current_ref_frame = None
        self.current_cam_frame = None
        self.current_ref_keypoints = None
        self.current_cam_keypoints = []

        # Reset face tracker
        self.face_tracker.reset()

        # Disable start button during game
        self.start_btn.config(state=tk.DISABLED)

        # Start threads
        threading.Thread(target=self.ref_video_thread, daemon=True).start()
        threading.Thread(target=self.webcam_thread, daemon=True).start()
        threading.Thread(target=self.cam_pose_detection_thread, daemon=True).start()
        threading.Thread(target=self.timer_thread, daemon=True).start()

        # Start UI update loop
        self.update_ui()

    def ref_video_thread(self):
        """Handle reference video playback with synchronized pose detection"""
        last_frame_time = time.time()

        while self.running:
            current_time = time.time()

            # Check if it's time for the next frame
            if current_time - last_frame_time >= self.ref_frame_time:
                ret, frame = self.cap_ref.read()
                if not ret:
                    # Loop video
                    self.cap_ref.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue

                # Resize for consistent performance
                frame = cv2.resize(frame, (640, 480))

                # Immediately detect pose for this frame (synchronous)
                keypoints = self.detect_keypoints_fast(frame)

                # Store frame and keypoints together
                with self.ref_frame_lock:
                    self.current_ref_frame = frame.copy()

                with self.keypoints_lock:
                    self.current_ref_keypoints = keypoints
                    self.reference_keypoints = keypoints  # For scoring

                last_frame_time = current_time

            # Sleep for a very short time to prevent busy waiting
            time.sleep(0.001)

    def webcam_thread(self):
        """Handle webcam capture"""
        while self.running:
            ret, frame = self.cap_webcam.read()
            if not ret:
                time.sleep(0.01)
                continue

            # Flip frame horizontally for mirror effect
            frame = cv2.flip(frame, 1)

            # Store frame safely
            with self.cam_frame_lock:
                self.current_cam_frame = frame.copy()

            # Control frame rate
            time.sleep(1 / 30)

    def cam_pose_detection_thread(self):
        """Handle pose detection for webcam with face tracking"""
        last_detection_time = time.time()
        detection_interval = 0.5  # Every 0.5 seconds as requested

        while self.running:
            current_time = time.time()

            if current_time - last_detection_time >= detection_interval:
                # Get current webcam frame
                with self.cam_frame_lock:
                    if self.current_cam_frame is not None:
                        cam_frame = self.current_cam_frame.copy()
                    else:
                        cam_frame = None

                if cam_frame is not None:
                    # Detect keypoints using face tracking
                    cam_keypoints = self.detect_all_keypoints_with_tracking(cam_frame)

                    # Store keypoints safely
                    with self.keypoints_lock:
                        self.current_cam_keypoints = cam_keypoints

                    # Score if game is active
                    if self.scoring_active and self.reference_keypoints:
                        self.score_webcam_keypoints(cam_keypoints)

                last_detection_time = current_time

            time.sleep(0.05)

    def timer_thread(self):
        """Handle game timer"""
        while self.running and self.game_started:
            elapsed = time.time() - self.start_time

            if elapsed >= self.time_limit:
                self.end_game()
                break

            time.sleep(0.5)

    def update_ui(self):
        """Main UI update loop - fast and responsive"""
        if not self.running:
            return

        # Update reference video display
        with self.ref_frame_lock:
            if self.current_ref_frame is not None:
                display_frame = self.current_ref_frame.copy()
            else:
                display_frame = None

        if display_frame is not None:
            # Add skeleton overlay
            with self.keypoints_lock:
                if self.current_ref_keypoints is not None:
                    display_frame = draw_skeleton(display_frame, self.current_ref_keypoints,
                                                  color_lines=(0, 255, 0), color_points=(0, 255, 0))

            update_tkinter_label(self.label_ref, display_frame, (480, 360))

        # Update webcam display
        with self.cam_frame_lock:
            if self.current_cam_frame is not None:
                display_frame = self.current_cam_frame.copy()
            else:
                display_frame = None

        if display_frame is not None:
            # Add skeleton overlay and scores
            with self.keypoints_lock:
                if self.current_cam_keypoints:
                    for person_id, keypoints in enumerate(self.current_cam_keypoints):
                        if keypoints:
                            display_frame = draw_skeleton(display_frame, keypoints)

                            # Display score only (no performance label)
                            if len(keypoints) > 0 and keypoints[0][0] > 0 and keypoints[0][1] > 0:
                                nose_x, nose_y = keypoints[0]
                                score = self.person_scores.get(person_id, 0)
                                cv2.putText(display_frame, f"Player {person_id + 1}: {int(score)}",
                                            (nose_x, nose_y - 30), cv2.FONT_HERSHEY_SIMPLEX,
                                            0.7, (0, 255, 255), 2)

            # Add debug information if enabled
            if self.debug_var.get():
                display_frame = self.face_tracker.draw_debug_info(display_frame, show_faces=True, show_player_ids=True)

            update_tkinter_label(self.label_webcam, display_frame, (480, 360))

        # Update timer
        if self.game_started:
            elapsed = time.time() - self.start_time
            remaining = max(0, self.time_limit - elapsed)
            self.timer_label.config(text=f"Time Left: {int(remaining)}s")

        # Schedule next update
        if self.running:
            self.root.after(33, self.update_ui)  # ~30 FPS UI updates

    def detect_keypoints_fast(self, frame):
        """Fast keypoint detection optimized for real-time performance"""
        try:
            # Use smaller resolution for faster processing
            small_frame = cv2.resize(frame, (256, 192))

            # Run detection with optimized settings
            results = model(small_frame, conf=0.15, verbose=False, imgsz=256)

            # Scale back to original frame size
            height, width = frame.shape[:2]
            scale_x = width / 256
            scale_y = height / 192

            best_person = None
            max_confidence = 0

            for result in results:
                if result.keypoints is not None:
                    keypoints_xyn = result.keypoints.xyn.cpu().numpy()
                    for person_kpts in keypoints_xyn:
                        # Scale keypoints back to original size
                        keypoints = [(int(x * 256 * scale_x), int(y * 192 * scale_y)) for x, y in person_kpts]

                        # Calculate confidence based on keypoint visibility
                        confidence = sum(1 for x, y in keypoints if x > 0 and y > 0)

                        if confidence > max_confidence:
                            max_confidence = confidence
                            best_person = keypoints

            return best_person
        except Exception as e:
            print(f"Error in fast pose detection: {e}")
            return None

    def detect_all_keypoints_with_tracking(self, frame):
        """Detect keypoints for all people using face tracking"""
        try:
            # Run YOLO pose detection
            results = model(frame, conf=0.3, verbose=False)
            height, width = frame.shape[:2]

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

            # Convert back to list format for compatibility
            stable_keypoints_list = []
            for i in range(len(stable_players)):
                if i in stable_players:
                    stable_keypoints_list.append(stable_players[i])
                else:
                    stable_keypoints_list.append([])  # Empty keypoints for missing players

            return stable_keypoints_list

        except Exception as e:
            print(f"Error in webcam pose detection: {e}")
            return []

    def score_webcam_keypoints(self, cam_keypoints):
        """Score all detected people against reference"""
        if not self.reference_keypoints:
            return

        for person_id, keypoints in enumerate(cam_keypoints):
            if keypoints and len(keypoints) >= 17:  # Ensure we have all keypoints
                score = self.compute_angle_similarity_score(
                    keypoints, self.reference_keypoints, angle_triplets
                )
                self.person_scores[person_id] = self.person_scores.get(person_id, 0) + score

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

    def compute_angle(self, p1, p2, p3):
        """Calculate angle between three points"""
        if not all(p1) or not all(p2) or not all(p3):
            return 0

        v1 = np.array(p1) - np.array(p2)
        v2 = np.array(p3) - np.array(p2)

        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)

        if norm_v1 == 0 or norm_v2 == 0:
            return 0

        cos_theta = np.dot(v1, v2) / (norm_v1 * norm_v2)
        cos_theta = np.clip(cos_theta, -1.0, 1.0)

        return np.degrees(np.arccos(cos_theta))

    def compute_angle_similarity_score(self, player_keypoints, ref_keypoints, triplets):
        """Calculate similarity score based on joint angles - normalized to 100 points maximum"""
        if not player_keypoints or not ref_keypoints:
            return 0

        diffs = []
        for a, b, c in triplets:
            if (a < len(player_keypoints) and b < len(player_keypoints) and c < len(player_keypoints) and
                    a < len(ref_keypoints) and b < len(ref_keypoints) and c < len(ref_keypoints)):

                angle_player = self.compute_angle(player_keypoints[a], player_keypoints[b], player_keypoints[c])
                angle_ref = self.compute_angle(ref_keypoints[a], ref_keypoints[b], ref_keypoints[c])

                if angle_player > 0 and angle_ref > 0:
                    diffs.append(abs(angle_player - angle_ref))

        if not diffs:
            return 0

        # Calculate average difference
        avg_diff = np.mean(diffs)

        # Get the max score per detection based on time limit
        max_score_per_detection = getattr(self, 'max_score_per_detection', 1.67)  # Default for 30 sec

        # Convert angle difference to score (0-180 degrees mapped to max_score_per_detection-0)
        score = max(0, max_score_per_detection * (1 - avg_diff / 180))

        return score

    def end_game(self):
        """End the game and show results"""
        self.running = False
        self.game_started = False
        self.scoring_active = False

        # Re-enable start button
        self.start_btn.config(state=tk.NORMAL)
        self.timer_label.config(text="Game Over!")

        # Show results with labels
        self.root.after(100, lambda: self.show_score_ranking_with_labels())

    def show_score_ranking_with_labels(self):
        """Show score ranking with performance labels and emojis"""
        if not self.person_scores:
            messagebox.showinfo("Results", "No scores to rank.")
            return

        # Sort by original numeric scores for proper ranking
        sorted_scores = sorted(self.person_scores.items(), key=lambda x: x[1], reverse=True)

        ranking_win = tk.Toplevel(self.root)
        ranking_win.title("Final Results")
        ranking_win.geometry("500x450")

        tk.Label(ranking_win, text="🏆 Final Rankings:", font=("Arial", 20, "bold")).pack(pady=10)

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

    def go_back(self):
        """Return to previous screen"""
        self.running = False
        self.game_started = False

        # Release resources
        if self.cap_ref:
            self.cap_ref.release()
        if self.cap_webcam:
            self.cap_webcam.release()

        # Clean up UI
        time.sleep(0.1)
        for widget in self.root.winfo_children():
            widget.destroy()

        self.go_back_callback()
