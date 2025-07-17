import cv2
import numpy as np
import mediapipe as mp
from scipy.spatial.distance import cdist


class FaceBasedPlayerTracker:
    """
    A player tracking system that uses MediaPipe face detection to maintain
    consistent player identities across frames, preventing player index swapping.
    """

    def __init__(self, max_distance_threshold=100, max_missing_frames=15, debug_mode=False):
        """
        Initialize the face-based player tracker.

        Args:
            max_distance_threshold (int): Maximum distance (pixels) to match faces between frames
            max_missing_frames (int): Maximum frames a player can be missing before removal
            debug_mode (bool): Whether to enable debug visualization
        """
        # Initialize MediaPipe face detection
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_drawing = mp.solutions.drawing_utils

        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=0,  # 0 for short-range (2m), 1 for full-range (5m)
            min_detection_confidence=0.5
        )

        # Player tracking data
        self.players = {}  # {player_id: {'face_center': (x, y), 'keypoints': [...], 'missing_count': 0}}
        self.next_player_id = 0

        # Configuration
        self.max_distance_threshold = max_distance_threshold
        self.max_missing_frames = max_missing_frames
        self.debug_mode = debug_mode

        # Statistics
        self.stats = {
            'total_frames': 0,
            'faces_detected': 0,
            'players_tracked': 0
        }

    def detect_faces(self, frame):
        """
        Detect faces in the frame and return their centers.

        Args:
            frame: BGR image frame

        Returns:
            list: List of face center coordinates [(x, y), ...]
        """
        try:
            # Convert BGR to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_detection.process(rgb_frame)

            face_centers = []
            face_boxes = []

            if results.detections:
                h, w = frame.shape[:2]

                for detection in results.detections:
                    # Get bounding box
                    bbox = detection.location_data.relative_bounding_box

                    # Convert to pixel coordinates
                    x = int(bbox.xmin * w)
                    y = int(bbox.ymin * h)
                    width = int(bbox.width * w)
                    height = int(bbox.height * h)

                    # Calculate face center
                    face_center = (x + width // 2, y + height // 2)
                    face_centers.append(face_center)

                    # Store box for debugging
                    face_boxes.append((x, y, width, height))

                self.stats['faces_detected'] += len(face_centers)

            return face_centers, face_boxes

        except Exception as e:
            print(f"Error in face detection: {e}")
            return [], []

    def get_pose_center(self, keypoints):
        """
        Calculate center point of pose using torso keypoints.

        Args:
            keypoints: List of (x, y) keypoint coordinates

        Returns:
            tuple: (x, y) center coordinates or None if invalid
        """
        if not keypoints or len(keypoints) < 13:
            return None

        # Use upper body keypoints for pose center
        valid_points = []

        # Shoulders (keypoints 5, 6)
        if len(keypoints) > 6:
            if keypoints[5][0] > 0 and keypoints[5][1] > 0:
                valid_points.append(keypoints[5])
            if keypoints[6][0] > 0 and keypoints[6][1] > 0:
                valid_points.append(keypoints[6])

        # Hips (keypoints 11, 12)
        if len(keypoints) > 12:
            if keypoints[11][0] > 0 and keypoints[11][1] > 0:
                valid_points.append(keypoints[11])
            if keypoints[12][0] > 0 and keypoints[12][1] > 0:
                valid_points.append(keypoints[12])

        # Neck (keypoint 1) - if available
        if len(keypoints) > 1:
            if keypoints[1][0] > 0 and keypoints[1][1] > 0:
                valid_points.append(keypoints[1])

        if len(valid_points) < 2:
            return None

        # Calculate weighted average (give more weight to torso points)
        avg_x = sum(p[0] for p in valid_points) / len(valid_points)
        avg_y = sum(p[1] for p in valid_points) / len(valid_points)

        return (int(avg_x), int(avg_y))

    def match_faces_to_poses(self, face_centers, pose_keypoints_list):
        """
        Match detected faces to pose keypoints based on proximity.

        Args:
            face_centers: List of face center coordinates
            pose_keypoints_list: List of pose keypoint arrays

        Returns:
            list: List of matched pairs [{'face_center': (x,y), 'keypoints': [...], 'pose_center': (x,y)}]
        """
        if not face_centers or not pose_keypoints_list:
            return []

        # Calculate pose centers
        pose_centers = []
        for keypoints in pose_keypoints_list:
            center = self.get_pose_center(keypoints)
            pose_centers.append(center)

        # Match faces to poses based on proximity
        matched_pairs = []

        if len(face_centers) > 0 and len(pose_centers) > 0:
            # Filter valid pose centers
            valid_pose_data = [(i, center, pose_keypoints_list[i])
                               for i, center in enumerate(pose_centers) if center is not None]

            if valid_pose_data:
                # Create coordinate arrays for distance calculation
                face_coords = np.array(face_centers)
                pose_coords = np.array([center for _, center, _ in valid_pose_data])

                # Calculate distances between all faces and poses
                distances = cdist(face_coords, pose_coords)

                # Greedy matching: assign each face to closest pose
                used_poses = set()

                for face_idx, face_center in enumerate(face_centers):
                    distances_for_face = distances[face_idx]

                    # Find closest unused pose
                    min_dist = float('inf')
                    best_pose_idx = -1

                    for pose_array_idx, (original_idx, pose_center, keypoints) in enumerate(valid_pose_data):
                        if original_idx not in used_poses:
                            dist = distances_for_face[pose_array_idx]
                            if dist < min_dist and dist < 200:  # Max 200 pixels distance
                                min_dist = dist
                                best_pose_idx = original_idx

                    if best_pose_idx != -1:
                        matched_pairs.append({
                            'face_center': face_center,
                            'keypoints': pose_keypoints_list[best_pose_idx],
                            'pose_center': pose_centers[best_pose_idx]
                        })
                        used_poses.add(best_pose_idx)

        return matched_pairs

    def update_players(self, frame, detected_keypoints_list):
        """
        Update player tracking with current frame data.

        Args:
            frame: Current video frame
            detected_keypoints_list: List of detected pose keypoints

        Returns:
            dict: {player_id: keypoints} mapping of stable player assignments
        """
        self.stats['total_frames'] += 1

        # Detect faces in current frame
        face_centers, face_boxes = self.detect_faces(frame)

        # Match faces to poses
        matched_pairs = self.match_faces_to_poses(face_centers, detected_keypoints_list)

        if not matched_pairs:
            # No matches found - increment missing count for all players
            for player_id in self.players:
                self.players[player_id]['missing_count'] += 1

            # Remove players missing for too long
            self._remove_missing_players()
            return {}

        # Match current detections to existing players
        matched_players = {}
        used_pairs = set()

        if self.players:
            # Get existing player data
            player_ids = list(self.players.keys())
            player_face_centers = [self.players[pid]['face_center'] for pid in player_ids]
            current_face_centers = [pair['face_center'] for pair in matched_pairs]

            if player_face_centers and current_face_centers:
                # Calculate distances between existing and current players
                distances = cdist(player_face_centers, current_face_centers)

                # Hungarian-style matching (greedy approximation)
                for _ in range(min(len(player_ids), len(matched_pairs))):
                    if distances.size == 0:
                        break

                    min_idx = np.unravel_index(np.argmin(distances), distances.shape)
                    player_idx, pair_idx = min_idx

                    if distances[player_idx, pair_idx] < self.max_distance_threshold:
                        player_id = player_ids[player_idx]
                        pair = matched_pairs[pair_idx]

                        # Update existing player
                        self.players[player_id].update({
                            'face_center': pair['face_center'],
                            'keypoints': pair['keypoints'],
                            'pose_center': pair['pose_center'],
                            'missing_count': 0
                        })

                        matched_players[player_id] = pair['keypoints']
                        used_pairs.add(pair_idx)

                        # Remove from consideration
                        distances[player_idx, :] = np.inf
                        distances[:, pair_idx] = np.inf

        # Create new players for unmatched detections
        for i, pair in enumerate(matched_pairs):
            if i not in used_pairs:
                player_id = self.next_player_id
                self.next_player_id += 1

                self.players[player_id] = {
                    'face_center': pair['face_center'],
                    'keypoints': pair['keypoints'],
                    'pose_center': pair['pose_center'],
                    'missing_count': 0
                }

                matched_players[player_id] = pair['keypoints']

        # Mark unmatched existing players as missing
        for player_id in self.players:
            if player_id not in matched_players:
                self.players[player_id]['missing_count'] += 1

        # Remove players missing for too long
        self._remove_missing_players()

        self.stats['players_tracked'] = len(self.players)
        return matched_players

    def _remove_missing_players(self):
        """Remove players that have been missing for too long."""
        to_remove = [pid for pid, data in self.players.items()
                     if data['missing_count'] > self.max_missing_frames]

        for pid in to_remove:
            del self.players[pid]

    def get_stable_player_mapping(self, frame, detected_keypoints_list):
        """
        Get stable player mapping that maintains consistent identities.

        Args:
            frame: Current video frame
            detected_keypoints_list: List of detected pose keypoints

        Returns:
            dict: {display_index: keypoints} mapping for game compatibility
        """
        # Update internal tracking
        matched_players = self.update_players(frame, detected_keypoints_list)

        # Convert to display index mapping (0, 1, 2, ...)
        stable_mapping = {}
        sorted_players = sorted(matched_players.items())  # Sort by player_id for consistency

        for display_index, (player_id, keypoints) in enumerate(sorted_players):
            stable_mapping[display_index] = keypoints

        return stable_mapping

    def draw_debug_info(self, frame, show_faces=True, show_player_ids=True):
        """
        Draw debug information on the frame.

        Args:
            frame: Image frame to draw on
            show_faces: Whether to show face detection boxes
            show_player_ids: Whether to show player ID labels

        Returns:
            numpy.ndarray: Frame with debug information
        """
        if not self.debug_mode:
            return frame

        debug_frame = frame.copy()

        if show_faces:
            # Detect faces for current frame
            face_centers, face_boxes = self.detect_faces(frame)

            # Draw face detection boxes
            for i, ((center_x, center_y), (x, y, w, h)) in enumerate(zip(face_centers, face_boxes)):
                # Draw bounding box
                cv2.rectangle(debug_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # Draw center point
                cv2.circle(debug_frame, (center_x, center_y), 3, (0, 255, 0), -1)

                # Draw face label
                cv2.putText(debug_frame, f"Face {i}", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        if show_player_ids:
            # Draw player assignments
            for player_id, data in self.players.items():
                face_center = data['face_center']
                pose_center = data.get('pose_center')

                # Draw face center with player ID
                cv2.circle(debug_frame, (int(face_center[0]), int(face_center[1])), 8, (255, 0, 0), 2)
                cv2.putText(debug_frame, f"P{player_id}",
                            (int(face_center[0]) - 15, int(face_center[1]) - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

                # Draw connection line if pose center exists
                if pose_center:
                    cv2.line(debug_frame,
                             (int(face_center[0]), int(face_center[1])),
                             (int(pose_center[0]), int(pose_center[1])),
                             (0, 0, 255), 2)

        return debug_frame

    def get_statistics(self):
        """
        Get tracking statistics.

        Returns:
            dict: Statistics about tracking performance
        """
        return {
            'total_frames': self.stats['total_frames'],
            'faces_detected': self.stats['faces_detected'],
            'players_tracked': self.stats['players_tracked'],
            'active_players': len(self.players),
            'avg_faces_per_frame': self.stats['faces_detected'] / max(1, self.stats['total_frames'])
        }

    def reset(self):
        """Reset the tracker state."""
        self.players = {}
        self.next_player_id = 0
        self.stats = {
            'total_frames': 0,
            'faces_detected': 0,
            'players_tracked': 0
        }

    def __del__(self):
        """Cleanup MediaPipe resources."""
        if hasattr(self, 'face_detection'):
            self.face_detection.close()