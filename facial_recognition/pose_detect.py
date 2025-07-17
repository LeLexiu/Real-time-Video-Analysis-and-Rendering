import numpy as np

# COCO skeleton connections (keypoint pairs)
SKELETON = [
    (0, 5), (0, 6),     # nose to shoulders
    (5, 6),             # shoulders
    (5, 7), (7, 9),     # left arm
    (6, 8), (8, 10),    # right arm
    (5, 11), (6, 12),   # torso sides
    (11, 12),           # hips
    (11, 13), (13, 15), # left leg
    (12, 14), (14, 16)  # right leg
]

def calculate_angle(point1, point2, point3):
    """Calculate angle between three points"""
    a = np.array(point1)
    b = np.array(point2)
    c = np.array(point3)
    
    ba = a - b
    bc = c - b
    
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)
    
    return np.degrees(angle)

def detect_hands_up(keypoints):
    """Detect if both hands are raised above shoulders"""
    if len(keypoints) < 11:
        return False
    
    left_shoulder = keypoints[5]
    right_shoulder = keypoints[6]
    left_wrist = keypoints[9]
    right_wrist = keypoints[10]
    
    return (left_wrist[1] < left_shoulder[1] and 
            right_wrist[1] < right_shoulder[1])

def detect_t_pose(keypoints):
    """Detect T-pose (arms horizontally spread)"""
    if len(keypoints) < 11:
        return False
    
    left_shoulder = keypoints[5]
    right_shoulder = keypoints[6]
    left_wrist = keypoints[9]
    right_wrist = keypoints[10]
    
    shoulder_y = (left_shoulder[1] + right_shoulder[1]) / 2
    y_threshold = 50  # Allowed y-axis deviation
    
    return (abs(left_wrist[1] - shoulder_y) < y_threshold and 
            abs(right_wrist[1] - shoulder_y) < y_threshold)

def detect_hands_overhead(keypoints):
    """Detect if hands are raised above head"""
    if len(keypoints) < 11:
        return False
    
    left_eye = keypoints[1]
    right_eye = keypoints[2]
    left_wrist = keypoints[9]
    right_wrist = keypoints[10]
    
    return (left_wrist[1] < left_eye[1] or right_wrist[1] < right_eye[1])

def detect_salute(keypoints):
    """Detect salute pose"""
    if len(keypoints) < 11:
        return False
    
    right_eye = keypoints[2]
    right_shoulder = keypoints[6]
    right_elbow = keypoints[8]
    right_wrist = keypoints[10]
    
    # Check if right hand is near the head
    wrist_eye_diff = abs(right_wrist[1] - right_eye[1])
    
    # Check if elbow is bent
    arm_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)
    
    return wrist_eye_diff < 30 and 15 < arm_angle < 45

def get_detected_poses(keypoints):
    """Get list of all detected poses for given keypoints"""
    detected_poses = []
    
    if detect_hands_up(keypoints):
        detected_poses.append("Hands Up")
    if detect_t_pose(keypoints):
        detected_poses.append("T-Pose")
    if detect_hands_overhead(keypoints):
        detected_poses.append("Hands Overhead")
    if detect_salute(keypoints):
        detected_poses.append("Salute")
        
    return detected_poses
