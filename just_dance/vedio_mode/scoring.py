import numpy as np
from collections import defaultdict

# Define joint triplets for angle calculation
JOINT_TRIPLETS = {
    "left_elbow": (5, 7, 9),
    "right_elbow": (6, 8, 10),
    "left_knee": (11, 13, 15),
    "right_knee": (12, 14, 16),
    "left_shoulder": (7, 5, 11),
    "right_shoulder": (8, 6, 12),
    "left_hip": (5, 11, 13),
    "right_hip": (6, 12, 14),
}

def calculate_angle(keypoints_pixels, p1_idx, p2_idx, p3_idx):
    """Calculates the angle between three keypoints."""
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
        return None # 无法形成有效角度

    # 计算角度的余弦值（使用点积）
    dot_product = np.dot(v1, v2)
    cosine_angle = dot_product / (magnitude_v1 * magnitude_v2)

    # 将余弦值限制在 [-1, 1] 范围内，以避免浮点误差
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)

    # 计算弧度，然后转换为度数
    angle_radians = np.arccos(cosine_angle)
    angle_degrees = np.degrees(angle_radians)

    # 确保角度在 [0, 180] 之间（关节角度的常见范围）
    # 原始代码有 if angle_degrees > 180.0: angle_degrees = 360 - angle_degrees
    # 但 np.arccos 结果范围是 [0, 180]，所以这行可能不需要或需要根据实际情况调整
    # 这里保持原始逻辑，但通常关节角度是0-180度
    if angle_degrees > 180.0:
        angle_degrees = 360 - angle_degrees


    return angle_degrees

def calculate_frame_score(video_angles, camera_angles, frame_index, score_threshold=20):
    """Calculates the score for a single frame."""
    frame_scores = []
    for joint_name in JOINT_TRIPLETS:
        video_angle_list = video_angles.get(joint_name, []) # 获取列表，避免 KeyError
        camera_angle_list = camera_angles.get(joint_name, [])

        if frame_index < len(video_angle_list) and frame_index < len(camera_angle_list):
            v_angle = video_angle_list[frame_index]
            c_angle = camera_angle_list[frame_index]
            if v_angle is not None and c_angle is not None:
                difference = abs(v_angle - c_angle)
                if difference < score_threshold:
                    frame_scores.append(1) # 1 表示该关节在该帧是“准确”的
                else:
                    frame_scores.append(0)
            else:
                frame_scores.append(0) # 如果任一角度为 None，则该关节该帧不得分
        else:
            frame_scores.append(0) # 如果索引超出范围，也不得分

    if frame_scores:
        return sum(frame_scores) / len(frame_scores)
    return 0 # 如果没有可比较的关节，返回0

def calculate_final_score(video_angles, camera_angles, score_threshold=20):
    """Calculates the final score based on the entire video."""
    # 确保字典不为空，并且有至少一个关节的角度列表
    if not video_angles or not camera_angles:
        return 0

    # 获取第一个关节的角度列表长度作为参考，假设所有关节的列表长度相同
    first_joint_video_len = len(next(iter(video_angles.values()))) if video_angles else 0
    first_joint_camera_len = len(next(iter(camera_angles.values()))) if camera_angles else 0

    min_len = min(first_joint_video_len, first_joint_camera_len)

    if min_len == 0:
        return 0

    all_joint_scores = []
    for joint_name in JOINT_TRIPLETS:
        video_angles_joint = video_angles.get(joint_name, [])[:min_len] # 安全获取并截断
        camera_angles_joint = camera_angles.get(joint_name, [])[:min_len]

        accuracy_count = 0
        for i in range(min_len):
            video_angle = video_angles_joint[i]
            camera_angle = camera_angles_joint[i]

            if video_angle is not None and camera_angle is not None:
                difference = abs(video_angle - camera_angle)
                if difference < score_threshold:
                    accuracy_count += 1

        if min_len > 0:
            joint_score = (accuracy_count / min_len) * 100
            all_joint_scores.append(joint_score)

    final_score = 0
    if all_joint_scores:
        final_score = np.mean(all_joint_scores)
        final_score += 20 # 奖励分

    return min(int(final_score), 100) # 确保分数不超过100

if __name__ == "__main__":
    # Example usage:
    video_angles_example = defaultdict(list)
    camera_angles_example = defaultdict(list)
    video_angles_example["left_elbow"].extend([90, 95, 100])
    camera_angles_example["left_elbow"].extend([88, 92, 105])
    video_angles_example["right_elbow"].extend([170, 165, 160])
    camera_angles_example["right_elbow"].extend([175, 160, 155])

    frame_score = calculate_frame_score(video_angles_example, camera_angles_example, 1)
    print(f"Frame score for index 1: {frame_score}")

    final_score = calculate_final_score(video_angles_example, camera_angles_example)
    print(f"Final score: {final_score}")

    # Test with missing data
    video_angles_incomplete = defaultdict(list)
    camera_angles_incomplete = defaultdict(list)
    video_angles_incomplete["left_elbow"].extend([90, 95])
    camera_angles_incomplete["left_elbow"].extend([88, 92, 105]) # Camera has more frames

    final_score_incomplete = calculate_final_score(video_angles_incomplete, camera_angles_incomplete)
    print(f"Final score with incomplete data: {final_score_incomplete}")