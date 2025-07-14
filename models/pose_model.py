from ultralytics import YOLO
import cv2
import numpy as np

# COCO skeleton connections
skeleton = [
    (0, 5), (0, 6),
    (5, 6),
    (5, 7), (7, 9),
    (6, 8), (8, 10),
    (5, 11), (6, 12),
    (11, 12),
    (11, 13), (13, 15),
    (12, 14), (14, 16)
]

def load_pose_model(model_path="models/yolov8n-pose.pt"):
    """Loads the YOLOv8 pose estimation model."""
    return YOLO(model_path)

def process_pose(model, frame, show_video_frame=True):
    """Performs pose estimation on a frame."""
    results = model(frame, conf=0.3)
    height, width = frame.shape[:2]

    overflow = frame.copy()
    if not show_video_frame:
        overflow = np.ones_like(frame) * 255 # 白色背景

    person_keypoints_pixels = None

    # 遍历检测到的人物（YOLOv8 可以检测多个人）
    # 这里我们取第一个检测到的人物（通常是置信度最高的）
    if results and results[0]: # 确保 results 不为空且有第一个结果
        first_result = results[0]
        if first_result.keypoints is not None and len(first_result.keypoints.xyn) > 0: # 确保 keypoints 和 xyn 存在
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
                    cv2.line(overflow, (x1, y1), (x2, y2), (0, 255, 0), 2) # 绿色线条

            for kp_x, kp_y in person_keypoints_pixels:
                cv2.circle(overflow, (kp_x, kp_y), 5, (0, 0, 255), -1) # 红色圆点

    return overflow, person_keypoints_pixels

if __name__ == "__main__":
    # Example usage:
    model = load_pose_model()
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    processed_frame, keypoints = process_pose(model, dummy_frame)
    if processed_frame is not None:
        cv2.imshow("Processed Frame", processed_frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("Error: Could not process dummy frame.")