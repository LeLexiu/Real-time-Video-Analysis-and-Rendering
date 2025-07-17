import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from collections import OrderedDict

# 定义面部关键点区域
FACIAL_LANDMARKS_IDXS = OrderedDict([
        ("mouth", (48, 68)),
        ("right_eyebrow", (17, 22)),
        ("left_eyebrow", (22, 27)),
        ("right_eye", (36, 42)),
        ("left_eye", (42, 48)),
        ("nose", (27, 36)),
        ("jaw", (0, 17))
    ])

def init_detectors():
    """
    初始化人脸检测器和关键点检测器
    """

    # 加载人脸检测器和关键点检测器
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    facemark = cv2.face.createFacemarkLBF()
    
    # 获取当前文件的绝对路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, "..", "facial_recognition", "lbfmodel.yaml")
    
    # print(f"Loading model from: {model_path}")
    facemark.loadModel(model_path)

    return face_cascade, facemark

def normalize_landmarks(landmarks):
    """
    将关键点坐标归一化，使用以下步骤：
    1. 将鼻尖(30)平移到坐标原点。
    2. 根据鼻尖(30)到下巴中点(8)的连线进行旋转对齐（使得鼻子到下巴的连线垂直向下）。
    3. 根据面部大小进行尺度归一化。
    """
    # 提取关键参考点
    nose_tip = landmarks[30]
    chin = landmarks[8]
    
    # 1. 计算用于旋转对齐的向量（从鼻尖到下巴）
    reference_vector_for_rotation = chin - nose_tip 
    
    # 错误检查：如果关键点检测失败导致距离过小，则返回None
    if np.linalg.norm(reference_vector_for_rotation) < 1e-6:
        print("Warning: Reference distance too small, possibly detection error. Returning None.")
        return None
        
    # 2. 计算旋转角度
    angle = -np.pi / 2 - np.arctan2(reference_vector_for_rotation[1], reference_vector_for_rotation[0])
    rotation_matrix = np.array([
        [np.cos(angle), -np.sin(angle)],
        [np.sin(angle), np.cos(angle)]
    ])
    
    # 开始对每个关键点进行归一化处理
    normalized = []
    for point in landmarks:
        # a. 平移：将鼻尖移动到坐标原点(0,0)
        centered = point - nose_tip
        
        # b. 旋转：根据计算出的角度旋转所有点
        rotated = np.dot(rotation_matrix, centered)
        normalized.append(rotated)
    
    normalized = np.array(normalized)
    
    # 3. 尺度归一化：使用面部大小进行归一化
    # 计算所有点到原点的最大距离作为缩放因子
    max_distance = np.max(np.linalg.norm(normalized, axis=1))
    if max_distance > 1e-6:
        normalized = normalized / max_distance
    
    return normalized

def visualize_normalized_landmarks(landmarks, output_path=None, title=None):
    """
    可视化归一化后的关键点
    使用白底黑点的效果，更清晰地展示面部特征点分布
    """
    plt.figure(figsize=(6, 6), facecolor='white')
    ax = plt.gca()
    ax.set_facecolor('white')
    
    # 重塑landmarks为nx2的数组
    landmarks_reshaped = landmarks.reshape(-1, 2)
    
    # 翻转y坐标使图像方向正确
    landmarks_reshaped[:, 1] = -landmarks_reshaped[:, 1]
    
    # 绘制所有点（黑色小点）
    plt.scatter(landmarks_reshaped[:, 0], landmarks_reshaped[:, 1], 
               c='black', s=20, alpha=1, marker='.')
    
    # 特别标记参考点（稍大一些的点）
    plt.scatter(landmarks_reshaped[30, 0], landmarks_reshaped[30, 1], 
               c='black', s=50, marker='o', label='Nose tip')
    plt.scatter(landmarks_reshaped[8, 0], landmarks_reshaped[8, 1], 
               c='black', s=50, marker='o', label='Chin')
    
    # 设置坐标轴范围，确保不同图片的显示范围一致
    # 由于移除了眼距归一化，调整显示范围以适应更大的坐标值
    plt.xlim(-150, 150)
    plt.ylim(-200, 200)
    
    # 设置标题
    if title:
        plt.title(title, fontsize=10)
    
    # 移除坐标轴
    plt.axis('off')
    
    # 保存或显示
    if output_path:
        plt.savefig(output_path, bbox_inches='tight', dpi=150, 
                   facecolor='white', edgecolor='none')
        plt.close()
    else:
        plt.show()

def visualize_sample_landmarks(features, image_names, emotion_label, output_dir):
    """
    可视化样本图片的归一化效果
    显示3x3的网格，共9张示例图
    """
    num_samples = min(9, len(features))
    if num_samples == 0:
        return
    
    # 创建3x3的子图
    fig = plt.figure(figsize=(15, 15), facecolor='white')
    fig.suptitle(f'Normalized Facial Landmarks - {emotion_label}', fontsize=16)
    
    # 随机选择9张图片
    indices = np.random.choice(len(features), num_samples, replace=False)
    
    for i, idx in enumerate(indices, 1):
        plt.subplot(3, 3, i)
        landmarks_reshaped = features[idx].reshape(-1, 2)
        
        # 绘制所有点
        plt.scatter(landmarks_reshaped[:, 0], landmarks_reshaped[:, 1], 
                   c='black', s=20, alpha=1, marker='.')
        
        # 标记参考点
        plt.scatter(landmarks_reshaped[30, 0], landmarks_reshaped[30, 1], 
                   c='black', s=50, marker='o')
        plt.scatter(landmarks_reshaped[8, 0], landmarks_reshaped[8, 1], 
                   c='black', s=50, marker='o')
        
        # 设置标题为图片名
        plt.title(image_names[idx], fontsize=8)
        
        # 设置显示范围和样式
        plt.xlim(-150, 150)
        plt.ylim(-200, 200)
        plt.axis('off')
    
    plt.tight_layout()
    
    # 保存结果
    output_path = os.path.join(output_dir, f"{emotion_label}_samples.png")
    plt.savefig(output_path, bbox_inches='tight', dpi=150, 
                facecolor='white', edgecolor='none')
    plt.close()
    print(f"样本可视化结果已保存至: {output_path}")

def check_face_symmetry(landmarks, threshold=0.15):
    """
    检查面部是否对称（是否为正脸）
    通过比较面部左右对称的关键点对的距离来判断
    
    Args:
        landmarks: 面部68个关键点坐标
        threshold: 不对称度阈值，越大表示允许的不对称程度越大
        
    Returns:
        bool: True表示基本对称（正脸），False表示不对称（侧脸）
    """
    # 定义对称点对（左右对称的关键点索引）
    symmetric_pairs = [
        # 眉毛对称点对
        (17, 26), (18, 25), (19, 24), (20, 23), (21, 22),
        # 眼睛对称点对
        (36, 45), (37, 44), (38, 43), (39, 42), (40, 47), (41, 46),
        # 嘴角对称点对
        (48, 54), (49, 53), (50, 52), (51, 51),
        # 下巴对称点对（除中点外）
        (0, 16), (1, 15), (2, 14), (3, 13), (4, 12), (5, 11), (6, 10), (7, 9)
    ]
    
    # 计算鼻尖（30）到人中（33）的向量作为面部中线参考
    nose_vector = landmarks[33] - landmarks[30]
    nose_length = np.linalg.norm(nose_vector)
    
    if nose_length < 1e-6:
        return False
        
    asymmetry_scores = []
    
    # 将2D向量转换为3D向量（添加z=0）
    def to_3d(point):
        return np.array([point[0], point[1], 0])
    
    nose_vector_3d = to_3d(nose_vector)
    nose_tip_3d = to_3d(landmarks[30])
    
    for left_idx, right_idx in symmetric_pairs:
        left_point_3d = to_3d(landmarks[left_idx])
        right_point_3d = to_3d(landmarks[right_idx])
        
        # 计算对称点对到面部中线的距离差异
        left_vector = left_point_3d - nose_tip_3d
        right_vector = right_point_3d - nose_tip_3d
        
        # 使用3D叉积计算
        left_dist = np.linalg.norm(np.cross(nose_vector_3d, left_vector)) / nose_length
        right_dist = np.linalg.norm(np.cross(nose_vector_3d, right_vector)) / nose_length
        
        # 计算不对称度分数
        asymmetry = abs(left_dist - right_dist) / ((left_dist + right_dist) / 2)
        asymmetry_scores.append(asymmetry)
    
    # 使用平均不对称度作为最终的评分
    mean_asymmetry = np.mean(asymmetry_scores)
    return mean_asymmetry <= threshold

def process_image(image_path, face_cascade, facemark):
    """
    处理单张图片，检测人脸和关键点
    """
    # 读取图片
    img = cv2.imread(image_path)
    if img is None:
        print(f"无法读取图片: {image_path}")
        return None
    
    # 获取原始尺寸
    original_height, original_width = img.shape[:2]
    
    # 放大图像到更大尺寸（保持宽高比）
    scale_factor = 4  # 放大4倍
    width = int(original_width * scale_factor)
    height = int(original_height * scale_factor)
    img_resized = cv2.resize(img, (width, height), interpolation=cv2.INTER_CUBIC)
    
    # 转换为灰度图
    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    
    # 增强对比度
    gray = cv2.equalizeHist(gray)
    
    # 使用更宽松的参数检测人脸
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.05,
        minNeighbors=3,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    
    if len(faces) == 0:
        # print(f"未检测到人脸: {image_path}")
        return None
    
    # 检查原始图像中的人脸大小（将放大后的人脸尺寸缩回原始尺寸）
    x, y, w, h = faces[0]
    original_face_width = int(w / scale_factor)
    original_face_height = int(h / scale_factor)
    
    # 如果原始图像中的人脸小于30x30像素，则跳过
    if original_face_width < 25 or original_face_height < 25:
        # print(f"人脸区域太小 ({original_face_width}x{original_face_height}像素): {image_path}")
        return None
    
    # 检测关键点
    success, landmarks = facemark.fit(gray, faces)
    
    if not success:
        print(f"关键点检测失败: {image_path}")
        return None
    
    # 将关键点坐标缩放回原始尺寸
    landmarks_original = landmarks[0][0] / scale_factor
    
    # 检查是否为正脸
    if not check_face_symmetry(landmarks_original):
        # print(f"检测到侧脸: {image_path}")
        return None
        
    return landmarks_original.astype(np.float32)

def process_dataset(train_dir, emotion_label):
    """
    处理数据集中的图片并保存关键点数据
    """
    # 初始化检测器
    face_cascade, facemark = init_detectors()
    
    # 获取所有图片文件
    image_files = [f for f in os.listdir(train_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
    total_images = len(image_files)
    
    if total_images == 0:
        print("未找到图片文件")
        return
    
    # 创建输出目录结构
    base_dir = os.path.join(os.path.dirname(train_dir), "landmarks_data")
    emotion_dir = os.path.join(base_dir, emotion_label)
    vis_dir = os.path.join(base_dir, "visualization")
    
    # 创建必要的目录
    for directory in [base_dir, emotion_dir, vis_dir]:
        if not os.path.exists(directory):
            os.makedirs(directory)
    
    # 用于存储所有特征和标签
    all_features = []
    all_image_names = []
    
    successful_count = 0
    failed_images = []
    
    print(f"开始处理 {total_images} 张图片...")
    
    for i, image_file in enumerate(image_files, 1):
        image_path = os.path.join(train_dir, image_file)
        # print(f"处理图片 {i}/{total_images}: {image_file}", end='\r')
        
        try:
            landmarks = process_image(image_path, face_cascade, facemark)
            
            if landmarks is not None:
                # 归一化关键点坐标
                normalized_landmarks = normalize_landmarks(landmarks)
                
                if normalized_landmarks is not None:
                    # 保存特征和图片名称
                    all_features.append(normalized_landmarks)
                    all_image_names.append(image_file)
                    successful_count += 1
                else:
                    failed_images.append(f"{image_file} (归一化失败)")
            else:
                failed_images.append(f"{image_file} (人脸检测失败)")
        except Exception as e:
            failed_images.append(f"{image_file} (错误: {str(e)})")
    
    # print("\n") # 清除进度显示的行
    
    # 将特征转换为numpy数组
    if successful_count > 0:
        features_array = np.array(all_features)
        labels_array = np.full(successful_count, emotion_label)
        
        # 保存特征、标签和图片名称到情绪特定的目录
        output_features = os.path.join(emotion_dir, "features.npy")
        output_labels = os.path.join(emotion_dir, "labels.npy")
        output_names = os.path.join(emotion_dir, "image_names.txt")
        
        np.save(output_features, features_array)
        np.save(output_labels, labels_array)
        with open(output_names, 'w') as f:
            f.write('\n'.join(all_image_names))
        
        print(f"\n特征维度: {features_array.shape}")
        print(f"特征数据已保存至: {output_features}")
        print(f"标签数据已保存至: {output_labels}")
        print(f"图片名称已保存至: {output_names}")
        
        # 生成样本可视化
        visualize_sample_landmarks(features_array, all_image_names, emotion_label, vis_dir)
    
    # 打印处理结果统计
    print("\n处理完成!")
    print(f"成功处理: {successful_count} 张图片")
    print(f"处理失败: {len(failed_images)} 张图片")
    
    if failed_images:
        failed_log_path = os.path.join(emotion_dir, "failed_images.txt")
        with open(failed_log_path, "w") as f:
            f.write('\n'.join(failed_images))
        print(f"失败图片列表已保存至: {failed_log_path}")

def main():
    # 基础路径
    base_dir = r"D:\my_project\vc-study\final_project\facial_expression_dataset\facial_expression_dataset\augmented_dataset"
    
    # emotion_label = 'disgust'
    # data_dir = os.path.join(base_dir, emotion_label)
    # process_dataset(data_dir, emotion_label)

    # 情绪类别列表
    emotions = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
    
    # 处理测试集的所有情绪类别
    for emotion in emotions:
        print(f"\n处理测试集 {emotion} 类别...")
        data_dir = os.path.join(base_dir, emotion)
        process_dataset(data_dir, emotion)

if __name__ == "__main__":
    main() 