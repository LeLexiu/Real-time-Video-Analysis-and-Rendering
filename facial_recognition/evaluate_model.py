import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import joblib

def load_features(data_dir, emotion):
    """加载特定情绪类别的特征数据"""
    feature_path = os.path.join(data_dir, emotion, "features.npy")
    if not os.path.exists(feature_path):
        print(f"Warning: 找不到{emotion}的特征文件: {feature_path}")
        return None, None
    
    features = np.load(feature_path)
    # 重塑特征数据从 (N, 68, 2) 到 (N, 136)
    features = features.reshape(features.shape[0], -1) # -1 means calculate the rest of the dimensions
    labels = np.full(len(features), emotion)
    return features, labels

def load_dataset(base_dir, emotions):
    """加载所有情绪类别的数据并组合"""
    all_features = []
    all_labels = []
    
    for emotion in emotions:
        features, labels = load_features(base_dir, emotion)
        if features is not None:
            all_features.append(features)
            all_labels.append(labels)
    
    if not all_features:
        raise ValueError("没有找到任何特征数据！")
    
    X = np.vstack(all_features)
    y = np.concatenate(all_labels)
    
    return X, y

def plot_confusion_matrix(cm, classes, save_path):
    """
    绘制并保存混淆矩阵
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def evaluate_model():
    # 设置数据路径
    train_dir = r""
    test_dir = r""

    # 定义情绪类别
    emotions = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
    
    # 创建结果目录
    results_dir = os.path.join(train_dir, "classifier_results")
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    # 加载训练数据
    print("加载训练数据...")
    X_train, y_train = load_dataset(train_dir, emotions)
    print(f"训练数据形状: {X_train.shape}")
    
    # 使用最佳参数创建模型
    best_params = {
        'max_depth': 30,
        'min_samples_leaf': 1,
        'min_samples_split': 2,
        'n_estimators': 300,
        'random_state': 42
    }
    
    print("\n使用最佳参数创建模型:")
    for param, value in best_params.items():
        print(f"{param}: {value}")
    
    # 训练模型
    print("\n开始训练模型...")
    model = RandomForestClassifier(**best_params)
    model.fit(X_train, y_train)
    
    # 加载测试数据
    print("\n加载测试数据...")
    X_test, y_test = load_dataset(test_dir, emotions)
    print(f"测试数据形状: {X_test.shape}")
    
    # 在测试集上评估模型
    print("\n在测试集上评估模型...")
    y_pred = model.predict(X_test)
    
    # 生成分类报告
    report = classification_report(y_test, y_pred)
    print("\n分类报告:")
    print(report)
    
    # 计算混淆矩阵
    cm = confusion_matrix(y_test, y_pred)
    
    # 保存结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = os.path.join(results_dir, f"evaluation_results_{timestamp}")
    os.makedirs(results_path, exist_ok=True)
    
    # 保存混淆矩阵图
    plot_confusion_matrix(cm, emotions, os.path.join(results_path, "confusion_matrix.png"))
    
    # 保存分类报告
    with open(os.path.join(results_path, "classification_report.txt"), 'w') as f:
        f.write("最佳参数:\n")
        for param, value in best_params.items():
            f.write(f"{param}: {value}\n")
        f.write("\n分类报告:\n")
        f.write(report)
    
    # 保存模型
    model_path = os.path.join(results_path, "random_forest_model.joblib")
    joblib.dump(model, model_path)
    print(f"\n模型和结果已保存至: {results_path}")

if __name__ == "__main__":
    evaluate_model() 