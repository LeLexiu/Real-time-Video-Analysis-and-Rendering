# Real-time-Video-Analysis-and-Rendering

This is a project for Visual Computing in 2025 NUS summer workshop

---

## Project Structure

```
Real-time-Video-Analysis-and-Rendering/
├── main.py                          # Entry point of the program (launch GUI or select module) 主程序入口（启动GUI或选择模块）
├── config.py                        # Global configurations (e.g., model paths, skeleton definitions, thresholds) 常量和全局配置（骨架连接、模型路径、阈值等）
├── requirements.txt                 # Dependency list Python 依赖项
│
├── models/
│   ├── pose_model.py                # Wrapper for loading YOLOv8 pose model 封装 YOLOv8 pose 模型加载
│   └── face_model.py                # Wrapper for loading face detection/recognition model 
│
├── gui/
│   ├── dance_gui.py                 # GUI for video dance mode (contains PoseApp class) 视频跟跳 GUI
│   └── face_gui.py                  # GUI for face recognition login/registration (if available) 人脸识别登录/注册界面（如有）
│
├── just_dance/
│   ├── freestyle_mode/             # Logic for freestyle dance mode 自由跳舞逻辑（独立模块）
│   └── video_mode/                 # Logic for video follow-along mode 视频跟跳逻辑
│
├── facial_recognition/
│   ├── 
│
├── assets/
│   ├── demo_videos/                # Example videos 示例视频
│   └── icons/                      # UI icons and visual resources UI图标资源
│
└── README.md                       # Project documentation 项目说明
```