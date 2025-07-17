import cv2
import numpy as np
import time
from PIL import Image
import os

class AnimatedEmoji:
    def __init__(self, gif_path, size=(64, 64)):
        """
        初始化动态表情
        size: (width, height) 元组，指定emoji的大小
        """
        self.frames = []
        self.durations = []
        self.current_frame_index = 0
        self.last_update_time = time.time()
        self.size = size
        
        try:
            # 加载GIF
            gif = Image.open(gif_path)
            
            # 获取所有帧
            try:
                while True:
                    # 转换为RGBA并调整大小
                    frame = gif.convert('RGBA')
                    frame = frame.resize(self.size, Image.Resampling.LANCZOS)
                    # 转换为numpy数组并将RGB转为BGR
                    frame_array = np.array(frame)
                    # RGB转BGR，但保持alpha通道不变
                    frame_array = cv2.cvtColor(frame_array, cv2.COLOR_RGBA2BGRA)
                    self.frames.append(frame_array)
                    
                    # 获取当前帧的持续时间（毫秒）
                    duration = gif.info.get('duration', 100)  # 默认100ms
                    self.durations.append(duration / 1000.0)  # 转换为秒
                    
                    # 移动到下一帧
                    gif.seek(gif.tell() + 1)
            except EOFError:
                pass  # 到达GIF末尾
                
            if not self.frames:  # 如果是静态图片
                frame = np.array(gif.convert('RGBA'))
                frame = cv2.resize(frame, self.size, interpolation=cv2.INTER_LANCZOS4)
                frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGRA)
                self.frames.append(frame)
                self.durations.append(0.1)  # 静态图片使用固定间隔
                
        except Exception as e:
            print(f"加载动态emoji出错 {gif_path}: {str(e)}")
            self.frames = []
            self.durations = []
    
    def get_current_frame(self):
        """
        获取当前应该显示的帧
        """
        if not self.frames:
            return None
            
        current_time = time.time()
        
        # 检查是否需要更新到下一帧
        if current_time - self.last_update_time > self.durations[self.current_frame_index]:
            self.current_frame_index = (self.current_frame_index + 1) % len(self.frames)
            self.last_update_time = current_time
            
        return self.frames[self.current_frame_index]


def load_emoji(emoji_path):
    """
    加载emoji图片并调整大小
    """
    try:
        # 使用PIL加载图片
        emoji = Image.open(emoji_path)
        
        # 如果是GIF，获取第一帧
        if emoji.format == 'GIF':
            emoji.seek(0)
        
        # 转换为RGBA模式
        emoji = emoji.convert('RGBA')
        
        # 统一调整大小为64x64
        emoji = emoji.resize((64, 64), Image.Resampling.LANCZOS)
        
        # 转换为numpy数组
        emoji = np.array(emoji)
        return emoji
    except Exception as e:
        print(f"加载emoji出错 {emoji_path}: {str(e)}")
        return None

def overlay_emoji(frame, animated_emoji, x, y):
    """
    将emoji叠加到frame上的指定位置
    """
    if animated_emoji is None:
        return frame
    
    try:
        # 获取当前帧
        emoji = animated_emoji.get_current_frame()
        if emoji is None:
            return frame
            
        # 获取emoji的尺寸
        h_emoji, w_emoji = emoji.shape[:2]
        
        # 确保位置在frame内
        y = max(0, min(y, frame.shape[0] - h_emoji))
        x = max(0, min(x, frame.shape[1] - w_emoji))
        
        # 创建叠加区域
        overlay = frame[y:y + h_emoji, x:x + w_emoji]
        
        # 调整emoji大小以匹配叠加区域
        if overlay.shape[:2] != (h_emoji, w_emoji):
            emoji = cv2.resize(emoji, (overlay.shape[1], overlay.shape[0]))
        
        # 使用alpha通道进行混合
        alpha = emoji[..., 3] / 255.0
        alpha = np.stack([alpha] * 3, axis=-1)
        
        # 混合emoji和原图
        overlay_new = overlay * (1 - alpha) + emoji[..., :3] * alpha
        
        # 更新frame
        frame[y:y + h_emoji, x:x + w_emoji] = overlay_new
        
    except Exception as e:
        print(f"叠加emoji出错: {str(e)}")
    
    return frame

def init_emojis():
    """
    初始化表情符号
    """
    emoji_dir = os.path.join(os.path.dirname(__file__), 'assets', 'emojis')
    extra_emoji_dir = os.path.join(os.path.dirname(__file__), 'assets', 'pic')
    emojis = {}
    extra_emojis = {}
    emotion_files = {
        'angry': 'angry.gif',
        'disgust': 'disgust.gif',
        'fear': 'fear.gif',
        'happy': 'happy.gif',
        'neutral': 'neutral.gif',
        'sad': 'sad.gif',
        'surprise': 'surprise.gif'
    }
    extra_emotion_files = {
        'happy':'balloon2.gif',
        'angry':'fire_head.gif',
        'sad':'tears.gif',
        'fear':'fear.gif',
        'surprise':'surprise.gif'
    }
    
    # 使用较小的尺寸
    size1 = (40, 40)  # emoji的大小
    size2 = (200, 200)  # 气球背景
    size3 = (200, 200)  # 泪滴
    size4 = (100, 100)  # 恐惧脸
    size5 = (900, 1200)  # 彩带
    
    for emotion, filename in emotion_files.items():
        emoji_path = os.path.join(emoji_dir, filename)
        if os.path.exists(emoji_path):
            emojis[emotion] = AnimatedEmoji(emoji_path, size=size1)
        else:
            print(f"警告: 找不到emoji文件 {emoji_path}")
    
    for emotion, filename in extra_emotion_files.items():
        emoji_path = os.path.join(extra_emoji_dir, filename)
        if os.path.exists(emoji_path):
            if emotion == 'happy':
                extra_emojis[emotion] = AnimatedEmoji(emoji_path, size=size2)
            elif emotion == 'sad':
                extra_emojis[emotion] = AnimatedEmoji(emoji_path, size=size3)
            elif emotion == 'fear':
                extra_emojis[emotion] = AnimatedEmoji(emoji_path, size=size4)
            elif emotion == 'surprise':
                extra_emojis[emotion] = AnimatedEmoji(emoji_path, size=size5)
            else:
                extra_emojis[emotion] = AnimatedEmoji(emoji_path, size=size1)
        else:
            print(f"警告: 找不到extra emoji文件 {emoji_path}")
    return emojis, extra_emojis

def apply_emotion_effects(frame, emotion, bbox, animated_emoji=None, extra_emoji=None, landmarks=None):
    """
    根据不同情绪添加视觉特效
    """
    x, y, w, h = bbox
    frame_with_effect = frame.copy()
    
    # 其他表情只添加表情emoji
    if animated_emoji is not None:
        emoji_x = x + w + 10
        emoji_y = y
        frame_with_effect = overlay_emoji(frame_with_effect, animated_emoji, emoji_x, emoji_y)
    
    if emotion == 'happy':
        if extra_emoji is not None:
            # 在四个角落添加气球
            positions = [
                (0, 0),  # 左上
                (frame.shape[1] - w, 0),  # 右上
                (0, frame.shape[0] - h),  # 左下
                (frame.shape[1] - w, frame.shape[0] - h)  # 右下
            ]
            # 一次性添加所有气球
            for balloon_x, balloon_y in positions:
                frame_with_effect = overlay_emoji(frame_with_effect, extra_emoji, balloon_x, balloon_y)
    
    elif emotion == 'angry':
        if extra_emoji is not None:
            #调小大小
            size = (40, 40)
            extra_emoji.size = size
            # 生气：添加红色边缘效果
            frame_with_effect = overlay_emoji(frame_with_effect, extra_emoji, x, y)
            overlay = frame_with_effect.copy()
            cv2.rectangle(overlay, (x-10, y-10), (x+w+10, y+h+10), (0, 0, 255), -1)
            cv2.addWeighted(overlay, 0.2, frame_with_effect, 0.8, 0, frame_with_effect)
    
    elif emotion == 'sad':
        # 悲伤：添加蓝色雨滴效果
        for _ in range(5):
            rain_x = np.random.randint(x, x + w)
            rain_y = np.random.randint(y - 50, y+20)
            cv2.line(frame_with_effect, (rain_x, rain_y), 
                    (rain_x + 2, rain_y + 10), (255, 0, 0), 2)
        
        # 添加泪滴
        if extra_emoji is not None:
            # 获取眼睛的关键点（假设landmarks是按照dlib的68点排列）
            # 左眼外角：36，左眼内角：39
            # 右眼外角：45，右眼内角：42
            left_eye_outer = landmarks[36]
            left_eye_inner = landmarks[39]
            # 计算每只眼睛的中心点
            left_eye_center = (
                int((left_eye_outer[0] + left_eye_inner[0]) / 2),
                int((left_eye_outer[1] + left_eye_inner[1]) / 2)
            )
            
            frame_with_effect = overlay_emoji(frame_with_effect, extra_emoji, x-w//4, y)
            
    elif emotion == 'surprise':
        # 惊讶：添加星星特效
        for _ in range(8):
            star_x = np.random.randint(x - 20, x + w + 20)
            star_y = np.random.randint(y - 20, y + h + 20)
            cv2.drawMarker(frame_with_effect, (star_x, star_y), 
                          (0, 255, 255), cv2.MARKER_STAR, 10, 2)
        # 彩带
        if extra_emoji is not None:
            frame_with_effect = overlay_emoji(frame_with_effect, extra_emoji, 0, 0)
    
    elif emotion == 'fear':
        # 恐惧：添加暗色阴影效果
        overlay = frame_with_effect.copy()
        cv2.rectangle(overlay, (x-15, y-15), (x+w+15, y+h+15), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.3, frame_with_effect, 0.7, 0, frame_with_effect)
        # 恐惧脸
        if extra_emoji is not None:
            #调节尺寸大小
            frame_with_effect = overlay_emoji(frame_with_effect, extra_emoji, x, y)

    
    
    return frame_with_effect