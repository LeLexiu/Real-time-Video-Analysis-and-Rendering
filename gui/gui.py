import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
from PIL import Image, ImageTk
import os
import cv2 # 确保导入 cv2

class MainWindow:
    def __init__(self, root, load_video_command, start_video_command, stop_video_command, toggle_video_display_command, start_cam_command, stop_cam_command):
        self.root = root
        self.root.title("Stickman Dance GUI")
        self.root.geometry("1200x600")
        s = ttk.Style()
        s.theme_use('vista')

        default_font = ("Segoe UI", 10)
        self.root.option_add("*Font", default_font)

        self.video_path = ""
        self.show_video_frame = True # 移到这里，由主程序控制

        self.left_frame = ttk.Frame(self.root)
        self.left_frame.pack(side=tk.LEFT, padx=10)
        self.right_frame = ttk.Frame(self.root) # 修正这里，应该是 self.root
        self.right_frame.pack(side=tk.RIGHT, padx=10)

        # Video File Window
        self.label_file = ttk.Label(self.left_frame)
        self.label_file.pack()
        self.controls_file = ttk.Frame(self.left_frame)
        self.controls_file.pack()
        ttk.Button(self.controls_file, text="Open Video", command=load_video_command).pack(side=tk.LEFT, padx=5) # 使用传入的 command
        ttk.Button(self.controls_file, text="Start Video", command=start_video_command).pack(side=tk.LEFT, padx=5)
        ttk.Button(self.controls_file, text="Stop Video", command=stop_video_command).pack(side=tk.LEFT, padx=5)
        ttk.Button(self.controls_file, text="Show/Hide Video", command=toggle_video_display_command).pack(side=tk.LEFT, padx=5)

        # Webcam Window
        self.label_cam = ttk.Label(self.right_frame)
        self.label_cam.pack()
        self.controls_cam = ttk.Frame(self.right_frame)
        self.controls_cam.pack()
        ttk.Button(self.controls_cam, text="Start Webcam", command=start_cam_command).pack(side=tk.LEFT, padx=5)
        ttk.Button(self.controls_cam, text="Stop Webcam", command=stop_cam_command).pack(side=tk.LEFT, padx=5)

    # 这里的 load_video 不再是类方法，而是由 main.py 调用 filedialog
    # def load_video(self):
    #     path = filedialog.askopenfilename(filetypes=[("MP4 files", "*.mp4"), ("All files", "*.*")])
    #     if path:
    #         self.video_path = path
    #         messagebox.showinfo("Video Selected", os.path.basename(path))
    #     return self.video_path

    def update_label(self, label, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        img = img.resize((640, 384))
        imgtk = ImageTk.PhotoImage(image=img)
        label.imgtk = imgtk
        label.configure(image=imgtk)

if __name__ == "__main__":
    # 仅用于测试 gui.py 自身，实际运行时不执行
    root = tk.Tk()
    # 传入 lambda 函数作为占位符
    main_window = MainWindow(root, lambda: print("Load Video Clicked"), lambda: print("Start Video Clicked"),
                             lambda: print("Stop Video Clicked"), lambda: print("Toggle Video Clicked"),
                             lambda: print("Start Cam Clicked"), lambda: print("Stop Cam Clicked"))
    root.mainloop()