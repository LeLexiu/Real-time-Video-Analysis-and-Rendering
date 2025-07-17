import cv2
import numpy as np
from PIL import Image, ImageTk

skeleton = [(0,5),(0,6),(5,6),(5,7),(7,9),(6,8),(8,10),
            (5,11),(6,12),(11,12),(11,13),(13,15),(12,14),(14,16)]

def draw_skeleton(frame, keypoints, color_lines=(0, 255, 0), color_points=(0, 0, 255), thickness=2, point_radius=5):
    """
    Draws skeleton lines and points on a frame given keypoints.
    - frame: np.ndarray, image frame (BGR)
    - keypoints: list of (x,y) tuples
    """
    overlay = frame.copy()
    for pt1, pt2 in skeleton:
        if pt1 < len(keypoints) and pt2 < len(keypoints):
            cv2.line(overlay, keypoints[pt1], keypoints[pt2], color_lines, thickness)
    for x, y in keypoints:
        cv2.circle(overlay, (x, y), point_radius, color_points, -1)
    return overlay

def update_tkinter_label(label, frame, size=None):
    """
    Converts a cv2 BGR frame to a Tkinter-compatible PhotoImage and updates the label.
    - label: tk.Label widget
    - frame: np.ndarray BGR image
    - size: (width, height) tuple to resize the image (optional)
    """
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(rgb)
    if size:
        img = img.resize(size)
    imgtk = ImageTk.PhotoImage(image=img)
    label.imgtk = imgtk
    label.config(image=imgtk)