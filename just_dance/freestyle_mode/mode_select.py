import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
from .mode_freestyle import FreestyleMode
from .mode_video import VideoMode

class ModeSelectPage:
    def __init__(self, root, go_back_callback=None):
        self.root = root
        self.go_back_callback = go_back_callback
        self.root.title("Select Game Mode")
        self.root.geometry("1200x600")

        style = ttk.Style()
        style.theme_use('clam')
        style.configure('TButton',
                        background='#d41f51',
                        foreground='white',
                        font=('Segoe UI', 24, 'bold'),
                        width=35,
                        borderwidth=0,
                        padding=10,
                        relief='groove')
        style.map('TButton',
                  background=[('active', '#32065c')])
        '''style.configure('TLabel',
                        background='transparent',
                        foreground='black',
                        font=('Segoe UI', 32, 'bold'),
                        padding=10,
                        width=35
                        )'''

        # Load and keep a reference to the background image
        bg_image = Image.open("mode_select_image.png")
        bg_image = bg_image.resize((1200, 600))
        self.bg_photo = ImageTk.PhotoImage(bg_image)

        # Create canvas with background image
        self.canvas = tk.Canvas(self.root, width=1200, height=600)
        self.canvas.pack(fill="both", expand=True)
        self.canvas.create_image(0, 0, image=self.bg_photo, anchor="nw")

        # Title label on canvas
        '''label = ttk.Label(self.root, text="Choose a Mode", style='TLabel')
        self.canvas.create_window(600, 150, window=label)'''
        self.canvas.create_text(600, 150, text="Choose a Mode", font=('Segoe UI', 40, 'bold'), fill='#fa821f', anchor='center')


        # Buttons on canvas
        btn1 = ttk.Button(self.root, text="Freestyle Mode", style='TButton',
                         command=self.start_freestyle)
        self.canvas.create_window(600, 250, window=btn1)

        btn2 = ttk.Button(self.root, text="Follow Video Mode", style='TButton',
                         command=self.start_video_mode)
        self.canvas.create_window(600, 330, window=btn2)

    def start_freestyle(self):
        self.clear()
        FreestyleMode(self.root, self.go_back_callback)

    def start_video_mode(self):
        self.clear()
        VideoMode(self.root, self.go_back_callback)

    def clear(self):
        for widget in self.root.winfo_children():
            widget.destroy()
