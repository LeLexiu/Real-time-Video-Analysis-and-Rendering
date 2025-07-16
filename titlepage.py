from PIL import Image, ImageTk
import tkinter as tk

class TitlePage:
    def __init__(self, root, start_callback):
        self.root = root
        self.root.title("Welcome to Stickman Dance!")
        self.root.geometry("1200x600")

        try:
            bg_image = Image.open("title_image_1.png").convert("RGB")
            bg_image = bg_image.resize((1200, 600))
            self.bg_photo = ImageTk.PhotoImage(bg_image)
        except Exception as e:
            print("Error loading image:", e)
            return

        self.canvas = tk.Canvas(root, width=1200, height=600)
        self.canvas.pack(fill="both", expand=True)
        self.bg_image_id = self.canvas.create_image(0, 0, image=self.bg_photo, anchor="nw")

        self.label = tk.Label(root, text="Stickman Dance", font=("Arial", 36, "bold"), bg="white", fg="black")
        self.canvas.create_window(600, 150, window=self.label)

        self.start_button = tk.Button(root, text="Game Start!", font=("Arial", 20), command=start_callback)
        self.canvas.create_window(600, 300, window=self.start_button)