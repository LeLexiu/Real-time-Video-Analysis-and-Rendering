from tkinter import ttk
import tkinter as tk

class ScoreDisplay:
    def __init__(self, root):
        self.score_label = ttk.Label(root, text="Score: 0", font=("Segoe UI", 24, "bold"))
        self.score_label.pack(side=tk.BOTTOM, pady=10)

    def update_score(self, score):
        self.score_label.config(text=f"Score: {int(score)}")

    def update_final_score(self, final_score):
        self.score_label.config(text=f"Final score: {int(final_score)}")

if __name__ == "__main__":
    # 仅用于测试 score_gui.py 自身
    root = tk.Tk()
    score_display = ScoreDisplay(root)
    score_display.update_score(85)
    root.mainloop()