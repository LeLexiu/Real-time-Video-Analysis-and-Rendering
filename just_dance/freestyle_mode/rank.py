import tkinter as tk
from tkinter import messagebox

def show_score_ranking(root, person_scores):
    if not person_scores:
        messagebox.showinfo("Results", "No scores to rank.")
        return

    def _show():
        sorted_scores = sorted(person_scores.items(), key=lambda x: x[1], reverse=True)

        ranking_win = tk.Toplevel(root)
        ranking_win.title("Final Results")
        ranking_win.geometry("400x300")

        tk.Label(ranking_win, text="🏆 Final Rankings:", font=("Arial", 20, "bold")).pack(pady=10)

        for i, (pid, score) in enumerate(sorted_scores, start=1):
            tk.Label(
                ranking_win,
                text=f"{i}. Player {pid + 1}: {int(score)} points",
                font=("Arial", 16)
            ).pack()

        tk.Button(ranking_win, text="Close", font=("Arial", 14), command=ranking_win.destroy).pack(pady=15)

    # Ensure it runs on the main thread
    try:
        root.after(0, _show)
    except Exception:
        _show()
