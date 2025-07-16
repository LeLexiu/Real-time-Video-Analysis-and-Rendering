import tkinter as tk
from titlepage import TitlePage
from just_dance.freestyle_mode.mode_select import ModeSelectPage

def main():
    root = tk.Tk()

    def go_to_mode_select():
        for widget in root.winfo_children():
            widget.destroy()
        # ModeSelectPage(root, go_to_title_page)
        root.mode_select_page = ModeSelectPage(root, go_to_title_page)  # optional, for consistency

    def go_to_title_page():
        for widget in root.winfo_children():
            widget.destroy()
        root.title_page = TitlePage(root, go_to_mode_select)

    go_to_title_page()
    root.mainloop()

if __name__ == "__main__":
    main()
