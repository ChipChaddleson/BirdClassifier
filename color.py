import tkinter as tk

def change_color(color):
    canvas.config(bg=color)

root = tk.Tk()
root.title("Color Changer")

canvas = tk.Canvas(root, width=200, height=50, bg="white")
canvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

frame = tk.Frame(root)
frame.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)

red_button = tk.Button(frame, text="Red", command=lambda: change_color("red"))
red_button.pack(side=tk.LEFT, padx=20, pady=10)

yellow_button = tk.Button(frame, text="green", command=lambda: change_color("green"))
yellow_button.pack(side=tk.LEFT, padx=20, pady=10)

blue_button = tk.Button(frame, text="Blue", command=lambda: change_color("blue"))
blue_button.pack(side=tk.LEFT, padx=20, pady=10)

root.mainloop()
