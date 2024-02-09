from tkinter import *
from tkinter import ttk
import colorsys

window = Tk()

nordBack = "#4C566A"
nordHigh = "#88C0D0"
nordDarkHigh = "#5E81AC"
nordVeryDark = "#2E3440"
hue = 0

def SubmitText(text):
    print(f"/{'='*len(text)}\\")
    print(f"|{text}|")
    print(f"\\{'='*len(text)}/")

f = Frame(
    window,
    bg=nordBack
)
f.grid(
    row=0,
    column=0,
    sticky="nsew"
)

lableBtnFrame = Frame(
    f,
    bg=nordBack
)
lableBtnFrame.grid(
    row=0,
    column=0,
    sticky="nsew"
)

disc = Label(
    lableBtnFrame,
    text="write some message below!!",
    background=nordBack,
    fg=nordHigh,
    width=20,
    anchor="w"
)
disc.grid(
    row=0,
    column=0,
    sticky="w"
)

btn = Button(
    lableBtnFrame,
    text="submit message",
    command=lambda: SubmitText(txtbox.get("1.0", "end-1c")),
    background=nordBack,
    foreground=nordHigh,
    activebackground=nordDarkHigh,
    borderwidth=1,
    relief="solid",
    padx=8,
)
btn.grid(
    row=0,
    column=1,
    sticky="e"
)

txtbox = Text(
    f,
    background=nordVeryDark,
    foreground="white"
)
txtbox.grid(
    row=1,
    column=0,
    sticky="nsew"
)

window.grid_rowconfigure(0, weight=1)
window.grid_columnconfigure(0, weight=1)

window.geometry("300x200+500+200")
window.title(string="GUI!")
window.mainloop()
