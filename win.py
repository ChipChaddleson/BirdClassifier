import tkinter as tk
from tkinter import ttk

def convertFeetToMeters():
    try:
        feet = float(feetEntry.get())
        meters = feet * 0.3048
        metersLabel.config(text=f"{meters:.2f} meter")
    except ValueError:
        metersLabel.config(text="invalid input")

root = tk.Tk()
root.title("feet to meters")

frame = ttk.Frame(
    root, 
    padding="20"
)
frame.grid(
    row=0, 
    column=0, 
    sticky=('NESW')
)

titleLabel = ttk.Label(
    frame, 
    text="feet to meters"
)

titleLabel.grid(
    row=0, 
    column=0, 
    columnspan=3, 
    pady=5
)

equivLabel = ttk.Label(
    frame, 
    text="is equivalent to"
)

equivLabel.grid(
    row=1, 
    column=1, 
    padx=5
)

feetLabel = ttk.Label(
    frame, 
    text="feet:"
)

feetLabel.grid(
    row=2, 
    column=0, 
    sticky=tk.W, 
    padx=5, 
    pady=5
)

feetEntry = ttk.Entry(
    frame
)
feetEntry.grid(
    row=2, 
    column=1, 
    sticky=tk.W, 
    padx=5, 
    pady=5
)

metersLabel = ttk.Label(
    frame, 
    text=""
)

metersLabel.grid(
    row=2, 
    column=2, 
    sticky=tk.W, 
    padx=5, 
    pady=5
)


convertButton = ttk.Button(
    frame, 
    text="convert", 
    command=convertFeetToMeters
)

convertButton.grid(
    row=3, 
    column=0, 
    columnspan=3, 
    pady=10
)

root.mainloop()
