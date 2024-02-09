from tkinter import *

window = Tk()

label1 = Label(window, text="Label 1")
label2 = Label(window, text="Label 2")
label3 = Label(window, text="Label 3")
label4 = Label(window, text="Label 4")

label1.grid(row=0, column=0, sticky="e")
label2.grid(row=0, column=1, sticky="w")
label3.grid(row=1, column=0, sticky="e", padx=(10, 10))
label4.grid(row=1, column=1, sticky="w", padx=(10, 10))



window.mainloop()
