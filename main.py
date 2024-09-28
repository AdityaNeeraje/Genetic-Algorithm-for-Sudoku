from sudoku_solver import SudokuSolver
from tkinter import *


# matrix = [[0,8,0,0,0,0,0,3,0],[2,0,0,6,0,7,0,0,1],[0,0,0,0,0,0,0,0,0],[0,6,0,2,0,1,0,7,0],[5,0,0,0,0,0,0,0,3],[9,0,0,7,0,5,0,0,8],[4,0,1,3,0,9,7,0,6],[0,2,0,0,0,0,0,1,0],[8,0,3,1,0,6,5,0,9]]
# matrix = ["000000000", "009000000", "000000000", "000000000", "000000000", "000000000", "000000000", "000000000", "000000000"]
# matrix = ["020100070", "000302000", "001080040", "900010007", "008060050", "000000000", "080036009", "005070006", "002000003"]
# matrix = [[int(char) for char in list(s)] for s in matrix]
# for row in matrix:
#     if len(row) != 9:
#         raise TypeError("Input is not a valid sudoku.")
# solver = SudokuSolver(matrix)
# if solver.matrix:
#     for row in solver.matrix:
#         print(row)
# else:
#     print("Sudoku is not solvable")

def check_entry(entry, number):
    global elements
    stringvar = elements[number].get()
    if len(stringvar) == 2:
        if entry.index(INSERT) == 1:
            if stringvar[0].isdigit():
                entry.delete(INSERT, END)
            else:
                entry.delete(0, INSERT)
        else:
            if stringvar[1].isdigit():
                entry.delete(0, 1)
            else:
                entry.delete(1, END)
    elif not elements[number].get().isdigit():
        position = entry.index(INSERT)
        if position > 0:
            entry.delete(position-1, position)


def move_to_next_entry(entry_num, direction=None):
    global boxes
    new_number = -1
    if (direction is None and entry_num != 80) or (direction == "Right" and entry_num % 9 != 8):
        new_number = entry_num+1
    elif direction == "Left" and entry_num % 9 != 0:
        new_number = entry_num-1
    elif direction == "Up" and int(entry_num/9) != 0:
        new_number = entry_num - 9
    elif direction == "Down" and int(entry_num/9) != 8:
        new_number = entry_num + 9
    if new_number != -1:
        boxes[new_number].focus()
        boxes[new_number].icursor(END)


def solve():
    global elements, submit_button
    submit_button["state"] = DISABLED
    matrix = [int(elements[i].get()) if elements[i].get() else 0 for i in range(81)]
    matrix = [[matrix[9*row + column] for column in range(9)] for row in range(9)]
    solver = SudokuSolver(matrix)
    if solver.matrix:
        for i in range(81):
            elements[i].set(str(solver.matrix[int(i/9)][i%9]))
            boxes[i]["state"] = DISABLED
    else:
        error_canvas.pack(fill=BOTH, expand=True)


def reset():
    global elements, submit_button
    for i in range(81):
        elements[i].set("")
        boxes[i]["state"] = NORMAL
    submit_button["state"] = NORMAL


def return_to_home():
    global error_canvas, boxes, submit_button
    error_canvas.pack_forget()
    for i in range(81):
        boxes[i]["state"] = NORMAL
    submit_button["state"] = NORMAL

root = Tk()
root.geometry("690x800")
root.title("Sudoku Solver")
root.config(bg="#326ba8")
elements = [StringVar() for i in range(81)]
boxes = [Entry(justify=CENTER, relief=GROOVE, textvariable=elements[i], width=4) for i in range(81)]

for i in range(81):
    entry = boxes[i]
    elements[i].trace("w", lambda name, index, mode, box=boxes[i], number=i: check_entry(box, number))
    entry.place(x=40 + 70 * (i % 9), y=40 + 70 * int(i / 9))
    entry.bind("<Return>", lambda event, number=i: move_to_next_entry(number))
    entry.bind("<Right>", lambda event, number=i, direction="Right": move_to_next_entry(number, direction))
    entry.bind("<Left>", lambda event, number=i, direction="Left": move_to_next_entry(number, direction))
    entry.bind("<Up>", lambda event, number=i, direction="Up": move_to_next_entry(number, direction))
    entry.bind("<Down>", lambda event, number=i, direction="Down": move_to_next_entry(number, direction))
    # entry.bind("<d>", lambda event, number=i, direction="Right": move_to_next_entry(number, direction))
    # entry.bind("<a>", lambda event, number=i, direction="Left": move_to_next_entry(number, direction))
    # entry.bind("<w>", lambda event, number=i, direction="Up": move_to_next_entry(number, direction))
    # entry.bind("<s>", lambda event, number=i, direction="Down": move_to_next_entry(number, direction))
    if (3 * int(int((i - i % 9) / 9) / 3) + int((i % 9) / 3)) % 2 == 0:
        entry.config(bg="slate gray", disabledbackground="slate gray", disabledforeground="black", fg="lightcyan", font={"Arial", 8, "bold"})
    else:
        entry.config(bg="lavender blush", disabledbackground="lavender blush", disabledforeground="black", fg="purple4", font={"Arial", 8, "bold"})

submit_button = Button(text="Submit", command=solve, justify=CENTER, font=("Arial", 10, "bold"), fg="blue", bg="goldenrod1")
submit_button.pack(side=BOTTOM, anchor="e", pady=50, padx=20)
reset_button = Button(text="Reset", command=reset, justify=CENTER, font=("Arial", 10, "bold"), fg="blue", bg="goldenrod1")
reset_button.pack(side=BOTTOM, anchor="w", padx=20)

error_canvas = Canvas(bg="red", width=root.winfo_width(), height=root.winfo_height())
error_label = Label(error_canvas, text="The Sudoku you have entered is unsolvable. Please check the input for errors", font={"Arial", 24, "bold"}, bg="red", wraplength=600)
error_label.pack()
home_button = Button(error_canvas, text="Return to Home", bg="red", borderwidth=1, relief=GROOVE, fg="yellow", command=return_to_home, font={"Arial", 8, "bold"})
home_button.pack(side=BOTTOM, pady=200)

root.mainloop()
