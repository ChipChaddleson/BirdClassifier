import tkinter as tk
from tkinter import messagebox

class TicTacToe:
    def __init__(self, root):
        self.root = root
        self.root.title("Tic-Tac-Toe")
        self.root.geometry("300x350")
        self.root.configure(bg="#333333")  # Dark background color

        self.current_player = "X"
        self.board = [""] * 9

        self.create_board_buttons()

    def create_board_buttons(self):
        for i in range(3):
            for j in range(3):
                button = tk.Button(self.root, text="", font=("Helvetica", 20),
                                   width=5, height=2, command=lambda row=i, col=j: self.on_click(row, col))
                button.grid(row=i, column=j, padx=5, pady=5)
                button.configure(bg="#444444", fg="#ffffff")  # Dark button colors

    def on_click(self, row, col):
        index = 3 * row + col

        if self.board[index] == "":
            self.board[index] = self.current_player
            self.update_button(row, col)

            if self.check_winner():
                messagebox.showinfo("Tic-Tac-Toe", f"Player {self.current_player} wins!")
                self.reset_game()
            elif "" not in self.board:
                messagebox.showinfo("Tic-Tac-Toe", "It's a draw!")
                self.reset_game()
            else:
                self.toggle_player()

    def update_button(self, row, col):
        index = 3 * row + col
        button = self.root.grid_slaves(row=row, column=col)[0]
        button["text"] = self.current_player

    def check_winner(self):
        # Check rows, columns, and diagonals for a win
        for i in range(3):
            if self.board[i] == self.board[i + 3] == self.board[i + 6] != "":
                return True
            if self.board[3 * i] == self.board[3 * i + 1] == self.board[3 * i + 2] != "":
                return True
        if self.board[0] == self.board[4] == self.board[8] != "":
            return True
        if self.board[2] == self.board[4] == self.board[6] != "":
            return True
        return False

    def toggle_player(self):
        self.current_player = "O" if self.current_player == "X" else "X"

    def reset_game(self):
        self.board = [""] * 9
        for button in self.root.grid_slaves():
            button["text"] = ""
        self.current_player = "X"

if __name__ == "__main__":
    root = tk.Tk()
    game = TicTacToe(root)
    root.mainloop()
