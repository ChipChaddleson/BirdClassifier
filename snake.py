import tkinter as tk
import random

class SnakeGame:
    def __init__(self, master):
        self.master = master
        self.master.title("Snake Game")
        self.master.geometry("400x400")
        self.canvas = tk.Canvas(self.master, bg="black", width=400, height=400)
        self.canvas.pack()
        
        self.snake = [(100, 100), (90, 100), (80, 100)]
        self.direction = "Right"
        self.food = self.create_food()
        
        self.master.bind("<Key>", self.change_direction)
        self.move()

    def create_food(self):
        x = random.randint(0, 39) * 10
        y = random.randint(0, 39) * 10
        self.food_item = self.canvas.create_rectangle(x, y, x+10, y+10, fill="red")
        return x, y

    def move(self):
        head = self.snake[0]
        if self.direction == "Up":
            new_head = (head[0], head[1] - 10)
        elif self.direction == "Down":
            new_head = (head[0], head[1] + 10)
        elif self.direction == "Left":
            new_head = (head[0] - 10, head[1])
        elif self.direction == "Right":
            new_head = (head[0] + 10, head[1])

        self.snake.insert(0, new_head)

        if new_head == self.food:
            self.canvas.delete(self.food_item)
            self.food = self.create_food()
        else:
            tail = self.snake.pop()
            self.canvas.delete(self.canvas.find_withtag(tail))

        self.draw_snake()

        if self.check_collision():
            self.game_over()
        else:
            self.master.after(100, self.move)

    def draw_snake(self):
        self.canvas.delete("snake")
        for segment in self.snake:
            x, y = segment
            self.canvas.create_rectangle(x, y, x+10, y+10, fill="green", tag="snake")

    def check_collision(self):
        head = self.snake[0]
        if (
            head[0] < 0 or head[0] >= 400 or
            head[1] < 0 or head[1] >= 400 or
            head in self.snake[1:]
        ):
            return True
        return False
    def change_direction(self, event):
        new_direction = event.keysym
        if (
            (new_direction == "Up" and not self.direction == "Down") or
            (new_direction == "Down" and not self.direction == "Up") or
            (new_direction == "Left" and not self.direction == "Right") or
            (new_direction == "Right" and not self.direction == "Left")
        ):
            self.direction = new_direction

    def game_over(self):
        self.canvas.create_text(200, 200, text="Game Over!", fill="white", font=("Helvetica", 16, "bold"))

if __name__ == "__main__":
    root = tk.Tk()
    snake_game = SnakeGame(root)
    root.mainloop()
