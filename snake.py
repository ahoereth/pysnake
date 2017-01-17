import numpy as np
from numpy import random as rand
import matplotlib
matplotlib.use('tkAgg')
from matplotlib import pyplot as plt
import sys


actions = {
    'UP': 0b00,  # 00
    'DOWN': 0b01,  # 01
    'LEFT': 0b10,  # 10
    'RIGHT': 0b11,  # 11
}


SNAKE = 2
FOOD = 1
BG = 0


class Snake:
    def __init__(self, size=8, onstep=lambda x: x):
        self.onstep = onstep
        self.size = size
        self.minsnakelen = 5

        self.steps = 0
        self.board = np.ones((self.size, self.size)) * BG
        self.body = [rand.randint(0, len(self.board) - 1)]
        self.board.flat[self.body[0]] = SNAKE
        self.board.flat[rand.choice(np.flatnonzero(self.board == BG))] = FOOD
        self.dir = list(actions.values())[np.random.randint(0, 4)]
        print(self.dir)

    def step(self, direction=None):
        if direction in actions.values() and direction ^ 1 != self.dir:
            self.dir = direction
        else:
            direction = self.dir

        r, c = self.get_head()
        if direction & 2:  # horizontal
            c = (c + (self.dir & 1) * 2 - 1) % self.size
        else:  # vertical
            r = (r + (self.dir & 1) * 2 - 1) % self.size

        head = np.ravel_multi_index((r, c), self.board.shape)
        if self.board.flat[head] == SNAKE:  # died
            return -1

        if self.board.flat[head] == FOOD:  # caught a special dot
            food = rand.choice(np.flatnonzero(self.board == BG))
            self.board.flat[food] = FOOD
        elif len(self.body) > self.minsnakelen:  # otherwise cut of tail
            self.board.flat[self.body.pop(0)] = BG

        self.steps += 1
        self.body.append(head)
        self.board.flat[head] = SNAKE
        self.onstep(board=self.board)
        return self.steps

    def get_highscore(self):
        return len(self.body) - self.minsnakelen

    def get_head(self, rc=True):
        head = self.body[len(self.body) - 1]
        return np.unravel_index(head, self.board.shape) if rc else head

    def get_food(self, rc=True):
        food = np.flatnonzero(self.body == FOOD)[0]
        return np.unravel_index(food, self.board.shape) if rc else food


class SnakeUI:
    def __init__(self):
        self.fig = plt.figure()
        plt.axis('off')
        self.img = None

    def redraw(self, board):
        if self.img is None:
            self.img = plt.imshow(board, cmap='gray_r', vmin=0, vmax=2)
        self.img.set_data(board)
        plt.draw()


class SnakeKeyboard:
    def __init__(self, snake, ui):
        self.snake = snake
        ui.fig.canvas.mpl_connect('key_press_event', self.keypress)
        plt.show()

    def keypress(self, event):
        key = (event.key or '').upper()
        if key == 'ESCAPE':
            sys.exit()
        if key in actions.keys():
            if self.snake.step(actions[key]) == -1:
                sys.exit()


if __name__ == '__main__':
    ui = SnakeUI()
    snake = Snake(size=4, onstep=ui.redraw)
    ui.redraw(snake.board)
    SnakeKeyboard(snake, ui)
