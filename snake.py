import numpy as np
from numpy import random as rand
from matplotlib import animation, pyplot as plt
import sys
import threading


dirs = {
    'UP': 0b00,  # 00
    'DOWN': 0b01,  # 01
    'LEFT': 0b10,  # 10
    'RIGHT': 0b11,  # 11
}


MINSNAKELEN = 5
SNAKE = 2
FOOD = 1
BG = 0


class Snake:
    def __init__(self, figure, speed=50, viz=True, listener=lambda: None):
        self.listener = listener
        self.fig = figure
        self.size = 32
        self.start()
        self.viz = viz
        self.speed = speed
        if viz:
            self.fig.canvas.mpl_connect('key_press_event', self.keypress)
            self.img = plt.imshow(self.board, cmap='gray_r', vmin=0, vmax=2)
            self.anim = animation.FuncAnimation(self.fig, self.step,
                                                interval=1100//speed,
                                                blit=True)
        else:
            thread = threading.Timer(1/self.speed, self.step)
            thread.start()


    def start(self):
        self.steps = 0
        self.board = np.ones((self.size, self.size)) * BG
        self.board.flat[rand.randint(0, len(self.board) - 1)] = FOOD
        self.body = [rand.randint(0, len(self.board) - 1)]
        self.dir = list(dirs.values())[np.random.randint(0, 4)]

    def step(self, **args):
        self.steps += 1
        r, c = self.get_head(rc=True)
        if self.dir & 2:  # horizontal
            c = (c + (self.dir & 1) * 2 - 1) % self.size
        else:  # vertical
            r = (r + (self.dir & 1) * 2 - 1) % self.size
        head = np.ravel_multi_index((r, c), self.board.shape)
        if self.board.flat[head] == SNAKE:
            self.event('lost')
            self.start()
        if self.board.flat[head] == FOOD:  # caught a special dot
            food = rand.choice(np.flatnonzero(self.board == BG))
            self.board.flat[food] = FOOD
        elif len(self.body) > MINSNAKELEN:  # otherwise cut of tail
            self.board.flat[self.body.pop(0)] = BG
        self.body.append(head)
        self.board.flat[head] = SNAKE
        self.event('highscore')
        if self.viz:
            self.img.set_data(self.board)
            return self.img,
        else:
            thread = threading.Timer(1/self.speed, self.step)
            thread.start()

    def get_highscore(self):
        return len(self.body) - MINSNAKELEN

    def get_head(self, rc=False):
        head = self.body[len(self.body) - 1]
        return np.unravel_index(head, self.board.shape) if rc else head

    def get_food(self, rc=False):
        food = np.flatnonzero(self.body == FOOD)[0]
        return np.unravel_index(food, self.board.shape) if rc else food

    def set_direction(self, direction):
        if direction in dirs.values() and direction ^ 1 != self.dir:
            self.dir = direction

    def keypress(self, event):
        key = (event.key or '').upper()
        if key == 'ESCAPE':
            sys.exit()
        if key in dirs.keys():
            self.set_direction(dirs[key])

    def event(self, name):
        self.listener(event=name, highscore=self.get_highscore(),
                      direction=self.dir, board=self.board,
                      head=self.get_head(), steps=self.steps)


if __name__ == '__main__':
    def listen(event, highscore, head, direction, **args):
        sys.stdout.write('\r%s | %s | %s' % (event, highscore, direction))
        if event == 'lost':
            sys.stdout.write('\n')
        sys.stdout.flush()

    fig = plt.figure()
    size = 256
    plt.axis('off')
    snake = Snake(fig, listener=listen)
    plt.show()
