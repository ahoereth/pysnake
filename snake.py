import numpy as np
from matplotlib import animation, pyplot as plt
import sys


dirs = {
    'UP': 0b00,  # 00
    'DOWN': 0b01,  # 01
    'LEFT': 0b10,  # 10
    'RIGHT': 0b11,  # 11
}


class Snake:
    def __init__(self, figure, speed=50, listener=lambda: None):
        self.listener = listener
        self.fig = figure
        self.size = 64
        self.fig.canvas.mpl_connect('key_press_event', self.input)
        self.start()
        self.img = plt.imshow(self.board, cmap='gray', vmin=0, vmax=2)
        self.anim = animation.FuncAnimation(self.fig, self.step,
                                            interval=1100//speed, blit=True)

    def start(self):
        self.board = np.ones((self.size, self.size))*255
        self.x = np.random.randint(0, self.size)
        self.y = np.random.randint(0, self.size)
        self.body = [(self.x, self.y)]
        self.dir = list(dirs.values())[np.random.randint(0, 4)]

    def step(self, i):
        if self.dir & 2:  # horizontal
            self.x = (self.x + ((self.dir & 1) * 2 - 1)) % self.size
        else:  # vertical
            self.y = (self.y + ((self.dir & 1) * 2 - 1)) % self.size
        if self.board[self.y, self.x] == 0:
            self.event('lost')
            self.start()
        self.body.append((self.x, self.y))
        self.board[self.y, self.x] = 0
        if i % 3 == 0:
            tailx, taily = self.body.pop(0)
            self.board[taily, tailx] = 255
        self.img.set_data(self.board)
        self.event('highscore')
        return self.img,

    def input(self, event):
        key = event.key.upper()
        if key == 'ESCAPE':
            sys.exit()
        if key in dirs.keys():
            direction = dirs[key]
            if direction ^ 1 != self.dir:
                self.dir = direction

    def event(self, name):
        self.listener(event=name, highscore=len(self.body), board=self.board)


if __name__ == '__main__':
    def listen(event, highscore):
        print(event, highscore)
    fig = plt.figure()
    size = 256
    plt.axis('off')
    snake = Snake(fig, listener=listen)
    plt.show()
