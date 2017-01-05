import numpy as np
from matplotlib import animation, pyplot as plt
import sys


class Snake:
    def __init__(self, figure, listener=lambda: None):
        self.listener = listener
        self.fig = figure
        self.size = 64
        self.fig.canvas.mpl_connect('key_press_event', self.input)
        self.start()
        self.img = plt.imshow(self.board, cmap='gray', vmin=0, vmax=2)
        self.anim = animation.FuncAnimation(self.fig, self.step,
                                            interval=11, blit=True)

    def start(self):
        self.body = []
        self.board = np.ones((self.size, self.size))*255
        self.x = np.random.randint(0, self.size)
        self.y = np.random.randint(0, self.size)
        self.dir = ('up', 'down', 'left', 'right')[np.random.randint(0, 4)]

    def step(self, i):
        getattr(self, self.dir)()
        if self.board[self.y, self.x] == 0:
            self.event('lost')
            self.start()
        self.body.append((self.y, self.x))
        self.board[self.y, self.x] = 0
        if i % 3 == 0:
            taily, tailx = self.body.pop(0)
            self.board[taily, tailx] = 255
        self.img.set_data(self.board)
        self.event('highscore')
        return self.img,

    def input(self, event):
        # print('you pressed', event.key)
        if event.key == 'escape':
            self.escape()
        if event.key in ('up', 'down', 'left', 'right'):
            if (
                (self.dir == 'up' and event.key == 'down') or
                (self.dir == 'down' and event.key == 'up') or
                (self.dir == 'right' and event.key == 'left') or
                (self.dir == 'left' and event.key == 'right')
            ):
                return
            self.dir = event.key

    def up(self):
        self.y = (self.y - 1) % self.size

    def down(self):
        self.y = (self.y + 1) % self.size

    def left(self):
        self.x = (self.x - 1) % self.size

    def right(self):
        self.x = (self.x + 1) % self.size

    def escape(self):
        sys.exit()

    def event(self, name):
        self.listener(event=name, highscore=len(self.body), board=self.board)


if __name__ == '__main__':
    def listen(event, highscore):
        print(event, highscore)
    fig = plt.figure()
    size = 256
    plt.axis('off')
    snake = Snake(fig, listen)
    plt.show()
