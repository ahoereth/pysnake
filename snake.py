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
        self.head = (np.random.randint(0, self.size),
                     np.random.randint(0, self.size))
        self.body = [self.head]
        self.dir = list(dirs.values())[np.random.randint(0, 4)]

    def step(self, i):
        x, y = self.head
        if self.dir & 2:  # horizontal
            x = x + ((self.dir & 1) * 2 - 1 % self.size)
        else:  # vertical
            y = y + ((self.dir & 1) * 2 - 1 % self.size)
        if self.board[y, x] == 0:
            self.event('lost')
            self.start()
        self.head = (x, y)
        self.body.append(self.head)
        self.board[y, x] = 0
        if i % 2 == 0:  # cut of tail every second step
            x, y = self.body.pop(0)
            self.board[y, x] = 255
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
        self.listener(event=name, highscore=len(self.body),
                      board=self.board, head=self.body[len(self.body)-1])


if __name__ == '__main__':
    def listen(event, highscore, head, **args):
        sys.stdout.write('\r%s | %s | %s' % (event, highscore, head))
        if event == 'lost':
            sys.stdout.write('\n')
        sys.stdout.flush()

    fig = plt.figure()
    size = 256
    plt.axis('off')
    snake = Snake(fig, listener=listen)
    plt.show()
