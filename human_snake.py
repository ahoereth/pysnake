import itertools
import sys

import numpy as np

import matplotlib
matplotlib.use('tkAgg')
from matplotlib import pyplot as plt  # noqa: E402
from matplotlib import animation as animation  # noqa: E402
from matplotlib import image as image  # noqa: E402

from snake import Snake  # noqa: E402


class HumanSnake:
    """
    This class can be used to control the snake via keyboard.
    """

    actions = {
        'UP': 0b00,  # 00
        'DOWN': 0b01,  # 01
        'LEFT': 0b10,  # 10
        'RIGHT': 0b11,  # 11
    }

    def __init__(self, snake, ui):
        self.snake = snake
        ui.fig.canvas.mpl_connect('key_press_event', self.keypress)
        plt.show()

    def keypress(self, event):
        key = (event.key or '').upper()
        if key == 'ESCAPE':
            sys.exit()
        if key in self.actions.keys():
            ret = self.snake.step(self.actions[key])
            if ret != 0:
                print('You {}!'.format('win' if ret > 0 else 'lose'))
                sys.exit()


class SnakeUI(animation.TimedAnimation):
    def __init__(self, snake):
        self.snake = snake

        self.fig = plt.figure()
        axes = self.fig.add_subplot(1, 1, 1)
        axes.set_xlim([-0.5, self.snake.board.shape[1] - 0.5])
        axes.set_ylim([self.snake.board.shape[0] - 0.5, -0.5])
        axes.set_aspect('equal')
        axes.axis('off')

        axesImage = image.AxesImage(axes, cmap='gray_r', interpolation='none')
        axesImage.set_clim(np.min(self.snake.board), np.max(self.snake.board))
        self.img = axes.add_image(axesImage)
        self.img.set_data(self.snake.board)

        super().__init__(self.fig, interval=50, blit=True)

    def new_frame_seq(self):
        return itertools.count()

    def _init_draw(self):
        pass

    def _draw_frame(self, frame_no):
        self.img.set_data(self.snake.board)


if __name__ == '__main__':
    snake = Snake(size=4)
    ui = SnakeUI(snake)
    HumanSnake(snake, ui)
