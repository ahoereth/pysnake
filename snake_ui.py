import atexit
import itertools
import sys

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation as animation
from matplotlib import image as image
from matplotlib.animation import FFMpegFileWriter as Writer


class SnakeUI(animation.TimedAnimation):
    def __init__(self, snake):
        """Initializes the snake visualization.

        Args:
            snake: A Snake simulation instance.
        """
        self.snake = snake
        self.fig = plt.figure(figsize=(4, 4))
        axes = self.fig.add_subplot(1, 1, 1)

        axes.set_xlim([-0.5, self.snake.board.shape[1] - 0.5])
        axes.set_ylim([self.snake.board.shape[0] - 0.5, -0.5])
        axes.set_aspect('equal')
        axes.get_xaxis().set_visible(False)
        axes.get_yaxis().set_visible(False)

        axesImage = image.AxesImage(axes, cmap='gray_r', interpolation='none')
        axesImage.set_clim(np.min(self.snake.board), np.max(self.snake.board))
        self.img = axes.add_image(axesImage)
        self.img.set_data(self.snake.board)

        self.save_movie = '--video' in sys.argv
        if self.save_movie:
            try:
                name = sys.argv[sys.argv.index('--video') + 1]
            except IndexError:
                name = 'pysnake.mp4'

            self.writer = Writer()
            self.writer.setup(self.fig, name, 72)

            atexit.register(self.finish)

        super().__init__(self.fig, interval=100 if not self.save_movie else 50,
                         blit=True)

    def new_frame_seq(self):
        """Counts frames infinitely."""
        return itertools.count()

    def _init_draw(self):
        pass

    def _draw_frame(self, frame_no):
        """Updates the current board."""
        self.img.set_data(self.snake.board)
        if self.save_movie:
            self.writer.grab_frame()

    def finish(self):
        if self.save_movie:
            self.writer.finish()
