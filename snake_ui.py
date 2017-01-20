import itertools

import numpy as np
from matplotlib import pyplot as plt  # noqa: E402
from matplotlib import animation as animation  # noqa: E402
from matplotlib import image as image  # noqa: E402


class SnakeUI(animation.TimedAnimation):
    def __init__(self, snake):
        """Initializes the snake visualization.

        Args:
            snake: A Snake simulation instance.
        """
        self.snake = snake

        self.fig = plt.figure()
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

        super().__init__(self.fig, interval=500, blit=True)

    def new_frame_seq(self):
        """Counts frames infinitely."""
        return itertools.count()

    def _init_draw(self):
        pass

    def _draw_frame(self, frame_no):
        """Updates the current board."""
        self.img.set_data(self.snake.board)
