import sys

import matplotlib
matplotlib.use('tkAgg')
from matplotlib import pyplot as plt  # noqa: E402

from snake import Snake  # noqa: E402
from snake_ui import SnakeUI  # noqa: E402


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
        """Handles keypresses."""
        key = (event.key or '').upper()
        if key == 'ESCAPE':
            sys.exit()
        if key in self.actions.keys():
            ret = self.snake.step(self.actions[key])
            if ret != 0:
                print('You {}!'.format('win' if ret > 0 else 'lose'))
                if '--store' in sys.argv:
                    # writes the score into the output file
                    try:
                        fname = 'participant{}.csv'.format(
                            sys.argv[sys.argv.index('--store') + 1])
                    except IndexError:
                        fname = 'participant_unspecified.csv'
                    with open(fname, 'a') as f:
                        print(self.snake.highscore, self.snake.steps, sep=',',
                              file=f)
                sys.exit()


if __name__ == '__main__':
    snake = Snake()
    ui = SnakeUI(snake)
    HumanSnake(snake, ui)
