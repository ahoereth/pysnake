import matplotlib
matplotlib.use('tkAgg')
from matplotlib import pyplot as plt  # noqa: E402

from snake import Snake  # noqa: E402
from snake_ui import SnakeUI  # noqa: E402


class SystematicSnake:
    """
    This class defines a systematic snake player to evaluate a baseline.
    It always moves in a specific pattern and will eventually win.
    Currently it only supports square boards.
    """

    def __init__(self, snake, ui):
        self.snake = snake
        self.max_y, self.max_x = self.snake.board.shape
        self.timer = ui.fig.canvas.new_timer(500, [(self, [], {})])
        self.timer.start()

    def __call__(self):
        """
        Calculates the next step and applies it.

        Returns:
            0 if done else None, to stop the GUI timer when done
        """
        next_step = self.calculate_step()
        ret = self.snake.step(next_step)
        if ret:
            print('You {}'.format('win' if ret > 0 else 'lose'))
            return 0

    def calculate_step(self):
        """Calculates the next step.

        Reads the current snake head position and moves in a spiraling pattern
        around the board:

        If at the main diagonal (x == y), change direction:
            even values: move right
            odd values: move left
        If above diagonal (y < x):
            even rows (y): move right, if at end move down
            odd rows: move left
        If below diagonal (y > x):
            even columns (x): move up
            odd columns: move down, if at end move right

        Returns:
            The directional value.
        """
        y, x = self.snake.head
        if y == x:  # diagonal: right on even, else down
            if y == self.max_y - 1:
                # exception: move right at bottom right
                return 3
            return 3 - 2 * (y & 1)
        if y < x:  # above diagonal: right on even row, else left
            if x == self.max_x - 1 and y & 1 == 0:
                # exception: move down at end of even row
                return 1
            return 3 - (y & 1)
        else:  # below diagonal: up on even column, else down
            if y == self.max_y - 1 and x & 1:
                # exception: move right at and of odd column
                return 3
            return 1 - (1 - (x & 1))


if __name__ == '__main__':
    snake = Snake(10)
    ui = SnakeUI(snake)
    SystematicSnake(snake, ui)
    plt.show()
