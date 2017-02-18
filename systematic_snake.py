import sys

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
        self.first_turn = True
        self.timer = ui.fig.canvas.new_timer(300 if len(sys.argv) <= 1 else 50,
                                             [(self, [], {})])
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
            if '--store' in sys.argv:
                # writes the score into the output file and exits
                with open('systematic.csv', 'a') as f:
                    print(self.snake.highscore, self.snake.steps, sep=',',
                          file=f)
                sys.exit()
            return 0

    def calculate_step(self):
        """Calculates the next step.

        For even board sizes, the SystematicSnake moves in a hamilton cycle
        across the board.
        If one side is odd, there is still a hamiltonian cycle to follow, but
        for now the SystematicSnake will crash.
        If both sides are odd, there is no hamiltonian cycle available and the
        snake will crash eventually.

        On the first turn the snake might correct its direction to avoid
        running into walls because it can't move towards itself.

        Returns:
            The directional value.
        """
        y, x = self.snake.head
        if self.first_turn:
            self.first_turn = False
            dir = self.calculate_first_step()
            if dir >= 0:
                return dir

        if x == 0:  # first col: down
            if y == self.max_y - 1:  # last row: right
                return 3
            return 1
        if y & 1:  # odd rows: right
            if x == self.max_x - 1:
                return 0
            return 3
        else:  # even rows: left
            if x == 1 and y > 0:
                return 0
            return 2

    def calculate_first_step(self):
        """Corrects the first movement, in case the initial snake
        direction is opposite to the optimal solution in cases where it
        does not resolve itself.

        Assumes the snake does not start on edges.

        In particular the problems to be solved are:
            odd rows with direction left
            even rows with direction right
        In both cases, the snake can just move up under the assumption
        mentioned above.

        Returns:
             1 if the snake should move up
            -1 otherwise
        """
        y, x = self.snake.head
        if (y & 1 and self.snake.dir == 2) or self.snake.dir == 3:
            return 0
        return -1


if __name__ == '__main__':
    snake = Snake()
    ui = SnakeUI(snake)
    SystematicSnake(snake, ui)
    plt.show()
