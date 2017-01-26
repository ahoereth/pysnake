import numpy as np
from numpy import random as rand


SNAKE = 2
FOOD = 1
BG = 0


class Snake:
    """A snake simulation."""

    def __init__(self, size=4):
        """Initializes a random square board of size size.

        Generates one snake and one food position.

        Args:
            size: the size of the board sides.
        """
        self.size = size
        self.minsnakelen = 5

        self.steps = 0
        self.board = np.ones((self.size, self.size)) * BG
        self.body = [rand.randint(0, len(self.board) - 1)]
        self.board.flat[self.body[0]] = SNAKE
        self.board.flat[rand.choice(np.flatnonzero(self.board == BG))] = FOOD
        self.dir = np.random.randint(0, 4)

    def step(self, direction=0):
        """Moves the snake into the specified direction.

        Appends a new snake marker in front and removes the last,
        if the snake would be longer than its maximum length.
        If food is consumed, the maximum length is increased by 1 and
        a new food is spawned.

        Checks if the snake wins or bites itself.

        Args:
            direction: The direction to move to as int.
                       1: up, 2: down, 3: left, 4: right

        Returns:
             1 if winning
            -1 if losing
             0 else
        """
        direction = abs(direction) % 4
        if direction ^ 1 != self.dir:
            self.dir = direction

        r, c = self.head
        if self.dir & 2:  # horizontal
            c = (c + (self.dir & 1) * 2 - 1) % self.size
        else:  # vertical
            r = (r + (self.dir & 1) * 2 - 1) % self.size

        head = np.ravel_multi_index((r, c), self.board.shape)
        if self.board.flat[head] == SNAKE:  # died
            return -1

        if self.board.flat[head] == FOOD:  # caught a special dot
            # check win condition
            new_location = np.flatnonzero(self.board == BG)
            if(len(new_location) == 0):
                return 1
            food = rand.choice(new_location)
            self.board.flat[food] = FOOD
        elif len(self.body) >= self.minsnakelen:  # otherwise cut off tail
            self.board.flat[self.body.pop(0)] = BG

        self.steps += 1
        self.body.append(head)
        self.board.flat[head] = SNAKE
        return 0

    @property
    def highscore(self):
        """The highscore is the number of foods eaten."""
        return len(self.body) - self.minsnakelen

    @property
    def head(self, rc=True):
        """Returns the current head position of the snake.

        Args:
            rc: True to return row and column (2D) coordinates.

        Return:
            2D coordinates if rc is True, otherwise 1D coordinate
        """
        head = self.body[len(self.body) - 1]
        return np.unravel_index(head, self.board.shape) if rc else head

    @property
    def food(self, rc=True):
        """Returns the position of the current food item.

        Args:
            rc: True to return row and column (2D) coordinates.

        Return:
            2D coordinates if rc is True, otherwise 1D coordinate
        """
        food = np.flatnonzero(self.body == FOOD)[0]
        return np.unravel_index(food, self.board.shape) if rc else food
