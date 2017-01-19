import numpy as np
from numpy import random as rand

SNAKE = 2
FOOD = 1
BG = 0


class Snake:

    def __init__(self, size=4):
        self.size = size
        self.minsnakelen = 5

        self.steps = 0
        self.board = np.ones((self.size, self.size)) * BG
        self.body = [rand.randint(0, len(self.board) - 1)]
        self.board.flat[self.body[0]] = SNAKE
        self.board.flat[rand.choice(np.flatnonzero(self.board == BG))] = FOOD
        self.dir = np.random.randint(0, 4)

    def step(self, direction=None):
        direction = abs(direction) % 4
        if direction ^ 1 != self.dir:
            self.dir = direction

        r, c = self.get_head()
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
        elif len(self.body) > self.minsnakelen:  # otherwise cut of tail
            self.board.flat[self.body.pop(0)] = BG

        self.steps += 1
        self.body.append(head)
        self.board.flat[head] = SNAKE
        return 0

    def get_highscore(self):
        return len(self.body) - self.minsnakelen

    def get_head(self, rc=True):
        head = self.body[len(self.body) - 1]
        return np.unravel_index(head, self.board.shape) if rc else head

    def get_food(self, rc=True):
        food = np.flatnonzero(self.body == FOOD)[0]
        return np.unravel_index(food, self.board.shape) if rc else food

