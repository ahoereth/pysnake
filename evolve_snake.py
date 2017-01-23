from multiprocessing import Pool

import numpy as np
import tensorflow as tf

from snake import Snake

BOARD_SIZE = 4
DIRECTIONS = 4

MAX_GAME_STEPS = 100
MAX_INDIVIDUALS = 5
MAX_GENERATIONS = 1  # 00

NUM_POOLS = 4

PRECISION = tf.float64


class EvolveSnake:
    def __init__(self, snake, weights=None):
        """Initializes a TensorFlow graph to play snake."""
        self.snake = snake
        self.weights = weights
        if self.weights is None:
            size = self.snake.board.size
            self.weights = np.random.random((size, DIRECTIONS))

    def init_network(self):
        board = tf.placeholder(PRECISION, [None, self.snake.board.size])
        w1 = tf.Variable(self.weights, name='weights')
        b = tf.Variable(tf.ones([DIRECTIONS], dtype=PRECISION))
        output_layer = tf.nn.relu_layer(board, w1, b, name='output')
        action = tf.argmax(tf.nn.softmax(output_layer), 1)
        return board, action

    def __call__(self):
        # run session here and return results
        board, action = self.init_network()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for i in range(MAX_GAME_STEPS):
                act = sess.run(action, {board: [self.snake.board.flatten()]})
                if self.snake.step(act) == -1:
                    break

        return self.weights, self.snake.highscore, self.snake.steps


def play_snake(snake):
    """Plays snake with the given snake."""
    return snake()


class SnakeTrainer:
    def generate_snakes(self, number):
        return [EvolveSnake(Snake(BOARD_SIZE)) for i in range(number)]


if __name__ == '__main__':
    trainer = SnakeTrainer()
    tensor_snakes = trainer.generate_snakes(MAX_INDIVIDUALS)

    for i in range(MAX_GENERATIONS):
        with Pool(NUM_POOLS) as p:
            results = p.map(play_snake, tensor_snakes)

    for w, h, s in results:
        print(h, s)
