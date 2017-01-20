from multiprocessing import Pool

import numpy as np
import tensorflow as tf

from snake import Snake

MAX_GAME_STEPS = 100
MAX_INDIVIDUALS = 20
MAX_GENERATIONS = 1  # 00
NUM_POOLS = 4


class TensorSnake:

    def __init__(self, snake, weights):
        """Initializes a TensorFlow graph to play snake."""
        self.snake = snake
        self.weights = weights

    def init_network(self):
        board = tf.placeholder(tf.float32, self.snake.board.shape)
        action = tf.constant(2)
        # build network here
        return board, action

    def __call__(self):
        # run session here and return results
        board, action = self.init_network()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for i in range(MAX_GAME_STEPS):
                my_action = sess.run(action, feed_dict={
                                                board: self.snake.board
                                             })
                self.snake.step(my_action)

        return self.weights, self.snake.highscore, self.snake.steps


def play_snake(snake):
    """Plays snake with the given snake."""
    return snake()


class SnakeTrainer:
    def generate_snakes(self, number):
        return [TensorSnake(Snake(), np.random.random(20))
                for i in range(number)]


if __name__ == '__main__':
    trainer = SnakeTrainer()
    tensor_snakes = trainer.generate_snakes(MAX_INDIVIDUALS)

    for i in range(MAX_GENERATIONS):
        with Pool(NUM_POOLS) as p:
            results = p.map(play_snake, tensor_snakes)

    for w, h, s in results:
        print(h, s)
