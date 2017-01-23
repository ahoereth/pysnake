import itertools
from multiprocessing import Pool

import numpy as np
import tensorflow as tf

from snake import Snake


BOARD_SIZE = 4
DIRECTIONS = 4

MAX_GAME_STEPS = 100
MAX_INDIVIDUALS = 22
WINNER_RATIO = .5
MAX_GENERATIONS = 500  # 00

NUM_POOLS = 4

PRECISION = tf.float64


class EvolveSnake:
    def __init__(self, snake, weights=None):
        """Initializes a TensorFlow graph to play snake."""
        self.snake = snake
        self.weights = weights
        if self.weights is None:
            size = self.snake.board.size + 1  # for bias term
            self.weights = np.random.random((size, DIRECTIONS))

    def init_network(self):
        board = tf.placeholder(PRECISION, [None, self.weights.size/DIRECTIONS])
        w1 = tf.Variable(self.weights, name='weights')
        output_layer = tf.nn.relu(tf.matmul(board, w1), name='output')
        action = tf.argmax(tf.nn.softmax(output_layer), 1)
        return board, action

    def __call__(self):
        # run session here and return results
        board, action = self.init_network()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for i in range(MAX_GAME_STEPS):
                act = sess.run(action, {board: [np.append(self.snake.board.flatten(),1)]})
                if self.snake.step(act) == -1:
                    break

        return self.weights, self.snake.highscore, self.snake.steps


def play_snake(snake):
    """Plays snake with the given snake."""
    return snake()


class SnakeTrainer:
    def generate_snakes(self, number, weights = None):
        if weights is None:
            snakes = [EvolveSnake(Snake(BOARD_SIZE)) for i in range(number)]
        else:
            snakes = [EvolveSnake(Snake(BOARD_SIZE),weight) for weight in weights]
        return snakes


    def get_offsprings(self, parents):
        babies = []
        for a, b in itertools.combinations(parents, 2):
            baby = (a+b)/2
            babies.append(baby)
        return babies

    def build_next_generation(self, results):
        # sort for highscore per step
        keep = int(np.ceil(MAX_INDIVIDUALS*WINNER_RATIO))
        ranked_individuals = sorted(results, key=lambda x: x[1]/x[2], reverse=True)

        best_individuals = [i[0] for i in ranked_individuals[0:keep]]
        offsprings = self.get_offsprings(best_individuals)
        random_new = self.generate_snakes(MAX_INDIVIDUALS-keep-len(offsprings))

        new_gen = self.generate_snakes(0, best_individuals + offsprings) + random_new
        return new_gen





if __name__ == '__main__':
    trainer = SnakeTrainer()
    tensor_snakes = trainer.generate_snakes(MAX_INDIVIDUALS)

    for i in range(MAX_GENERATIONS):
        with Pool(NUM_POOLS) as p:
            results = p.map(play_snake, tensor_snakes)
        tensor_snakes = trainer.build_next_generation(results)

    for w, h, s in results:
        print(h, s)