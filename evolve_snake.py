import itertools
import collections
import sys
import pickle
from os import path
from multiprocessing import Pool
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # hacky way for tensorflow to ignore the fact that I didn't build it locally

import tensorflow as tf
import numpy as np

from matplotlib import pyplot as plt

from snake import Snake
from snake_ui import SnakeUI


MAX_GAME_STEPS = 100
MAX_INDIVIDUALS = 20
MAX_GENERATIONS = 2  # 00

HIDDEN = 8
LAYERS = [Snake.size**2 + 1, HIDDEN, Snake.directions]

MUTATIONRATE = 0.001
KEEP = 10  # The permutation of this is also kept in the 'children'

NUM_POOLS = 4
PRECISION = tf.float64

CKPTNAME = 'evo_snake-'


class EvolveSnake:
    def __init__(self, snake, weights=None):
        """Initializes a TensorFlow graph to play snake."""
        self.snake = snake
        self.layers = LAYERS
        self.weights = weights
        if self.weights is None:
            self.weights = [np.random.random((size, self.layers[i + 1]))
                            for i, size in enumerate(self.layers[:-1])]

    def init_network(self):
        board = tf.placeholder(PRECISION, (None, self.layers[0]))
        w1 = tf.Variable(self.weights[0], name='hidden_weights')
        h1 = tf.nn.relu(tf.matmul(board, w1), name='hidden_layer')
        w2 = tf.Variable(self.weights[1], name='output_weights')
        output_layer = tf.nn.relu(tf.matmul(h1, w2), name='output')
        action = tf.argmax(tf.nn.softmax(output_layer), 1)
        return board, action

    def get_action(self, board):
        board, action = self.init_network()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            act = sess.run(action, {board: [np.append(self.snake.board.flat, 1)]})

        return act

    def __call__(self):
        # run session here and return results
        board, action = self.init_network()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for i in range(MAX_GAME_STEPS):
                act = sess.run(action,
                               {board: [np.append(self.snake.board.flat, 1)]})
                if self.snake.step(act) == -1:
                    break

        return self.weights, self.snake.highscore, self.snake.steps


class SnakeTrainer:
    def generate_snakes(self, number_or_weights):
        """
        This function generates Snakes with random or predefined weights.

        Args:
            number_or_weights: if an integer argument is given, this number of
                               new Snakes is generated
                               if a list of weights is given, those are used to
                               initialize the network

        Returns:
            the generated Snakes
        """
        try:
            snakes = [EvolveSnake(Snake(), weights)
                      for weights in number_or_weights]
        except TypeError:
            snakes = [EvolveSnake(Snake())
                      for i in range(number_or_weights)]
        return snakes

    def flatten(l):
        """
        Flattens a list containing several sublists into a flat list.

        Returns:
            a flattened representation of the given list
        """
        for el in l:
            if isinstance(el, collections.Iterable) and \
                    not isinstance(el, (str, bytes)):
                yield from SnakeTrainer.flatten(el)
            else:
                yield el

    def get_offsprings(self, parents):
        """
        Generates the offspring from two controller Networks.

        Args:
            parents: The parents which weight values are being used to generate
                     the offspring

        Returns:
            the generated offspring networks
        """
        babies = []
        for mum, dad in itertools.combinations(parents, 2):

            flat_mum = list(SnakeTrainer.flatten(mum))
            flat_dad = list(SnakeTrainer.flatten(dad))

            baby_flat = []
            for i in range(0, len(flat_mum)):
                # Step 1: Random Crossover
                # choose with equal probability the weight for the child from
                # mum and dad
                baby_flat.append(flat_mum[i] if np.random.random_sample() < 0.5
                                 else flat_dad[i])
                # Step 2: Mutation
                if np.random.random_sample() < MUTATIONRATE:
                    baby_flat[i] = np.random.random_sample()

            # reshaping back into layered form
            start_w = 0
            baby = []
            for i in range(len(mum)):
                layer = np.array(baby_flat)[start_w: start_w +
                                            mum[i].size].reshape(mum[i].shape)
                start_w = start_w + mum[i].size
                baby.append(layer)

            babies.append(baby)
        return babies

    def build_next_generation(self, results):
        """
        Builds the next generation from the previous training results.

        :param results: the individuals and performances from the prior run
        :return: a new generation to be used
        """
        # sort for highscore per step
        ranked_individuals = sorted(results, key=lambda x: x[1] / x[2] if x[2] is not 0 else x[1], reverse=True)

        best_individuals = [i[0] for i in ranked_individuals[0:KEEP]]
        offsprings = self.get_offsprings(best_individuals)
        spawning = MAX_INDIVIDUALS - KEEP - len(offsprings)

        new = []
        if (spawning > 0):
            new = self.generate_snakes(spawning)

        new_gen = self.generate_snakes(best_individuals + offsprings) + new

        return new_gen


def save_snake(results, keep=5, generation=0):
    """
    Saves Snakes from the evolution run to a given location.

    :param results: results in the form of [weights, highscore, step]
    :param keep: the number of snakes to be saved
    :param generation: the generation count to be used (default = 0)
    :return: Nothing! But saves the best x snakes in a file ;)
    """
    ranked_individuals = sorted(results, key=lambda x: x[1] / x[2], reverse=True)
    # only save the best performing snake
    with open(CKPTNAME+str(generation)+'.np', 'ab') as f:
        pickle.dump(np.asarray(ranked_individuals[0:keep]), f)


def load_snake(file):
    """
    Loads in the given file and uses numpy to convert it to an array
    :param file: the file to be loaded
    :return: an array containing the files input
    """
    return pickle.load(open(file, 'rb'))


def play_snake(snake):
    """Plays snake with the given snake."""
    return snake()


def replay_snake(snake_file, individual=0):
    """
    Plays snake with the given snake.
    :param snake_file the file from which the snake should be loaded
    :param individual the individual from the file
    """

    weights_flat = np.array(load_snake(snake_file)[individual])
    weights = []

    for i in range(weights_flat.shape[0] - 1):
        print(i)
        print(weights_flat[individual][i])
        layer = weights_flat[individual][i]
        weights.append(layer)

    player = EvolveSnake(Snake(), weights)
    game = Snake()

    def step():
        ret = game.step(player.get_action(game.board))
        if ret:
            print('You {}'.format('win' if ret > 0 else 'lose'))
            return 0

    ui = SnakeUI(game)
    timer = ui.fig.canvas.new_timer(500, [(step, [], {})])
    timer.start()
    plt.show()


def main(args):
    """
    Main method that can either train or replay a snake with evolution.

    :param args: define whether it is training or replay
    :return: Nothing
    """
    if len(args) > 0 and args[0] == 'train':
        trainer = SnakeTrainer()
        tensor_snakes = trainer.generate_snakes(MAX_INDIVIDUALS)

        for i in range(MAX_GENERATIONS):
            with Pool(NUM_POOLS) as p:
                results = p.map(play_snake, tensor_snakes)
            tensor_snakes = trainer.build_next_generation(results)

        for w, h, s in results:
            print(h, s)

        save_snake(results, 5)

    elif len(args) > 0 and args[0] == 'play':

        if len(args) > 1 and args[1].lower() != 'latest':
            file = path.realpath(args[1])
        else:
            file = path.realpath('evo_snake-0.np')

        replay_snake(file)
    else:
        print('Please provide either the `train` or `play` positional arguments.')


if __name__ == '__main__':
    main(sys.argv[1:])
