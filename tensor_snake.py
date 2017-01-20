from multiprocessing import Pool

import numpy as np
import tensorflow as tf

from snake import Snake

MAX_GAME_STEPS = 100
MAX_INDIVIDUALS = 5
MAX_GENERATIONS = 1  # 00
NUM_POOLS = 4


class TensorSnake:

    def __init__(self, snake, weights):
        """Initializes a TensorFlow graph to play snake."""
        self.snake = snake
        self.weights = weights

    def init_network(self):

        board = tf.placeholder(tf.float32, [None,self.snake.board.size])
        # build network here
        w1 = tf.Variable(self.weights, name='weights')
        b = tf.Variable(tf.ones([4]))
        output_layer = tf.nn.relu_layer(board, w1, b, name='output')
        action = tf.argmax(tf.nn.softmax(output_layer), 1)
        return board, action

    def __call__(self):
        # run session here and return results
        board, action = self.init_network()
        with tf.Session() as sess:
            sess.run(tf.initialize_all_variables())
            for i in range(MAX_GAME_STEPS):
                my_action = sess.run(action, feed_dict={
                                                board: [self.snake.board.flatten().astype(np.float32)]
                                             })
                if self.snake.step(my_action) == -1:
                    break

        return self.weights, self.snake.highscore, self.snake.steps


def play_snake(snake):
    """Plays snake with the given snake."""
    return snake()


class SnakeTrainer:
    def generate_snakes(self, number):
        return [TensorSnake(Snake(), np.random.random([16,4]).astype(np.float32))
                for i in range(number)]


if __name__ == '__main__':
    trainer = SnakeTrainer()
    tensor_snakes = trainer.generate_snakes(MAX_INDIVIDUALS)

    for i in range(MAX_GENERATIONS):
        with Pool(NUM_POOLS) as p:
            results = p.map(play_snake, tensor_snakes)

    for w, h, s in results:
        print(h, s)
