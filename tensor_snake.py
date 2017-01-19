from multiprocessing import Pool

import tensorflow as tf

from snake import Snake

MAX_STEPS = 100


class TensorSnake:
    def __init__(self, weights, size=4):
        self.snake = Snake(size)
        self.weights = weights

    def init_network(self):
        board = tf.placeholder(tf.float32)  # Adjust dimension here
        action = tf.constant(2)
        # build network here
        return board, action

    def __call__(self):
        # run session here and return results
        board, action = self.init_network()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for i in range(10):
                my_action = sess.run(action, feed_dict={
                                                board: 0  # self.snake.board
                                             })
                self.snake.step(my_action)

        return self.weights, self.snake.highscore, self.snake.steps


def play_snake(snake):
    return snake()


if __name__ == '__main__':
    max_gens = 100  # how many generations do we want to train?
    max_ind = 6    # how many individuals do we have per generation?

    with Pool(5) as p:
        print(p.map(play_snake, [TensorSnake(i) for i in range(max_ind)]))
