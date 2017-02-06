import sys
from os import path
from collections import deque, namedtuple
from datetime import datetime

import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use('tkAgg')
from matplotlib import pyplot as plt

from snake_ui import SnakeUI
from snake import Snake


BOARD_SIZE = 4
DIRECTIONS = 4

EPOCHS = 1000
Q_DECAY = .975
MAX_GAME_STEPS = 100

ACTIONS = 4
MEMORY_SIZE = 80
BATCHSIZE = 40


def conv_layer(x, shape, kernel, stride=1):
    weights = tf.Variable(tf.truncated_normal(kernel + shape, stddev=0.1))
    bias = tf.Variable(tf.ones((shape[1],))/100)
    conv = tf.nn.conv2d(x, weights, (1, stride, stride, 1), 'SAME')
    return tf.nn.relu(conv + bias)


def dens_layer(x, shape, act_func=lambda x: x):
    weights = tf.Variable(tf.truncated_normal(shape, stddev=0.1))
    bias = tf.Variable(tf.ones((shape[1],))/10)
    return act_func(tf.matmul(x, weights) + bias)


Q_Graph = namedtuple('Q_Graph', [
    'state', 'target_q', 'out_q', 'max_q', 'action', 'train',
])


class Q_Snake:
    def __init__(self, checkpoint=None):
        self.checkpoint = checkpoint

        self.start = datetime.now().strftime('%Y%m%d-%H%M%S')
        self.graph = self._model()
        self.session = tf.Session()
        self.saver = tf.train.Saver()

        self.session.run(tf.global_variables_initializer())
        if checkpoint is not None:
            self.saver.restore(self.session, checkpoint)


    def _model(self):
        state = tf.placeholder(tf.float32, (None, BOARD_SIZE, BOARD_SIZE))
        target_q = tf.placeholder(tf.float32, (None, 4))

        net = tf.reshape(state, (-1, BOARD_SIZE, BOARD_SIZE, 1))  # for conv
        net = net / tf.reduce_max(net)  # let input range from 0 to 1

        net = conv_layer(net, (1, 8), (4, 4))
        net = conv_layer(net, (8, 12), (3, 3))
        net = tf.reshape(net, (-1, BOARD_SIZE * BOARD_SIZE * 12))
        net = dens_layer(net, (BOARD_SIZE * BOARD_SIZE * 12, 1024), tf.nn.relu)

        out_q = dens_layer(net, (1024, 4), tf.nn.softsign)
        max_q = tf.reduce_max(out_q, axis=1)
        action = tf.argmax(out_q, axis=1)

        loss = tf.reduce_mean(tf.square(target_q - out_q))
        train = tf.train.RMSPropOptimizer(0.03).minimize(loss)

        return Q_Graph(state, target_q, out_q, max_q, action, train)


    def train(self, epochs=EPOCHS, random_action_probability=1):
        memory = deque([], MEMORY_SIZE)

        for epoch in range(1, epochs + 1):
            game = Snake(BOARD_SIZE, markhead=True)
            last_highscore = 0
            for _ in range(MAX_GAME_STEPS):
                state = np.copy(game.board)
                qs, = self.session.run(self.graph.out_q, {
                    self.graph.state: [state],
                })

                # Select an action a
                if np.random.rand(1) < random_action_probability:
                    act = np.random.randint(0, ACTIONS)
                else:
                    act = np.argmax(qs)

                # Carry out action
                game_state = game.step(act)
                state_new = np.copy(game.board)

                # Observe reward
                if game.highscore > last_highscore:
                    reward = 1
                    last_highscore = game.highscore
                else:
                    reward = game_state # -0.25 if snake_state == 0 else -1

                # Store in replay memory
                memory.append((state, act, reward, state_new))

                # Did we see enough moves to start learning?
                if len(memory) == memory.maxlen:
                    train_states = []
                    train_target_qs = []
                    samples = np.random.permutation(memory.maxlen)[:BATCHSIZE]
                    for i in samples:
                        state_old, action, reward, state_new = memory[i]
                        out_q, = self.session.run(self.graph.out_q, {
                            self.graph.state: [state_old],
                        })
                        max_q, = self.session.run(self.graph.max_q, {
                            self.graph.state: [state_new],
                        })

                        if reward == -1:  # terminal state
                            out_q[action] = -1
                        else:
                            out_q[action] = np.min([reward + (Q_DECAY * max_q), 1])

                        train_states.append(state_old)
                        train_target_qs.append(out_q)

                    self.session.run(self.graph.train, {
                        self.graph.state: train_states,
                        self.graph.target_q: train_target_qs,
                    })

                # Exit game loop if game ended.
                if game_state != 0:
                    break

            if epoch % 100 == 0:
                print('epoch {} of {}'.format(epoch, epochs))
                self.saver.save(self.session,
                    'q_snake-{}'.format(self.start),
                    global_step=epoch
                )

            if random_action_probability > 0.1:
                random_action_probability -= (1/epochs)

    def get_action(self, state):
        return self.session.run(self.graph.action, {
            self.graph.state: state,
        })[0]


def play(checkpoint):
    player = Q_Snake(checkpoint=path.realpath(checkpoint))
    game = Snake(BOARD_SIZE, markhead=True)

    def step():
        ret = game.step(player.get_action([game.board]))
        if ret:
            print('You {}'.format('win' if ret > 0 else 'lose'))
            return 0

    ui = SnakeUI(game)
    timer = ui.fig.canvas.new_timer(500, [(step, [], {})])
    timer.start()
    plt.show()


def train():
    player = Q_Snake()
    player.train()


if __name__ == '__main__':
    args = sys.argv[1:]
    if len(args) > 0:
        play(args[0])
    else:
        train()
