#! /usr/bin/env python3
import sys
from os import path
from glob import glob
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

EPOCHS = 10000
Q_DECAY = .975
MAX_NO_REWARD_STATES = BOARD_SIZE**2

ACTIONS = 4
MEMORY_SIZE = 80
BATCHSIZE = 40

CKPTNAME = 'q_snake-'


def conv_layer(x, shape, kernel, stride=1):
    weights = tf.Variable(tf.truncated_normal(kernel + shape, stddev=0.1))
    bias = tf.Variable(tf.ones((shape[1],))/100)
    conv = tf.nn.conv2d(x, weights, (1, stride, stride, 1), 'SAME')
    return tf.nn.relu(conv + bias)


def dens_layer(x, shape, act_func=None):
    weights = tf.Variable(tf.truncated_normal(shape, stddev=0.1))
    bias = tf.Variable(tf.ones((shape[1],))/10)
    if callable(act_func):
        return act_func(tf.matmul(x, weights) + bias)
    else:
        return tf.matmul(x, weights) + bias


Q_Graph = namedtuple('Q_Graph', [
    'x', 'target_q', 'out_q', 'max_q', 'action', 'train',
])


def flatten(l):
    return [j for i in l for j in i]


class Q_Snake:
    def __init__(self, checkpoint=None):
        self.start = datetime.now().strftime('%Y%m%d-%H%M%S')
        self.graph = self._model()
        self.session = tf.Session()
        self.saver = tf.train.Saver(keep_checkpoint_every_n_hours=.25,
                                    max_to_keep=20)

        self.session.run(tf.global_variables_initializer())
        if checkpoint is not None:
            self.saver.restore(self.session, checkpoint)

    def _model(self):
        x = tf.placeholder(tf.float32, (None, BOARD_SIZE, BOARD_SIZE))
        target_q = tf.placeholder(tf.float32, (None, 4))

        net = tf.reshape(x, (-1, BOARD_SIZE, BOARD_SIZE, 1))  # for conv
        net = net / tf.reduce_max(net)  # let input range from 0 to 1

        net = conv_layer(net, (1, 8), (4, 4))
        net = conv_layer(net, (8, 12), (3, 3))
        net = tf.reshape(net, (-1, BOARD_SIZE * BOARD_SIZE * 12))
        net = dens_layer(net, (BOARD_SIZE * BOARD_SIZE * 12, 1024), tf.nn.relu)

        out_q = dens_layer(net, (1024, 4))
        max_q = tf.reduce_max(out_q, axis=1)
        action = tf.argmax(out_q, axis=1)

        loss = tf.reduce_mean(tf.square(target_q - out_q))
        train = tf.train.RMSPropOptimizer(0.03).minimize(loss)

        return Q_Graph(x, target_q, out_q, max_q, action, train)

    def train(self, epochs=EPOCHS, random_action_probability=1, tpath='.'):
        memory = deque([], MEMORY_SIZE)

        for epoch in range(1, epochs + 1):
            game = Snake(BOARD_SIZE, markhead=True)
            no_reward_states = 0
            last_highscore = 0
            state = np.zeros(game.board.shape)

            while 42:
                state_old = state

                state = np.copy(game.board)
                qs, action = flatten(self.session.run(
                    [self.graph.out_q, self.graph.action],
                    {self.graph.x: [state * 2 - state_old]}
                ))

                # Maybe take a random action instead.
                # TODO: Move this to tensorflow?
                if np.random.rand(1) < random_action_probability:
                    action = np.random.randint(0, ACTIONS)

                # Carry out action
                game_status = game.step(action)
                state_new = np.copy(game.board)

                # Observe reward
                if game.highscore > last_highscore:
                    reward = 1
                    no_reward_states = 0
                    last_highscore = game.highscore
                else:
                    reward = game_status

                # Punish staying alive without getting a reward.
                if reward == 0 and no_reward_states >= MAX_NO_REWARD_STATES:
                    reward = -1
                no_reward_states += 1

                # Store in replay memory
                # TODO: Store memory in tensorflow.
                memory.append((state * 2 - state_old, action, reward,
                               state_new * 2 - state))

                # Did we see enough moves to start learning?
                if len(memory) == memory.maxlen:
                    train_states = []
                    train_target_qs = []
                    samples = np.random.permutation(memory.maxlen)[:BATCHSIZE]
                    for i in samples:
                        diff_state, action, reward, diff_state_new = memory[i]
                        out_q, = self.session.run(self.graph.out_q, {
                            self.graph.x: [diff_state],
                        })
                        max_q, = self.session.run(self.graph.max_q, {
                            self.graph.x: [diff_state_new],
                        })

                        if reward == -1:  # terminal state
                            out_q[action] = -1
                        else:
                            out_q[action] = reward + (Q_DECAY * max_q)

                        train_states.append(diff_state)
                        train_target_qs.append(out_q)

                    self.session.run(self.graph.train, {
                        self.graph.x: train_states,
                        self.graph.target_q: train_target_qs,
                    })

                # Exit game loop if game ended.
                if game_status != 0:
                    break

            if epoch % 100 == 0:
                print('epoch {} of {}'.format(epoch, epochs))
                save_path = path.join(tpath, CKPTNAME + self.start)
                self.saver.save(self.session, save_path, global_step=epoch)

            if random_action_probability > 0.1:
                random_action_probability -= (1/epochs)

    def get_action(self, state):
        action, = self.session.run(self.graph.action, {
            self.graph.state: state,
        })
        return action


def play(checkpoint=None):
    if checkpoint == 'LATEST':
        checkpoints = sorted(glob('./{}*'.format(CKPTNAME)))
        checkpoint = checkpoints[-1].replace('.meta', '')

    player = Q_Snake(checkpoint)
    game = Snake(BOARD_SIZE)

    def step():
        ret = game.step(player.get_action([game.board]))
        if ret:
            print('You {}'.format('win' if ret > 0 else 'lose'))
            return 0

    ui = SnakeUI(game)
    timer = ui.fig.canvas.new_timer(500, [(step, [], {})])
    timer.start()
    plt.show()


def train(tpath='.'):
    player = Q_Snake()
    player.train(tpath=tpath)


def main(args):
    if len(args) > 0 and args[0] == 'train':
        if len(args) > 1:
            train(tpath=args[1])
        train()
    elif len(args) > 0 and args[0] == 'play':
        checkpoint = 'LATEST'
        if len(args) > 1 and args[1].lower() != 'latest':
            checkpoint = path.realpath(args[1])
        play(checkpoint)
    else:
        print('Please provide either the `train CHECKPOINT_PATH` or '
              '`play CHECKPOINT_PATH` positional arguments. ')


if __name__ == '__main__':
    main(sys.argv[1:])
