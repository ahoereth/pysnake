#! /usr/bin/env python3
import sys
from os import path
from glob import glob
from collections import deque, namedtuple
from datetime import datetime

import numpy as np
import tensorflow as tf
import matplotlib

from snake import Snake

try:
    matplotlib.use('TkAgg')
    from matplotlib import pyplot as plt
    from snake_ui import SnakeUI
except:
    tkagg_available = False
    pass
else:
    tkagg_available = True


DIRECTIONS = 4

EPOCHS = 100000
Q_DECAY = .8
MAX_NO_REWARD_STATES = Snake.size ** 2

ACTIONS = 4
MEMORY_SIZE = 5000
BATCHSIZE = 32

CKPTNAME = 'q_snake-'


def flatten(l):
    return [j for i in l for j in i]

def conv_layer(x, shape, kernel, stride=1, pad='SAME', act_func=tf.nn.relu):
    weights = tf.Variable(tf.truncated_normal(kernel + shape, stddev=.1))
    bias = tf.Variable(tf.constant(.01, shape=shape[1:2]))
    conv = tf.nn.conv2d(x, weights, (1, stride, stride, 1), pad)
    return act_func(conv + bias)

def dens_layer(x, shape, act_func=None):
    weights = tf.Variable(tf.truncated_normal(shape, stddev=.1))
    bias = tf.Variable(tf.constant(.01, shape=shape[1:2]))
    y = tf.matmul(x, weights) + bias
    return act_func(y) if callable(act_func) else y


Q_Graph = namedtuple('Q_Graph', [
    'x',
    'q_target',
    'q',
    'q_max',
    'action',
    'train',
    'loss',
])


class Q_Snake:
    def __init__(self, checkpoint=None):
        self.start = datetime.now().strftime('%Y%m%d-%H%M%S')
        self.graph = self._model()
        self.session = tf.Session()
        self.saver = tf.train.Saver(keep_checkpoint_every_n_hours=.1,
                                    max_to_keep=20)

        self.session.run(tf.global_variables_initializer())
        if checkpoint is not None:
            self.saver.restore(self.session, checkpoint)

    def _model(self):
        x = tf.placeholder(tf.float32, (None, Snake.size, Snake.size))
        q_target = tf.placeholder(tf.float32, (None, 4))

        M = 24
        N = 36
        O = Snake.size ** 2 * N
        P = 256

        net = tf.reshape(x, (-1, Snake.size, Snake.size, 1))  # for conv

        net = conv_layer(net, (1, M), (3, 3))
        net = conv_layer(net, (M, N), (3, 3))
        net = tf.reshape(net, (-1, O))
        net = dens_layer(net, (O, P), tf.nn.relu)

        q = dens_layer(net, (P, ACTIONS))
        q_max = tf.reduce_max(q, axis=1)
        action = tf.argmax(q, axis=1)

        loss = tf.reduce_mean(tf.square(q_target - q))
        train = tf.train.RMSPropOptimizer(0.0005).minimize(loss)

        return Q_Graph(x, q_target, q, q_max, action, train, loss)

    def train(self, epochs=EPOCHS, random_action_probability=1, tpath='.'):
        memory = deque([], MEMORY_SIZE)
        random_action_decay = (.9 / (epochs * .5))

        for epoch in range(1, epochs + 1):
            game = Snake()
            last_highscore = 0
            state = np.zeros(game.board.shape)

            while 42:
                state_old = state

                state = np.copy(game.board)
                qs, action = flatten(self.session.run(
                    [self.graph.q, self.graph.action],
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

                # Store in replay memory
                # TODO: Store memory in tensorflow.
                memory.append((state * 2 - state_old, action, reward,
                               state_new * 2 - state))

                # Did we see enough moves to start learning?
                if len(memory) > BATCHSIZE:
                    train_states = []
                    train_q_targets = []
                    samples = np.random.permutation(len(memory))[:BATCHSIZE]
                    for i in samples:
                        diff_state, action, reward, diff_state_new = memory[i]
                        q, = self.session.run(self.graph.q, {
                            self.graph.x: [diff_state],
                        })
                        q_max, = self.session.run(self.graph.q_max, {
                            self.graph.x: [diff_state_new],
                        })

                        if reward == -1:  # terminal state
                            q[action] = -1
                        else:
                            q[action] = reward + (Q_DECAY * q_max)

                        train_states.append(diff_state)
                        train_q_targets.append(q)

                    self.session.run(self.graph.train, {
                        self.graph.x: train_states,
                        self.graph.q_target: train_q_targets,
                    })

                # Exit game loop if game ended.
                if game_status != 0:
                    break

            if epoch % 100 == 0 and len(memory) > BATCHSIZE:
                print('epoch {} of {}'.format(epoch, epochs))
                save_path = path.join(tpath, CKPTNAME + self.start)
                self.saver.save(self.session, save_path, global_step=epoch)

            if random_action_probability > 0.1:
                random_action_probability -= random_action_decay

    def get_action(self, state):
        try:
            self.state_old
        except:
            self.state_old = np.zeros(state.shape)

        action, = self.session.run(self.graph.action, {
            self.graph.x: [state * 2 - self.state_old],
        })

        self.state_old = np.copy(state)
        return action


def play(checkpoint=None):
    if checkpoint == 'LATEST':
        checkpoints = sorted(glob('./{}*'.format(CKPTNAME)))
        checkpoint = checkpoints[-1].replace('.meta', '')

    player = Q_Snake(checkpoint)
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


def train(tpath='.'):
    player = Q_Snake()
    player.train(tpath=tpath)


def main(args):
    if len(args) > 0 and args[0] == 'train':
        train(tpath=args[1] if len(args) > 1 else '.')
    elif len(args) > 0 and args[0] == 'play':
        if not tkagg_available:
            print('TkAgg Matplotlib backend not available, cannot visualize '
                  'gameplay.')
            sys.exit(1)
        checkpoint = 'LATEST'
        if len(args) > 1 and args[1].lower() != 'latest':
            checkpoint = path.realpath(args[1])
        play(checkpoint)
    else:
        print('Please provide either the `train CHECKPOINT_PATH` or '
              '`play CHECKPOINT_PATH` positional arguments. ')


if __name__ == '__main__':
    main(sys.argv[1:])
