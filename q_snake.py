from collections import deque

import numpy as np
import tensorflow as tf

from snake import Snake


BOARD_SIZE = 4
DIRECTIONS = 4

RAND_ACT_EPS = 1
EPOCHS = 1000
y = .975
MAX_GAME_STEPS = 500

saved_model = False  # './model-400'

ACTIONS = 4


def conv_layer(x, shape, kernel, stride=1):
    weights = tf.Variable(tf.truncated_normal(kernel + shape, stddev=0.1))
    bias = tf.Variable(tf.ones((shape[1],))/100)
    conv = tf.nn.conv2d(x, weights, (1, stride, stride, 1), 'SAME')
    return tf.nn.relu(conv + bias)

def dens_layer(x, shape, act_func=lambda x: x):
    weights = tf.Variable(tf.truncated_normal(shape, stddev=0.1))
    bias = tf.Variable(tf.ones((shape[1],))/10)
    return act_func(tf.matmul(x, weights) + bias)


x = tf.placeholder(tf.float32, (None, BOARD_SIZE, BOARD_SIZE))
next_q = tf.placeholder(tf.float32, (None, 4))

net = tf.reshape(x, (-1, BOARD_SIZE, BOARD_SIZE, 1)) / tf.reduce_max(x)

net = conv_layer(net, (1, 8), (4, 4))
net = conv_layer(net, (8, 12), (3, 3))
net = tf.reshape(net, (-1, BOARD_SIZE * BOARD_SIZE * 12))
net = dens_layer(net, (BOARD_SIZE * BOARD_SIZE * 12, 512), tf.nn.relu)
net = dens_layer(net, (512, 4))

max_q = tf.reduce_max(net, axis=1)
action = tf.argmax(net, axis=1)

loss = tf.reduce_mean(tf.square(next_q - net))
train = tf.train.RMSPropOptimizer(0.03).minimize(loss)


BATCHSIZE = 40
memory = deque([], 80)

saver = tf.train.Saver()
sess = tf.Session()
sess.run(tf.global_variables_initializer())


if not saved_model:
    for epoch in range(EPOCHS):
        game = Snake(BOARD_SIZE, markhead=True)
        last_highscore = 0
        for _ in range(MAX_GAME_STEPS):
            state = np.copy(game.board)
            qs = sess.run(net, {x: [state]})[0]

            # Select an action a
            if np.random.rand(1) < RAND_ACT_EPS:
                act = np.random.randint(0, ACTIONS)
            else:
                act = np.argmax(qs)

            # Carry out action
            snake_state = game.step(act)
            state_new = np.copy(game.board)

            # Observe reward
            if game.highscore > last_highscore:
                reward = 1
                last_highscore = game.highscore
            else:
                reward = snake_state

            # Store in replay memory
            memory.append((state, act, reward, state_new))

            # Did we see enough moves to start learning?
            if len(memory) == memory.maxlen:
                train_x = []
                train_y = []
                samples = np.random.permutation(memory.maxlen)[:BATCHSIZE]
                for i in samples:
                    state_old, act, reward, state_new = memory[i]
                    qs = sess.run(net, {x: [state_old]})[0]
                    mq = sess.run(max_q, {x: [state_new]})[0]
                    qs[act] = -1 if reward == -1 else reward + (y * mq)
                    train_x.append(state_old)
                    train_y.append(qs)
                sess.run(train, {x: train_x, next_q: train_y})

            if snake_state != 0:
                break

        highscores.append(game.highscore)
        if epoch % 100 == 0:
            saver.save(sess, 'model', global_step=epoch)

        if RAND_ACT_EPS > 0.1:
            RAND_ACT_EPS -= (1/EPOCHS)


if saved_model:
    saver.restore(sess, saved_model)
    import matplotlib
    matplotlib.use('tkAgg')
    from matplotlib import pyplot as plt
    from snake_ui import SnakeUI


    for _ in range(10):
        game = Snake(BOARD_SIZE, markhead=True)
        def step():
            act = sess.run(action, {x: [game.board]})[0]
            print(sess.run(net, {x: [game.board]})[0])
            ret = game.step(act)
            if ret:
                print('You {}'.format('win' if ret > 0 else 'lose'))
                return 0
        ui = SnakeUI(game)
        timer = ui.fig.canvas.new_timer(500, [(step, [], {})])
        timer.start()
        plt.show()
