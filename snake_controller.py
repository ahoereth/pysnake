from matplotlib import pyplot as plt
import sys
import tensorflow as tf

class SnakeKeyboard:
    '''
    This class can be used to control the Snake via keyboard and have the original game.
    '''

    def __init__(self, snake, ui):
        self.snake = snake
        ui.fig.canvas.mpl_connect('key_press_event', self.keypress)
        plt.show()

    def keypress(self, event):
        key = (event.key or '').upper()
        if key == 'ESCAPE':
            sys.exit()
        if key in self.snake.actions.keys():
            if self.snake.step(self.snake.actions[key]) == -1:
                sys.exit()

class SnakeNeuralNetwork():
    '''
    How is this supposed to work? I imagine this class to be initialized with a set of weights, from
    which it constructs the network. It can also play a game of snake and report the performance value.
    '''

    def play_game(self, iterations):
        '''
        Plays a game of snake with the given controller.

        :param iterations: number of times the game is played
        :return: a (mean?) performance value for the games played
        '''


    def build_network(self, weights):
        '''
        Generates the network for this controller from the weights

        :param weights: the weights to use
        :return network
        '''

        #tensorflow computational graph will be constructed here
        return None

    def __init__(self, weights):
        '''
        Defines the network of the controller

        :param weights: the weights to use for that
        '''
        self.weights = weights
        self.net = self.build_network(weights)