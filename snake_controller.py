import tensorflow as tf
import snake

class SnakeController():
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