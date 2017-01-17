'''
Aim: we want to find a feedforward net that plays snake.
How: the agents are optimized with genetic algorithms and not standard tf backpropagation. PseudoCode for this
would look roughly like this:

1) initialize population
2) let each of the individuals play a certain number of games
3) most performand individuals (based on the highscore) pass on to the next generation
4) produce offsprings and new individuals
5) go to 2

The process is repeated for a certain number of generations
'''

def initialize_gen(individuals):
    '''
    Randomly generates networks that control the player(snake) in the game.

    :param individuals: the number of individuals that should be created
    :return: the individuals
    '''
    return None

def play_snake(gen):
    perf = None
    for individual in gen:
        #play the game and calculate performance
        print('playing ..')
    return perf

def get_next_gen(parent, performance):
    '''
    Produces the next generation of controller networks from the given networks
    and their performance

    :param parent: the parent generation from which the offsprings are produced
    :param performance: the performance value per individual
    :return: the next generation of networks
    '''
    next_gen = None
    return next_gen

if __name__ == '__main__':
    max_gens = 100 #how many generations do we want to train?
    max_ind = 20   #how many individuals do we have per generation?

    cur_gen = initialize_gen(max_ind)

    for gen in range(max_gens):
        perf = play_snake(cur_gen)
        cur_gen = get_next_gen(cur_gen, perf)
