import numpy as np
from strategies import *
from policies import *
from collections import defaultdict
from environments import Environment


class Agent:
    def __init__(self, strategy, num_actions):
        '''
        Params:

            strategy (str): agent's learning strategy
            num_actions (int): number of actions in the environment
        '''
        self._strategy = self.init_strategy(strategy)
        self._nA = num_actions
        self.Q = defaultdict(lambda: np.zeros(num_actions))
        # Hyperparameters
        self._alpha = 0.0001
        self._gamma = 1.0
        self.eps = 0.001

    def step(self, state, action, reward, next_state, done):
        ''' 
        Update the agent's knowledge, using the most recently sampled tuple.

        Input:
        - state: the previous state of the environment
        - action: the agent's previous choice of action
        - reward: last reward received
        - next_state: the current state of the environment
        - done: whether the episode is complete (True or False)

        Updates agent's Q-table
        '''
        pass

    def init_strategy(self, strategy):
        return StrategyFactory().init_strategy(strategy)

    
