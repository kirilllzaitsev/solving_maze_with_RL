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
        self._strategy = StrategyFactory().init_strategy(strategy)
        self._nA = num_actions
        self.Q = defaultdict(lambda: np.zeros(num_actions))
        # Hyperparameters
        self._alpha = 0.001
        self._gamma = 1.0
        self._eps_decay = 0.999
        self._eps_min = 0.005
        self.eps = 1

        @property
        def strategy(self) -> Strategy:
            return self._strategy

        @strategy.setter
        def strategy(self, strategy: Strategy) -> None:
            self._strategy = strategy
