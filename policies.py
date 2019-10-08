import numpy as np


class Policy:


    def __init__(self):
        self._name = 'I am agent\'s policy'

    def get_action(self):
        pass


class Epsilon(Policy):


    def __init__(self):
        super().__init__()

    def get_action(self, Q, state, nA=None, eps=None):
        '''
        
        Inputs:
        - Q (dict): current agent's Q-table
        - state (tuple): environment's state
        - nA (int): number of actions
        - eps (float): probability of selecting non-greedy action

        Returns:
        - action (int): action, chosen with given state and epsilon
        '''
        if np.random.random_sample() > eps:
            return np.argmax(Q[state])
        else:
            return np.random.choice(np.arange(nA))


class Greedy(Policy):


    def __init__(self):
        super().__init__()

    def get_action(self, Q, state, nA=None, eps=None):
        '''
        
        Inputs:
        - Q (dict): current agent's Q-table
        - state (tuple): environment's state

        Returns:
        - action (int): action that maximizes reward in given state
        '''
        return np.argmax(Q[state])
        

class PolicyFactory:
   def init_policy(self, typ):
      targetclass = typ.capitalize()
      return globals()[targetclass]()