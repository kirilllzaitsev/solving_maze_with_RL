import numpy as np
from policies import *

       

class Strategy:


    def __init__(self, policy=None):
        self.name = 'I am strategy'
        if policy is not None:
            self._policy = self.init_policy(policy)

    def init_policy(self, policy):
       return PolicyFactory().init_policy(policy)

    def update(self):
        print('I am stratege\'s update')


class Sarsamax(Strategy):


    def __init__(self, policy='Greedy'):
        super().__init__(policy)

    def update(self, alpha, gamma, Q, state, action, reward, next_state = None, next_action = None):
        '''
        Update rule for Sarsamax.

        Inputs:
        - Q (dict): current Q-table
        - state (tuple): current state
        - action (int): number in range(0, num_actions)
        - reward (int): reward after action
        - alpha, gamma (float): learning hyperparameters

        Returns:
        - new_value (int): updated action-value for Q[state][action]
        '''
        Q_sarsa_next = Q[next_state][next_action] if next_state is not None else 0
        new_value = Q[state][action] + \
            alpha*(reward+gamma*Q_sarsa_next-Q[state][action])
        return new_value

class ExpectedSarsa(Strategy):


    def __init__(self, policy='Epsilon'):
        super().__init__(policy)

    def update(self, alpha, gamma, Q, state, action, reward, next_state = None, next_action = None):
        self.eps = max(self.eps*self.eps_decay, self.eps_min)**2
        current = self.Q[state][action]
        policy_s = np.ones(self.nA)*self.eps/self.nA
        best_a = np.argmax(self.Q[next_state])
        policy_s[best_a] = 1 - self.eps + self.eps/self.nA
        next_distrib = np.dot(policy_s, self.Q[next_state])
        next_action = self._policy.get_action(Q, next_state)
        Q_sarsa_next = Q[next_state][next_action]
        new_value = Q[state][action] + alpha*(reward+self.gamma*Q_sarsa_next-current)
        return new_value

class StrategyFactory:
   def init_strategy(self, typ):
      targetclass = typ.capitalize()
      return globals()[targetclass]()
