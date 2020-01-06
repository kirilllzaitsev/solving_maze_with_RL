from policies import *
from abc import abstractmethod
       

class Strategy:

    def __init__(self, policy=None):
        if policy is not None:
            self._policy = PolicyFactory().init_policy(policy)

    @property
    def policy(self) -> Policy:
         return self._policy

    @policy.setter
    def policy(self, policy):
        self._policy = policy

    @abstractmethod
    def update(self, *args):
        pass


class Sarsamax(Strategy):

    def __init__(self, policy='EpsilonGreedy'):
        super().__init__(policy)

    def update(self, alpha, gamma, Q, state, action, reward, next_state = None, next_action = None, eps = None):
        """
        Update rule for Sarsamax.

        Inputs:
        - Q (dict): current Q-table
        - state (tuple): current state
        - action (int): number in range(0, num_actions)
        - reward (int): reward after action
        - alpha, gamma (float): learning hyperparameters

        Returns:
        - new_value (int): updated action-value for Q[state][action]
        """
        next_action = np.argmax(Q[next_state])
        Q_sarsa_next = Q[next_state][next_action] if next_state is not None else 0
        new_value = Q[state][action] + \
            alpha*(reward+gamma*Q_sarsa_next-Q[state][action])
        return new_value


class ExpectedSarsa(Strategy):

    def __init__(self, policy='Epsilon'):
        super().__init__(policy)

    def update(self, alpha, gamma, Q, state, action, reward, next_state=None, next_action=None, eps=None):
        nA=len(Q[next_state])
        current = Q[state][action]
        policy_s = np.ones(nA)*eps/nA
        best_a = np.argmax(Q[next_state])
        exp_sarsa_next = np.dot(policy_s, Q[next_state])+(1-eps)*best_a
        new_value = Q[state][action] + alpha*(reward+gamma*exp_sarsa_next-current)
        return new_value


class Dqn(Strategy):

    def __init__(self, policy=None):
        super().__init__(policy)

    def update(self, alpha, gamma, Q, state, action, reward, next_state = None, next_action=None, eps=None):
        pass


class StrategyFactory:
    @staticmethod
    def init_strategy(typ):
        target_class = typ.capitalize()
        return globals()[target_class]()
