from SimpleAgents import Agent
from environments import Environment
import sys
from collections import deque
import numpy as np
import matplotlib.pyplot as plt
import pprint
import seaborn as sns
import pandas as pd
from Trainers import *

pp = pprint.PrettyPrinter(indent=4)

class Manager:
    __instance = None
    
    def __init__(self, env_size, strategy, num_episodes, num_actions):
        if not Manager.__instance:
            self._num_episodes = num_episodes
            self._env = self.init_env(env_size)
            self._agent = self.init_agent(strategy, num_actions)
            self._history = []
        else:
            print("Instance already created:", self.getInstance(strategy, env_size, num_episodes, num_actions))

    @classmethod
    def getInstance(cls, env_size, strategy, num_episodes, num_actions):
        if not cls.__instance:
            cls.__instance = Manager(env_size, strategy, num_episodes, num_actions)
        return cls.__instance

    def init_agent(self, strategy, num_actions):
        agent = Agent(strategy, num_actions)
        return agent

    def init_env(self, env_size):
        env = Environment(env_size)
        return env
        
    def start_learning(self, plot_every=100):
        Q = self._agent.Q
        nA = self._agent._nA
        env = self._env
        alpha = self._agent._alpha
        gamma = self._agent._gamma
        eps = self._agent.eps
        eps_decay = self._agent._eps_decay
        eps_min = self._agent._eps_min

        
        # monitor performance
        tmp_scores = deque(maxlen=plot_every)     # deque for keeping track of scores
        avg_scores = deque(maxlen=self._num_episodes)   # average scores over every plot_every episodes
        for i_episode in range(1, self._num_episodes+1):
        # monitor progress
            if i_episode % 1000 == 0:
                print("\rEpisode {}/{}".format(i_episode, self._num_episodes), end="")
                sys.stdout.flush()   
            score = 0                                             
            state = env.reset()                                   # start episode
            print('RESET ENV')
            eps = max(eps*eps_decay, eps_min)**2                               
            action = self._agent._strategy._policy.get_action(Q, state, nA, eps)            
            

            while True:
             
                next_state, reward, done = env.step(state, action) # take action, observe reward and state
                print('Action: ', action, 'State: ', state,\
                    'Reward: ', reward, 'Next state: ', next_state)
                score += reward  
                
                                           
                if not done:
                    Q[state][action] = self._agent._strategy.update(alpha, gamma, Q, \
                                                    state, action, reward, next_state=next_state, next_action=None, eps=eps)
                    next_action = self._agent._strategy._policy.get_action(Q, next_state, nA, eps) 
                    
                    state = next_state     
                    action = next_action
                    # print(state, action, reward)   
                if done:
                    Q[state][action] = self._agent._strategy.update(alpha, gamma, Q, \
                                                    state, action, reward)
                      
                    break
                tmp_scores.append(score)  
            if (i_episode % plot_every == 0):
                avg_scores.append(np.mean(tmp_scores))

        # plot performance
        self._history = avg_scores
        # print best 100-episode performance
        print(f'Best Average Reward over {plot_every} Episodes: {np.max(avg_scores)}')
        return Q
        
    def display_current_policy(self, parameter_list):
        pass


def main(strategy, epochs, env, env_size=(10, 10)):
    # if "DQN" is in strategy:
    if env == 'Custom':
        manager = Manager.getInstance(env_size, strategy, epochs, 4)
        Q = manager.start_learning()
        print('Resulting table of (state, action) value-pairs: ')
        pp.pprint(Q)
        df = pd.DataFrame.from_dict(Q)

        plt.plot(np.linspace(0, manager._num_episodes, len(manager._history), endpoint=False),
                 np.asarray(manager._history))
        plt.xlabel('Episode Number')
        plt.ylabel('Average Reward (Over Next %d Episodes)' % 300)

        fig = plt.figure(figsize=(11, 6))

        ax1 = fig.add_subplot(2, 2, 1)
        ax2 = fig.add_subplot(2, 2, 2)
        ax3 = fig.add_subplot(3, 1, 3)

        manager._env.display_env(ax1)
        manager._env.plot_optimal_path(Q, ax2)

        ax3.set_title('Learned Q-table')
        sns.heatmap(df, cmap='coolwarm',  annot=False, fmt='g', ax=ax3)

        plt.show()
    else:
        gt = GymTrainer()
        gt.train()

