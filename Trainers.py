import gym
import gym_maze
import torch
import numpy as np
from collections import deque
import random
import AdvAgents
from environments import Environment


class Trainer:

    def __init__(self, env, agent, n_episodes=10, t_steps=500,
                 eps_start=1, eps_min=0.001, eps_decay=.95):
        self.env = env
        self.agent = agent
        self.n_episodes = n_episodes
        self.t_steps = t_steps
        self.eps_start = eps_start
        self.eps_min = eps_min
        self.eps_decay = eps_decay

    def train(self):
        scores = []
        scores_window = deque(maxlen=100)
        eps = self.eps_start
        for i_episode in range(1, self.n_episodes + 1):
            state = np.array(self.env.reset())
            score = 0
            for t in range(self.t_steps):
                action = self.agent.act(state, eps)
                self.env.render()
                next_state, reward, done, _ = self.step(state, action)
                self.agent.step(state, action, reward, next_state, done)
                state = next_state
                score += reward
                if done:
                    break
            scores_window.append(score)
            scores.append(score)
            eps = max(self.eps_min, eps * self.eps_decay)
            print('Episode {}\tAverage Score: {:.2f}\n'.format(i_episode, np.mean(scores_window)), end="")
            if i_episode % 100 == 0:
                print("Episode:{}, Average:{}".format(i_episode, np.mean(scores_window)))
            if np.mean(scores_window) >= 200:
                print("Environment solved! in {} episodes. Average: {}".format(i_episode, np.mean(scores_window)))
                torch.save(self.agent.qnetwork_local.state_dict(), 'checkpoint.ckpt')
                break
        return scores

    def step(self, *args):
        pass


class GymTrainer(Trainer):
    def __init__(self, env, agent, n_episodes):
        super().__init__(env, agent, n_episodes)
        self._action_space = ['N', 'E', 'S', 'W']
        self.env = gym.make('maze-random-5x5-v0')

    def step(self, state, action):
        return self.env.step(self._action_space[action])

    def demo(self):
        state = self.env.reset()
        for j in range(200):
            action = self.select_action(state)
            self.env.render()
            next_state, action, done, _ = self.env.step(action)
            if done:
                break
            state = next_state

    def select_action(self, state, explore_rate=0.):
        # Select a random action
        if random.random() < explore_rate:
            action = self.env.action_space.sample()
        # Select the action with the highest q
        else:
            action = int(random.choice(np.arange(self.env.action_space.n)))
        return action


class CustomTrainer(Trainer):
    def __init__(self, env, agent, n_episodes):
        super().__init__(env, agent, n_episodes)
        self._action_space = [0, 1, 2, 3]

    def step(self, state, action):
        return self.env.step(state, self._action_space[action])



