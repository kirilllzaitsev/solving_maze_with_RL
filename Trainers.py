import gym
import gym_maze
import torch
import numpy as np
from collections import deque
import random
import AdvAgents
from environments import Environment
from abc import ABC, abstractmethod
import sys
import msvcrt


class Trainer(ABC):
    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def step(self, *args):
        pass


class TrainerStd(Trainer):
    def __init__(self, env, agent, n_episodes=10, t_steps=500,
                 eps_start=1, eps_min=0.001, eps_decay=.95):
        self._env = env
        self._agent = agent
        self.n_episodes = n_episodes
        self.t_steps = t_steps
        self.eps_start = eps_start
        self.eps_min = eps_min
        self.eps_decay = eps_decay
        self._history = []

    @property
    def history(self):
        return self._history

    @property
    def env(self):
        return self._env

    @property
    def agent(self):
        return self._agent

    @history.setter
    def history(self, value):
        self._history = value

    @env.setter
    def env(self, value):
        self._env = value

    def train(self, plot_every=100, t_steps=200):
        Q = self.agent.Q
        nA = self.agent.nA
        env = self.env
        alpha = self.agent.alpha
        gamma = self.agent.gamma
        eps = self.agent.eps
        eps_decay = self.agent.eps_decay
        eps_min = self.agent.eps_min

        # monitor performance
        tmp_scores = deque(maxlen=plot_every)  # deque for keeping track of scores
        avg_scores = deque(maxlen=self.n_episodes)  # average scores over every plot_every episodes
        for i_episode in range(1, self.n_episodes + 1):
            # monitor progress
            if i_episode % 1000 == 0:
                print("\rEpisode {}/{}".format(i_episode, self.n_episodes), end="")
                sys.stdout.flush()
            score = 0
            state = env.reset()  # start episode
            state = tuple(state)
            print('RESET ENV')
            eps = max(eps * eps_decay, eps_min) ** 2
            action = self.agent.strategy.policy.get_action(Q, state, nA, eps)

            for t in range(t_steps):
                if msvcrt.kbhit():
                    try:
                        self.env.close()
                    except:
                        pass
                    return Q
                self.env.render()
                next_state, reward, done, _ = self.step(state, action)  # take action, observe reward and state
                next_state = tuple(next_state)
                print('Action: ', action, 'State: ', state,
                      'Reward: ', reward, 'Next state: ', next_state)
                score += reward

                if not done:
                    Q[state][action] = self.agent.strategy.update(alpha, gamma, Q,
                                                                  state, action, reward, next_state=next_state,
                                                                  next_action=None, eps=eps)
                    next_action = self.agent.strategy.policy.get_action(Q, next_state, nA, eps)

                    state = next_state
                    action = next_action
                    # print(state, action, reward)
                if done:
                    Q[state][action] = self.agent.strategy.update(alpha, gamma, Q,
                                                                  state, action, reward)

                    break
                tmp_scores.append(score)

            if i_episode % plot_every == 0:
                avg_scores.append(np.mean(tmp_scores))
        # plot performance
        self._history = avg_scores
        # print best 100-episode performance
        try:
            print(f'Best Average Reward over {plot_every} Episodes: {np.max(avg_scores)}')
        except:
            pass

        return Q

    @abstractmethod
    def step(self, *args):
        pass


class GymTrainerStd(TrainerStd):
    def __init__(self, env, agent, n_episodes):
        super().__init__(env, agent, n_episodes)
        self._action_space = ['N', 'E', 'S', 'W']

    def step(self, state, action):
        return self.env.step(self._action_space[int(action)])

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


class CustomTrainerStd(TrainerStd):
    def __init__(self, env, agent, n_episodes):
        super().__init__(env, agent, n_episodes)
        self._action_space = [0, 1, 2, 3]

    def step(self, state, action):
        return self.env.step(state, self._action_space[int(action)])


class TrainerDQN(ABC):
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
                if msvcrt.kbhit():
                    try:
                        self.env.close()
                    except:
                        pass
                    return scores
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

    @abstractmethod
    def step(self, *args):
        pass


class GymTrainerDQN(TrainerDQN):
    def __init__(self, env, agent, n_episodes):
        super().__init__(env, agent, n_episodes)
        self._action_space = ['N', 'E', 'S', 'W']

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


class CustomTrainerDQN(TrainerDQN):
    def __init__(self, env, agent, n_episodes):
        super().__init__(env, agent, n_episodes)
        self._action_space = [0, 1, 2, 3]

    def step(self, state, action):
        return self.env.step(state, self._action_space[action])



