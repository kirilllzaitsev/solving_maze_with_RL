import gym
import gym_maze
import torch
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
import random
import AdvAgents

from environments import Environment


def DeepQNetwork(env, agent, n_episodes=100, t_steps=1000, eps_start=0.5, eps_min=0.01, eps_decay=.995, select_action=False):

    scores = []
    scores_window = deque(maxlen=100)
    eps = eps_start
    for i_episode in range(1, n_episodes + 1):
        state = np.array(env.reset())
        score = 0
        for t in range(t_steps):
            action = agent.act(state, eps)
            # env.render()
            next_state, reward, done = env.step(state, action)
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                break
        scores_window.append(score)
        scores.append(score)
        eps = max(eps_min, eps * eps_decay)
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
        if i_episode % 100 == 0:
            print("Episode:{}, Average:{}".format(i_episode, np.mean(scores_window)))
        if np.mean(scores_window) >= 200:
            print("Environment solved! in {} episodes. Average: {}".format(i_episode, np.mean(scores_window)))
            torch.save(agent.qnetwork_local.state_dict(), 'checkpoint.ckpt')
            break
    return scores


class GymTrainer:
    def __init__(self):
        self._action_space = ['N', 'E', 'S', 'W']
        self._env = gym.make('maze-random-10x10-v0')

    def train(self):

        self._env.seed(0)
        print("State shape: ", self._env.observation_space)
        print('Action space: ', self._env.action_space)

        torch.cuda.current_device()
        import AdvAgents

        agent = AdvAgents.DQNAgent(state_size=2, action_size=4, seed=0)
        def DeepQNetwork(n_episodes=2, t_steps=500, eps_start=1, eps_min=0.01, eps_decay=.9995):
            scores = []
            scores_window = deque(maxlen=100)
            eps = eps_start
            for i_episode in range(1, n_episodes + 1):

                state = np.array(self._env.reset())
                score = 0
                for t in range(t_steps):
                    # action = select_action(state)
                    action = agent.act(state, eps)
                    self._env.render()
                    next_state, reward, done, _ = self._env.step(self._action_space[action])
                    agent.step(state, action, reward, next_state, done)
                    state = next_state
                    score += reward
                    if done:
                        break
                scores_window.append(score)
                scores.append(score)
                eps = max(eps_min, eps * eps_decay)
                print('\nEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
                if i_episode % 100 == 0:
                    print(f"Episode:{i_episode}, Average:{np.mean(scores_window)}")
                if np.mean(scores_window) >= 200:
                    print(f"Environment solved! in {i_episode} episodes. Average: {np.mean(scores_window)}")
                    torch.save(agent.qnetwork_local.state_dict(), 'checkpoint.ckpt')
                    break
            return scores

        scores = DeepQNetwork()
        self._env.close()
        fgr = plt.figure()
        ax = plt.subplot(111)
        plt.plot(np.arange(len(scores)), scores)
        plt.xlabel('Episodes')
        plt.ylabel('Scores')
        plt.show()

    def demo(self):
        state = self._env.reset()
        for j in range(200):
            action = self.select_action(state)
            self._env.render()
            next_state, action, done, _ = self._env.step(action)
            if done:
                break
            state = next_state

    def select_action(self, state, explore_rate=0.):
        # Select a random action
        if random.random() < explore_rate:
            action = self._env.action_space.sample()
        # Select the action with the highest q
        else:
            action = int(random.choice(np.arange(self._env.action_space.n)))
        return action


class CustomTrainer:
    def __init__(self, env_size, action_size):
        self._state_space = env_size
        self._action_space = action_size
        self._seed = 42

    def train(self):
        torch.cuda.current_device()
        agent = AdvAgents.DQNAgent(state_size=len(self._state_space), action_size=self._action_space, seed=self._seed)

        env = Environment(self._state_space)

        scores = DeepQNetwork(env, agent)
        # env.close()
        fgr = plt.figure()
        ax = plt.subplot(111)
        plt.plot(np.arange(len(scores)), scores)
        plt.xlabel('Episodes')
        plt.ylabel('Scores')
        plt.show()



