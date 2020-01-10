import collections
import pprint
from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from SimpleAgents import Agent
from Trainers import *

pp = pprint.PrettyPrinter(indent=4)


class AbstractFactory(ABC):

    @abstractmethod
    def create_trainer_std(self, env, agent, epochs) -> TrainerStd:
        pass

    @abstractmethod
    def create_trainer_dqn(self, env, agent, epochs) -> TrainerDQN:
        pass


class FactoryCustomEnv(AbstractFactory):
    def create_trainer_std(self, env, agent, epochs) -> CustomTrainerStd:
        return CustomTrainerStd(env, agent, epochs)

    def create_trainer_dqn(self, env, agent, epochs) -> CustomTrainerDQN:
        return CustomTrainerDQN(env, agent, epochs)


class FactoryGymEnv(AbstractFactory):
    def create_trainer_std(self, env, agent, epochs) -> GymTrainerStd:
        return GymTrainerStd(env, agent, epochs)

    def create_trainer_dqn(self, env, agent, epochs) -> GymTrainerDQN:
        return GymTrainerDQN(env, agent, epochs)


class Manager:
    __instance = None
    _on_init = None
    _on_finish = None
    _validator = None

    def __init__(self, env_size, strategy, num_episodes, num_actions):
        if not Manager.__instance:
            self._num_episodes = num_episodes
            self._env = self.init_env(env_size)
            self._agent = self.init_agent(strategy, num_actions)
            self._history = []
        else:
            print("Instance already created:", self.get_instance(strategy, env_size, num_episodes, num_actions))

    @property
    def history(self):
        return self._history

    @property
    def agent(self):
        return self._agent

    @property
    def env(self):
        return self._env

    @property
    def num_episodes(self):
        return self._num_episodes
    
    @property
    def on_init(self):
        return self._on_init
    
    @property
    def on_finish(self):
        return self._on_finish

    @property
    def validator(self):
        return self._validator

    @classmethod
    def get_instance(cls, env_size, strategy, num_episodes, num_actions):
        if not cls.__instance:
            cls.__instance = Manager(env_size, strategy, num_episodes, num_actions)
        return cls.__instance

    @staticmethod
    def init_agent(strategy, num_actions):
        agent = Agent(strategy, num_actions)
        return agent

    @staticmethod
    def init_env(env_size):
        env = Environment(env_size)
        return env

    @staticmethod
    def run(strategy_type, factory, env, agent, epochs):
        if strategy_type == 'std':
            trainer = factory.create_trainer_std(env, agent, epochs)
        else:
            trainer = factory.create_trainer_dqn(env, agent, epochs)
        return trainer

    def set_on_init(self, command):
        self._on_init = command

    def set_on_finish(self, command):
        self._on_finish = command

    def set_validator(self, command):
        self._validator = command


class Plotter:
    @staticmethod
    def plot_std(trainer, Q):
        if trainer.n_episodes <= 100:
            print('\nNot enough epochs to calculate statistics')
            return
        df = pd.DataFrame.from_dict(Q)
        plt.plot(np.linspace(0, trainer.n_episodes, len(trainer.history), endpoint=False),
                 np.asarray(trainer.history))
        plt.xlabel('Episode Number')
        plt.ylabel('Average Reward (Over Next %d Episodes)' % 300)
        fig = plt.figure(figsize=(11, 6))
        ax1 = fig.add_subplot(2, 2, 1)
        ax2 = fig.add_subplot(2, 2, 2)
        ax3 = fig.add_subplot(3, 1, 3)
        trainer.env.render(ax1)
        try:
            trainer.env.plot_optimal_path(Q, ax2)
        except AttributeError:
            pass
        ax3.set_title('Learned Q-table')
        sns.heatmap(df, cmap='coolwarm', annot=False, fmt='g', ax=ax3)
        plt.show()

    @staticmethod
    def plot_dqn(scores):
        fgr = plt.figure()
        ax = plt.subplot(111)
        plt.plot(np.arange(len(scores)), scores)
        plt.xlabel('Episodes')
        plt.ylabel('Scores')
        plt.show()


class Command(ABC):
    @abstractmethod
    def execute(self) -> None:
        pass


class Descriptor(Command):
    def __init__(self, env_size, strategy, num_episodes, num_actions) -> None:
        self._env_size = env_size
        self._strategy = strategy
        self._num_episodes = num_episodes
        self._num_actions = num_actions

    def execute(self) -> None:
        print(f"Manager was initialised with the following parameters: \n"
              f"Maze size: {self._env_size}.\nStrategy: {self._strategy}.\n"
              f"Epochs: {self._num_episodes}.\nActions: {self._num_actions}.\n")


class Summarizer(Command):
    def __init__(self, receiver, trainer, result) -> None:
        self._receiver = receiver
        self._trainer = trainer
        self._result = result

    def execute(self) -> None:
        if isinstance(self._result, collections.defaultdict):
            self._receiver.plot_std(self._trainer, self._result)
        if isinstance(self._result, list):
            self._receiver.plot_dqn(self._result)


class Validator(Command):
    def __init__(self, message='', response=''):
        self.message = message
        self.response = response

    def execute(self):
        print('Error found: ', self.message,
              '\n', self.response)


def main(strategy, epochs, env, env_size=(10, 10), n_actions=4, seed=42):
    manager = Manager.get_instance(env_size, strategy, epochs, 4)
    manager.set_on_init(Descriptor(env_size, strategy, epochs, 4))
    manager.on_init.execute()

    if env == 'Custom':
        factory = FactoryCustomEnv()
        if 'Sarsa' in strategy:
            strategy_type = 'std'
            trainer = manager.run(strategy_type, factory, manager.env, manager.agent, epochs)
            Q = trainer.train()
            manager.set_on_finish(Summarizer(
                Plotter, trainer, Q))
        else:
            strategy_type = 'dqn'
            torch.cuda.current_device()
            agent = AdvAgents.DQNAgent(state_size=len(env_size), action_size=n_actions, seed=seed)
            env = Environment(env_size)
            trainer = Manager.run(strategy_type, factory, env, agent, epochs)
            scores = trainer.train()
            manager.set_on_finish(Summarizer(
                Plotter, trainer, scores))
    else:
        factory = FactoryGymEnv()
        if env_size[0] not in [5, 10]:
            manager.set_validator(Validator("maze size is not valid",
                                            "To use Gym environment, size must be equal to 5 or 10"))
            manager.validator.execute()
            return
        if 'Sarsa' in strategy:
            strategy_type = 'std'
            env = gym.make(f'maze-random-{env_size[0]}x{env_size[1]}-v0')
            trainer = manager.run(strategy_type, factory, env, manager.agent, epochs)
            Q = trainer.train()
            manager.set_on_finish(Summarizer(
                Plotter, trainer, Q))
        else:
            strategy_type = 'dqn'
            torch.cuda.current_device()
            agent = AdvAgents.DQNAgent(state_size=len(env_size), action_size=n_actions, seed=seed)
            env = gym.make(f'maze-random-{env_size[0]}x{env_size[1]}-v0')
            trainer = Manager.run(strategy_type, factory, env, agent, epochs)
            scores = trainer.train()
            manager.set_on_finish(Summarizer(
                Plotter, trainer, scores))

    manager.on_finish.execute()



