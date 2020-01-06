from SimpleAgents import Agent
import sys
import numpy as np
import matplotlib.pyplot as plt
import pprint
import seaborn as sns
import pandas as pd
from Trainers import *
from abc import ABC, abstractmethod
import collections

pp = pprint.PrettyPrinter(indent=4)


class AbstractProductA(ABC):
    @abstractmethod
    def useful_function_a(self) -> str:
        pass


class AbstractProductB(ABC):
    @abstractmethod
    def useful_function_b(self) -> None:
        pass

    @abstractmethod
    def another_useful_function_b(self, collaborator: AbstractProductA) -> None:
        pass


class AbstractFactory(ABC):

    @abstractmethod
    def create_trainer_std(self, env, agent, epochs) -> AbstractProductA:
        pass

    @abstractmethod
    def create_trainer_dqn(self, env, agent, epochs) -> AbstractProductB:
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
    _on_start = None
    _on_finish = None

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

    def set_on_start(self, command):
        self._on_start = command

    def set_on_finish(self, command):
        self._on_finish = command


class Plotter:
    @staticmethod
    def plot_std(trainer, Q):
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
    """
    Интерфейс Команды объявляет метод для выполнения команд.
    """

    @abstractmethod
    def execute(self) -> None:
        pass


class SimpleCommand(Command):
    """
    Некоторые команды способны выполнять простые операции самостоятельно.
    """

    def __init__(self, env_size, strategy, num_episodes, num_actions) -> None:
        self._env_size = env_size
        self._strategy = strategy
        self._num_episodes = num_episodes
        self._num_actions = num_actions

    def execute(self) -> None:
        print(f"Manager was initialised with the following parameters: \n"
              f"Maze size: {self._env_size}.\nStrategy: {self._strategy}.\n")


class ComplexCommand(Command):
    """
    Но есть и команды, которые делегируют более сложные операции другим
    объектам, называемым «получателями».
    """

    def __init__(self, receiver, trainer, result) -> None:
        """
        Сложные команды могут принимать один или несколько объектов-получателей
        вместе с любыми данными о контексте через конструктор.
        """

        self._receiver = receiver
        self._trainer = trainer
        self._result = result

    def execute(self) -> None:
        """
        Команды могут делегировать выполнение любым методам получателя.
        """

        print("ComplexCommand: Complex stuff should be done by a receiver object", end="")
        print(1+1)
        print(type(self._result))
        if isinstance(self._result, collections.defaultdict):
            self._receiver.plot_std(self._trainer, self._result)
        if isinstance(self._result, list):
            self._receiver.plot_dqn(self._result)
        # self._receiver.do_something_else(self._b)


def main(strategy, epochs, env, env_size=(10, 10), n_actions=4, seed=42):
    manager = Manager.get_instance(env_size, strategy, epochs, 4)
    manager.set_on_start(SimpleCommand("Say Hi!"))
    manager._on_start.execute()

    if env == 'Custom':
        factory = FactoryCustomEnv()
        if 'Sarsa' in strategy:
            strategy_type = 'std'
            trainer = manager.run(strategy_type, factory, manager.env, manager.agent, epochs)
            Q = trainer.train()
            manager.set_on_finish(ComplexCommand(
                Plotter, trainer, Q))
            # print('Resulting table of (state, action) value-pairs: ')
            # pp.pprint(Q)
            # Plotter.plot_std(trainer, Q)
        else:
            strategy_type = 'dqn'
            torch.cuda.current_device()
            agent = AdvAgents.DQNAgent(state_size=len(env_size), action_size=n_actions, seed=seed)
            env = Environment(env_size)
            trainer = Manager.run(strategy_type, factory, env, agent, epochs)
            scores = trainer.train()
            manager.set_on_finish(ComplexCommand(
                Plotter, trainer, scores))
            # Plotter.plot_dqn(scores)
    else:
        factory = FactoryGymEnv()
        if 'Sarsa' in strategy:
            # manager = Manager.get_instance(env_size, strategy, epochs, 4)
            strategy_type = 'std'
            env = gym.make('maze-random-5x5-v0')
            trainer = manager.run(strategy_type, factory, env, manager.agent, epochs)
            Q = trainer.train()
            manager.set_on_finish(ComplexCommand(
                trainer, "Send email", "Save report"))
            print('Resulting table of (state, action) value-pairs: ')
            pp.pprint(Q)
            Plotter.plot_std(trainer, Q)
        else:
            strategy_type = 'dqn'
            torch.cuda.current_device()
            agent = AdvAgents.DQNAgent(state_size=len(env_size), action_size=n_actions, seed=seed)
            env = Environment(env_size)
            trainer = Manager.run(strategy_type, factory, env, agent, epochs)
            scores = trainer.train()
            manager.set_on_finish(ComplexCommand(
                trainer, "Send email", "Save report"))
            Plotter.plot_dqn(scores)

    manager._on_finish.execute()



