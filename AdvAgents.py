from abc import ABC, abstractmethod
from model import QNetwork
from buffers import *

import torch
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = int(1e4)  # replay buffer size
BATCH_SIZE = 32  # minibatch size
GAMMA = 0.9  # discount factor
TAU = 1e-3  # for soft update of target parameters
LR = 5e-4  # learning rate
UPDATE_EVERY = 4  # how often to update the network
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Agent(ABC):
    def __init__(self, state_size, action_size, seed):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)
        self.t_step = 0
        self.seed = random.seed(seed)

    @abstractmethod
    def step(self, *args):
        pass

    @abstractmethod
    def act(self, *args):
        pass

    @abstractmethod
    def learn(self, *args):
        pass


class DQNAgent(Agent):
    """Agent interacts learns via vanilla DQN.
       Memory is not implemented so the importance of this technique is emphasized.
    """

    def __init__(self, state_size, action_size, seed):
        """Initialize a DQNAgent object.

        Params
        ======
            seed (int): random seed
        """
        super().__init__(state_size, action_size, seed)
        self.qnetwork = QNetwork(state_size, action_size, seed).to(device)
        self.optimizer = optim.Adam(self.qnetwork.parameters(), lr=LR)

    def step(self, state, action, reward, next_state, done):
        experience = state, action, reward, next_state, done
        self.learn(experience, GAMMA)

    def act(self, state, eps=0.05):
        """Returns actions for given state as per current policy.

        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = np.array(state)
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork.eval()
        with torch.no_grad():
            action_values = self.qnetwork(state)
        self.qnetwork.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experience, gamma):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """
        state, action, reward, next_state, done = experience
        state = torch.tensor(state).float().unsqueeze(0).to(device)
        next_state = torch.tensor(next_state).float().unsqueeze(0).to(device)
        action = torch.tensor([[action]]).long().to(device)
        Q_targets_next = self.qnetwork(next_state).detach().max(1)[0].unsqueeze(1)

        # Compute Q targets for current states
        Q_targets = reward + (gamma * Q_targets_next * (1 - done))
        # Get expected Q values from local model
        Q_expected = self.qnetwork(state).gather(1, action)

        loss = F.mse_loss(Q_targets, Q_expected)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


class DQNAgent_ExpReplay(Agent):
    """Agent learns via using a memory buffer"""

    def __init__(self, state_size, action_size, seed):
        super().__init__(state_size, action_size, seed)

        # Q-Network
        self.qnetwork = QNetwork(state_size, action_size, seed).to(device)
        self.optimizer = optim.Adam(self.qnetwork.parameters(), lr=LR)

    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)

        if len(self.memory) > BATCH_SIZE:
            experiences = self.memory.sample()
            self.learn(experiences, GAMMA)

    def act(self, state, eps=0.05):
        state = np.array(state)
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork.eval()
        with torch.no_grad():
            action_values = self.qnetwork(state)
        self.qnetwork.train()

        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences, gamma):
        states, actions, rewards, next_states, dones = experiences

        Q_targets_next = self.qnetwork(next_states).detach().max(1)[0].unsqueeze(1)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        Q_expected = self.qnetwork(states).gather(1, actions)

        loss = F.mse_loss(Q_expected, Q_targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


class DQNAgent_DoubleDQN(Agent):
    """Agent uses memory + two networks to prohibit
       'donkey driving' in case of one DQN
    """
    def __init__(self, state_size, action_size, seed):
        super().__init__(state_size, action_size, seed)

        self.qnetwork_local = QNetwork(state_size, action_size, seed).to(device)
        self.qnetwork_target = QNetwork(state_size, action_size, seed).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)

    def step(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)

        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            if len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA)

    def act(self, state, eps=0.05):
        state = np.array(state)
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences, gamma):
        states, actions, rewards, next_states, dones = experiences

        # Get max predicted Q values (for next states) from target model
        Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        # Compute Q targets for current states 
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        # Get expected Q values from local model
        Q_expected = self.qnetwork_local(states).gather(1, actions)

        # Compute loss
        loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)

    def soft_update(self, local_model, target_model, tau):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)


class DQNAgent_PrioritizedExpReplay(Agent):
    """Agent learns through setting priorities to memories
       and samples experiences according to them
    """

    def __init__(self, state_size, action_size, seed):
        super().__init__(state_size, action_size, seed)
        self.memory = PrioritizedReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)
        self.qnetwork = QNetwork(state_size, action_size, seed).to(device)
        self.optimizer = optim.Adam(self.qnetwork.parameters(), lr=LR)

    def step(self, state, action, reward, next_state, done):
        # Create tmp-variables to run a network
        tmp_state = torch.tensor(state).float().unsqueeze(0).to(device)
        tmp_next_state = torch.tensor(next_state).float().unsqueeze(0).to(device)
        tmp_action = torch.tensor([[action]]).long().to(device)

        sigma = reward + (GAMMA * self.qnetwork(tmp_next_state).detach().max(1)[0].unsqueeze(1) -
                          self.qnetwork(tmp_state).gather(1, tmp_action)).item() * (1 - done)

        self.memory.add(state, action, reward, next_state, done, sigma)

        if len(self.memory) > BATCH_SIZE:
            experiences = self.memory.sample()
            self.learn(experiences, GAMMA)

    def act(self, state, exploration_rate=0.05):
        state = np.array(state)
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork.eval()
        with torch.no_grad():
            action_values = self.qnetwork(state)
        self.qnetwork.train()

        if random.random() > exploration_rate:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences, gamma, eps=1e-6):
        states, actions, rewards, next_states, dones, sigmas = experiences

        Q_targets_next = self.qnetwork(next_states).detach().max(1)[0].unsqueeze(1)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        Q_expected = self.qnetwork(states).gather(1, actions)

        loss = F.mse_loss(Q_expected, Q_targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
