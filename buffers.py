import torch
import random
import numpy as np
from collections import deque, namedtuple
from abc import abstractmethod

SAMPLING_PROB = 0.5
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Buffer:
    def __init__(self, action_size, buffer_size, batch_size):
        """Initialize a Buffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size

    @abstractmethod
    def add(self, *args):
        pass

    @abstractmethod
    def sample(self):
        pass

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)


class ReplayBuffer(Buffer):
    """Fixed-size buffer to store experience in tuples <S, A, R, S>."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.

        Params
        ======
            seed (int): random seed
        """
        super().__init__(action_size, buffer_size, batch_size)
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])) \
            .float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)) \
            .float().to(device)

        return states, actions, rewards, next_states, dones


class PrioritizedReplayBuffer(Buffer):
    """Fixed-size buffer to store experience in tuples <S, A, R, S, p>."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        super().__init__(action_size, buffer_size, batch_size)
        self.experience = namedtuple("Experience",
                                     field_names=["state", "action", "reward", "next_state", "done", "sigma"])
        self.seed = random.seed(seed)
        self.priority_norm = 0

    def add(self, state, action, reward, next_state, done, p):
        e = self.experience(state, action, reward, next_state, done, p)
        self.memory.append(e)

    def count_probs(self, eps=1e-6):
        probs = list()

        for *_, s in self.memory:
            p = abs(s) + eps
            probs.append(p ** SAMPLING_PROB)
        probs = np.array(probs, dtype='float64')
        probs /= probs.sum()
        return probs

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        indices = np.random.choice(np.arange(len(self.memory)),
                                   size=self.batch_size, p=self.count_probs())
        experiences = [self.memory[x] for x in indices]

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])) \
            .float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)) \
            .float().to(device)
        sigmas = torch.tensor(np.vstack([e.sigma for e in experiences if e is not None])) \
            .float().to(device)
        return states, actions, rewards, next_states, dones, sigmas
