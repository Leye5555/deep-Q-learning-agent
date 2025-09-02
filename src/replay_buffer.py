from collections import deque, namedtuple
import random

class ReplayBuffer:
    def __init__(self, size=10000):
        self.buffer = deque(maxlen=size)
        self.Transition = namedtuple('Transition', ['state', 'action', 'reward', 'next_state', 'done'])

    def push(self, *args):
        self.buffer.append(self.Transition(*args))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        batch = self.Transition(*zip(*batch))
        return batch

    def __len__(self):
        return len(self.buffer)
