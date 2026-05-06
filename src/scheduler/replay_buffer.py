# src/scheduler/replay_buffer.py
import numpy as np
import torch
from .config import ENV, DQN

class ReplayBuffer:
    def __init__(self, capacity=DQN.replay_capacity, state_dim=ENV.state_dim):
        self.capacity = capacity
        self.ptr = 0
        self.size = 0

        # Khởi tạo các mảng NumPy tĩnh để tối ưu tốc độ
        self.state      = np.zeros((capacity, state_dim), dtype=np.float32)
        self.action     = np.zeros((capacity, 1), dtype=np.int64)
        self.reward     = np.zeros((capacity, 1), dtype=np.float32)
        self.next_state = np.zeros((capacity, state_dim), dtype=np.float32)
        self.done       = np.zeros((capacity, 1), dtype=np.float32)

    def push(self, state, action, reward, next_state, done):
        self.state[self.ptr]      = state
        self.action[self.ptr]     = action
        self.reward[self.ptr]     = reward
        self.next_state[self.ptr] = next_state
        self.done[self.ptr]       = done

        # Cơ chế cuốn chiếu (Circular)
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size=DQN.batch_size):
        # Bốc ngẫu nhiên indices để học
        indices = np.random.choice(self.size, batch_size, replace=False)

        return (
            torch.FloatTensor(self.state[indices]),
            torch.LongTensor(self.action[indices]),
            torch.FloatTensor(self.reward[indices]),
            torch.FloatTensor(self.next_state[indices]),
            torch.FloatTensor(self.done[indices])
        )

    def __len__(self):
        return self.size
