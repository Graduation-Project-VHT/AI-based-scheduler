# src/scheduler/q_network.py
import torch
import torch.nn as nn
from .config import ENV, DQN

class QNetwork(nn.Module):
    def __init__(self, state_dim=ENV.state_dim, action_dim=ENV.n_ues, hidden_dims=DQN.hidden_dims):
        super(QNetwork, self).__init__()

        # Tách danh sách [256, 256] thành 2 biến h1, h2
        h1, h2 = hidden_dims[0], hidden_dims[1]

        self.network = nn.Sequential(
            nn.Linear(state_dim, h1),
            nn.ReLU(),
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Linear(h2, action_dim)
        )

    def forward(self, state):
        return self.network(state)
