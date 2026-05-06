# src/scheduler/dqn_agent.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import os

from .config import DQN, ENV
from .q_network import QNetwork         # Import code của Quý
from .replay_buffer import ReplayBuffer # Import code của Quý

class DQNAgent:
    def __init__(self, state_dim=ENV.state_dim, n_actions=ENV.n_ues, device="cpu"):
        self.device = torch.device(device)
        self.n_actions = n_actions

        # 1. Khởi tạo 2 mạng nơ-ron (Sử dụng đúng tham số của Quý)
        self.online_net = QNetwork(state_dim=state_dim, action_dim=n_actions, hidden_dims=DQN.hidden_dims).to(self.device)
        self.target_net = QNetwork(state_dim=state_dim, action_dim=n_actions, hidden_dims=DQN.hidden_dims).to(self.device)
        
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval() # Target net chỉ để tính Bellman target

        # 2. Khởi tạo Optimizer
        self.optimizer = optim.Adam(self.online_net.parameters(), lr=DQN.learning_rate)

        # 3. Khởi tạo Replay Buffer (Sử dụng code của Quý)
        self.memory = ReplayBuffer(capacity=DQN.replay_capacity, state_dim=state_dim)

    def select_action(self, state, epsilon):
        """ Epsilon-greedy action selection """
        if random.random() < epsilon:
            # Exploration
            return random.randint(0, self.n_actions - 1)
        else:
            # Exploitation
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_values = self.online_net(state_tensor)
            return int(q_values.argmax(dim=1).item())

    def train_step(self):
        """ Rút data từ Replay Buffer và cập nhật Online Network """
        if len(self.memory) < DQN.batch_size:
            return None, None 

        # Rút batch (Code của Quý ĐÃ TRẢ VỀ PYTORCH TENSORS, không phải numpy array nữa!)
        states, actions, rewards, next_states, dones = self.memory.sample(DQN.batch_size)

        # Đẩy thẳng vào GPU/CPU. 
        # LƯU Ý: Code của Quý đã định nghĩa action, reward, done có shape là (batch, 1)
        # NÊN TUYỆT ĐỐI KHÔNG DÙNG .unsqueeze(1) ở đây nữa để tránh lỗi chiều (Dimension Error)!
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)

        # --- TÍNH TOÁN LOSS ---
        current_q_values = self.online_net(states).gather(1, actions)

        with torch.no_grad(): # FOOTGUN 3: Gradient leak footgun
            # Lấy Max Q của next_state từ Target Network (FOOTGUN 1)
            next_q_values = self.target_net(next_states).max(1)[0].unsqueeze(1)
            
            # Tính Bellman Target có (1 - dones) (FOOTGUN 2)
            target_q_values = rewards + (DQN.gamma * next_q_values * (1 - dones))

        loss = nn.MSELoss()(current_q_values, target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online_net.parameters(), max_norm=1.0)
        self.optimizer.step()

        return loss.item(), current_q_values.mean().item()

    def sync_target(self):
        self.target_net.load_state_dict(self.online_net.state_dict())

    def save_checkpoint(self, filepath):
        torch.save(self.online_net.state_dict(), filepath)

    def load_checkpoint(self, filepath):
        self.online_net.load_state_dict(torch.load(filepath))
        self.sync_target()