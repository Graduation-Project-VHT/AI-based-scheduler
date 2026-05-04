"""
config.py — Single source of truth for all hyperparameters and environment settings.
Every other file imports from here. Never hardcode numbers anywhere else.
"""
from dataclasses import dataclass, field
from typing import List


@dataclass
class EnvConfig:
    # LTE cell topology
    n_ues: int = 15             # Number of User Equipments in the cell
    n_rbs: int = 50             # Total Resource Blocks in 5 MHz FDD
    n_usable_rbs: int = 43     # After PDCCH/CRS overhead (~7 RBs reserved)

    # State vector dimensions
    # Each UE contributes 5 features: CQI, buffer, HOL delay, EWMA tput, QCI class
    # Plus 2 global: Jain's Fairness Index, remaining RBs this TTI
    features_per_ue: int = 5
    n_global_features: int = 2
    # Total = 20*5 + 2 = 102

    @property
    def state_dim(self) -> int:
        return self.n_ues * self.features_per_ue + self.n_global_features  # = 102

    # CQI range
    cqi_min: int = 1
    cqi_max: int = 15

    # Buffer (bytes)
    buffer_min: int = 0
    buffer_max: int = 100_000   # 100 KB max queue

    # HOL delay (ms)
    hol_min: float = 0.0
    hol_max: float = 500.0      # 500 ms is considered very bad

    # Reward weights
    # r = w1*ΔThroughput + w2*ΔFairness - w3*ΔDelay
    # These are the most important knobs — wrong weights = bad policy
    w1: float = 1.0   # throughput weight
    w2: float = 0.5   # fairness weight
    w3: float = 0.3   # delay penalty weight

    # QCI traffic classes
    # QCI 1 = voice (delay-sensitive), QCI 4 = video, QCI 9 = best-effort
    qci_classes: List[int] = field(default_factory=lambda: [1, 4, 9])

    # Episode settings
    max_steps_per_episode: int = 1000   # 1000 TTIs = 1 second of simulated LTE


@dataclass
class DQNConfig:
    hidden_dims: List[int] = field(default_factory=lambda: [256, 256])
    learning_rate: float = 1e-4
    gamma: float = 0.99          # discount factor
    replay_capacity: int = 100_000
    batch_size: int = 64
    target_sync_interval: int = 1000   # copy online→target every N steps
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 50_000
    checkpoint_interval: int = 100     # save model every N episodes


# Instantiate once — everyone imports these
ENV = EnvConfig()
DQN = DQNConfig()
