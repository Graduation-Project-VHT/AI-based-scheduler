"""
stub_env.py — Synthetic LTE environment for offline DQN prototyping.
  - state:   np.ndarray, shape (102,), dtype float32, all values in [0, 1]
  - action:  int in [0, n_ues), meaning "give next RB to UE #action"
  - reward:  float, typically in [-2.0, 2.0] (calibrate carefully!)
  - done:    bool, True after max_steps_per_episode TTI-steps
  - info:    dict with raw (un-normalized) metrics for debugging
"""

import numpy as np
from config import ENV

# Modulation scheme lookup table
# Source: 3GPP TS 36.213 Table 7.2.3-1
CQI_BYTES_PER_RB = {
    1:   2,   # QPSK,   efficiency 0.1523
    2:   4,   # QPSK,   efficiency 0.2344
    3:   6,   # QPSK,   efficiency 0.3770
    4:  10,   # QPSK,   efficiency 0.6016
    5:  15,   # QPSK,   efficiency 0.8770
    6:  21,   # QPSK,   efficiency 1.1758
    7:  26,   # 16-QAM, efficiency 1.4766
    8:  34,   # 16-QAM, efficiency 1.9141
    9:  43,   # 16-QAM, efficiency 2.4063
    10: 49,   # 64-QAM, efficiency 2.7305
    11: 59,   # 64-QAM, efficiency 3.3223
    12: 70,   # 64-QAM, efficiency 3.9023
    13: 81,   # 64-QAM, efficiency 4.5234
    14: 92,   # 64-QAM, efficiency 5.1152
    15: 100,  # 64-QAM, efficiency 5.5547
}

# Convert to numpy array for vectorized lookup (index 0 unused, CQI starts at 1)
_CQI_BYTES_ARRAY = np.array(
    [0] + [CQI_BYTES_PER_RB[cqi] for cqi in range(1, 16)],
    dtype=np.float32
)

class StubLTEEnv:
    """
    A synthetic 4G LTE cell environment.

    One "step" = one RB allocation decision within a TTI.
    One "episode" = max_steps_per_episode allocation decisions.

    Why per-RB and not per-TTI?
    The action space per TTI is "allocate ALL RBs for ALL UEs simultaneously."
    That's combinatorially huge. Instead, we decompose it:
    each step allocates ONE RB to ONE UE, and we repeat until
    all usable RBs in this TTI are exhausted — then we move to the next TTI.
    This keeps |action| = n_ues = 20, tractable for DQN.
    """

    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)  # seeded RNG for reproducibility
        self.n_ues = ENV.n_ues
        self.n_usable_rbs = ENV.n_usable_rbs

        # Internal state — raw (un-normalized) values
        self._cqi = np.zeros(self.n_ues, dtype=np.float32)
        self._buffer = np.zeros(self.n_ues, dtype=np.float32)
        self._hol_delay = np.zeros(self.n_ues, dtype=np.float32)
        self._ewma_tput = np.zeros(self.n_ues, dtype=np.float32)
        self._qci = np.zeros(self.n_ues, dtype=np.float32)

        # Episode counters
        self._step_count = 0              # total steps in this episode
        self._rbs_remaining = 0           # RBs left in current TTI
        self._rbs_allocated = np.zeros(self.n_ues, dtype=np.int32)

        # Previous-TTI metrics for computing deltas (reward is a *change*)
        self._prev_total_tput = 0.0
        self._prev_fairness = 0.0
        self._prev_avg_delay = 0.0

    # PUBLIC API (the Gym contract)

    def reset(self) -> np.ndarray:
        """
        Start a new episode. Randomize initial UE conditions.
        Returns the initial state vector.
        """
        self._step_count = 0
        self._rbs_remaining = self.n_usable_rbs
        self._rbs_allocated[:] = 0

        # Random initial CQIs: uniform integer in [1, 15]
        self._cqi = self.rng.integers(ENV.cqi_min, ENV.cqi_max + 1,
                                      size=self.n_ues).astype(np.float32)

        # Random initial buffers: some UEs have data, some don't
        self._buffer = self.rng.uniform(0, ENV.buffer_max * 0.5,
                                        size=self.n_ues).astype(np.float32)

        # Random initial HOL delays
        self._hol_delay = self.rng.uniform(ENV.hol_min, ENV.hol_max * 0.3,
                                           size=self.n_ues).astype(np.float32)

        # EWMA throughput starts at zero (no history yet)
        self._ewma_tput = np.zeros(self.n_ues, dtype=np.float32)

        # Assign random QCI class to each UE
        self._qci = self.rng.choice(ENV.qci_classes,
                                    size=self.n_ues).astype(np.float32)

        # Initialize prev-TTI metrics
        self._prev_total_tput = 0.0
        self._prev_fairness = self._compute_fairness()
        self._prev_avg_delay = float(np.mean(self._hol_delay))

        return self._build_state()

    def step(self, action: int):
        """
        Allocate one RB to UE #action.

        Args:
            action: int in [0, n_ues). Which UE gets this RB.

        Returns:
            next_state: np.ndarray shape (102,) float32
            reward:     float
            done:       bool
            info:       dict  ← raw metrics for debugging/logging
        """
        assert 0 <= action < self.n_ues, f"Invalid action {action}"

        # 1. Apply the allocation
        self._rbs_allocated[action] += 1
        self._rbs_remaining -= 1

        # 2. If TTI is complete (all RBs allocated), simulate what happens
        if self._rbs_remaining == 0:
            self._simulate_tti()         # update buffers, delays, throughput
            self._rbs_remaining = self.n_usable_rbs   # start fresh next TTI
            self._rbs_allocated[:] = 0

        # 3. Compute reward
        reward = self._compute_reward()

        # 4. Advance step counter and check episode end
        self._step_count += 1
        done = (self._step_count >= ENV.max_steps_per_episode)

        # 5. Build next state
        next_state = self._build_state()

        info = {
            "cqi": self._cqi.copy(),
            "buffer": self._buffer.copy(),
            "hol_delay": self._hol_delay.copy(),
            "ewma_tput": self._ewma_tput.copy(),
            "fairness": self._compute_fairness(),
            "rbs_remaining": self._rbs_remaining,
            "step": self._step_count,
        }

        return next_state, reward, done, info

    # PRIVATE HELPERS

    def _simulate_tti(self):
        """
        After all RBs are assigned, simulate the outcome of this TTI.
        This is the 'physics' of the fake LTE cell.

        Real OAI does this automatically. Here we approximate it:
        - Each allocated RB delivers bytes proportional to CQI (higher CQI = more data)
        - Buffer drains by the delivered bytes
        - HOL delay increases if buffer is non-empty, resets if drained
        - EWMA throughput updates with α=0.1 (slow moving average)
        """
        # bytes_per_rb = self._cqi * 50.0   # shape (n_ues,)

        # Bytes delivered per RB is proportional to CQI
        # Vectorized lookup: for each UE, look up its CQI in the table
        cqi_indices = self._cqi.astype(np.int32)          # float CQI → int index
        bytes_per_rb = _CQI_BYTES_ARRAY[cqi_indices]      # shape (n_ues,)

        # Total bytes delivered this TTI per UE
        bytes_delivered = self._rbs_allocated * bytes_per_rb   # shape (n_ues,)

        # Drain buffer (can't deliver more than what's in the buffer)
        self._buffer = np.maximum(0, self._buffer - bytes_delivered)

        # Update HOL delay: grows if buffer non-empty, resets if buffer empty
        # (1ms per TTI growth for non-empty queues)
        has_data = self._buffer > 0
        self._hol_delay = np.where(has_data,
                                   self._hol_delay + 1.0,   # +1ms per TTI
                                   0.0)                       # reset when empty
        self._hol_delay = np.clip(self._hol_delay, ENV.hol_min, ENV.hol_max)

        # New data arrives randomly each TTI (simulate traffic generator)
        new_arrivals = self.rng.exponential(
            scale=5000,   # avg 5000 bytes per TTI per UE
            size=self.n_ues
        ).astype(np.float32)
        self._buffer = np.clip(self._buffer + new_arrivals, 0, ENV.buffer_max)

        # EWMA throughput update: α=0.1
        # ewma_new = 0.9 * ewma_old + 0.1 * current_throughput
        alpha = 0.1
        current_tput = bytes_delivered * 8 / 1e3   # convert to kbps (1ms TTI)
        self._ewma_tput = (1 - alpha) * self._ewma_tput + alpha * current_tput

        # Slowly drift CQIs (simulate channel variation)
        cqi_drift = self.rng.integers(-1, 2, size=self.n_ues)   # -1, 0, or +1
        self._cqi = np.clip(self._cqi + cqi_drift, ENV.cqi_min, ENV.cqi_max)

    def _build_state(self) -> np.ndarray:
        """
        Assemble the 102-dim state vector and NORMALIZE everything to [0, 1].

        Normalization is critical — neural networks behave poorly when inputs
        have wildly different scales (e.g., buffer in bytes [0, 100000] vs
        CQI [1, 15]). Normalizing makes the gradient well-behaved.

        Layout:
          [0:20]   CQI per UE         → divide by 15
          [20:40]  buffer per UE      → divide by buffer_max
          [40:60]  HOL delay per UE   → divide by hol_max
          [60:80]  EWMA tput per UE   → divide by max possible tput
          [80:100] QCI class per UE   → map {1,4,9} → {0, 0.5, 1.0}
          [100]    Jain's Fairness Index  → already in [0, 1]
          [101]    Remaining RBs      → divide by n_usable_rbs
        """
        # Per-UE features normalized
        cqi_norm = self._cqi / ENV.cqi_max
        buffer_norm = self._buffer / ENV.buffer_max
        hol_norm = self._hol_delay / ENV.hol_max

        # Max possible EWMA tput: CQI=15, all RBs, in kbps
        # max_tput = ENV.cqi_max * 50.0 * self.n_usable_rbs * 8 / 1e3
        # tput_norm = np.clip(self._ewma_tput / max_tput, 0.0, 1.0)

        # Max possible EWMA tput: CQI=15 (100 bytes/RB), all 43 RBs, in kbps
        # 100 bytes × 43 RBs × 8 bits / 1ms TTI = 34,400 kbps per UE (theoretical max)
        max_tput = CQI_BYTES_PER_RB[15] * ENV.n_usable_rbs * 8 / 1.0  # kbps
        tput_norm = np.clip(self._ewma_tput / max_tput, 0.0, 1.0)

        # QCI normalization: map integer class to [0, 1]
        qci_map = {1: 0.0, 4: 0.5, 9: 1.0}
        qci_norm = np.array([qci_map[int(q)] for q in self._qci], dtype=np.float32)

        # Global features
        fairness = self._compute_fairness()          # already in [0, 1]
        rbs_norm = self._rbs_remaining / self.n_usable_rbs

        state = np.concatenate([
            cqi_norm,           # 20 values
            buffer_norm,        # 20 values
            hol_norm,           # 20 values
            tput_norm,          # 20 values
            qci_norm,           # 20 values
            [fairness],         # 1 value
            [rbs_norm],         # 1 value
        ]).astype(np.float32)

        assert state.shape == (ENV.state_dim,), f"State shape mismatch: {state.shape}"
        return state

    def _compute_fairness(self) -> float:
        """
        Range: [1/n, 1.0] where 1.0 = perfectly fair, 1/n = totally unfair.

        We use EWMA throughput as the 'resource' each UE is receiving.
        If all throughputs are 0 (start of episode), return 1.0 (trivially fair).
        """
        x = self._ewma_tput
        sum_x = np.sum(x)
        sum_x2 = np.sum(x ** 2)
        if sum_x2 == 0:
            return 1.0
        return float((sum_x ** 2) / (self.n_ues * sum_x2))

    def _compute_reward(self) -> float:
        """
        All deltas are normalized so weights are comparable:
        - ΔThroughput in [0, 1] range (change in total normalized throughput)
        - ΔFairness in [-1, 1] (change in Jain's index)
        - ΔDelay in [-1, 1] (penalized — positive delta means delay got worse)
        """
        # Current total throughput (normalized)
        max_tput = ENV.cqi_max * 50.0 * self.n_usable_rbs * 8 / 1e3
        total_tput = float(np.sum(self._ewma_tput)) / (self.n_ues * max_tput)

        fairness = self._compute_fairness()
        avg_delay = float(np.mean(self._hol_delay)) / ENV.hol_max

        delta_tput = total_tput - self._prev_total_tput
        delta_fairness = fairness - self._prev_fairness
        delta_delay = avg_delay - self._prev_avg_delay

        # Update previous-TTI baseline
        self._prev_total_tput = total_tput
        self._prev_fairness = fairness
        self._prev_avg_delay = avg_delay

        reward = (ENV.w1 * delta_tput
                  + ENV.w2 * delta_fairness
                  - ENV.w3 * delta_delay)

        return float(reward)
