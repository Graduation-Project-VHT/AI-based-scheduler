"""
stub_env.py — Synthetic LTE environment for offline DQN prototyping.
  - state:   np.ndarray, shape (102,), dtype float32, all values in [0, 1]
  - action:  int in [0, n_ues), meaning "give next RB to UE #action"
  - reward:  float, typically in [-2.0, 2.0] (calibrate carefully!)
  - done:    bool, True after max_steps_per_episode TTI-steps
  - info:    dict with raw (un-normalized) metrics for debugging
"""

import numpy as np
from .config import ENV

# Modulation scheme lookup table (3GPP TS 36.213 Table 7.2.3-1)
CQI_BYTES_PER_RB = {
    1:   2, 2:   4, 3:   6, 4:  10, 5:  15, 
    6:  21, 7:  26, 8:  34, 9:  43, 10: 49, 
    11: 59, 12: 70, 13: 81, 14: 92, 15: 100,
}

_CQI_BYTES_ARRAY = np.array(
    [0] + [CQI_BYTES_PER_RB[cqi] for cqi in range(1, 16)],
    dtype=np.float32
)

class StubLTEEnv:
    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)
        self.n_ues = ENV.n_ues
        self.n_usable_rbs = ENV.n_usable_rbs

        # Internal state
        self._cqi = np.zeros(self.n_ues, dtype=np.float32)
        self._buffer = np.zeros(self.n_ues, dtype=np.float32)
        self._hol_delay = np.zeros(self.n_ues, dtype=np.float32)
        self._ewma_tput = np.zeros(self.n_ues, dtype=np.float32)
        self._qci = np.zeros(self.n_ues, dtype=np.float32)

        self._step_count = 0              
        self._rbs_remaining = 0           
        self._rbs_allocated = np.zeros(self.n_ues, dtype=np.int32)
        
        # [FIX] Thêm mảng track số Bytes thực sự truyền được trong 1 TTI
        self._bytes_delivered_this_tti = np.zeros(self.n_ues, dtype=np.float32)

        self._prev_total_tput = 0.0
        self._prev_fairness = 0.0
        self._prev_avg_delay = 0.0

    def reset(self) -> np.ndarray:
        self._step_count = 0
        self._rbs_remaining = self.n_usable_rbs
        self._rbs_allocated[:] = 0
        self._bytes_delivered_this_tti[:] = 0.0

        self._cqi = self.rng.integers(ENV.cqi_min, ENV.cqi_max + 1, size=self.n_ues).astype(np.float32)
        self._buffer = self.rng.uniform(0, ENV.buffer_max * 0.5, size=self.n_ues).astype(np.float32)
        self._hol_delay = self.rng.uniform(ENV.hol_min, ENV.hol_max * 0.3, size=self.n_ues).astype(np.float32)
        self._ewma_tput = np.zeros(self.n_ues, dtype=np.float32)
        self._qci = self.rng.choice(ENV.qci_classes, size=self.n_ues).astype(np.float32)

        self._prev_fairness = self._compute_fairness()
        self._prev_avg_delay = float(np.mean(self._hol_delay))
        self._prev_total_tput = 0.0

        return self._build_state()

    def step(self, action: int):
        assert 0 <= action < self.n_ues, f"Invalid action {action}"

        # =====================================================================
        # [FIX 1] XỬ LÝ VẬT LÝ TỨC THỜI (IMMEDIATE PHYSICS UPDATE)
        # =====================================================================
        cqi_idx = int(self._cqi[action])
        bytes_capacity = _CQI_BYTES_ARRAY[cqi_idx]
        
        # Tính số byte thực sự chở được (không vượt quá Buffer)
        actual_tx_bytes = min(self._buffer[action], bytes_capacity)
        
        # TRỪ BUFFER NGAY LẬP TỨC để AI không bị ảo giác ở step sau
        self._buffer[action] -= actual_tx_bytes
        self._rbs_allocated[action] += 1
        self._rbs_remaining -= 1
        self._bytes_delivered_this_tti[action] += actual_tx_bytes

        # =====================================================================
        # [FIX 2] DENSE REWARD & PENALTY (CHỐNG LÃNG PHÍ TÀI NGUYÊN)
        # =====================================================================
        if actual_tx_bytes == 0:
            # PHẠT NẶNG: Cấp RB cho thằng không có data (lãng phí phổ tần)
            reward = -1.0 
        else:
            # Nếu truyền được data, tính Reward bình thường theo công thức của nhóm
            reward = self._compute_reward()

        # =====================================================================
        # KẾT THÚC TTI (Cập nhật Kênh truyền, Traffic mới, EWMA)
        # =====================================================================
        if self._rbs_remaining == 0:
            self._simulate_tti_end()
            self._rbs_remaining = self.n_usable_rbs   
            self._rbs_allocated[:] = 0
            self._bytes_delivered_this_tti[:] = 0.0 # Reset cho TTI sau

        self._step_count += 1
        done = (self._step_count >= ENV.max_steps_per_episode)
        next_state = self._build_state()

        # =====================================================================
        # [FIX 3] BỔ SUNG DATA CHO FILE INFERENCE.PY ĐỂ VẼ BIỂU ĐỒ
        # =====================================================================
        info = {
            "ue_cqis": self._cqi.copy(),
            "allocated_rbs": int(self._rbs_allocated[action]), # Trả về để inference bóc tách
            "tbs_bytes": float(actual_tx_bytes),               # Trả về để inference bóc tách
            "buffer": self._buffer.copy(),
            "hol_delay": self._hol_delay.copy(),
            "ewma_tput": self._ewma_tput.copy(),
            "fairness": self._compute_fairness(),
            "rbs_remaining": self._rbs_remaining,
            "step": self._step_count,
        }

        return next_state, reward, done, info

    def _simulate_tti_end(self):
        """ Chỉ cập nhật các yếu tố thay đổi theo thời gian (1ms) """
        
        # 1. Update HOL delay
        has_data = self._buffer > 0
        self._hol_delay = np.where(has_data, self._hol_delay + 1.0, 0.0)
        self._hol_delay = np.clip(self._hol_delay, ENV.hol_min, ENV.hol_max)

        # 2. Sinh Traffic mới (New Arrivals)
        new_arrivals = self.rng.exponential(scale=5000, size=self.n_ues).astype(np.float32)
        self._buffer = np.clip(self._buffer + new_arrivals, 0, ENV.buffer_max)

        # 3. Update EWMA Throughput dựa trên tổng số Bytes đã truyền trong TTI vừa qua
        alpha = 0.1
        current_tput = self._bytes_delivered_this_tti * 8 / 1e3   # Đổi ra kbps
        self._ewma_tput = (1 - alpha) * self._ewma_tput + alpha * current_tput

        # 4. Fading kênh truyền (Trôi CQI)
        cqi_drift = self.rng.integers(-1, 2, size=self.n_ues)   
        self._cqi = np.clip(self._cqi + cqi_drift, ENV.cqi_min, ENV.cqi_max)

    def _build_state(self) -> np.ndarray:
        cqi_norm = self._cqi / ENV.cqi_max
        buffer_norm = self._buffer / ENV.buffer_max
        hol_norm = self._hol_delay / ENV.hol_max

        # Chuẩn hóa Throughput
        max_tput = CQI_BYTES_PER_RB[15] * ENV.n_usable_rbs * 8 / 1.0  
        tput_norm = np.clip(self._ewma_tput / max_tput, 0.0, 1.0)

        qci_map = {1: 0.0, 4: 0.5, 9: 1.0}
        qci_norm = np.array([qci_map[int(q)] for q in self._qci], dtype=np.float32)

        fairness = self._compute_fairness()          
        rbs_norm = self._rbs_remaining / self.n_usable_rbs

        state = np.concatenate([
            cqi_norm, buffer_norm, hol_norm, tput_norm, qci_norm, 
            [fairness], [rbs_norm],
        ]).astype(np.float32)

        return state

    def _compute_fairness(self) -> float:
        x = self._ewma_tput
        sum_x = np.sum(x)
        sum_x2 = np.sum(x ** 2)
        if sum_x2 == 0:
            return 1.0
        return float((sum_x ** 2) / (self.n_ues * sum_x2))

    def _compute_reward(self) -> float:
        # Chuẩn hóa để tính delta
        max_tput = CQI_BYTES_PER_RB[15] * ENV.n_usable_rbs * 8 / 1.0
        total_tput = float(np.sum(self._ewma_tput)) / (self.n_ues * max_tput)

        fairness = self._compute_fairness()
        avg_delay = float(np.mean(self._hol_delay)) / ENV.hol_max

        delta_tput = total_tput - self._prev_total_tput
        delta_fairness = fairness - self._prev_fairness
        delta_delay = avg_delay - self._prev_avg_delay

        self._prev_total_tput = total_tput
        self._prev_fairness = fairness
        self._prev_avg_delay = avg_delay

        reward = (ENV.w1 * delta_tput + ENV.w2 * delta_fairness - ENV.w3 * delta_delay)
        return float(np.clip(reward, -5.0, 5.0))
