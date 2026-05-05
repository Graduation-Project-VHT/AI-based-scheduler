"""
Tests for stub_env.py — the ground truth of Stage A.
Run with: python -m pytest tests/ -v
"""
import numpy as np
import pytest
from scheduler.stub_env import StubLTEEnv
from scheduler.config import ENV


class TestReset:
    def test_state_shape(self):
        """State must match with the config."""
        env = StubLTEEnv(seed=0)
        state = env.reset()
        # assert state.shape == (102,), f"Expected (102,), got {state.shape}"
        assert state.shape == (ENV.state_dim,),f"Expected ({ENV.state_dim},), got {state.shape}"

    def test_state_dtype(self):
        """State must be float32 — PyTorch's default dtype."""
        env = StubLTEEnv(seed=0)
        state = env.reset()
        assert state.dtype == np.float32, f"Expected float32, got {state.dtype}"

    def test_state_normalized(self):
        """All state values must be in [0, 1]. Unnormalized values crash NN training."""
        env = StubLTEEnv(seed=0)
        state = env.reset()
        assert np.all(state >= 0.0), f"State has negative values: {state.min()}"
        assert np.all(state <= 1.0), f"State has values > 1: {state.max()}"

    def test_deterministic_with_seed(self):
        """Same seed must produce identical state. Required for reproducible debugging."""
        env1 = StubLTEEnv(seed=42)
        env2 = StubLTEEnv(seed=42)
        np.testing.assert_array_equal(env1.reset(), env2.reset())

    def test_different_seeds_differ(self):
        """Different seeds should produce different states."""
        env1 = StubLTEEnv(seed=1)
        env2 = StubLTEEnv(seed=2)
        assert not np.array_equal(env1.reset(), env2.reset())


class TestStep:
    def test_step_return_types(self):
        """step() must return (ndarray, float, bool, dict)."""
        env = StubLTEEnv(seed=0)
        env.reset()
        next_state, reward, done, info = env.step(0)
        assert isinstance(next_state, np.ndarray)
        assert isinstance(reward, float)
        assert isinstance(done, bool)
        assert isinstance(info, dict)

    def test_next_state_shape_and_dtype(self):
        env = StubLTEEnv(seed=0)
        env.reset()
        next_state, _, _, _ = env.step(0)
        assert next_state.shape == (ENV.state_dim,) # Reflects the defined shape in config file
        assert next_state.dtype == np.float32

    def test_next_state_normalized(self):
        env = StubLTEEnv(seed=0)
        env.reset()
        for action in range(ENV.n_ues):
            next_state, _, done, _ = env.step(action % ENV.n_ues)
            assert np.all(next_state >= 0.0)
            assert np.all(next_state <= 1.0)
            if done:
                break

    def test_invalid_action_raises(self):
        """Action outside [0, n_ues) must raise AssertionError."""
        env = StubLTEEnv(seed=0)
        env.reset()
        with pytest.raises(AssertionError):
            env.step(ENV.n_ues)      # one past the end
        with pytest.raises(AssertionError):
            env.step(-1)

    def test_done_at_max_steps(self):
        """Episode must terminate after exactly max_steps_per_episode steps."""
        env = StubLTEEnv(seed=0)
        env.reset()
        done = False
        steps = 0
        while not done:
            _, _, done, _ = env.step(steps % ENV.n_ues)
            steps += 1
        assert steps == ENV.max_steps_per_episode


class TestReward:
    def test_reward_is_finite(self):
        """Reward must never be NaN or Inf — those silently destroy training."""
        env = StubLTEEnv(seed=0)
        env.reset()
        for i in range(200):
            _, reward, done, _ = env.step(i % ENV.n_ues)
            assert np.isfinite(reward), f"Non-finite reward at step {i}: {reward}"
            if done:
                break

    def test_reward_in_reasonable_range(self):
        """
        Reward scale matters enormously for DQN stability.
        If rewards are too large (e.g., 1000s), gradients explode.
        If too small (e.g., 1e-6), the agent learns nothing.
        We expect [-5, 5] to be safe.
        """
        env = StubLTEEnv(seed=0)
        env.reset()
        rewards = []
        for i in range(500):
            _, reward, done, _ = env.step(i % ENV.n_ues)
            rewards.append(reward)
            if done:
                break
        assert max(rewards) <= 5.0, f"Reward too large: {max(rewards)}"
        assert min(rewards) >= -5.0, f"Reward too negative: {min(rewards)}"


class TestFairness:
    def test_fairness_in_unit_interval(self):
        """Jain's index must always be in [0, 1]."""
        env = StubLTEEnv(seed=0)
        env.reset()
        for i in range(100):
            _, _, done, info = env.step(i % ENV.n_ues)
            f = info["fairness"]
            assert 0.0 <= f <= 1.0, f"Fairness out of range: {f}"
            if done:
                break

    def test_perfect_fairness_on_equal_tput(self):
        """If all UEs have equal throughput, Jain's index should be 1.0."""
        env = StubLTEEnv(seed=0)
        env.reset()
        # Manually set equal EWMA throughput
        env._ewma_tput[:] = 100.0
        assert abs(env._compute_fairness() - 1.0) < 1e-6

    def test_zero_tput_fairness_is_one(self):
        """If all throughputs are 0, fairness should be 1.0 (trivially fair)."""
        env = StubLTEEnv(seed=0)
        env.reset()
        env._ewma_tput[:] = 0.0
        assert env._compute_fairness() == 1.0
