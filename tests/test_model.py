# tests/test_model.py
import torch
import numpy as np

# Updated imports pointing to the src/scheduler directory
from src.scheduler.config import ENV, DQN
from src.scheduler.q_network import QNetwork
from src.scheduler.replay_buffer import ReplayBuffer

def test_q_network_nan():
    print("Testing Q-Network for NaN outputs...")
    model = QNetwork()

    # Generate mock state using the dynamic state_dim from ENV config
    mock_state = torch.randn(10, ENV.state_dim)

    # Deliberately multiply by a huge number to test gradient explosion / NaN limits
    extreme_state = mock_state * 1e6
    output = model(extreme_state)

    assert not torch.isnan(output).any(), "ERROR: Network returned NaN values!"
    print("-> Q-Network is stable, no NaNs detected.")

def test_replay_buffer_circular():
    print("Testing Replay Buffer circular (wrap-around) logic...")

    # Create a tiny buffer of size 5 to test the overwrite mechanic
    buffer = ReplayBuffer(capacity=5, state_dim=ENV.state_dim)

    # Push 7 items into a buffer of size 5
    for i in range(7):
        mock_state = np.ones(ENV.state_dim) * i  # Mark state with the loop index
        buffer.push(mock_state, i, 1.0, mock_state, 0.0)

    assert len(buffer) == 5, "ERROR: Buffer size exceeded capacity!"
    assert buffer.ptr == 2, "ERROR: Pointer is at the wrong position after wrap-around!"

    # Check if the oldest records (0 and 1) were correctly overwritten by the newest (5 and 6)
    assert buffer.state[0][0] == 5.0, "ERROR: Circular overwrite failed at index 0!"
    assert buffer.state[1][0] == 6.0, "ERROR: Circular overwrite failed at index 1!"

    print("-> Replay Buffer correctly evicts old experiences and wraps around.")

if __name__ == "__main__":
    test_q_network_nan()
    test_replay_buffer_circular()
