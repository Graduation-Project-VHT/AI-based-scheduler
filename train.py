# train.py (Đặt ở ngoài cùng thư mục dự án)
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import os
import time

# Sửa lại đường dẫn import cho khớp với cây thư mục của nhóm
from src.scheduler.config import ENV, DQN
from src.scheduler.stub_env import StubLTEEnv
from src.scheduler.dqn_agent import DQNAgent

def get_epsilon(global_step, total_decay_steps):
    """ 
    Hàm tính Epsilon động: Tự động co giãn theo tổng số Steps 
    """
    if global_step >= total_decay_steps:
        return DQN.epsilon_end
    
    decay_rate = (DQN.epsilon_start - DQN.epsilon_end) / total_decay_steps
    return DQN.epsilon_start - (global_step * decay_rate)

def main():
    run_name = f"DQN_StageA_{int(time.time())}"
    writer = SummaryWriter(f"runs/{run_name}")
    os.makedirs("checkpoints", exist_ok=True)

    env = StubLTEEnv(seed=42)
    # Khởi tạo Agent
    agent = DQNAgent(state_dim=ENV.state_dim, n_actions=ENV.n_ues)

    global_step = 0
    N_EPISODES = 5000

    # [FIX] Tự động tính toán Epsilon Decay theo tổng số bước
    # Ép AI dành 80% chặng đường để vừa khám phá vừa học, 20% cuối để khai thác
    TOTAL_STEPS = N_EPISODES * ENV.max_steps_per_episode
    EXPLORATION_STEPS = int(TOTAL_STEPS * 0.8)

    print(f"🚀 Starting Training: {run_name}")
    print(f"📊 State dim: {ENV.state_dim}, Actions: {ENV.n_ues}")
    print(f"🎯 Total Episodes: {N_EPISODES} | Exploration Steps: {EXPLORATION_STEPS}")

    for episode in range(N_EPISODES):
        state = env.reset()
        episode_reward = 0
        episode_losses = []
        episode_q_values =[]
        done = False

        while not done:
            # Dùng EXPLORATION_STEPS vừa tính toán thay vì đọc từ config
            epsilon = get_epsilon(global_step, EXPLORATION_STEPS)
            action = agent.select_action(state, epsilon)

            next_state, reward, done, info = env.step(action)

            # Đẩy vào Replay Buffer của Quý
            agent.memory.push(state, action, reward, next_state, done)

            # Huấn luyện
            loss, avg_q = agent.train_step()
            if loss is not None:
                episode_losses.append(loss)
                episode_q_values.append(avg_q)

            # Sync Target Network
            if global_step % DQN.target_sync_interval == 0:
                agent.sync_target()

            state = next_state
            episode_reward += reward
            global_step += 1

        # Ghi log TensorBoard
        avg_loss = np.mean(episode_losses) if episode_losses else 0
        avg_q = np.mean(episode_q_values) if episode_q_values else 0

        writer.add_scalar("Train/Episode_Reward", episode_reward, episode)
        writer.add_scalar("Train/Loss", avg_loss, episode)
        writer.add_scalar("Train/Average_Q", avg_q, episode)
        writer.add_scalar("Hyperparams/Epsilon", epsilon, episode)

        # [FIX] Giảm tần suất in log xuống mỗi 50 episodes để console đỡ rối
        if episode % 50 == 0:
            print(f"Ep {episode:4d} | Reward: {episode_reward:7.2f} | Loss: {avg_loss:5.3f} | Epsilon: {epsilon:.3f} | Step: {global_step}")

        # Lưu checkpoint (Không cần lưu mốc 0)
        if episode % DQN.checkpoint_interval == 0 and episode > 0:
            agent.save_checkpoint(f"checkpoints/dqn_ep{episode}.pth")

    # Lưu Model Cuối cùng
    agent.save_checkpoint(f"checkpoints/dqn_final.pth")
    writer.close()
    print("✅ Training Complete! Model saved successfully.")

if __name__ == "__main__":
    main()