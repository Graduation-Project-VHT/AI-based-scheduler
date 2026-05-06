# tests/test_smoke.py
import numpy as np

from scheduler.config import ENV, DQN
from scheduler.stub_env import StubLTEEnv
from scheduler.dqn_agent import DQNAgent

def test_end_to_end_smoke():
    """ End-to-End Smoke Test: Trivial 2-UE scenario """
    
    # 1. Ép cấu hình Trivial
    ENV.n_ues = 2
    #ENV.state_dim = 2 * ENV.features_per_ue + ENV.n_global_features
    DQN.batch_size = 32
    DQN.epsilon_decay_steps = 100 * ENV.max_steps_per_episode 
    
    env = StubLTEEnv(seed=123)
    agent = DQNAgent(state_dim=ENV.state_dim, n_actions=ENV.n_ues)

    global_step = 0
    rewards_history =[]

    print("\nStarting End-to-End Smoke Test (200 episodes)...")
    for episode in range(200):
        state = env.reset()
        ep_reward = 0
        done = False
        
        while not done:
            eps = max(0.01, 1.0 - (global_step / DQN.epsilon_decay_steps))
            action = agent.select_action(state, eps)
            
            next_state, reward, done, _ = env.step(action)
            agent.memory.push(state, action, reward, next_state, done)
            
            agent.train_step()
            
            if global_step % 500 == 0:
                agent.sync_target()
                
            state = next_state
            ep_reward += reward
            global_step += 1
            
        rewards_history.append(ep_reward)

    avg_reward_start = np.mean(rewards_history[:20])
    avg_reward_end = np.mean(rewards_history[-20:])

    print(f"Smoke Test | Start Reward: {avg_reward_start:.2f} -> End Reward: {avg_reward_end:.2f}")

    # CHỨNG MINH AGENT CÓ HỌC: Reward cuối phải lớn hơn Reward ban đầu
    assert avg_reward_end > avg_reward_start, "ERROR: Agent failed to learn in a trivial 2-UE scenario!"
    
    print("-> Smoke Test PASSED! Agent is learning correctly.")

    # Dọn dẹp (Trả lại config cũ)
    ENV.n_ues = 15 
    #ENV.state_dim = 15 * ENV.features_per_ue + ENV.n_global_features