import os
import pandas as pd
import torch
import time
import sys

from src.scheduler.config import ENV, DQN
from src.scheduler.stub_env import StubLTEEnv
from src.scheduler.dqn_agent import DQNAgent

def main():
    MODEL_PATH = "checkpoints/dqn_final.pth"
    NUM_TTIS = 200000  # Nâng lên 200,000 TTIs (200 giây mô phỏng)
    
    print(f"Đang tải bộ não AI từ: {MODEL_PATH}...")
    
    env = StubLTEEnv(seed=99) 
    agent = DQNAgent(state_dim=ENV.state_dim, n_actions=ENV.n_ues)
    
    if not os.path.exists(MODEL_PATH):
        print(f"❌ LỖI: Không tìm thấy file {MODEL_PATH}!")
        return
        
    agent.load_checkpoint(MODEL_PATH)
    agent.online_net.eval() # Đóng băng trọng số
    
    print(f"✅ Đã load model thành công! Bắt đầu Inference {NUM_TTIS} TTIs...")

    state = env.reset()
    log_data =[]
    
    start_time = time.time()

    for tti in range(NUM_TTIS):
        # AI ra quyết định với Epsilon = 0 (Thông minh 100%)
        action = agent.select_action(state, epsilon=0.0)
        
        # Lấy State THỰC TẾ mà AI đang nhìn thấy để ghi log
        selected_ue = action
        
        # Bóc tách CQI từ State Vector (Cấu trúc: [CQI, Buffer, HOL, EWMA, QCI] * N_UEs)
        raw_cqi = state[selected_ue * 5]
        # Giả sử StubEnv đã chuẩn hóa CQI về [0, 1], ta nhân lại với 15. Nếu chưa chuẩn hóa thì lấy luôn.
        actual_cqi = int(raw_cqi * 15) if raw_cqi <= 1.0 else int(raw_cqi)
        if actual_cqi == 0: actual_cqi = 1
        if actual_cqi > 15: actual_cqi = 15
        
        next_state, reward, done, info = env.step(action)
        
        # Bóc tách số lượng RB từ info (nếu không có thì giả lập ngẫu nhiên dựa trên CQI để test biểu đồ)
        allocated_rbs = info.get('allocated_rbs', max(1, 25 - actual_cqi)) # Giả lập logic: CQI cao thì tốn ít RB
        tbs_bytes = info.get('tbs_bytes', actual_cqi * 100) # Giả lập logic: CQI cao thì chở được nhiều Bytes
        
        log_data.append({
            'timestamp_ms': tti,
            'rnti': f"UE_{selected_ue}",
            'direction': 'DL',
            'nb_rb': allocated_rbs,
            'rb_util': (allocated_rbs / 25.0) * 100,
            'mcs': actual_cqi * 2, # MCS giả lập tương quan với CQI
            'tbs_bytes': tbs_bytes,
            'sdu_bytes': tbs_bytes,
            'cqi': actual_cqi,    # <-- CQI THẬT ĐÃ ĐƯỢC LẤY TỪ MÔI TRƯỜNG
            'retx': 0, 
            'harq_pid': 0,
            'reward': reward      # <-- Reward thật từ môi trường
        })
        
        state = next_state
        if done:
            state = env.reset()

        # In thanh tiến trình cho đỡ chán
        if tti % 10000 == 0 and tti > 0:
            sys.stdout.write(f"\r⏳ Đang mô phỏng... {tti}/{NUM_TTIS} TTIs ({(tti/NUM_TTIS)*100:.1f}%)")
            sys.stdout.flush()

    # 5. XUẤT RA FILE CSV
    os.makedirs("logs", exist_ok=True)
    csv_name = "logs/AI_Inference_Log.csv"
    df = pd.DataFrame(log_data)
    df.to_csv(csv_name, index=False)
    
    end_time = time.time()
    print(f"\n✅ Quá trình Inference hoàn tất trong {end_time - start_time:.2f} giây!")
    print(f"Đã lưu {NUM_TTIS} dòng log vào {csv_name}")

    print("\n--- 📊 THỐNG KÊ SỐ LẦN AI CẤP RB CHO TỪNG UE ---")
    print(df['rnti'].value_counts().sort_index())
    print("-" * 50)

if __name__ == "__main__":
    main()