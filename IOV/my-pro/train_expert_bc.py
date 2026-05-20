import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import time
from residual_ppo_agent import Prior_DNN

def train_bc_native_gpu():
    # 强制使用 CPU (因为 RTX 5070 sm_120 过于先进，当前 PyTorch wheel 尚不兼容)
    device = torch.device("cpu")
    print(f"=== Starting Ultra-Fast Native GPU Behavioral Cloning ===")
    print(f"Using Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")

    # 2. 绕过 DataLoader，直接将全量数据常驻显存！(彻底消灭 CPU 搬运瓶颈)
    print("Loading raw data from disk...")
    raw_data = np.load('data/expert_dataset.npy', allow_pickle=True)
    
    states_np = np.array([sample['state'] for sample in raw_data], dtype=np.float32)
    actions_np = np.array([sample['action'] for sample in raw_data], dtype=np.int64)
    
    # 【核心创新】：打碎集权大脑，把 10 万条全局经验拆解成 60 万条单车本能经验！
    if len(states_np.shape) == 2 and states_np.shape[1] == 24:
        states_np = states_np.reshape(-1, 4)
    else:
        states_np = states_np[:, :, :4].reshape(-1, 4)
    actions_np = actions_np.reshape(-1)
    
    print(f"Reshaped Dataset Shape for Parameter Sharing: States {states_np.shape}, Actions {actions_np.shape}")
    
    state_dim = 4
    action_dim = 4 # 每辆车 4 个 Logits
    
    print("Moving ENTIRE dataset to GPU VRAM at once...")
    # 核心：一次性把所有数据扔进显存，不给 CPU 留任何活！
    states_gpu = torch.FloatTensor(states_np).to(device)
    actions_gpu = torch.LongTensor(actions_np).to(device)

    # 3. 初始化网络与优化器
    model = Prior_DNN(state_dim, action_dim).to(device)
    for param in model.parameters():
        param.requires_grad = True # 解冻进行预训练
        
    optimizer = optim.Adam(model.parameters(), lr=5e-3)
    criterion = nn.CrossEntropyLoss()
    
    epochs = 300
    # 由于数据已在显存，我们可以使用极度疯狂的 Batch Size (例如半数数据)
    batch_size = 50000 
    dataset_size = states_gpu.size(0)
    
    print(f"Training started with GPU native batching (Batch Size: {batch_size})...")
    start_time = time.time()
    
    # 4. 纯显存训练循环 (CPU 处于闲置状态)
    for epoch in range(1, epochs + 1):
        model.train()
        
        # 显存内极速打乱索引
        permutation = torch.randperm(dataset_size, device=device)
        
        total_loss = 0.0
        correct_preds = 0
        total_preds = 0
        
        for i in range(0, dataset_size, batch_size):
            indices = permutation[i:i+batch_size]
            
            # 直接在显存中切片，极速！
            batch_states = states_gpu[indices]
            batch_actions = actions_gpu[indices]
            
            optimizer.zero_grad()
            
            # 前向传播
            logits = model(batch_states) 
            
            # 由于重塑后数据变成了 4 维到 4 维的映射，无需在计算 Loss 时展平
            loss = criterion(logits, batch_actions)
            
            # 反向传播  loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * batch_states.size(0)
            
            # 计算准确率 (全在显存内完成计算)
            predicted = torch.argmax(logits, dim=1)
            correct_preds += (predicted == batch_actions).sum().item()
            total_preds += batch_actions.size(0)
            
        avg_loss = total_loss / dataset_size
        accuracy = correct_preds / total_preds * 100.0
        
        if epoch % 50 == 0 or epoch == 1:
            print(f"Epoch [{epoch}/{epochs}] | Loss: {avg_loss:.4f} | Accuracy: {accuracy:.2f}%")
            
    end_time = time.time()
    print(f"\n[OK] 300 Epochs completed in {end_time - start_time:.2f} seconds!")
            
    # 5. 保存模型
    os.makedirs('model', exist_ok=True)
    save_path = 'model/prior_dnn_expert.pth'
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

if __name__ == "__main__":
    train_bc_native_gpu()
