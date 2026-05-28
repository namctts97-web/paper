import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import os
import time

# 从残差代理中直接引出 Prior_DNN，确保模型定义 100% 对齐
from residual_ppo_agent_v3 import Prior_DNN

def train_bc():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=========================================")
    print(f"=== Prior DNN 行为克隆 (Behavioral Cloning) ===")
    print(f"=== Oracle 知识蒸馏 (Knowledge Distillation) ===")
    print(f"Using Device: {device}")
    print(f"=========================================")
    
    data_path = 'data/expert_dataset_v3_clean.npy'
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found!")
        return
        
    print("Loading raw data from disk...")
    raw_data = np.load(data_path, allow_pickle=True).item()
    states_np = raw_data['states']   # Shape: (N, 74)
    actions_np = raw_data['actions'] # Shape: (N, 10)
    
    num_samples = states_np.shape[0]
    state_dim = states_np.shape[1]
    num_vehicles = actions_np.shape[1]
    action_dim = num_vehicles * 4 # 10辆车 * 4种动作 = 40
    
    print(f"Dataset Shape: States {states_np.shape}, Actions {actions_np.shape}")
    print(f"State Dim: {state_dim}, Total Action Logits: {action_dim}")
    
    # 划分验证集 (20%) 防止过拟合
    indices = np.random.permutation(num_samples)
    val_size = int(num_samples * 0.2)
    train_idx, val_idx = indices[val_size:], indices[:val_size]
    
    X_train = torch.FloatTensor(states_np[train_idx]).to(device)
    Y_train = torch.LongTensor(actions_np[train_idx]).to(device)
    X_val = torch.FloatTensor(states_np[val_idx]).to(device)
    Y_val = torch.LongTensor(actions_np[val_idx]).to(device)
    
    # 初始化模型
    model = Prior_DNN(state_dim, action_dim).to(device)
    for param in model.parameters():
        param.requires_grad = True # 解冻进行预训练
        
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    # 多维 Cross Entropy: Y_train 是 (Batch, 10)，需要将 logits 变成 (Batch, 4, 10)
    criterion = nn.CrossEntropyLoss()
    
    epochs = 300
    batch_size = 128
    
    train_dataset = TensorDataset(X_train, Y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    val_dataset = TensorDataset(X_val, Y_val)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    print("Training started...")
    start_time = time.time()
    
    best_val_loss = float('inf')
    
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            logits = model(batch_x) # (Batch, 40)
            logits = logits.view(-1, num_vehicles, 4) # (Batch, 10, 4)
            # CrossEntropyLoss expects (Batch, C, d1, d2...) and target (Batch, d1, d2...)
            # We transpose logits to (Batch, 4, 10)
            logits = logits.transpose(1, 2) 
            
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * batch_x.size(0)
            
        avg_train_loss = total_loss / len(train_dataset)
        
        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                logits = model(batch_x)
                logits = logits.view(-1, num_vehicles, 4)
                
                preds = torch.argmax(logits, dim=2) # (Batch, 10)
                correct += (preds == batch_y).sum().item()
                total += batch_y.numel()
                
                logits = logits.transpose(1, 2)
                loss = criterion(logits, batch_y)
                val_loss += loss.item() * batch_x.size(0)
                
        avg_val_loss = val_loss / len(val_dataset)
        val_acc = correct / total * 100.0
        
        if epoch % 20 == 0 or epoch == 1:
            print(f"Epoch [{epoch}/{epochs}] | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.2f}%")
            
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            os.makedirs('model', exist_ok=True)
            torch.save(model.state_dict(), 'model/prior_dnn_expert.pth')
            
    end_time = time.time()
    print(f"\n[OK] Training completed in {end_time - start_time:.2f} seconds!")
    print(f"Best Val Loss: {best_val_loss:.4f}")
    print("Model saved to model/prior_dnn_expert.pth")

if __name__ == "__main__":
    train_bc()
