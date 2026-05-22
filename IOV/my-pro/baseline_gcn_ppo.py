import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np

# 从用户原有 GNN 思路精简而来的 GCN 算子 (适用于我们连续动作的状态矩阵提取)
class SimpleGCNLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        
    def forward(self, x, adj):
        # 简化的 GCN 传播公式: H' = \sigma(D^{-1/2} A D^{-1/2} H W)
        # 这里用最基础的聚合代表拓扑感知
        support = self.linear(x)
        output = torch.matmul(adj, support)
        return torch.relu(output)

class GCN_ActorCritic(nn.Module):
    def __init__(self, N_vehicles, feature_dim, action_dim):
        super(GCN_ActorCritic, self).__init__()
        
        self.N = N_vehicles
        
        # 拓扑感知层
        self.gcn1 = SimpleGCNLayer(feature_dim, 32)
        self.gcn2 = SimpleGCNLayer(32, 64)
        
        # 将 GCN 输出压平
        self.flatten_dim = self.N * 64
        
        # Actor
        self.actor = nn.Sequential(
            nn.Linear(self.flatten_dim, 128),
            nn.ReLU(),
            nn.Linear(128, self.N * action_dim)
        )
        
        # Critic
        self.critic = nn.Sequential(
            nn.Linear(self.flatten_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
    def forward(self, state, adj):
        # state: (batch, N, feature_dim)
        # adj: (batch, N, N)
        x = self.gcn1(state, adj)
        x = self.gcn2(x, adj)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Actor 输出类别对数
        action_logits = self.actor(x)
        action_logits = action_logits.view(-1, self.N, 4)
        
        value = self.critic(x)
        
        return action_logits, value

class GCN_PPO_Agent:
    def __init__(self, N_vehicles=6, feature_dim=6, action_dim=4, lr=3e-4, gamma=0.99, K_epochs=4, eps_clip=0.2):
        self.device = torch.device("cpu")
        self.N = N_vehicles
        
        self.policy = GCN_ActorCritic(N_vehicles, feature_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        
        self.MseLoss = nn.MSELoss()
        
    def build_adjacency_matrix(self, state):
        # 简单全连接拓扑（根据距离反比构建边权重）
        batch_size = state.shape[0]
        adj = torch.ones((batch_size, self.N, self.N)).to(self.device)
        # 可以提取坐标 state[:, :, 0:2] 计算真实距离矩阵，这里简化为一个基础完全图来完成聚合
        return adj / self.N

    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        adj = self.build_adjacency_matrix(state)
        
        with torch.no_grad():
            action_logits, _ = self.policy(state, adj)
            dist = Categorical(logits=action_logits[0])
            action = dist.sample()
            
        return action.cpu().numpy()
