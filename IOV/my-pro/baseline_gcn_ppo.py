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
    def __init__(self, N_vehicles=6, feature_dim=7, action_dim=4, lr=1e-4, gamma=0.99, K_epochs=4, eps_clip=0.2):
        self.device = torch.device("cpu")
        self.N = N_vehicles
        
        self.policy = GCN_ActorCritic(N_vehicles, feature_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        
        self.MseLoss = nn.MSELoss()
        self.buffer = []
        
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
            action_logits, value = self.policy(state, adj)
            dist = Categorical(logits=action_logits[0])
            action = dist.sample()
            action_logprob = dist.log_prob(action).sum()
            
        return action.cpu().numpy(), action_logprob.item(), value.item()

    def store_transition(self, transition):
        self.buffer.append(transition)

    def update(self):
        if len(self.buffer) == 0:
            return
            
        states = torch.FloatTensor(np.array([t[0] for t in self.buffer])).to(self.device)
        actions = torch.FloatTensor(np.array([t[1] for t in self.buffer])).to(self.device)
        old_logprobs = torch.FloatTensor(np.array([t[2] for t in self.buffer])).to(self.device)
        rewards = [t[3] / 10.0 for t in self.buffer]
        is_terminals = [t[4] for t in self.buffer]
        
        returns = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(rewards), reversed(is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            returns.insert(0, discounted_reward)
            
        returns = torch.FloatTensor(returns).to(self.device)
        
        for _ in range(self.K_epochs):
            adj = self.build_adjacency_matrix(states)
            action_logits, values = self.policy(states, adj)
            dist = Categorical(logits=action_logits)
            
            logprobs = dist.log_prob(actions).sum(dim=1)
            dist_entropy = dist.entropy().sum(dim=1)
            
            advantages = returns - values.squeeze().detach()
            if advantages.std() > 1e-6:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                
            ratios = torch.exp(logprobs - old_logprobs)
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages
            
            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(values.squeeze(), returns) - 0.01 * dist_entropy
            
            self.optimizer.zero_grad()
            loss.mean().backward()
            nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)
            self.optimizer.step()
            
        self.buffer = []

    def save_model(self, path):
        torch.save(self.policy.state_dict(), path)

    def load_model(self, path):
        self.policy.load_state_dict(torch.load(path, map_location=self.device))
