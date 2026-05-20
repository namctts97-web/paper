import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np

class Prior_DNN(nn.Module):
    """先验网络：系统的“本能反应”，权重冻结"""
    def __init__(self, state_dim, action_dim):
        super(Prior_DNN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, action_dim)
        )
        # 严格冻结：不参与反向传播
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, state):
        return self.net(state)

class Residual_ActorCritic(nn.Module):
    """残差策略网络：系统的“免疫调节器” (CTDE 架构)"""
    def __init__(self, N_vehicles=6, feature_dim=6, action_dim=4):
        super(Residual_ActorCritic, self).__init__()
        # Decentralized Actor: 6维进，4维出 (参数共享，切断串扰)
        self.actor = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
            # 【核心修正】：删掉 nn.Tanh()，打破边界死锁！允许 Residual 产生足够大的值去反杀 Prior
        )
        # 初始化索引改为 -1 (因为去掉了 Tanh)
        nn.init.orthogonal_(self.actor[-1].weight, gain=0.01)
        nn.init.constant_(self.actor[-1].bias, 0.0)
        
        # Centralized Critic: 接收展平的 (N*6) 维全战局信息，统揽全局
        self.critic = nn.Sequential(
            nn.Linear(N_vehicles * feature_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

class ResidualPPOAgent:
    def __init__(self, state_dim, action_dim, lr=1e-4, gamma=0.99, K_epochs=4, eps_clip=0.2):
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.residual_scale = 2.0 # 退回 2.0，避免过度放大标将导致策略崩溃
        
        # 强制使用 CPU (规避 RTX 5070 的 sm_120 兼容性报错)
        self.device = torch.device("cpu")
        
        # 任务 1: 定义先验网络 (已冻结) 并移动到设备
        # 左脑(先验)是 4进4出 的纯粹微型本能网络
        self.prior_net = Prior_DNN(4, 4).to(self.device)
        import os
        if os.path.exists('model/prior_dnn_expert.pth'):
            print("=> Loading pre-trained Prior DNN from expert data...", flush=True)
            self.prior_net.load_state_dict(torch.load('model/prior_dnn_expert.pth', map_location=self.device))
            
        # 任务 2: 定义 CTDE 残差策略网络 并移动到设备
        self.residual_net = Residual_ActorCritic(N_vehicles=6, feature_dim=6, action_dim=4).to(self.device)
        
        self.optimizer = optim.Adam(self.residual_net.parameters(), lr=lr)
        self.MseLoss = nn.MSELoss()
        
        self.buffer = []

    def select_action(self, state):
        # state shape: (6, 6)
        state = torch.FloatTensor(state).to(self.device)
        N = state.shape[0]
        with torch.no_grad():
            # 1. 左脑本能 (提取前4维特征)
            prior_x = state[:, :4].clone() # shape: (6, 4)
            # 【护目镜】：截断 D_i (索引2) 防止 OOD 击穿冻结的网络
            prior_x[:, 2] = torch.clamp(prior_x[:, 2], max=5.0)
            logits_prior = self.prior_net(prior_x) # shape: (6, 4)
            
            # 2. 右脑修正 (完整6维特征)
            delta_logits = self.residual_net.actor(state) # shape: (6, 4)
            
            # 融合
            logits_final = logits_prior + self.residual_scale * delta_logits # shape: (6, 4)
            
            # 3. CTDE 上帝视角评估 (展平 36 维)
            value = self.residual_net.critic(state.view(-1))
            
            # 采样动作
            dist = Categorical(logits=logits_final)
            action = dist.sample() # shape: (6,)
            action_logprob = dist.log_prob(action).sum() # 所有动作概率的联合对数
            
        return action.cpu().numpy(), action_logprob.item(), value.item()

    def store_transition(self, transition):
        self.buffer.append(transition)

    def update(self):
        if len(self.buffer) == 0:
            return
            
        # 转换 buffer 数据并移动到设备
        states = torch.FloatTensor(np.array([t[0] for t in self.buffer])).to(self.device)
        actions = torch.FloatTensor(np.array([t[1] for t in self.buffer])).to(self.device)
        old_logprobs = torch.FloatTensor(np.array([t[2] for t in self.buffer])).to(self.device)
        # 核心修正：缩放 Reward (除以 10.0) 稳定 Critic 价值网络
        rewards = [t[3] / 10.0 for t in self.buffer]
        is_terminals = [t[4] for t in self.buffer]
        
        # 计算回报和优势
        returns = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(rewards), reversed(is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            returns.insert(0, discounted_reward)
            
        returns = torch.FloatTensor(returns).to(self.device)
        # ！！！【核心修复 1】：彻底删掉对 returns 的归一化代码！！！
        
        # PPO 优化循环
        for _ in range(self.K_epochs):
            # states shape: (batch, 6, 6)
            batch_size = states.shape[0]
            N = states.shape[1]
            
            # 1. 批量重塑送入左脑
            prior_states = states[:, :, :4].contiguous().view(-1, 4).clone() # (batch*6, 4)
            prior_states[:, 2] = torch.clamp(prior_states[:, 2], max=5.0)
            logits_prior = self.prior_net(prior_states).view(batch_size, N, 4)
            
            # 2. 批量重塑送入右脑 Actor (切断串扰！)
            actor_x = states.view(-1, 6) # (batch*6, 6)
            delta_logits = self.residual_net.actor(actor_x).view(batch_size, N, 4)
            
            logits_final = logits_prior + self.residual_scale * delta_logits
            
            # 3. Centralized Critic 上帝视角评估
            critic_x = states.view(batch_size, -1) # (batch, 36)
            values = self.residual_net.critic(critic_x).squeeze() # (batch,)
            
            # 计算分布和优势
            dist = Categorical(logits=logits_final)
            logprobs = dist.log_prob(actions).sum(dim=1) # (batch,)
            dist_entropy = dist.entropy().sum(dim=1)
            
            advantages = returns - values.detach()
            
            # ！！！【核心修复 2】：PPO 的标准做法，在这里对 Advantage 进行归一化 ！！！
            if advantages.std() > 1e-6:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            
            # PPO Loss
            ratios = torch.exp(logprobs - old_logprobs)
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages
            
            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(values.squeeze(), returns) - 0.01 * dist_entropy
            
            self.optimizer.zero_grad()
            loss.mean().backward()
            
            # 【核心修复 3】：加入梯度裁剪 (Gradient Clipping)，防止灾后巨大方差导致梯度爆炸和策略失忆
            nn.utils.clip_grad_norm_(self.residual_net.parameters(), max_norm=0.5)
            
            self.optimizer.step()
            
        self.buffer = []
