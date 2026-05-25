import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
import os
import random

class Prior_DNN(nn.Module):
    """先验网络：系统的“本能反应”，权重冻结"""
    def __init__(self, state_dim, action_dim):
        super(Prior_DNN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, action_dim)
        )
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, state):
        return self.net(state)

class Residual_ActorCritic(nn.Module):
    def __init__(self, N_vehicles=6, feature_dim=7, action_dim=4): # 状态维度从6增加到7
        super(Residual_ActorCritic, self).__init__()
        # 1. Feature Extractor
        self.feature = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU()
        )
        # 2. Action Head (OOD Compensation Strategy)
        self.action_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
            nn.Tanh()
        )
        nn.init.orthogonal_(self.action_head[-2].weight, gain=0.01)
        nn.init.constant_(self.action_head[-2].bias, 0.0)
        
        # 3. Gate Head (Dynamic activation confidence)
        self.gate_head = nn.Sequential(
            nn.Linear(128, action_dim),
            nn.Sigmoid()
        )
        
        # Centralized Critic
        self.critic = nn.Sequential(
            nn.Linear(N_vehicles * feature_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward_actor(self, state):
        feat = self.feature(state)
        action_comp = self.action_head(feat)
        gate = self.gate_head(feat)
        # 核心数学机制：用软门控动态缩放策略，允许门控学习在和平期闭合，而动作头被 EWC 冻结
        return gate * action_comp

class ResidualPPOAgent:
    def __init__(self, state_dim=42, action_dim=24, lr=1e-4, gamma=0.99, K_epochs=4, eps_clip=0.2, ablation_mode='ours'):
        self.ablation_mode = ablation_mode
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.residual_scale = 15.0
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 提取前4维给左脑 (去除预警特征)
        self.prior_net = Prior_DNN(4, 4).to(self.device)
        self.residual_net = Residual_ActorCritic(N_vehicles=6, feature_dim=7, action_dim=4).to(self.device)
        self.optimizer = optim.Adam(self.residual_net.parameters(), lr=lr)
        self.MseLoss = nn.MSELoss()
        
        self.ood_memory = []
        self.buffer = []
        
        # EWC 记忆容器
        self.ewc_fisher = {}
        self.ewc_anchor = {}

    def update_ewc(self):
        """[导师学术修正] 精确计算二阶梯度的近似（Fisher Information Matrix）并锚定核心参数"""
        if len(self.ood_memory) < 50:
            print("[EWC] OOD memory too small to compute Fisher Information.", flush=True)
            return
            
        print("\n[EWC] Computing Fisher Information Matrix to freeze disaster memory...", flush=True)
        # [精细化手术] 仅仅保护 feature 和 action_head，让 gate_head 和 critic 绝对保持塑性！
        self.ewc_anchor = {}
        self.ewc_fisher = {}
        for n, p in self.residual_net.named_parameters():
            if 'gate_head' not in n and 'critic' not in n:
                self.ewc_anchor[n] = p.detach().clone()
                self.ewc_fisher[n] = torch.zeros_like(p)
        
        sampled = random.sample(self.ood_memory, min(200, len(self.ood_memory)))
        
        for s in sampled:
            s_state = s[0].unsqueeze(0).to(self.device)
            s_action = s[1].unsqueeze(0).to(self.device)
            N = s_state.shape[1]
            
            self.optimizer.zero_grad()
            
            prior_x = s_state[:, :, :4].clone()
            with torch.no_grad():
                logits_prior = self.prior_net(prior_x.view(-1, 4)).view(1, N, 4)
                logits_prior = torch.clamp(logits_prior, min=-5.0, max=5.0)
                
            delta_logits = self.residual_net.forward_actor(s_state.view(-1, 7)).view(1, N, 4)
            logits_final = logits_prior + self.residual_scale * delta_logits
            
            dist = Categorical(logits=logits_final)
            logprob = dist.log_prob(s_action).sum()
            
            # 真实 Empirical Fisher：对数似然概率梯度平方的期望
            logprob.backward()
            
            for n, p in self.residual_net.named_parameters():
                if 'gate_head' not in n and 'critic' not in n:
                    if p.grad is not None:
                        self.ewc_fisher[n] += (p.grad.detach() ** 2) / len(sampled)
                    
        self.optimizer.zero_grad()
        print("[EWC] Fisher Information Matrix computed successfully.\n", flush=True)

    def load_expert(self, path):
        if os.path.exists(path):
            self.prior_net.load_state_dict(torch.load(path, map_location=self.device))
            print(f"=> Loaded Expert Prior from {path}", flush=True)

    def save_model(self, path):
        torch.save(self.residual_net.state_dict(), path)

    def load_model(self, path):
        if os.path.exists(path):
            self.residual_net.load_state_dict(torch.load(path, map_location=self.device))
            print(f"=> Loaded Agent Model from {path}", flush=True)

    def select_action(self, state):
        state = torch.FloatTensor(state).to(self.device)
        N = state.shape[0]
        with torch.no_grad():
            # 1. 左脑
            prior_x = state[:, :4].clone()
            logits_prior = self.prior_net(prior_x)
            logits_prior = torch.clamp(logits_prior, min=-5.0, max=5.0) 
            
            # 【移除】暴力的 ood_mask 如果机制，全权交由残差网络调节
            
            # 2. 右脑 (替换为自门控 Actor)
            delta_logits = self.residual_net.forward_actor(state)
            
            if self.ablation_mode == 'prior':
                logits_final = logits_prior.clone()
                brain_wave = torch.abs(delta_logits).mean(dim=0).cpu().numpy()
            elif self.ablation_mode == 'ppo':
                logits_final = self.residual_scale * delta_logits.clone() 
                brain_wave = torch.abs(delta_logits).mean(dim=0).cpu().numpy()
            else: # ours
                logits_final = logits_prior + self.residual_scale * delta_logits
                brain_wave = torch.abs(self.residual_scale * delta_logits).mean(dim=0).cpu().numpy()
            
            value = self.residual_net.critic(state.view(-1))
            dist = Categorical(logits=logits_final)
            action = dist.sample()
            action_logprob = dist.log_prob(action).sum()
            
        return action.cpu().numpy(), action_logprob.item(), value.item(), brain_wave

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
        
        # [导师学术修正] 提取 OOD 记忆：废除物理阈值硬编码，改为基于内在优势的数学动态筛选
        with torch.no_grad():
            initial_values = self.residual_net.critic(states.view(states.shape[0], -1)).squeeze()
            initial_adv = torch.abs(returns - initial_values)
            
        if len(initial_adv) > 0:
            threshold = torch.quantile(initial_adv, 0.8) # 提取网络预测误差最大的 Top 20%
            for i in range(len(states)):
                if initial_adv[i] > threshold:
                    # 必须保存 old_logprobs 用于后续的 Importance Sampling
                    self.ood_memory.append((states[i].cpu(), actions[i].cpu(), returns[i].cpu(), old_logprobs[i].cpu()))
                    
        if len(self.ood_memory) > 2000:
            self.ood_memory = self.ood_memory[-2000:]
            
        # PPO 优化循环
        for _ in range(self.K_epochs):
            batch_size = states.shape[0]
            N = states.shape[1]
            
            prior_states = states[:, :, :4].contiguous().view(-1, 4).clone()
            logits_prior = self.prior_net(prior_states).view(batch_size, N, 4)
            logits_prior = torch.clamp(logits_prior, min=-5.0, max=5.0) 
            
            actor_x = states.view(-1, 7)
            delta_logits = self.residual_net.forward_actor(actor_x).view(batch_size, N, 4)
            
            if self.ablation_mode == 'prior':
                logits_final = logits_prior.clone()
            elif self.ablation_mode == 'ppo':
                logits_final = self.residual_scale * delta_logits.clone()
            else:
                logits_final = logits_prior + self.residual_scale * delta_logits
            
            critic_x = states.view(batch_size, -1)
            values = self.residual_net.critic(critic_x).squeeze()
            
            dist = Categorical(logits=logits_final)
            logprobs = dist.log_prob(actions).sum(dim=1)
            dist_entropy = dist.entropy().sum(dim=1)
            
            advantages = returns - values.detach()
            if advantages.std() > 1e-6:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            
            ratios = torch.exp(logprobs - old_logprobs)
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages
            
            l2_penalty = (delta_logits ** 2).mean()
            loss_ppo = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(values.squeeze(), returns) - 0.01 * dist_entropy 
            if self.ablation_mode == 'ours':
                loss_ppo += 0.05 * l2_penalty
                
            # === 导师修正：优势加权辅助损失 (Advantage-Weighted CL Loss) 引入 IS 比率 ===
            loss_cl = torch.tensor(0.0).to(self.device)
            if self.ablation_mode == 'ours' and len(self.ood_memory) > 100:
                sampled = random.sample(self.ood_memory, int(len(states) * 0.2))
                s_states = torch.stack([s[0] for s in sampled]).to(self.device)
                s_actions = torch.stack([s[1] for s in sampled]).to(self.device)
                s_returns = torch.stack([s[2] for s in sampled]).to(self.device)
                s_old_logprobs = torch.stack([s[3] for s in sampled]).to(self.device)
                
                s_batch = s_states.shape[0]
                s_critic_x = s_states.view(s_batch, -1)
                s_values = self.residual_net.critic(s_critic_x).squeeze()
                
                # 计算优势，不使用单调 ReLU，保留双向梯度
                s_adv = s_returns - s_values.detach()
                if s_adv.std() > 1e-6:
                    s_adv = (s_adv - s_adv.mean()) / (s_adv.std() + 1e-8)
                
                # 提取动作的对数概率
                s_prior_states = s_states[:, :, :4].contiguous().view(-1, 4).clone()
                with torch.no_grad():
                    s_logits_prior = self.prior_net(s_prior_states).view(s_batch, N, 4)
                    s_logits_prior = torch.clamp(s_logits_prior, min=-5.0, max=5.0)
                
                s_actor_x = s_states.view(-1, 7)
                s_delta_logits = self.residual_net.forward_actor(s_actor_x).view(s_batch, N, 4)
                s_logits_final = s_logits_prior + self.residual_scale * s_delta_logits
                
                s_dist = Categorical(logits=s_logits_final)
                s_logprobs = s_dist.log_prob(s_actions).sum(dim=1)
                
                # 计算重要性采样比率 (Importance Sampling Ratio)
                ratio = torch.exp(s_logprobs - s_old_logprobs)
                
                # 严格使用带截断的 PPO 目标，防止 Off-Policy 数据造成梯度爆炸
                surr1_cl = ratio * s_adv
                surr2_cl = torch.clamp(ratio, 1-self.eps_clip, 1+self.eps_clip) * s_adv
                loss_cl = - torch.min(surr1_cl, surr2_cl).mean() * 0.1 # 辅助损失权重
            
            # === 导师修正：注入 EWC 曲率惩罚 ===
            loss_ewc = torch.tensor(0.0).to(self.device)
            if self.ablation_mode == 'ours' and len(self.ewc_fisher) > 0:
                for n, p in self.residual_net.named_parameters():
                    if n in self.ewc_fisher:
                        loss_ewc += (self.ewc_fisher[n] * (p - self.ewc_anchor[n]) ** 2).sum()
                # 放松 EWC 系数，允许网络在灾难时做出策略大漂移，而不是被锁死在和平时期的动作分布
                loss_ewc = 100.0 * loss_ewc
                
            loss_total = loss_ppo.mean() + loss_cl + loss_ewc
            
            self.optimizer.zero_grad()
            loss_total.backward()
            nn.utils.clip_grad_norm_(self.residual_net.parameters(), max_norm=0.5)
            self.optimizer.step()
            
        self.buffer = []
