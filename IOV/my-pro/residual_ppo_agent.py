import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
import os
import random

class RunningMeanStd(nn.Module):
    """
    [终极学术修正] Welford's Online Algorithm 用于特征维度无关的 Z-Score 白化
    保证物理异构维度的严格正交与计算图隔离
    """
    def __init__(self, shape, epsilon=1e-8):
        super(RunningMeanStd, self).__init__()
        self.register_buffer('mean', torch.zeros(shape))
        self.register_buffer('var', torch.ones(shape))
        self.register_buffer('count', torch.tensor(1e-4))
        self.epsilon = epsilon

    @torch.no_grad()
    def update(self, x):
        if x.dim() > len(self.mean.shape) + 1:
            x = x.reshape(-1, *self.mean.shape) if len(self.mean.shape) > 0 else x.reshape(-1)
            
        batch_mean = x.mean(dim=0)
        batch_var = x.var(dim=0, unbiased=False)
        batch_count = x.shape[0]
        
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count
        
        self.mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + torch.square(delta) * self.count * batch_count / tot_count
        self.var = M2 / tot_count
        self.count = tot_count

    def forward(self, x, update=True):
        if update and self.training:
            self.update(x)
        # [工程防火墙] 强制 detach，绝对禁止统计过程污染策略梯度反向传播
        current_mean = self.mean.detach()
        current_var = self.var.detach()
        return (x - current_mean) / torch.sqrt(current_var + self.epsilon)

class StateAutoencoder(nn.Module):
    def __init__(self, feature_dim):
        super(StateAutoencoder, self).__init__()
        # [学术修正] 废除 NLP 领域的 LayerNorm，引入强化学习标准的 RunningMeanStd
        self.norm = RunningMeanStd(shape=(feature_dim,))
        self.encoder = nn.Sequential(
            nn.Linear(feature_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 8)
        )
        self.decoder = nn.Sequential(
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, feature_dim)
        )

    def forward(self, x):
        norm_x = self.norm(x)
        encoded = self.encoder(norm_x)
        decoded = self.decoder(encoded)
        return decoded, norm_x

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
    def __init__(self, feature_dim, action_dim, gate_temperature=50.0, ood_threshold=0.05, ablation_mode='ours'):
        super(Residual_ActorCritic, self).__init__()
        self.ablation_mode = ablation_mode
        self.gate_temperature = gate_temperature
        self.ood_threshold = ood_threshold
        
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
        
        # 3. OOD Detector (Zero-Shot Gate)
        self.autoencoder = StateAutoencoder(feature_dim)
        
        # Centralized Critic (Critic 仍然需要全局视野，所以使用 flatten 的 N*feature_dim)
        self.critic = nn.Sequential(
            nn.Linear(6 * feature_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        # [学术修正] 戴着镣铐跳舞的自适应温度参数
        self.alpha = nn.Parameter(torch.zeros(1))
        self.R_max = 5.0

    def forward_actor(self, state, logits_prior):
        x = self.feature(state)
        delta_logits = self.action_head(x)
        delta_logits = delta_logits.view(logits_prior.size())
        
        # [学术修正] 无监督物理异常检测 (使用白化后的状态)
        reconstructed, norm_state = self.autoencoder(state)
        recon_error = torch.mean((norm_state - reconstructed) ** 2, dim=-1, keepdim=True)
        
        # [学术修正] 移除魔法数字，使用网格搜索超参数
        gate = torch.sigmoid(self.gate_temperature * (recon_error.detach() - self.ood_threshold))
        
        if hasattr(self, 'ablation_mode') and self.ablation_mode == 'ablation_nogate':
            # [消融实验 B] 强制完全开启残差，破坏冷启动
            gate = torch.ones_like(gate)
        
        # [代数坍缩修正] 绝对禁止在 forward_actor 内部再次叠加 logits_prior！
        # 这里只允许输出纯净的门控残差修正量，并带有物理防爆边界 R_max
        adaptive_scale = self.R_max * torch.sigmoid(self.alpha)
        return adaptive_scale * gate * delta_logits, gate, recon_error

class ResidualPPOAgent:
    def __init__(self, state_dim=42, action_dim=24, lr=1e-4, gamma=0.99, K_epochs=4, eps_clip=0.2, ablation_mode='ours', lambda_ewc=100.0, ood_threshold=0.05, gate_temperature=50.0):
        self.ablation_mode = ablation_mode
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.residual_scale = 15.0
        self.lambda_ewc = lambda_ewc
        
        self.device = torch.device("cpu")
        
        # 提取前4维给左脑 (去除预警特征)
        self.prior_net = Prior_DNN(4, 4).to(self.device)
        self.residual_net = Residual_ActorCritic(7, 4, gate_temperature, ood_threshold, ablation_mode).to(self.device)
        self.optimizer = optim.Adam(self.residual_net.parameters(), lr=lr)
        self.MseLoss = nn.MSELoss()
        
        # [学术修正] 独立的回报归一化器，根治 Critic 价值饥饿
        self.ret_rms = RunningMeanStd(shape=()).to(self.device)
        
        self.ood_memory = []
        self.buffer = []
        
        # EWC 记忆容器
        self.ewc_fisher = {}
        self.ewc_anchor = {}

    def init_ewc_anchor(self):
        """[学术修正] EWC 逻辑拨乱反正：锚定初始基线（即等价于 Prior DNN）"""
        print("\n[EWC] Initializing Fisher Anchor to left-brain safe baseline...", flush=True)
        self.ewc_anchor = {}
        self.ewc_fisher = {}
        for n, p in self.residual_net.named_parameters():
            if 'autoencoder' not in n and 'critic' not in n:
                self.ewc_anchor[n] = p.detach().clone()
                self.ewc_fisher[n] = torch.ones_like(p) # 使用 L2 正则近似 Empirical Fisher 以确保稳定

    def update_ewc(self):
        """灾难发生时，计算 EWC 的 Fisher 矩阵保护 Prior 知识"""
        if len(self.ood_memory) < 100:
            return
            
        print("\n[EWC] Computing Fisher Information Matrix to freeze disaster memory...", flush=True)
        # [学术修正] 扩大 Fisher 采样规模，降低 Empirical Fisher 的高维方差
        sample_size = min(1000, len(self.ood_memory))
        sampled = random.sample(self.ood_memory, sample_size)
        
        self.ewc_anchor = {}
        self.ewc_fisher = {}
        for n, p in self.residual_net.named_parameters():
            if 'autoencoder' not in n and 'critic' not in n:
                self.ewc_anchor[n] = p.detach().clone()
                self.ewc_fisher[n] = torch.zeros_like(p)
        
        for s in sampled:
            s_state = s[0].unsqueeze(0).to(self.device)
            s_action = s[1].unsqueeze(0).to(self.device)
            
            self.optimizer.zero_grad()
            
            prior_x = s_state[:, :, :4].clone()
            with torch.no_grad():
                logits_prior = self.prior_net(prior_x)
                logits_prior = torch.clamp(logits_prior, min=-5.0, max=5.0)
                
            delta_logits, _, _ = self.residual_net.forward_actor(s_state, logits_prior)
            logits_final = logits_prior + self.residual_scale * delta_logits
            
            dist = Categorical(logits=logits_final)
            logprob = dist.log_prob(s_action).sum()
            
            logprob.backward()
            
            for n, p in self.residual_net.named_parameters():
                if 'autoencoder' not in n and 'critic' not in n:
                    if p.grad is not None:
                        self.ewc_fisher[n] += (p.grad.detach() ** 2) / sample_size
                    
        self.optimizer.zero_grad()
        print("[EWC] Fisher Information Matrix computed successfully.\n", flush=True)

    def load_expert(self, path):
        if os.path.exists(path):
            self.prior_net.eval()
            self.prior_net.load_state_dict(torch.load(path, map_location='cpu'))
            self.init_ewc_anchor() # 加载完专家后立即初始化 EWC 锚点
            print(f"=> Loaded Expert Prior from {path}", flush=True)

    def save_model(self, path):
        torch.save(self.residual_net.state_dict(), path)

    def load_model(self, path):
        if os.path.exists(path):
            self.residual_net.load_state_dict(torch.load(path, map_location='cpu'), strict=False)
            print(f"=> Loaded Agent Model from {path}", flush=True)

    def select_action(self, state):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        with torch.no_grad():
            prior_x = state[:, :, :4].clone()
            logits_prior = self.prior_net(prior_x)
            logits_prior = torch.clamp(logits_prior, min=-5.0, max=5.0) 
            
            delta_logits, gate, _ = self.residual_net.forward_actor(state, logits_prior)
            
            if self.ablation_mode == 'prior':
                logits_final = logits_prior.clone()
            elif self.ablation_mode == 'ppo':
                # PPO baseline also gets the adaptive scale from forward_actor
                logits_final = delta_logits.clone() 
            else: 
                logits_final = logits_prior + delta_logits
            
            value = self.residual_net.critic(state.view(state.size(0), -1))
            dist = Categorical(logits=logits_final)
            action = dist.sample()
            action_logprob = dist.log_prob(action).sum()
            
        return action.squeeze(0).cpu().numpy(), action_logprob.item(), value.item(), gate.mean().item()

    def store_transition(self, transition):
        self.buffer.append(transition)

    def update(self):
        if len(self.buffer) == 0:
            return 0.0, 0.0
            
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
        
        if self.ablation_mode == 'ablation_hardclip':
            # [消融实验 A] 关闭 Return Normalization，验证价值饥饿
            normalized_returns = returns
        else:
            # [学术修正] 根治价值饥饿 (Value Starvation)
            # 强制将压缩后的 Return 展宽至方差为 1.0 的尺度
            self.ret_rms.update(returns)
            normalized_returns = (returns - self.ret_rms.mean.detach()) / torch.sqrt(self.ret_rms.var.detach() + self.ret_rms.epsilon)
        
        with torch.no_grad():
            initial_values = self.residual_net.critic(states.view(states.size(0), -1)).squeeze()
            # Initial_values 已经是展宽后尺度的预测，计算残差 Advantage
            initial_adv = torch.abs(normalized_returns - initial_values)
            
        if len(initial_adv) > 0:
            threshold = torch.quantile(initial_adv, 0.8)
            for i in range(len(states)):
                if initial_adv[i] > threshold:
                    self.ood_memory.append((states[i].cpu(), actions[i].cpu(), returns[i].cpu(), old_logprobs[i].cpu()))
                    
        if len(self.ood_memory) > 2000:
            self.ood_memory = self.ood_memory[-2000:]
            
        epoch_mse = []
        epoch_entropy = []
        
        for _ in range(self.K_epochs):
            prior_states = states[:, :, :4].clone()
            logits_prior = self.prior_net(prior_states)
            logits_prior = torch.clamp(logits_prior, min=-5.0, max=5.0) 
            
            delta_logits, gate, recon_error = self.residual_net.forward_actor(states, logits_prior)
            
            # forward_actor 已经包含了 self.alpha 和 gate
            logits_final = logits_prior + delta_logits
            
            values = self.residual_net.critic(states.view(states.size(0), -1)).squeeze()
            
            dist = Categorical(logits=logits_final)
            logprobs = dist.log_prob(actions).sum(dim=1)
            dist_entropy = dist.entropy().sum(dim=1)
            
            # 使用 Normalized Returns 计算 Advantage
            advantages = normalized_returns - values.detach()
            if advantages.std() > 1e-6:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            
            ratios = torch.exp(logprobs - old_logprobs)
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages
            
            # 使用 Normalized Returns 计算 Critic MSE Loss (彻底解除 Loss 过小不收敛的问题)
            critic_loss = self.MseLoss(values, normalized_returns)
            loss_ppo = -torch.min(surr1, surr2) + critic_loss - 0.01 * dist_entropy 
            
            epoch_mse.append(critic_loss.mean().item())
            epoch_entropy.append(dist_entropy.mean().item())
            
            loss_cl = torch.tensor(0.0).to(self.device)
            if self.ablation_mode == 'ours' and len(self.ood_memory) > 100:
                sampled = random.sample(self.ood_memory, int(len(states) * 0.2))
                s_states = torch.stack([s[0] for s in sampled]).to(self.device)
                s_actions = torch.stack([s[1] for s in sampled]).to(self.device)
                s_returns = torch.stack([s[2] for s in sampled]).to(self.device)
                s_old_logprobs = torch.stack([s[3] for s in sampled]).to(self.device)
                
                # 对灾难记忆池的 Return 同样执行展宽 (如果是硬截断消融则不执行)
                if self.ablation_mode == 'ablation_hardclip':
                    s_norm_returns = s_returns
                else:
                    s_norm_returns = (s_returns - self.ret_rms.mean.detach()) / torch.sqrt(self.ret_rms.var.detach() + self.ret_rms.epsilon)
                
                s_values = self.residual_net.critic(s_states.view(s_states.size(0), -1)).squeeze()
                s_adv = s_norm_returns - s_values.detach()
                s_adv = (s_adv - s_adv.mean()) / (s_adv.std() + 1e-8)
                
                with torch.no_grad():
                    s_logits_prior = self.prior_net(s_states[:, :, :4])
                    s_logits_prior = torch.clamp(s_logits_prior, min=-5.0, max=5.0)
                
                s_delta_logits, _, _ = self.residual_net.forward_actor(s_states, s_logits_prior)
                s_logits_final = s_logits_prior + s_delta_logits
                
                s_dist = Categorical(logits=s_logits_final)
                s_logprobs = s_dist.log_prob(s_actions).sum(dim=1)
                
                ratio = torch.exp(s_logprobs - s_old_logprobs)
                loss_cl = - torch.min(ratio * s_adv, torch.clamp(ratio, 1-self.eps_clip, 1+self.eps_clip) * s_adv).mean() * 0.1
            
            loss_ewc = torch.tensor(0.0).to(self.device)
            mean_gate = gate.mean().item()
            if self.ablation_mode == 'ours' and len(self.ewc_fisher) > 0:
                for n, p in self.residual_net.named_parameters():
                    if n in self.ewc_fisher:
                        loss_ewc += (self.ewc_fisher[n] * (p - self.ewc_anchor[n]) ** 2).sum()
                if loss_ewc > 0:
                    # [学术修正] 动态 EWC 阻尼：灾难越严重(Gate越大)，惩罚越小
                    lambda_dynamic = self.lambda_ewc * (1.0 - mean_gate)
                    loss_ewc = lambda_dynamic * loss_ewc
                
            normal_mask = (initial_adv <= threshold).unsqueeze(-1).expand(-1, 6).reshape(-1, 1)
            recon_error_flat = recon_error.view(-1, 1)
            if normal_mask.sum() > 0:
                loss_ae = recon_error_flat[normal_mask].mean() * 10.0
            else:
                loss_ae = torch.tensor(0.0).to(self.device)
                
            # TODO: Concept Drift Adaptation Handler
            # if mean_gate > 0.8 and disaster_duration > T_drift:
            #     # trigger ultra-slow momentum update for AE (EMA: tau=0.0001)
            #     pass
                
            loss_total = loss_ppo.mean() + loss_cl + loss_ewc + loss_ae
            
            self.optimizer.zero_grad()
            loss_total.backward()
            nn.utils.clip_grad_norm_(self.residual_net.parameters(), max_norm=0.5)
            self.optimizer.step()
            
        self.buffer.clear()
        
        return np.mean(epoch_mse) if len(epoch_mse) > 0 else 0.0, np.mean(epoch_entropy) if len(epoch_entropy) > 0 else 0.0
