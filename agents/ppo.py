"""
PPO算法 v3.1 - 修复版

严格按照 RESEARCH_PROPOSAL_v3.1_FINAL.md 第11节配置
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from collections import deque


@dataclass
class Experience:
    """单步经验"""
    obs: Dict[str, np.ndarray]
    action: Dict[str, np.ndarray]
    reward: float
    done: bool
    value: float
    log_prob: float


class RolloutBuffer:
    """经验缓冲区"""
    
    def __init__(self):
        self.experiences: List[Experience] = []
    
    def add(self, exp: Experience):
        self.experiences.append(exp)
    
    def clear(self):
        self.experiences = []
    
    def __len__(self):
        return len(self.experiences)
    
    def get_all(self) -> List[Experience]:
        return self.experiences


class PPO:
    """
    PPO算法实现
    
    配置（按方案11）：
    - gamma: 0.99
    - gae_lambda: 0.95
    - clip_eps: 0.2
    - lr: 3e-4
    - n_epochs: 10
    - entropy_coef: 0.01
    - value_coef: 0.5
    - max_grad_norm: 0.5
    """
    
    def __init__(self, policy_network: nn.Module, config: Dict = None, device: str = 'cpu'):
        self.policy = policy_network.to(device)
        self.device = device
        self.config = config or {}
        
        ppo_config = self.config.get('ppo', {})
        
        self.gamma = ppo_config.get('gamma', 0.99)
        self.gae_lambda = ppo_config.get('gae_lambda', 0.95)
        self.clip_eps = ppo_config.get('clip_eps', 0.2)
        self.lr = ppo_config.get('lr', 3e-4)
        self.n_epochs = ppo_config.get('n_epochs', 10)
        self.mini_batch_size = ppo_config.get('mini_batch_size', 64)
        self.entropy_coef = ppo_config.get('entropy_coef', 0.01)
        self.value_coef = ppo_config.get('value_coef', 0.5)
        self.max_grad_norm = ppo_config.get('max_grad_norm', 0.5)
        
        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.lr, eps=1e-5)
        
        # 统计
        self.episode_rewards = deque(maxlen=100)
        self.episode_lengths = deque(maxlen=100)
    
    def compute_gae(
        self, 
        rewards: List[float], 
        values: List[float], 
        dones: List[bool], 
        last_value: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        计算GAE优势估计
        
        Args:
            rewards: 奖励序列
            values: 价值估计序列
            dones: 终止标志序列
            last_value: 最后状态的价值估计
        
        Returns:
            advantages: 优势估计
            returns: 回报
        """
        n = len(rewards)
        advantages = np.zeros(n, dtype=np.float32)
        returns = np.zeros(n, dtype=np.float32)
        
        gae = 0.0
        for t in reversed(range(n)):
            if t == n - 1:
                next_value = last_value
                next_non_terminal = 1.0 - float(dones[t])
            else:
                next_value = values[t + 1]
                next_non_terminal = 1.0 - float(dones[t])
            
            delta = rewards[t] + self.gamma * next_value * next_non_terminal - values[t]
            gae = delta + self.gamma * self.gae_lambda * next_non_terminal * gae
            advantages[t] = gae
            returns[t] = advantages[t] + values[t]
        
        return advantages, returns
    
    def update(self, buffer: RolloutBuffer, last_value: float = 0.0) -> Dict[str, float]:
        """
        执行PPO更新
        
        Args:
            buffer: 经验缓冲区
            last_value: 最后一个状态的价值估计
        
        Returns:
            训练统计
        """
        if len(buffer) == 0:
            return {}
        
        experiences = buffer.get_all()
        n = len(experiences)
        
        # 收集数据
        rewards = [e.reward for e in experiences]
        values = [e.value for e in experiences]
        dones = [e.done for e in experiences]
        
        # 计算GAE
        advantages, returns = self.compute_gae(rewards, values, dones, last_value)
        
        # 归一化优势
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # 准备批数据
        obs_dict = self._collate_obs([e.obs for e in experiences])
        actions = self._collate_actions([e.action for e in experiences])
        old_log_probs = torch.tensor([e.log_prob for e in experiences], 
                                     dtype=torch.float32, device=self.device)
        advantages_t = torch.tensor(advantages, dtype=torch.float32, device=self.device)
        returns_t = torch.tensor(returns, dtype=torch.float32, device=self.device)
        
        # 训练统计
        stats = {
            'policy_loss': 0.0,
            'value_loss': 0.0,
            'entropy': 0.0,
            'approx_kl': 0.0,
            'clip_fraction': 0.0
        }
        
        n_updates = 0
        
        # 多轮更新
        for epoch in range(self.n_epochs):
            indices = np.random.permutation(n)
            
            for start in range(0, n, self.mini_batch_size):
                end = min(start + self.mini_batch_size, n)
                batch_indices = indices[start:end]
                
                # 获取batch
                batch_obs = self._get_batch_obs(obs_dict, batch_indices)
                batch_actions = self._get_batch_actions(actions, batch_indices)
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages_t[batch_indices]
                batch_returns = returns_t[batch_indices]
                
                # 前向传播
                log_probs, values_pred, entropy = self.policy.evaluate_actions(
                    batch_obs, batch_actions
                )
                
                # 计算比率
                new_log_probs = log_probs['type']
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                
                # 策略损失（PPO-Clip）
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # 价值损失
                value_loss = F.mse_loss(values_pred, batch_returns)
                
                # 总损失
                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy.mean()
                
                # 更新
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()
                
                # 统计
                with torch.no_grad():
                    approx_kl = ((ratio - 1) - torch.log(ratio)).mean().item()
                    clip_fraction = ((ratio - 1).abs() > self.clip_eps).float().mean().item()
                
                stats['policy_loss'] += policy_loss.item()
                stats['value_loss'] += value_loss.item()
                stats['entropy'] += entropy.mean().item()
                stats['approx_kl'] += approx_kl
                stats['clip_fraction'] += clip_fraction
                n_updates += 1
        
        # 平均
        if n_updates > 0:
            for key in stats:
                stats[key] /= n_updates
        
        return stats
    
    def _collate_obs(self, obs_list: List[Dict]) -> Dict[str, torch.Tensor]:
        """整理观察"""
        result = {}
        keys = obs_list[0].keys()
        
        for key in keys:
            arrays = [obs[key] for obs in obs_list]
            stacked = np.stack(arrays, axis=0)
            result[key] = torch.tensor(stacked, dtype=torch.float32, device=self.device)
        
        return result
    
    def _collate_actions(self, action_list: List[Dict]) -> Dict[str, torch.Tensor]:
        """整理动作"""
        result = {}
        result['action_type'] = torch.tensor(
            [a['action_type'] for a in action_list], 
            dtype=torch.long, device=self.device
        )
        return result
    
    def _get_batch_obs(self, obs_dict: Dict, indices: np.ndarray) -> Dict[str, torch.Tensor]:
        """获取batch观察"""
        return {key: val[indices] for key, val in obs_dict.items()}
    
    def _get_batch_actions(self, actions: Dict, indices: np.ndarray) -> Dict[str, torch.Tensor]:
        """获取batch动作"""
        return {key: val[indices] for key, val in actions.items()}
    
    def save(self, path: str):
        """保存模型"""
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)
    
    def load(self, path: str):
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
