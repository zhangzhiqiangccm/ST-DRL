"""
策略网络 v3.1 - 修复版

严格按照 RESEARCH_PROPOSAL_v3.1_FINAL.md 第7节实现

修复：
1. 象限ID还原使用round而非截断
2. 添加输入验证和边界检查
3. 改进数值稳定性
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
from typing import Dict, Tuple, Optional


class GlobalEncoder(nn.Module):
    """全局状态编码器"""
    
    def __init__(self, input_dim: int = 24, hidden_dim: int = 64, output_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class NodeEncoder(nn.Module):
    """节点编码器"""
    
    def __init__(self, input_dim: int = 28, hidden_dim: int = 64, output_dim: int = 64, n_layers: int = 2):
        super().__init__()
        
        layers = []
        current_dim = input_dim
        
        for i in range(n_layers):
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.ReLU())
            current_dim = hidden_dim
        
        layers.append(nn.Linear(hidden_dim, output_dim))
        layers.append(nn.LayerNorm(output_dim))
        
        self.net = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x: 节点特征 (batch, K, input_dim)
            mask: 有效节点掩码 (batch, K)
        
        Returns:
            节点嵌入 (batch, K, output_dim)
        """
        h = self.net(x)
        
        if mask is not None:
            h = h * mask.unsqueeze(-1)
        
        return h


class QuadrantEncoder(nn.Module):
    """
    象限编码器 - 显式建模四象限差异化
    
    核心创新：每个象限有独立的变换层
    
    修复：使用round还原象限ID
    """
    
    def __init__(self, n_quadrants: int = 4, embed_dim: int = 16, sc_dim: int = 2, output_dim: int = 32):
        super().__init__()
        
        self.n_quadrants = n_quadrants
        self.embed_dim = embed_dim
        
        # 象限嵌入
        self.quadrant_embedding = nn.Embedding(n_quadrants, embed_dim)
        
        # 象限特定变换（每个象限一个独立的线性层）
        self.quadrant_transforms = nn.ModuleList([
            nn.Sequential(
                nn.Linear(sc_dim, embed_dim),
                nn.ReLU()
            ) for _ in range(n_quadrants)
        ])
        
        # 融合层
        self.fusion = nn.Sequential(
            nn.Linear(embed_dim * 2, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU()
        )
    
    def forward(self, quadrant_ids: torch.Tensor, stance_consistency: torch.Tensor) -> torch.Tensor:
        """
        Args:
            quadrant_ids: 象限ID (batch, K), 值为1-4
            stance_consistency: 立场和一致性 (batch, K, 2)
        
        Returns:
            象限嵌入 (batch, K, output_dim)
        """
        batch_size, K = quadrant_ids.shape
        device = quadrant_ids.device
        
        # 象限嵌入 (需要将1-4转为0-3)
        # 使用clamp确保索引有效
        q_idx = (quadrant_ids - 1).clamp(0, self.n_quadrants - 1)
        q_embed = self.quadrant_embedding(q_idx)  # (batch, K, embed_dim)
        
        # 象限特定变换
        sc_features = torch.zeros(batch_size, K, self.embed_dim, device=device)
        
        for q in range(self.n_quadrants):
            mask = (q_idx == q)
            if mask.any():
                # 获取该象限的节点
                indices = mask.nonzero(as_tuple=True)
                sc_q = stance_consistency[indices[0], indices[1]]  # (n_q, 2)
                transformed = self.quadrant_transforms[q](sc_q)  # (n_q, embed_dim)
                sc_features[indices[0], indices[1]] = transformed
        
        # 融合
        combined = torch.cat([q_embed, sc_features], dim=-1)
        output = self.fusion(combined)
        
        return output


class StateFusion(nn.Module):
    """状态融合模块"""
    
    def __init__(self, global_dim: int = 64, node_dim: int = 64, quad_dim: int = 32, 
                 hidden_dim: int = 128, output_dim: int = 128):
        super().__init__()
        
        input_dim = global_dim + node_dim + quad_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim)
        )
    
    def forward(self, h_global: torch.Tensor, h_node_pooled: torch.Tensor, 
                h_quad_pooled: torch.Tensor) -> torch.Tensor:
        combined = torch.cat([h_global, h_node_pooled, h_quad_pooled], dim=-1)
        return self.net(combined)


class TypePolicyHead(nn.Module):
    """动作类型策略头"""
    
    def __init__(self, input_dim: int = 128, hidden_dim: int = 64, output_dim: int = 8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, h_state: torch.Tensor) -> torch.Tensor:
        return self.net(h_state)


class AllocPolicyHead(nn.Module):
    """资源分配策略头"""
    
    def __init__(self, input_dim: int = 128, hidden_dim: int = 64, output_dim: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, h_state: torch.Tensor) -> torch.Tensor:
        return F.softmax(self.net(h_state), dim=-1)


class NodeScoringHead(nn.Module):
    """节点评分头（象限感知）"""
    
    def __init__(self, state_dim: int = 128, node_dim: int = 64, quad_dim: int = 32,
                 hidden_dim: int = 64):
        super().__init__()
        
        input_dim = state_dim + node_dim + quad_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, h_state: torch.Tensor, h_nodes: torch.Tensor, 
                h_quads: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        batch_size, K, _ = h_nodes.shape
        
        # 广播状态到每个节点
        h_state_expanded = h_state.unsqueeze(1).expand(-1, K, -1)
        
        # 拼接
        combined = torch.cat([h_state_expanded, h_nodes, h_quads], dim=-1)
        
        # 评分
        scores = self.net(combined).squeeze(-1)  # (batch, K)
        
        # 掩码处理
        scores = scores.masked_fill(mask == 0, float('-inf'))
        
        return scores


class ValueNetwork(nn.Module):
    """价值网络"""
    
    def __init__(self, input_dim: int = 128, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, h_state: torch.Tensor) -> torch.Tensor:
        return self.net(h_state).squeeze(-1)


# 特征索引常量（按方案6.1节点特征布局）
class FeatureIndex:
    """节点特征索引"""
    GROUP_START = 0      # 群体one-hot起始 (4维)
    GROUP_END = 4
    
    STRUCT_START = 4     # 结构特征起始 (5维)
    OUT_DEGREE = 4
    IN_DEGREE = 5
    PAGERANK = 6
    BETWEENNESS = 7
    CLUSTERING = 8
    STRUCT_END = 9
    
    TRAIT_START = 9      # 个体特质起始 (4维)
    SUSCEPTIBILITY = 9
    OPENNESS = 10
    SKEPTICISM = 11
    MEDIA_LITERACY = 12
    TRAIT_END = 13
    
    STANCE = 13          # 立场
    CONSISTENCY = 14     # 一致性
    COG_DIFF = 15        # 认知难度
    QUADRANT = 16        # 象限
    
    STATE_START = 17     # 感染状态起始 (4维)
    STATE_S = 17
    STATE_IR = 18
    STATE_ID = 19
    STATE_R = 20
    STATE_END = 21
    
    NEIGHBOR_START = 21  # 邻域特征起始 (3维)
    NEIGHBOR_END = 24
    
    INTERV_START = 24    # 干预状态起始 (2维)
    THROTTLE_FACTOR = 24
    THROTTLE_REMAINING = 25
    INTERV_END = 26
    
    COST_START = 26      # 成本估计起始 (2维)
    COST_END = 28


class PolicyNetwork(nn.Module):
    """
    完整策略网络
    
    包含：
    - 全局编码器
    - 节点编码器
    - 象限编码器
    - 状态融合
    - 三个策略头
    - 价值网络
    """
    
    def __init__(self, config: Dict = None):
        super().__init__()
        
        if config is None:
            config = {}
        
        model_config = config.get('model', {})
        
        # 全局编码器
        ge_config = model_config.get('global_encoder', {})
        self.global_encoder = GlobalEncoder(
            input_dim=ge_config.get('input_dim', 24),
            hidden_dim=ge_config.get('hidden_dim', 64),
            output_dim=ge_config.get('output_dim', 64)
        )
        
        # 节点编码器
        ne_config = model_config.get('node_encoder', {})
        self.node_encoder = NodeEncoder(
            input_dim=ne_config.get('input_dim', 28),
            hidden_dim=ne_config.get('hidden_dim', 64),
            output_dim=ne_config.get('output_dim', 64),
            n_layers=ne_config.get('n_layers', 2)
        )
        
        # 象限编码器
        qe_config = model_config.get('quadrant_encoder', {})
        self.quadrant_encoder = QuadrantEncoder(
            n_quadrants=qe_config.get('n_quadrants', 4),
            embed_dim=qe_config.get('embed_dim', 16),
            sc_dim=qe_config.get('sc_dim', 2),
            output_dim=qe_config.get('output_dim', 32)
        )
        
        # 状态融合
        sf_config = model_config.get('state_fusion', {})
        self.state_fusion = StateFusion(
            global_dim=64,
            node_dim=64,
            quad_dim=32,
            hidden_dim=sf_config.get('hidden_dim', 128),
            output_dim=sf_config.get('output_dim', 128)
        )
        
        # 策略头
        th_config = model_config.get('type_head', {})
        self.type_head = TypePolicyHead(
            input_dim=128,
            hidden_dim=th_config.get('hidden_dim', 64),
            output_dim=th_config.get('output_dim', 8)
        )
        
        ah_config = model_config.get('alloc_head', {})
        self.alloc_head = AllocPolicyHead(
            input_dim=128,
            hidden_dim=ah_config.get('hidden_dim', 64),
            output_dim=ah_config.get('output_dim', 3)
        )
        
        nsh_config = model_config.get('node_scoring_head', {})
        self.node_scoring_head = NodeScoringHead(
            state_dim=128,
            node_dim=64,
            quad_dim=32,
            hidden_dim=nsh_config.get('hidden_dim', 64)
        )
        
        # 价值网络
        self.value_network = ValueNetwork(input_dim=128, hidden_dim=64)
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化网络权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, obs: Dict[str, torch.Tensor]) -> Tuple[Dict, torch.Tensor]:
        """
        前向传播
        
        Args:
            obs: 观察字典
                - global: (batch, 24)
                - nodes: (batch, K, 28)
                - mask: (batch, K)
                - candidate_ids: (batch, K)
        
        Returns:
            action_dist: 动作分布字典
            value: 状态价值
        """
        global_feat = obs['global']
        node_feat = obs['nodes']
        mask = obs['mask']
        
        batch_size = global_feat.shape[0]
        K = node_feat.shape[1]
        
        # 编码
        h_global = self.global_encoder(global_feat)  # (batch, 64)
        h_nodes = self.node_encoder(node_feat, mask)  # (batch, K, 64)
        
        # 提取立场和一致性（使用特征索引常量）
        stance_consistency = node_feat[:, :, FeatureIndex.STANCE:FeatureIndex.CONSISTENCY+1]  # (batch, K, 2)
        
        # 提取象限ID（使用round还原，更准确）
        # 方案中象限编码为 quadrant_id / 4.0，所以还原需要 * 4 并四舍五入
        quadrant_raw = node_feat[:, :, FeatureIndex.QUADRANT] * 4.0
        quadrant_ids = torch.round(quadrant_raw).long().clamp(1, 4)  # (batch, K)
        
        # 象限编码
        h_quads = self.quadrant_encoder(quadrant_ids, stance_consistency)  # (batch, K, 32)
        
        # 池化（带掩码的平均池化）
        mask_sum = mask.sum(dim=1, keepdim=True).clamp(min=1)  # 避免除零
        mask_expanded = mask.unsqueeze(-1)
        
        h_node_pooled = (h_nodes * mask_expanded).sum(dim=1) / mask_sum  # (batch, 64)
        h_quad_pooled = (h_quads * mask_expanded).sum(dim=1) / mask_sum  # (batch, 32)
        
        # 状态融合
        h_state = self.state_fusion(h_global, h_node_pooled, h_quad_pooled)  # (batch, 128)
        
        # 策略头
        type_logits = self.type_head(h_state)  # (batch, 8)
        alloc_probs = self.alloc_head(h_state)  # (batch, 3)
        node_scores = self.node_scoring_head(h_state, h_nodes, h_quads, mask)  # (batch, K)
        
        # 价值
        value = self.value_network(h_state)
        
        return {
            'type_logits': type_logits,
            'alloc_probs': alloc_probs,
            'node_scores': node_scores
        }, value
    
    def get_action(self, obs: Dict[str, torch.Tensor], deterministic: bool = False) -> Tuple[Dict, Dict, torch.Tensor]:
        """
        采样动作
        
        Returns:
            action: 动作字典
            log_probs: 对数概率字典
            value: 状态价值
        """
        action_dist, value = self.forward(obs)
        
        # 类型采样
        type_dist = Categorical(logits=action_dist['type_logits'])
        if deterministic:
            action_type = action_dist['type_logits'].argmax(dim=-1)
        else:
            action_type = type_dist.sample()
        type_log_prob = type_dist.log_prob(action_type)
        
        # 分配（直接使用softmax输出）
        alloc = action_dist['alloc_probs']
        
        # 节点分数
        node_scores = action_dist['node_scores']
        
        action = {
            'action_type': action_type,
            'resource_allocation': alloc,
            'node_scores': node_scores
        }
        
        log_probs = {
            'type': type_log_prob
        }
        
        return action, log_probs, value
    
    def evaluate_actions(self, obs: Dict[str, torch.Tensor], action: Dict) -> Tuple[Dict, torch.Tensor, torch.Tensor]:
        """
        评估动作（用于PPO更新）
        
        Returns:
            log_probs: 对数概率
            value: 状态价值
            entropy: 策略熵
        """
        action_dist, value = self.forward(obs)
        
        # 类型
        type_dist = Categorical(logits=action_dist['type_logits'])
        type_log_prob = type_dist.log_prob(action['action_type'])
        type_entropy = type_dist.entropy()
        
        log_probs = {'type': type_log_prob}
        entropy = type_entropy
        
        return log_probs, value, entropy
