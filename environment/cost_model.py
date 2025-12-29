"""
混合成本模型 v3.1

严格按照 RESEARCH_PROPOSAL_v3.1_FINAL.md 第4节实现

混合函数形式：
- 限流：乘法形式（各因子相对独立）
- 封禁：对数线性形式（指数增长反映舆论风险）
- 辟谣：与接受概率挂钩（难说服→高成本）
"""

import numpy as np
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from .node import NodeState, NodeRole, InfectionState, GroupType


class ActionType(Enum):
    """干预动作类型"""
    NONE = 0
    THROTTLE = 1
    BAN = 2
    DEBUNK = 3


@dataclass
class CostBreakdown:
    """
    成本分解
    
    用于分析成本的各组成部分
    """
    base_cost: float
    tg_factor: float          # 时间-群体因子
    sc_factor: float          # 立场-一致性因子
    ns_factor: float          # 网络结构因子
    total_cost: float
    action_type: str
    
    def to_dict(self) -> Dict[str, float]:
        return {
            'base_cost': self.base_cost,
            'tg_factor': self.tg_factor,
            'sc_factor': self.sc_factor,
            'ns_factor': self.ns_factor,
            'total_cost': self.total_cost,
            'action_type': self.action_type
        }


class CostModel:
    """
    混合成本模型
    
    根据动作类型采用不同的成本计算方式：
    - 限流：乘法形式
    - 封禁：对数线性形式
    - 辟谣：与接受概率挂钩
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config: 配置字典
        """
        self.config = config
        cost_config = config.get('cost', {})
        
        # 基础成本
        base_config = cost_config.get('base', {})
        self.base_costs = {
            ActionType.THROTTLE: base_config.get('throttle', 1.0),
            ActionType.BAN: base_config.get('ban', 5.0),
            ActionType.DEBUNK: base_config.get('debunk', 2.0)
        }
        
        # 成本范围约束
        bounds_config = cost_config.get('bounds', {})
        self.cost_bounds = {
            ActionType.THROTTLE: (
                bounds_config.get('throttle', {}).get('min', 0.5),
                bounds_config.get('throttle', {}).get('max', 8.0)
            ),
            ActionType.BAN: (
                bounds_config.get('ban', {}).get('min', 2.0),
                bounds_config.get('ban', {}).get('max', 30.0)
            ),
            ActionType.DEBUNK: (
                bounds_config.get('debunk', {}).get('min', 1.0),
                bounds_config.get('debunk', {}).get('max', 15.0)
            )
        }
        
        # 立场-一致性因子参数 (kappa)
        cog_config = cost_config.get('cognitive', {})
        self.kappa1 = cog_config.get('kappa1', 0.5)  # 限流-认知难度权重
        self.kappa2 = cog_config.get('kappa2', 0.3)  # 封禁-影响力权重
        self.kappa3 = cog_config.get('kappa3', 0.4)  # 封禁-认知难度权重
        self.kappa4 = cog_config.get('kappa4', 0.6)  # 辟谣-认知难度权重
        
        # 网络结构因子参数 (lambda)
        struct_config = cost_config.get('structure', {})
        self.lambda1 = struct_config.get('lambda1', 0.5)  # PageRank权重
        self.lambda2 = struct_config.get('lambda2', 0.3)  # 度数权重
        
        # 数值稳定常数
        self.epsilon = cost_config.get('epsilon', 0.1)
        
        # 辟谣接受概率参数（用于成本计算）
        prop_config = config.get('propagation', {})
        debunk_config = prop_config.get('debunk', {})
        self.gamma0 = debunk_config.get('gamma0', 0.5)
        self.gamma1 = debunk_config.get('gamma1', 2.0)
        self.gamma2 = debunk_config.get('gamma2', 1.0)
        self.gamma3 = debunk_config.get('gamma3', 1.5)
        
        # 网络统计量（需要在初始化后设置）
        self.max_pagerank = 1.0
        self.max_out_degree = 1
        self.max_influence = 1.0
    
    def set_network_stats(self, max_pagerank: float, max_out_degree: int, max_influence: float):
        """设置网络统计量（用于归一化）"""
        self.max_pagerank = max(max_pagerank, 1e-6)
        self.max_out_degree = max(max_out_degree, 1)
        self.max_influence = max(max_influence, 1e-6)
    
    def compute_cost(
        self,
        node_state: NodeState,
        action_type: ActionType,
        group_activity: float,
        debunk_accept_prob: Optional[float] = None
    ) -> float:
        """
        计算干预成本
        
        Args:
            node_state: 节点状态
            action_type: 动作类型
            group_activity: 当前群体活跃度 A_g(t)
            debunk_accept_prob: 辟谣接受概率（仅辟谣动作需要）
        
        Returns:
            总成本
        """
        if action_type == ActionType.NONE:
            return 0.0
        
        if action_type == ActionType.THROTTLE:
            cost = self._compute_throttle_cost(node_state, group_activity)
        elif action_type == ActionType.BAN:
            cost = self._compute_ban_cost(node_state, group_activity)
        elif action_type == ActionType.DEBUNK:
            cost = self._compute_debunk_cost(node_state, group_activity, debunk_accept_prob)
        else:
            cost = 0.0
        
        # 应用边界约束
        bounds = self.cost_bounds.get(action_type, (0, float('inf')))
        cost = np.clip(cost, bounds[0], bounds[1])
        
        return cost
    
    def compute_cost_with_breakdown(
        self,
        node_state: NodeState,
        action_type: ActionType,
        group_activity: float,
        debunk_accept_prob: Optional[float] = None
    ) -> CostBreakdown:
        """
        计算成本并返回分解
        
        Args:
            node_state: 节点状态
            action_type: 动作类型
            group_activity: 当前群体活跃度
            debunk_accept_prob: 辟谣接受概率
        
        Returns:
            CostBreakdown对象
        """
        if action_type == ActionType.NONE:
            return CostBreakdown(0, 1, 1, 1, 0, 'none')
        
        base = self.base_costs[action_type]
        
        if action_type == ActionType.THROTTLE:
            tg, sc, ns = self._get_throttle_factors(node_state, group_activity)
            raw_cost = base * tg * sc * ns
        elif action_type == ActionType.BAN:
            tg, sc, ns = self._get_ban_factors(node_state, group_activity)
            raw_cost = base * np.exp(tg + sc + ns)
        elif action_type == ActionType.DEBUNK:
            tg, sc, ns = self._get_debunk_factors(node_state, group_activity, debunk_accept_prob)
            raw_cost = base * tg * sc  # ns已包含在sc中
        else:
            return CostBreakdown(0, 1, 1, 1, 0, 'unknown')
        
        # 应用边界约束
        bounds = self.cost_bounds.get(action_type, (0, float('inf')))
        total_cost = np.clip(raw_cost, bounds[0], bounds[1])
        
        return CostBreakdown(
            base_cost=base,
            tg_factor=tg,
            sc_factor=sc,
            ns_factor=ns,
            total_cost=total_cost,
            action_type=action_type.name.lower()
        )
    
    def _compute_throttle_cost(self, node_state: NodeState, group_activity: float) -> float:
        """
        计算限流成本（乘法形式）
        
        C_throttle = C_base × φ_TG × φ_SC × φ_NS
        """
        base = self.base_costs[ActionType.THROTTLE]
        tg, sc, ns = self._get_throttle_factors(node_state, group_activity)
        return base * tg * sc * ns
    
    def _get_throttle_factors(
        self, 
        node_state: NodeState, 
        group_activity: float
    ) -> Tuple[float, float, float]:
        """获取限流成本的各因子"""
        # 时间-群体因子: φ_TG = 1 / (A_g(t) + ε)
        tg_factor = 1.0 / (group_activity + self.epsilon)
        
        # 立场-一致性因子: φ_SC = 1 + κ1 × D_i^cog
        cog_diff = node_state.cognitive_difficulty
        sc_factor = 1.0 + self.kappa1 * cog_diff
        
        # 网络结构因子: φ_NS = 1 + λ1 × PR_norm
        pr_norm = node_state.pagerank / self.max_pagerank
        ns_factor = 1.0 + self.lambda1 * pr_norm
        
        return tg_factor, sc_factor, ns_factor
    
    def _compute_ban_cost(self, node_state: NodeState, group_activity: float) -> float:
        """
        计算封禁成本（对数线性形式）
        
        C_ban = C_base × exp(f_TG + f_SC + f_NS)
        """
        base = self.base_costs[ActionType.BAN]
        tg, sc, ns = self._get_ban_factors(node_state, group_activity)
        return base * np.exp(tg + sc + ns)
    
    def _get_ban_factors(
        self, 
        node_state: NodeState, 
        group_activity: float
    ) -> Tuple[float, float, float]:
        """获取封禁成本的各因子（对数形式）"""
        # 时间-群体因子: f_TG = -log(A_g(t) + ε)
        tg_factor = -np.log(group_activity + self.epsilon)
        
        # 立场-一致性因子: f_SC = κ2 × Inf_norm + κ3 × D_i^cog
        inf_norm = node_state.influence / self.max_influence
        cog_diff = node_state.cognitive_difficulty
        sc_factor = self.kappa2 * inf_norm + self.kappa3 * cog_diff
        
        # 网络结构因子: f_NS = λ1 × PR_norm + λ2 × log(1+d_out)/log(1+d_max)
        pr_norm = node_state.pagerank / self.max_pagerank
        degree_norm = np.log(1 + node_state.out_degree) / np.log(1 + self.max_out_degree)
        ns_factor = self.lambda1 * pr_norm + self.lambda2 * degree_norm
        
        return tg_factor, sc_factor, ns_factor
    
    def _compute_debunk_cost(
        self, 
        node_state: NodeState, 
        group_activity: float,
        debunk_accept_prob: Optional[float] = None
    ) -> float:
        """
        计算辟谣成本（与接受概率挂钩）
        
        C_debunk = C_base × φ_TG × (1 + κ4 × D_i^cog / (P_accept + ε))
        """
        base = self.base_costs[ActionType.DEBUNK]
        tg, sc, _ = self._get_debunk_factors(node_state, group_activity, debunk_accept_prob)
        return base * tg * sc
    
    def _get_debunk_factors(
        self, 
        node_state: NodeState, 
        group_activity: float,
        debunk_accept_prob: Optional[float] = None
    ) -> Tuple[float, float, float]:
        """获取辟谣成本的各因子"""
        # 时间-群体因子: φ_TG = 1 / (A_g(t) + ε)
        tg_factor = 1.0 / (group_activity + self.epsilon)
        
        # 如果没有提供接受概率，使用估计值
        if debunk_accept_prob is None:
            debunk_accept_prob = self._estimate_debunk_accept_prob(node_state)
        
        # 立场-一致性因子: 1 + κ4 × D_i^cog / (P_accept + ε)
        cog_diff = node_state.cognitive_difficulty
        sc_factor = 1.0 + self.kappa4 * cog_diff / (debunk_accept_prob + self.epsilon)
        
        # 网络结构因子（辟谣成本中不单独使用，已合并到SC）
        ns_factor = 1.0
        
        return tg_factor, sc_factor, ns_factor
    
    def _estimate_debunk_accept_prob(self, node_state: NodeState) -> float:
        """
        估计辟谣接受概率
        
        P_accept^D = σ(γ0 + γ1×lit + γ2×open - γ3×s×c×I[I_R])
        """
        role = node_state.role
        
        # 基础项
        logit = self.gamma0
        logit += self.gamma1 * role.media_literacy
        logit += self.gamma2 * role.openness
        
        # 感染抵抗项
        if node_state.infection_state == InfectionState.INFECTED_RUMOR:
            logit -= self.gamma3 * role.stance * role.consistency
        
        # Sigmoid
        prob = 1.0 / (1.0 + np.exp(-logit))
        return prob
    
    def get_all_costs(
        self, 
        node_state: NodeState, 
        group_activity: float,
        debunk_accept_prob: Optional[float] = None
    ) -> Dict[str, float]:
        """
        获取所有动作类型的成本
        
        Returns:
            动作类型到成本的映射
        """
        return {
            'throttle': self.compute_cost(node_state, ActionType.THROTTLE, group_activity),
            'ban': self.compute_cost(node_state, ActionType.BAN, group_activity),
            'debunk': self.compute_cost(node_state, ActionType.DEBUNK, group_activity, debunk_accept_prob)
        }
    
    def get_normalized_costs(
        self,
        node_state: NodeState,
        group_activity: float,
        max_costs: Optional[Dict[str, float]] = None
    ) -> np.ndarray:
        """
        获取归一化成本（用于特征）
        
        Args:
            node_state: 节点状态
            group_activity: 群体活跃度
            max_costs: 最大成本（用于归一化）
        
        Returns:
            归一化成本数组 [throttle, ban, debunk]
        """
        if max_costs is None:
            max_costs = {
                'throttle': self.cost_bounds[ActionType.THROTTLE][1],
                'ban': self.cost_bounds[ActionType.BAN][1],
                'debunk': self.cost_bounds[ActionType.DEBUNK][1]
            }
        
        costs = self.get_all_costs(node_state, group_activity)
        
        return np.array([
            costs['throttle'] / max_costs['throttle'],
            costs['ban'] / max_costs['ban'],
            costs['debunk'] / max_costs['debunk']
        ])
