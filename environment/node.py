"""
节点模块 v3.1 - 立场-一致性模型与四象限划分

严格按照 RESEARCH_PROPOSAL_v3.1_FINAL.md 第3节实现

核心概念：
1. 观点立场 (Stance): s_i ∈ [-1, 1]
2. 观点一致性 (Consistency): c_i ∈ [0, 1]
3. 认知干预难度: D_i^cog = c_i × (1 + |s_i|) × ψ(S_i)
4. 四象限划分: Q1-Q4
"""

import numpy as np
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum


class GroupType(Enum):
    """群体类型"""
    ELDERLY = "elderly"
    WORKER = "worker"
    STUDENT = "student"
    FREELANCER = "freelancer"


class InfectionState(Enum):
    """感染状态（按方案5.1）"""
    SUSCEPTIBLE = "S"           # 易感
    INFECTED_RUMOR = "I_R"      # 谣言感染
    INFECTED_DEBUNK = "I_D"     # 辟谣接受
    RECOVERED = "R"             # 免疫
    BANNED = "B"                # 封禁


class Quadrant(Enum):
    """
    四象限划分（按方案3.3）
    
    基于立场(stance)和一致性(consistency)划分：
    - Q1: 顽固易感者 (s>0, c>0.5)
    - Q2: 摇摆易感者 (s>0, c≤0.5) - 优先辟谣目标
    - Q3: 摇摆理性者 (s≤0, c≤0.5)
    - Q4: 坚定理性者 (s≤0, c>0.5)
    """
    Q1_STUBBORN_SUSCEPTIBLE = 1
    Q2_SWING_SUSCEPTIBLE = 2
    Q3_SWING_RATIONAL = 3
    Q4_FIRM_RATIONAL = 4


def get_quadrant(stance: float, consistency: float) -> Quadrant:
    """
    根据立场和一致性判定象限（按方案附录A）
    
    Args:
        stance: 立场值 ∈ [-1, 1]
        consistency: 一致性值 ∈ [0, 1]
    
    Returns:
        象限枚举
    """
    if stance > 0:
        if consistency > 0.5:
            return Quadrant.Q1_STUBBORN_SUSCEPTIBLE
        else:
            return Quadrant.Q2_SWING_SUSCEPTIBLE
    else:
        if consistency > 0.5:
            return Quadrant.Q4_FIRM_RATIONAL
        else:
            return Quadrant.Q3_SWING_RATIONAL


def get_quadrant_id(stance: float, consistency: float) -> int:
    """返回象限ID (1-4)"""
    return get_quadrant(stance, consistency).value


@dataclass
class NodeRole:
    """
    节点角色属性
    
    包含静态属性和动态认知状态
    """
    # 群体归属
    group: GroupType
    
    # 个体特质向量 θ_i（静态）
    susceptibility: float = 0.5      # 易感性 θ^sus ∈ [0, 1]
    openness: float = 0.5            # 开放性 θ^open ∈ [0, 1]
    skepticism: float = 0.5          # 怀疑性 θ^skep ∈ [0, 1]
    media_literacy: float = 0.5      # 媒体素养 θ^lit ∈ [0, 1]
    
    # 立场-一致性状态（动态，核心创新）
    stance: float = 0.0              # 观点立场 s_i ∈ [-1, 1]
    consistency: float = 0.5         # 观点一致性 c_i ∈ [0, 1]
    
    @property
    def quadrant(self) -> Quadrant:
        """当前所属象限"""
        return get_quadrant(self.stance, self.consistency)
    
    @property
    def quadrant_id(self) -> int:
        """当前象限ID (1-4)"""
        return self.quadrant.value
    
    @property
    def cognitive_difficulty_base(self) -> float:
        """
        基础认知干预难度（不含状态因子）
        
        D_base = c_i × (1 + |s_i|)
        
        范围: [0, 2]
        - c=0, s=0: D=0 (最易干预)
        - c=1, s=±1: D=2 (最难干预)
        """
        return self.consistency * (1 + abs(self.stance))
    
    def get_cognitive_difficulty(self, infection_state: InfectionState) -> float:
        """
        完整认知干预难度（按方案3.2）
        
        D_i^cog = c_i × (1 + |s_i|) × ψ(S_i)
        
        状态因子 ψ(S_i):
        - Susceptible: 1.0
        - Infected_Rumor: 1.5 (确认偏误，更难改变)
        - Infected_Debunk: 0.5 (认知一致，更易强化)
        - Recovered/Banned: 1.0
        """
        psi = self._get_state_factor(infection_state)
        return self.cognitive_difficulty_base * psi
    
    def _get_state_factor(self, infection_state: InfectionState) -> float:
        """获取状态因子 ψ(S_i)"""
        state_factors = {
            InfectionState.SUSCEPTIBLE: 1.0,
            InfectionState.INFECTED_RUMOR: 1.5,
            InfectionState.INFECTED_DEBUNK: 0.5,
            InfectionState.RECOVERED: 1.0,
            InfectionState.BANNED: 1.0
        }
        return state_factors.get(infection_state, 1.0)
    
    def update_stance(
        self,
        delta_s: float,
        source_influence: float,
        edge_weight: float,
        gamma_c: float = 0.9
    ):
        """
        更新立场（按方案3.4）
        
        s_i(t+1) = clip(s_i(t) + η_i × Δs × w_ji × Inf(v_j), -1, 1)
        
        其中学习率: η_i = (1 - c_i) × θ^open
        
        Args:
            delta_s: 信息影响增量（+δ谣言, -δ辟谣）
            source_influence: 来源影响力 Inf(v_j)
            edge_weight: 边权重 w_ji
            gamma_c: 一致性衰减因子
        """
        # 保存旧立场
        old_stance = self.stance
        
        # 计算学习率
        eta = (1 - self.consistency) * self.openness
        
        # 更新立场
        stance_change = eta * delta_s * edge_weight * source_influence
        self.stance = np.clip(self.stance + stance_change, -1.0, 1.0)
        
        # 更新一致性（指数移动平均）
        # c_i(t+1) = γ_c × c_i(t) + (1 - γ_c) × clip(1 - |Δs|, 0, 1)
        stance_diff = abs(self.stance - old_stance)
        target_consistency = max(0.0, 1.0 - stance_diff)
        self.consistency = gamma_c * self.consistency + (1 - gamma_c) * target_consistency
        self.consistency = np.clip(self.consistency, 0.0, 1.0)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'group': self.group.value,
            'susceptibility': self.susceptibility,
            'openness': self.openness,
            'skepticism': self.skepticism,
            'media_literacy': self.media_literacy,
            'stance': self.stance,
            'consistency': self.consistency,
            'quadrant': self.quadrant_id,
            'cognitive_difficulty_base': self.cognitive_difficulty_base
        }


@dataclass
class NodeState:
    """
    节点完整状态
    
    包含角色属性、感染状态、网络位置、干预状态
    """
    # 节点ID
    node_id: int
    
    # 角色属性
    role: NodeRole
    
    # 感染状态
    infection_state: InfectionState = InfectionState.SUSCEPTIBLE
    
    # 感染时间步（-1表示未感染）
    infection_step: int = -1
    
    # 网络结构特征
    in_degree: int = 0
    out_degree: int = 0
    pagerank: float = 0.0
    betweenness: float = 0.0
    clustering: float = 0.0
    
    # 影响力（综合指标）
    influence: float = 0.0
    
    # 干预状态
    throttle_factor: float = 1.0      # 限流因子（1.0表示无限流）
    throttle_remaining: int = 0        # 限流剩余步数
    
    # 干预历史
    n_throttled: int = 0
    n_debunked: int = 0
    
    @property
    def is_infected(self) -> bool:
        """是否处于感染状态（谣言或辟谣）"""
        return self.infection_state in [
            InfectionState.INFECTED_RUMOR,
            InfectionState.INFECTED_DEBUNK
        ]
    
    @property
    def is_rumor_infected(self) -> bool:
        """是否感染谣言"""
        return self.infection_state == InfectionState.INFECTED_RUMOR
    
    @property
    def is_debunk_infected(self) -> bool:
        """是否接受辟谣"""
        return self.infection_state == InfectionState.INFECTED_DEBUNK
    
    @property
    def is_active(self) -> bool:
        """是否处于活跃状态（未被封禁）"""
        return self.infection_state != InfectionState.BANNED
    
    @property
    def quadrant(self) -> Quadrant:
        """当前象限"""
        return self.role.quadrant
    
    @property
    def quadrant_id(self) -> int:
        """当前象限ID"""
        return self.role.quadrant_id
    
    @property
    def cognitive_difficulty(self) -> float:
        """当前认知干预难度"""
        return self.role.get_cognitive_difficulty(self.infection_state)
    
    def apply_throttle(self, factor: float, duration: int):
        """
        应用限流
        
        Args:
            factor: 限流因子（0-1，越小限制越强）
            duration: 持续步数
        """
        self.throttle_factor = factor
        self.throttle_remaining = duration
        self.n_throttled += 1
    
    def update_throttle(self):
        """更新限流状态（每步调用）"""
        if self.throttle_remaining > 0:
            self.throttle_remaining -= 1
            if self.throttle_remaining == 0:
                self.throttle_factor = 1.0
    
    def ban(self):
        """封禁节点"""
        self.infection_state = InfectionState.BANNED
        self.throttle_factor = 0.0
        self.throttle_remaining = 0
    
    def infect_rumor(self, step: int):
        """感染谣言"""
        if self.infection_state == InfectionState.SUSCEPTIBLE:
            self.infection_state = InfectionState.INFECTED_RUMOR
            self.infection_step = step
    
    def accept_debunk(self, step: int):
        """接受辟谣"""
        if self.infection_state == InfectionState.SUSCEPTIBLE:
            self.infection_state = InfectionState.INFECTED_DEBUNK
            self.infection_step = step
        elif self.infection_state == InfectionState.INFECTED_RUMOR:
            self.infection_state = InfectionState.RECOVERED
            self.n_debunked += 1
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'node_id': self.node_id,
            'role': self.role.to_dict(),
            'infection_state': self.infection_state.value,
            'infection_step': self.infection_step,
            'in_degree': self.in_degree,
            'out_degree': self.out_degree,
            'pagerank': self.pagerank,
            'influence': self.influence,
            'throttle_factor': self.throttle_factor,
            'throttle_remaining': self.throttle_remaining,
            'quadrant': self.quadrant_id,
            'cognitive_difficulty': self.cognitive_difficulty
        }


class NodeGenerator:
    """
    节点生成器
    
    根据配置生成节点角色属性
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config: 配置字典
        """
        self.config = config
        
        # 群体分布
        network_config = config.get('network', {})
        self.group_distribution = network_config.get('group_distribution', {
            'elderly': 0.15,
            'worker': 0.40,
            'student': 0.25,
            'freelancer': 0.20
        })
        
        # 立场-一致性配置
        opinion_config = config.get('opinion', {})
        
        # 立场分布参数
        stance_config = opinion_config.get('stance', {})
        self.stance_params = {
            GroupType.ELDERLY: {
                'mean': stance_config.get('elderly', {}).get('mean', 0.15),
                'std': stance_config.get('elderly', {}).get('std', 0.25)
            },
            GroupType.WORKER: {
                'mean': stance_config.get('worker', {}).get('mean', -0.05),
                'std': stance_config.get('worker', {}).get('std', 0.30)
            },
            GroupType.STUDENT: {
                'mean': stance_config.get('student', {}).get('mean', 0.05),
                'std': stance_config.get('student', {}).get('std', 0.35)
            },
            GroupType.FREELANCER: {
                'mean': stance_config.get('freelancer', {}).get('mean', -0.10),
                'std': stance_config.get('freelancer', {}).get('std', 0.30)
            }
        }
        
        # 一致性分布参数 Beta(alpha, beta)
        cons_config = opinion_config.get('consistency', {})
        self.consistency_params = {
            GroupType.ELDERLY: {
                'alpha': cons_config.get('elderly', {}).get('alpha', 4),
                'beta': cons_config.get('elderly', {}).get('beta', 3)
            },
            GroupType.WORKER: {
                'alpha': cons_config.get('worker', {}).get('alpha', 3),
                'beta': cons_config.get('worker', {}).get('beta', 3)
            },
            GroupType.STUDENT: {
                'alpha': cons_config.get('student', {}).get('alpha', 2),
                'beta': cons_config.get('student', {}).get('beta', 4)
            },
            GroupType.FREELANCER: {
                'alpha': cons_config.get('freelancer', {}).get('alpha', 3),
                'beta': cons_config.get('freelancer', {}).get('beta', 4)
            }
        }
        
        # 个体特质参数（按群体）
        self.trait_params = {
            GroupType.ELDERLY: {
                'susceptibility': (0.6, 0.15),
                'openness': (0.3, 0.15),
                'skepticism': (0.4, 0.15),
                'media_literacy': (0.3, 0.15)
            },
            GroupType.WORKER: {
                'susceptibility': (0.4, 0.15),
                'openness': (0.5, 0.15),
                'skepticism': (0.5, 0.15),
                'media_literacy': (0.6, 0.15)
            },
            GroupType.STUDENT: {
                'susceptibility': (0.5, 0.2),
                'openness': (0.7, 0.15),
                'skepticism': (0.4, 0.2),
                'media_literacy': (0.7, 0.15)
            },
            GroupType.FREELANCER: {
                'susceptibility': (0.45, 0.15),
                'openness': (0.6, 0.15),
                'skepticism': (0.55, 0.15),
                'media_literacy': (0.65, 0.15)
            }
        }
    
    def generate_roles(self, n_nodes: int, rng: np.random.Generator = None) -> Dict[int, NodeRole]:
        """
        生成所有节点的角色属性
        
        Args:
            n_nodes: 节点数量
            rng: 随机数生成器
        
        Returns:
            节点ID到角色的映射
        """
        if rng is None:
            rng = np.random.default_rng()
        
        roles = {}
        
        # 分配群体
        groups = self._assign_groups(n_nodes, rng)
        
        for node_id in range(n_nodes):
            group = groups[node_id]
            role = self._generate_single_role(group, rng)
            roles[node_id] = role
        
        return roles
    
    def _assign_groups(self, n_nodes: int, rng: np.random.Generator) -> Dict[int, GroupType]:
        """分配群体"""
        groups = {}
        
        # 计算各群体节点数
        group_counts = {}
        remaining = n_nodes
        
        group_list = list(GroupType)
        for i, group in enumerate(group_list[:-1]):
            count = int(n_nodes * self.group_distribution.get(group.value, 0.25))
            group_counts[group] = count
            remaining -= count
        
        # 最后一个群体获得剩余节点
        group_counts[group_list[-1]] = remaining
        
        # 创建节点列表并打乱
        node_ids = list(range(n_nodes))
        rng.shuffle(node_ids)
        
        # 分配
        idx = 0
        for group, count in group_counts.items():
            for _ in range(count):
                if idx < n_nodes:
                    groups[node_ids[idx]] = group
                    idx += 1
        
        return groups
    
    def _generate_single_role(self, group: GroupType, rng: np.random.Generator) -> NodeRole:
        """生成单个节点的角色属性"""
        # 生成个体特质
        trait_p = self.trait_params[group]
        
        susceptibility = np.clip(rng.normal(*trait_p['susceptibility']), 0, 1)
        openness = np.clip(rng.normal(*trait_p['openness']), 0, 1)
        skepticism = np.clip(rng.normal(*trait_p['skepticism']), 0, 1)
        media_literacy = np.clip(rng.normal(*trait_p['media_literacy']), 0, 1)
        
        # 生成立场（正态分布）
        stance_p = self.stance_params[group]
        stance = np.clip(rng.normal(stance_p['mean'], stance_p['std']), -1, 1)
        
        # 生成一致性（Beta分布）
        cons_p = self.consistency_params[group]
        consistency = rng.beta(cons_p['alpha'], cons_p['beta'])
        
        return NodeRole(
            group=group,
            susceptibility=susceptibility,
            openness=openness,
            skepticism=skepticism,
            media_literacy=media_literacy,
            stance=stance,
            consistency=consistency
        )
    
    def get_group_stats(self, roles: Dict[int, NodeRole]) -> Dict[str, Any]:
        """获取群体统计信息"""
        stats = {g.value: {'count': 0, 'stances': [], 'consistencies': [], 'quadrants': {1: 0, 2: 0, 3: 0, 4: 0}}
                for g in GroupType}
        
        for role in roles.values():
            g = role.group.value
            stats[g]['count'] += 1
            stats[g]['stances'].append(role.stance)
            stats[g]['consistencies'].append(role.consistency)
            stats[g]['quadrants'][role.quadrant_id] += 1
        
        # 计算均值和标准差
        for g in stats:
            if stats[g]['count'] > 0:
                stats[g]['stance_mean'] = np.mean(stats[g]['stances'])
                stats[g]['stance_std'] = np.std(stats[g]['stances'])
                stats[g]['consistency_mean'] = np.mean(stats[g]['consistencies'])
                stats[g]['consistency_std'] = np.std(stats[g]['consistencies'])
            del stats[g]['stances']
            del stats[g]['consistencies']
        
        return stats
