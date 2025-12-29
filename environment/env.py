"""
强化学习环境 v3.1 - 修复版

严格按照 RESEARCH_PROPOSAL_v3.1_FINAL.md 实现

修复：
1. 干预逻辑防止重复干预
2. 奖励计算数值稳定性
3. 特征维度一致性
4. 候选节点选择改进
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass

from .node import (
    NodeState, NodeRole, NodeGenerator, InfectionState, GroupType,
    get_quadrant_id
)
from .network import NetworkData, NetworkGenerator
from .time_model import TimeModel, TimeContext
from .cost_model import CostModel, ActionType
from .propagation import PropagationModel, PropagationResult


@dataclass
class InterventionRecord:
    """干预记录"""
    step: int
    node_id: int
    action_type: str
    cost: float
    success: bool
    quadrant: int
    group: str


class InterventionEnv(gym.Env):
    """
    谣言干预强化学习环境
    
    状态空间：
    - 全局状态 (24维)
    - 节点状态 (K×28维)
    
    动作空间：
    - 动作类型 (8种组合)
    - 资源分配 (3维)
    - 节点选择分数 (K维)
    """
    
    metadata = {'render_modes': ['human']}
    
    def __init__(self, config: Dict[str, Any], seed: int = None):
        super().__init__()
        
        self.config = config
        self.seed_value = seed
        self.rng = np.random.default_rng(seed)
        
        # 基础配置
        self.n_nodes = config.get('network', {}).get('n_nodes', 2000)
        self.episode_length = config.get('time', {}).get('episode_length', 48)
        self.candidate_pool_size = config.get('intervention', {}).get('candidate_pool', 50)
        
        # 预算
        budget_config = config.get('budget', {})
        self.total_budget = budget_config.get('total', 200)
        self.step_budget = budget_config.get('per_step', 20)
        
        # 干预配置
        interv_config = config.get('intervention', {})
        self.throttle_duration = interv_config.get('throttle_duration', 6)
        self.max_targets = interv_config.get('max_targets', 10)
        self.throttle_factor = interv_config.get('throttle_factor', 0.3)
        
        # 奖励参数（按方案6.4）
        reward_config = config.get('reward', {})
        self.alpha1 = reward_config.get('alpha1', 10.0)
        self.alpha2 = reward_config.get('alpha2', 2.0)
        self.alpha3 = reward_config.get('alpha3', 5.0)
        self.alpha4 = reward_config.get('alpha4', 3.0)
        self.alpha5 = reward_config.get('alpha5', 1.0)
        self.alpha6 = reward_config.get('alpha6', 50.0)
        self.alpha7 = reward_config.get('alpha7', 20.0)
        self.alpha8 = reward_config.get('alpha8', 10.0)
        self.omega1 = reward_config.get('omega1', 0.6)
        self.omega2 = reward_config.get('omega2', 0.4)
        
        # 特征维度
        self.global_dim = 24
        self.node_dim = 28
        
        # 初始化组件
        self.network_generator = NetworkGenerator(config)
        self.node_generator = NodeGenerator(config)
        self.time_model = TimeModel(config)
        self.cost_model = CostModel(config)
        self.propagation_model = PropagationModel(config)
        
        # 动作和状态空间
        self._init_spaces()
        
        # 状态变量（将在reset中初始化）
        self.network: Optional[NetworkData] = None
        self.node_states: Optional[Dict[int, NodeState]] = None
        self.current_step = 0
        self.remaining_budget = 0.0
        self.intervention_records: List[InterventionRecord] = []
        self.trajectory: List[Dict] = []
        self._step_interventions: set = set()  # 当前步已干预的节点
    
    def _init_spaces(self):
        """初始化动作和状态空间"""
        self.observation_space = spaces.Dict({
            'global': spaces.Box(low=-np.inf, high=np.inf, 
                               shape=(self.global_dim,), dtype=np.float32),
            'nodes': spaces.Box(low=-np.inf, high=np.inf, 
                              shape=(self.candidate_pool_size, self.node_dim), dtype=np.float32),
            'mask': spaces.Box(low=0, high=1, 
                             shape=(self.candidate_pool_size,), dtype=np.float32),
            'candidate_ids': spaces.Box(low=0, high=self.n_nodes, 
                                       shape=(self.candidate_pool_size,), dtype=np.int32)
        })
        
        self.action_space = spaces.Dict({
            'action_type': spaces.Discrete(8),
            'resource_allocation': spaces.Box(low=0, high=1, shape=(3,), dtype=np.float32),
            'node_scores': spaces.Box(low=-np.inf, high=np.inf, 
                                     shape=(self.candidate_pool_size,), dtype=np.float32)
        })
    
    def reset(self, seed: int = None, options: dict = None) -> Tuple[Dict, Dict]:
        """重置环境"""
        if seed is not None:
            self.rng = np.random.default_rng(seed)
            self.seed_value = seed
        
        # 生成网络
        network_seed = int(self.rng.integers(0, 2**31))
        self.network = self.network_generator.generate(seed=network_seed)
        
        # 设置成本模型网络统计
        max_influence = max(self.network.pagerank.max(), 1e-6)
        self.cost_model.set_network_stats(
            self.network.max_pagerank,
            self.network.max_out_degree,
            max_influence
        )
        
        # 生成节点角色
        roles = self.node_generator.generate_roles(self.n_nodes, self.rng)
        
        # 初始化节点状态
        self.node_states = {}
        for node_id in range(self.n_nodes):
            role = roles[node_id]
            state = NodeState(
                node_id=node_id,
                role=role,
                in_degree=int(self.network.in_degrees[node_id]),
                out_degree=int(self.network.out_degrees[node_id]),
                pagerank=float(self.network.pagerank[node_id]),
                betweenness=float(self.network.betweenness[node_id]),
                clustering=float(self.network.clustering[node_id]),
                influence=float(self.network.pagerank[node_id])
            )
            self.node_states[node_id] = state
        
        # 初始感染
        self._init_infection()
        
        # 重置状态
        self.current_step = 0
        self.remaining_budget = float(self.total_budget)
        self.intervention_records = []
        self.trajectory = []
        self._step_interventions = set()
        
        obs = self._get_observation()
        info = self._get_info()
        
        return obs, info
    
    def _init_infection(self):
        """初始化感染"""
        init_config = self.config.get('initial_infection', {})
        method = init_config.get('method', 'high_degree')
        count = init_config.get('count', 40)
        min_degree = init_config.get('min_degree', 3)
        
        # 候选节点
        candidates = [
            nid for nid, state in self.node_states.items()
            if state.out_degree >= min_degree
        ]
        
        if len(candidates) < count:
            candidates = list(range(self.n_nodes))
        
        if method == 'high_degree':
            candidates.sort(key=lambda x: self.node_states[x].out_degree, reverse=True)
        elif method == 'high_pagerank':
            candidates.sort(key=lambda x: self.node_states[x].pagerank, reverse=True)
        elif method == 'random':
            self.rng.shuffle(candidates)
        
        # 感染
        for nid in candidates[:count]:
            self.node_states[nid].infect_rumor(0)
    
    def step(self, action: Dict) -> Tuple[Dict, float, bool, bool, Dict]:
        """执行一步"""
        time_ctx = self.time_model.get_time_context(self.current_step)
        
        # 记录干预前状态
        n_infected_before = self._count_infected()
        
        # 清空当前步已干预节点记录
        self._step_interventions = set()
        
        # 执行干预
        intervention_result = self._execute_intervention(action, time_ctx)
        
        # 更新限流状态
        for state in self.node_states.values():
            state.update_throttle()
        
        # 传播
        prop_result = self.propagation_model.propagate(
            self.network, self.node_states, time_ctx.group_activity,
            self.current_step, record_events=False, rng=self.rng
        )
        
        # 记录干预后状态
        n_infected_after = self._count_infected()
        
        # 计算奖励
        reward = self._compute_reward(
            n_infected_before, n_infected_after,
            intervention_result, time_ctx
        )
        
        # 记录轨迹
        self.trajectory.append({
            'step': self.current_step,
            'n_infected': n_infected_after,
            'n_new_infected': prop_result.total_new_infections,
            'n_debunked': prop_result.total_new_debunks,
            'cost': intervention_result['total_cost'],
            'reward': reward
        })
        
        self.current_step += 1
        terminated = self.current_step >= self.episode_length
        truncated = False
        
        # 终局奖励
        if terminated:
            reward += self._compute_terminal_reward()
        
        obs = self._get_observation()
        info = self._get_info()
        info['prop_result'] = {
            'new_rumor': prop_result.total_new_infections,
            'new_debunk': prop_result.total_new_debunks
        }
        info['intervention'] = {
            'total_cost': intervention_result['total_cost'],
            'throttle': intervention_result['throttle_success'],
            'ban': intervention_result['ban_success'],
            'debunk_success': intervention_result['debunk_success'],
            'debunk_fail': intervention_result['debunk_fail']
        }
        
        return obs, reward, terminated, truncated, info
    
    def _count_infected(self) -> int:
        """统计谣言感染数"""
        return sum(1 for s in self.node_states.values() if s.is_rumor_infected)
    
    def _execute_intervention(self, action: Dict, time_ctx: TimeContext) -> Dict:
        """执行干预动作"""
        action_type = action.get('action_type', 0)
        if isinstance(action_type, np.ndarray):
            action_type = int(action_type.item())
        
        resource_alloc = np.asarray(action.get('resource_allocation', [0.33, 0.33, 0.34]))
        node_scores = np.asarray(action.get('node_scores', np.zeros(self.candidate_pool_size)))
        
        result = {
            'total_cost': 0.0,
            'throttle_success': 0, 'throttle_fail': 0,
            'ban_success': 0, 'ban_fail': 0,
            'debunk_success': 0, 'debunk_fail': 0,
            'group_success': {g.value: 0 for g in GroupType},
            'group_fail': {g.value: 0 for g in GroupType},
        }
        
        if action_type == 0:  # 不干预
            return result
        
        # 解析动作类型
        do_throttle = action_type in [1, 4, 5, 7]
        do_ban = action_type in [2, 4, 6, 7]
        do_debunk = action_type in [3, 5, 6, 7]
        
        # 归一化资源分配
        alloc_sum = resource_alloc.sum()
        if alloc_sum > 0:
            resource_alloc = resource_alloc / alloc_sum
        else:
            resource_alloc = np.array([0.33, 0.33, 0.34])
        
        # 分配预算
        available_budget = min(self.step_budget, self.remaining_budget)
        budget_throttle = available_budget * resource_alloc[0] if do_throttle else 0
        budget_ban = available_budget * resource_alloc[1] if do_ban else 0
        budget_debunk = available_budget * resource_alloc[2] if do_debunk else 0
        
        # 获取候选节点
        candidates = self._get_candidate_nodes()
        
        # 按分数排序
        n_candidates = min(len(candidates), len(node_scores))
        scored = [(candidates[i], float(node_scores[i])) for i in range(n_candidates)]
        scored.sort(key=lambda x: x[1], reverse=True)
        
        # 执行干预（每个节点只能被干预一次）
        for node_id, _ in scored[:self.max_targets]:
            if node_id in self._step_interventions:
                continue
            
            state = self.node_states[node_id]
            
            # 已封禁节点跳过
            if state.infection_state == InfectionState.BANNED:
                continue
            
            group = state.role.group.value
            activity = time_ctx.group_activity.get(group, 0.5)
            
            intervened = False
            
            # 限流（只对感染节点）
            if do_throttle and not intervened and budget_throttle > 0 and state.is_rumor_infected:
                cost = self.cost_model.compute_cost(state, ActionType.THROTTLE, activity)
                if cost <= budget_throttle:
                    state.apply_throttle(self.throttle_factor, self.throttle_duration)
                    budget_throttle -= cost
                    result['total_cost'] += cost
                    result['throttle_success'] += 1
                    result['group_success'][group] += 1
                    self._record_intervention(state, 'throttle', cost, True)
                    self._step_interventions.add(node_id)
                    intervened = True
            
            # 封禁（只对感染节点）
            if do_ban and not intervened and budget_ban > 0 and state.is_rumor_infected:
                cost = self.cost_model.compute_cost(state, ActionType.BAN, activity)
                if cost <= budget_ban:
                    state.ban()
                    budget_ban -= cost
                    result['total_cost'] += cost
                    result['ban_success'] += 1
                    result['group_success'][group] += 1
                    self._record_intervention(state, 'ban', cost, True)
                    self._step_interventions.add(node_id)
                    intervened = True
            
            # 辟谣（对易感和感染节点）
            if do_debunk and not intervened and budget_debunk > 0:
                if state.infection_state in [InfectionState.SUSCEPTIBLE, InfectionState.INFECTED_RUMOR]:
                    accept_prob = self.propagation_model.get_debunk_accept_prob(state)
                    cost = self.cost_model.compute_cost(state, ActionType.DEBUNK, activity, accept_prob)
                    if cost <= budget_debunk:
                        success = self.rng.random() < accept_prob
                        if success:
                            state.accept_debunk(self.current_step)
                            result['debunk_success'] += 1
                            result['group_success'][group] += 1
                        else:
                            result['debunk_fail'] += 1
                            result['group_fail'][group] += 1
                        budget_debunk -= cost
                        result['total_cost'] += cost
                        self._record_intervention(state, 'debunk', cost, success)
                        self._step_interventions.add(node_id)
        
        self.remaining_budget -= result['total_cost']
        return result
    
    def _record_intervention(self, state: NodeState, action_type: str, cost: float, success: bool):
        """记录干预"""
        self.intervention_records.append(InterventionRecord(
            step=self.current_step,
            node_id=state.node_id,
            action_type=action_type,
            cost=cost,
            success=success,
            quadrant=state.quadrant_id,
            group=state.role.group.value
        ))
    
    def _compute_reward(
        self, 
        n_before: int, 
        n_after: int, 
        interv_result: Dict, 
        time_ctx: TimeContext
    ) -> float:
        """计算奖励"""
        N = self.n_nodes
        eps = 1e-8  # 数值稳定
        
        # R_base: 基础遏制
        new_infected = max(0, n_after - n_before)
        r_base = -self.alpha1 * (new_infected / N) - self.alpha2 * (n_after / N)
        
        # R_TG: 时间-群体利用奖励
        r_tg = 0.0
        for g in GroupType:
            gv = g.value
            activity = time_ctx.group_activity.get(gv, 0.5)
            n_g = sum(1 for s in self.node_states.values() if s.role.group == g)
            if n_g > 0:
                succ = interv_result['group_success'].get(gv, 0)
                fail = interv_result['group_fail'].get(gv, 0)
                # 高活跃时成功得分，低活跃时失败惩罚
                r_tg += activity * (succ / n_g) - (1 - activity) * (fail / n_g)
        r_tg *= self.alpha3
        
        # R_SC: 立场-一致性利用奖励
        r_sc = 0.0
        current_step_records = [r for r in self.intervention_records if r.step == self.current_step]
        for rec in current_step_records:
            if rec.success:
                state = self.node_states[rec.node_id]
                # 影响力归一化
                inf_norm = state.influence / (self.network.max_pagerank + eps)
                # 认知难度（归一化到合理范围）
                cog_diff = min(state.cognitive_difficulty, 3.0)  # 限制最大值
                # 奖励 = 影响力 + 效率（难度越高成功越有价值，但要限制）
                r_sc += self.omega1 * inf_norm + self.omega2 / (cog_diff + 0.5)
        r_sc *= self.alpha4
        
        # R_cost: 成本效率惩罚
        if self.step_budget > 0:
            r_cost = -self.alpha5 * (interv_result['total_cost'] / self.step_budget)
        else:
            r_cost = 0.0
        
        return r_base + r_tg + r_sc + r_cost
    
    def _compute_terminal_reward(self) -> float:
        """计算终局奖励"""
        eps = 1e-8
        
        # FIR
        n_infected = self._count_infected()
        fir = n_infected / self.n_nodes
        
        # ER
        total_interv = len(self.intervention_records)
        succ_interv = sum(1 for r in self.intervention_records if r.success)
        er = succ_interv / max(total_interv, 1)
        
        # GFI (Group Fairness Index)
        group_firs = {}
        for g in GroupType:
            g_nodes = [s for s in self.node_states.values() if s.role.group == g]
            if g_nodes:
                g_infected = sum(1 for s in g_nodes if s.is_rumor_infected)
                group_firs[g.value] = g_infected / len(g_nodes)
        
        if group_firs and len(group_firs) > 1:
            fir_values = list(group_firs.values())
            mean_fir = np.mean(fir_values)
            std_fir = np.std(fir_values)
            gfi = max(0, 1 - std_fir / (mean_fir + eps))
        else:
            gfi = 1.0
        
        return self.alpha6 * (1 - fir) + self.alpha7 * er + self.alpha8 * gfi
    
    def _get_candidate_nodes(self) -> List[int]:
        """获取候选节点池"""
        candidates = []
        
        # 1. 所有谣言感染节点
        infected = [nid for nid, s in self.node_states.items() 
                   if s.infection_state == InfectionState.INFECTED_RUMOR]
        candidates.extend(infected)
        
        # 2. 感染节点的易感邻居
        for nid in infected[:min(20, len(infected))]:
            neighbors = self.network.get_neighbors(nid, 'out')
            for n in neighbors:
                if self.node_states[n].infection_state == InfectionState.SUSCEPTIBLE:
                    candidates.append(n)
        
        # 去重并排序
        candidates = list(set(candidates))
        candidates.sort(key=lambda x: self.node_states[x].influence, reverse=True)
        candidates = candidates[:self.candidate_pool_size]
        
        # 补充（如果不足）
        if len(candidates) < self.candidate_pool_size:
            others = [n for n in range(self.n_nodes) 
                     if n not in candidates 
                     and self.node_states[n].infection_state not in 
                     [InfectionState.BANNED, InfectionState.RECOVERED]]
            self.rng.shuffle(others)
            candidates.extend(others[:self.candidate_pool_size - len(candidates)])
        
        return candidates
    
    def _get_observation(self) -> Dict[str, np.ndarray]:
        """获取观察"""
        time_ctx = self.time_model.get_time_context(self.current_step)
        candidates = self._get_candidate_nodes()
        
        global_feat = self._get_global_features(time_ctx)
        
        node_feat = np.zeros((self.candidate_pool_size, self.node_dim), dtype=np.float32)
        mask = np.zeros(self.candidate_pool_size, dtype=np.float32)
        ids = np.zeros(self.candidate_pool_size, dtype=np.int32)
        
        for i, nid in enumerate(candidates[:self.candidate_pool_size]):
            node_feat[i] = self._get_node_features(nid, time_ctx)
            mask[i] = 1.0
            ids[i] = nid
        
        return {
            'global': global_feat,
            'nodes': node_feat,
            'mask': mask,
            'candidate_ids': ids
        }
    
    def _get_global_features(self, time_ctx: TimeContext) -> np.ndarray:
        """获取全局特征 (24维)"""
        feat = []
        
        # 时间编码 (2)
        hour = time_ctx.hour
        feat.append(np.sin(2 * np.pi * hour / 24))
        feat.append(np.cos(2 * np.pi * hour / 24))
        
        # 日期类型 (2)
        feat.append(1.0 if time_ctx.day_type == 'weekday' else 0.0)
        feat.append(1.0 if time_ctx.day_type == 'weekend' else 0.0)
        
        # 群体活跃度 (4)
        for g in GroupType:
            feat.append(time_ctx.group_activity.get(g.value, 0.5))
        
        # 群体感染率 (4)
        for g in GroupType:
            g_nodes = [s for s in self.node_states.values() if s.role.group == g]
            fir_g = sum(1 for s in g_nodes if s.is_rumor_infected) / max(len(g_nodes), 1)
            feat.append(fir_g)
        
        # 部分群体平均立场和一致性 (4)
        for g in [GroupType.ELDERLY, GroupType.WORKER]:
            g_nodes = [s for s in self.node_states.values() if s.role.group == g]
            if g_nodes:
                mean_stance = np.mean([s.role.stance for s in g_nodes])
                mean_cons = np.mean([s.role.consistency for s in g_nodes])
            else:
                mean_stance, mean_cons = 0.0, 0.5
            feat.append(mean_stance)
            feat.append(mean_cons)
        
        # 时间异质性 (1)
        feat.append(time_ctx.time_heterogeneity)
        
        # 感染者统计
        infected = [s for s in self.node_states.values() if s.is_rumor_infected]
        feat.append(np.var([s.role.stance for s in infected]) if len(infected) > 1 else 0.0)
        feat.append(np.mean([s.role.consistency for s in infected]) if infected else 0.5)
        
        # 预算和进度 (2)
        feat.append(self.remaining_budget / max(self.total_budget, 1))
        feat.append(self.current_step / max(self.episode_length, 1))
        
        # 感染增长率 (1)
        if self.trajectory:
            curr = self._count_infected()
            prev = self.trajectory[-1]['n_infected']
            feat.append((curr - prev) / self.n_nodes)
        else:
            feat.append(0.0)
        
        # 辟谣覆盖率 (1)
        n_debunk = sum(1 for s in self.node_states.values() 
                      if s.infection_state in [InfectionState.INFECTED_DEBUNK, InfectionState.RECOVERED])
        feat.append(n_debunk / self.n_nodes)
        
        # 成本效率 (1)
        total_cost = sum(r.cost for r in self.intervention_records)
        feat.append(total_cost / max(self.total_budget - self.remaining_budget + 1, 1))
        
        return np.array(feat[:self.global_dim], dtype=np.float32)
    
    def _get_node_features(self, node_id: int, time_ctx: TimeContext) -> np.ndarray:
        """获取节点特征 (28维)"""
        state = self.node_states[node_id]
        role = state.role
        feat = []
        
        # 群体 one-hot (4)
        group_list = list(GroupType)
        group_idx = group_list.index(role.group) if role.group in group_list else 0
        group_onehot = [0.0] * 4
        group_onehot[group_idx] = 1.0
        feat.extend(group_onehot)
        
        # 结构特征 (5)
        feat.append(self.network.out_degrees_norm[node_id])
        feat.append(self.network.in_degrees_norm[node_id])
        feat.append(self.network.pagerank_norm[node_id])
        feat.append(self.network.betweenness_norm[node_id])
        feat.append(self.network.clustering[node_id])
        
        # 个体特质 (4)
        feat.extend([role.susceptibility, role.openness, role.skepticism, role.media_literacy])
        
        # 立场和一致性 (2)
        feat.extend([role.stance, role.consistency])
        
        # 认知干预难度和象限 (2)
        feat.append(min(state.cognitive_difficulty / 3.0, 1.0))
        feat.append(state.quadrant_id / 4.0)
        
        # 感染状态 one-hot (4)
        state_onehot = [0.0] * 4
        state_map = {InfectionState.SUSCEPTIBLE: 0, InfectionState.INFECTED_RUMOR: 1,
                    InfectionState.INFECTED_DEBUNK: 2, InfectionState.RECOVERED: 3}
        if state.infection_state in state_map:
            state_onehot[state_map[state.infection_state]] = 1.0
        feat.extend(state_onehot)
        
        # 邻域特征 (3)
        neighbors = self.network.get_neighbors(node_id, 'in')
        if neighbors:
            total = len(neighbors)
            n_s = sum(1 for n in neighbors if self.node_states[n].infection_state == InfectionState.SUSCEPTIBLE)
            n_ir = sum(1 for n in neighbors if self.node_states[n].infection_state == InfectionState.INFECTED_RUMOR)
            n_id = sum(1 for n in neighbors if self.node_states[n].infection_state in 
                      [InfectionState.INFECTED_DEBUNK, InfectionState.RECOVERED])
            feat.extend([n_s/total, n_ir/total, n_id/total])
        else:
            feat.extend([0.0, 0.0, 0.0])
        
        # 干预状态 (2)
        feat.append(state.throttle_factor)
        feat.append(state.throttle_remaining / max(self.throttle_duration, 1))
        
        # 成本估计 (2)
        activity = time_ctx.group_activity.get(role.group.value, 0.5)
        costs = self.cost_model.get_normalized_costs(state, activity)
        feat.extend([costs[0], costs[2]])
        
        return np.array(feat[:self.node_dim], dtype=np.float32)
    
    def _get_info(self) -> Dict:
        """获取信息"""
        counts = {s: 0 for s in InfectionState}
        for state in self.node_states.values():
            counts[state.infection_state] += 1
        
        return {
            'step': self.current_step,
            'n_susceptible': counts[InfectionState.SUSCEPTIBLE],
            'n_infected_rumor': counts[InfectionState.INFECTED_RUMOR],
            'n_infected_debunk': counts[InfectionState.INFECTED_DEBUNK],
            'n_recovered': counts[InfectionState.RECOVERED],
            'n_banned': counts[InfectionState.BANNED],
            'remaining_budget': self.remaining_budget,
            'fir': counts[InfectionState.INFECTED_RUMOR] / self.n_nodes
        }
    
    def render(self):
        """渲染环境状态"""
        info = self._get_info()
        print(f"Step {info['step']:3d}: IR={info['n_infected_rumor']:4d}, "
              f"ID={info['n_infected_debunk']:4d}, R={info['n_recovered']:4d}, "
              f"B={info['n_banned']:4d}, Budget={info['remaining_budget']:.1f}")
    
    def get_metrics(self) -> Dict[str, float]:
        """获取评估指标"""
        n_infected = self._count_infected()
        fir = n_infected / self.n_nodes
        
        if self.trajectory:
            pir = max(t['n_infected'] for t in self.trajectory) / self.n_nodes
            auc_ir = np.mean([t['n_infected'] for t in self.trajectory]) / self.n_nodes
        else:
            pir = auc_ir = fir
        
        total_interv = len(self.intervention_records)
        succ_interv = sum(1 for r in self.intervention_records if r.success)
        er = succ_interv / max(total_interv, 1)
        
        total_reward = sum(t['reward'] for t in self.trajectory) if self.trajectory else 0
        
        return {
            'fir': fir, 'pir': pir, 'auc_ir': auc_ir, 'er': er,
            'total_reward': total_reward, 'total_interventions': total_interv,
            'successful_interventions': succ_interv
        }


def make_env(config: Dict[str, Any], seed: int = None) -> InterventionEnv:
    """创建环境实例"""
    return InterventionEnv(config, seed)
