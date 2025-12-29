"""
传播模型 v3.1 - 信息传播动力学

严格按照 RESEARCH_PROPOSAL_v3.1_FINAL.md 第5节实现
"""

import numpy as np
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field

from .node import NodeState, InfectionState


@dataclass
class PropagationEvent:
    """传播事件记录"""
    step: int
    source_id: int
    target_id: int
    info_type: str
    success: bool
    probability: float
    target_quadrant: int


@dataclass
class PropagationResult:
    """单步传播结果"""
    new_rumor_infected: List[int] = field(default_factory=list)
    new_debunk_accepted: List[int] = field(default_factory=list)
    new_recovered: List[int] = field(default_factory=list)
    events: List[PropagationEvent] = field(default_factory=list)
    stance_updates: int = 0
    
    @property
    def total_new_infections(self) -> int:
        return len(self.new_rumor_infected)
    
    @property
    def total_new_debunks(self) -> int:
        return len(self.new_debunk_accepted) + len(self.new_recovered)


class PropagationModel:
    """信息传播模型"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        prop_config = config.get('propagation', {})
        
        # 谣言参数
        rumor_config = prop_config.get('rumor', {})
        self.beta0 = rumor_config.get('beta0', -1.0)
        self.beta1 = rumor_config.get('beta1', 2.0)
        self.beta2 = rumor_config.get('beta2', 3.0)
        self.beta3 = rumor_config.get('beta3', 1.5)
        self.beta4 = rumor_config.get('beta4', 1.0)
        
        # 辟谣参数
        debunk_config = prop_config.get('debunk', {})
        self.gamma0 = debunk_config.get('gamma0', 0.5)
        self.gamma1 = debunk_config.get('gamma1', 2.0)
        self.gamma2 = debunk_config.get('gamma2', 1.0)
        self.gamma3 = debunk_config.get('gamma3', 1.5)
        self.rho_d = debunk_config.get('rho_d', 0.7)
        
        # 立场更新
        stance_config = prop_config.get('stance_update', {})
        self.delta_s = stance_config.get('delta', 0.1)
        
        opinion_config = config.get('opinion', {})
        self.gamma_c = opinion_config.get('consistency_decay', 0.9)
        self.epsilon = 0.01
    
    def propagate(
        self,
        network,
        node_states: Dict[int, NodeState],
        group_activities: Dict[str, float],
        current_step: int,
        record_events: bool = False,
        rng: np.random.Generator = None
    ) -> PropagationResult:
        if rng is None:
            rng = np.random.default_rng()
        
        result = PropagationResult()
        
        # 谣言源
        rumor_sources = [
            nid for nid, state in node_states.items()
            if state.infection_state == InfectionState.INFECTED_RUMOR
            and state.throttle_factor > 0
        ]
        
        # 辟谣源
        debunk_sources = [
            nid for nid, state in node_states.items()
            if state.infection_state in [InfectionState.INFECTED_DEBUNK, InfectionState.RECOVERED]
        ]
        
        # 谣言传播
        for source_id in rumor_sources:
            source_state = node_states[source_id]
            neighbors = network.get_neighbors(source_id, direction='out')
            
            for target_id in neighbors:
                target_state = node_states[target_id]
                if target_state.infection_state != InfectionState.SUSCEPTIBLE:
                    continue
                
                edge_weight = network.get_edge_weight(source_id, target_id)
                group = target_state.role.group.value
                activity = group_activities.get(group, 0.5)
                
                social_proof = self._compute_social_proof(target_id, network, node_states, 'rumor')
                accept_prob = self._compute_rumor_accept_prob(target_state, social_proof)
                spread_prob = edge_weight * activity * accept_prob * source_state.throttle_factor
                
                if rng.random() < spread_prob:
                    target_state.infect_rumor(current_step)
                    result.new_rumor_infected.append(target_id)
                    self._update_stance(target_state, source_state, edge_weight, network, self.delta_s)
                    result.stance_updates += 1
                
                if record_events:
                    result.events.append(PropagationEvent(
                        step=current_step, source_id=source_id, target_id=target_id,
                        info_type='rumor', success=target_id in result.new_rumor_infected,
                        probability=spread_prob, target_quadrant=target_state.quadrant_id
                    ))
        
        # 辟谣传播
        for source_id in debunk_sources:
            source_state = node_states[source_id]
            neighbors = network.get_neighbors(source_id, direction='out')
            
            for target_id in neighbors:
                target_state = node_states[target_id]
                if target_state.infection_state not in [InfectionState.SUSCEPTIBLE, InfectionState.INFECTED_RUMOR]:
                    continue
                
                edge_weight = network.get_edge_weight(source_id, target_id)
                group = target_state.role.group.value
                activity = group_activities.get(group, 0.5)
                
                accept_prob = self._compute_debunk_accept_prob(target_state)
                spread_prob = edge_weight * activity * self.rho_d * accept_prob
                
                if rng.random() < spread_prob:
                    old_state = target_state.infection_state
                    target_state.accept_debunk(current_step)
                    
                    if old_state == InfectionState.SUSCEPTIBLE:
                        result.new_debunk_accepted.append(target_id)
                    else:
                        result.new_recovered.append(target_id)
                    
                    self._update_stance(target_state, source_state, edge_weight, network, -self.delta_s)
                    result.stance_updates += 1
        
        return result
    
    def _compute_social_proof(self, node_id: int, network, node_states: Dict[int, NodeState], info_type: str) -> float:
        in_neighbors = network.get_neighbors(node_id, direction='in')
        if len(in_neighbors) == 0:
            return 0.0
        
        target_state = InfectionState.INFECTED_RUMOR if info_type == 'rumor' else InfectionState.INFECTED_DEBUNK
        infected_weight = 0.0
        total_weight = 0.0
        
        for neighbor_id in in_neighbors:
            neighbor_state = node_states[neighbor_id]
            edge_weight = network.get_edge_weight(neighbor_id, node_id)
            
            eff_weight = edge_weight * (0.5 if neighbor_state.infection_state == InfectionState.BANNED else 1.0)
            total_weight += eff_weight
            
            if neighbor_state.infection_state == target_state:
                infected_weight += eff_weight
        
        return infected_weight / (total_weight + self.epsilon)
    
    def _compute_rumor_accept_prob(self, node_state: NodeState, social_proof: float) -> float:
        role = node_state.role
        logit = self.beta0 + self.beta1 * role.susceptibility + self.beta2 * social_proof
        logit -= self.beta3 * role.media_literacy
        logit += self.beta4 * role.stance
        return 1.0 / (1.0 + np.exp(-np.clip(logit, -20, 20)))
    
    def _compute_debunk_accept_prob(self, node_state: NodeState) -> float:
        role = node_state.role
        logit = self.gamma0 + self.gamma1 * role.media_literacy + self.gamma2 * role.openness
        if node_state.infection_state == InfectionState.INFECTED_RUMOR:
            logit -= self.gamma3 * role.stance * role.consistency
        return 1.0 / (1.0 + np.exp(-np.clip(logit, -20, 20)))
    
    def get_debunk_accept_prob(self, node_state: NodeState) -> float:
        return self._compute_debunk_accept_prob(node_state)
    
    def _update_stance(self, target_state: NodeState, source_state: NodeState, edge_weight: float, network, delta: float):
        source_influence = 0.5 + 0.5 * source_state.pagerank / max(network.max_pagerank, 1e-6)
        target_state.role.update_stance(delta, source_influence, edge_weight, self.gamma_c)
    
    def _compute_influence(self, node_state: NodeState, network) -> float:
        return 0.5 + 0.5 * node_state.pagerank / max(network.max_pagerank, 1e-6)
