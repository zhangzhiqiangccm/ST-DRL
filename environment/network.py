"""
网络模块 v3.1 - 社交网络生成与管理
"""

import numpy as np
import networkx as nx
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field


@dataclass
class NetworkData:
    """网络数据结构"""
    n_nodes: int
    graph: nx.DiGraph
    edge_weights: Dict[Tuple[int, int], float] = field(default_factory=dict)
    
    # 结构特征
    out_degrees: np.ndarray = None
    in_degrees: np.ndarray = None
    pagerank: np.ndarray = None
    betweenness: np.ndarray = None
    clustering: np.ndarray = None
    
    # 归一化特征
    out_degrees_norm: np.ndarray = None
    in_degrees_norm: np.ndarray = None
    pagerank_norm: np.ndarray = None
    betweenness_norm: np.ndarray = None
    
    # 统计量
    max_out_degree: int = 0
    max_in_degree: int = 0
    max_pagerank: float = 0.0
    max_betweenness: float = 0.0
    
    def get_neighbors(self, node_id: int, direction: str = 'out') -> List[int]:
        if direction == 'out':
            return list(self.graph.successors(node_id))
        else:
            return list(self.graph.predecessors(node_id))
    
    def get_edge_weight(self, source: int, target: int) -> float:
        return self.edge_weights.get((source, target), 0.0)
    
    def get_node_features(self, node_id: int) -> np.ndarray:
        return np.array([
            self.out_degrees_norm[node_id],
            self.in_degrees_norm[node_id],
            self.pagerank_norm[node_id],
            self.betweenness_norm[node_id],
            self.clustering[node_id]
        ])


class NetworkGenerator:
    """网络生成器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        net_config = config.get('network', {})
        
        self.n_nodes = net_config.get('n_nodes', 2000)
        self.network_type = net_config.get('type', 'barabasi_albert')
        self.m = net_config.get('m', 5)
        self.directed = net_config.get('directed', True)
        self.mutual_prob = net_config.get('mutual_prob', 0.7)
        
        edge_config = net_config.get('edge_weight', {})
        self.weight_min = edge_config.get('min', 0.05)
        self.weight_max = edge_config.get('max', 0.25)
    
    def generate(self, seed: int = None) -> NetworkData:
        if seed is not None:
            np.random.seed(seed)
        rng = np.random.default_rng(seed)
        
        # 生成基础图
        if self.network_type == 'barabasi_albert':
            G = nx.barabasi_albert_graph(self.n_nodes, self.m, seed=int(seed) if seed else None)
        elif self.network_type == 'watts_strogatz':
            k = self.config.get('network', {}).get('k', 6)
            p = self.config.get('network', {}).get('p', 0.3)
            G = nx.watts_strogatz_graph(self.n_nodes, k, p, seed=int(seed) if seed else None)
        elif self.network_type == 'erdos_renyi':
            p = self.config.get('network', {}).get('p', 0.01)
            G = nx.erdos_renyi_graph(self.n_nodes, p, seed=int(seed) if seed else None)
        else:
            G = nx.barabasi_albert_graph(self.n_nodes, self.m, seed=int(seed) if seed else None)
        
        # 转为有向图
        if self.directed:
            DG = nx.DiGraph()
            DG.add_nodes_from(range(self.n_nodes))
            
            for u, v in G.edges():
                DG.add_edge(u, v)
                if rng.random() < self.mutual_prob:
                    DG.add_edge(v, u)
            G = DG
        else:
            G = G.to_directed()
        
        # 生成边权重
        edge_weights = {}
        for u, v in G.edges():
            edge_weights[(u, v)] = rng.uniform(self.weight_min, self.weight_max)
        
        # 创建网络数据
        network = NetworkData(n_nodes=self.n_nodes, graph=G, edge_weights=edge_weights)
        
        # 计算结构特征
        self._compute_features(network)
        
        return network
    
    def _compute_features(self, network: NetworkData):
        G = network.graph
        n = network.n_nodes
        
        # 度数
        network.out_degrees = np.array([G.out_degree(i) for i in range(n)])
        network.in_degrees = np.array([G.in_degree(i) for i in range(n)])
        
        # PageRank
        pr = nx.pagerank(G, alpha=0.85)
        network.pagerank = np.array([pr.get(i, 0) for i in range(n)])
        
        # Betweenness（采样加速）
        if n > 500:
            bc = nx.betweenness_centrality(G, k=min(100, n))
        else:
            bc = nx.betweenness_centrality(G)
        network.betweenness = np.array([bc.get(i, 0) for i in range(n)])
        
        # Clustering
        cc = nx.clustering(G.to_undirected())
        network.clustering = np.array([cc.get(i, 0) for i in range(n)])
        
        # 统计量
        network.max_out_degree = max(network.out_degrees.max(), 1)
        network.max_in_degree = max(network.in_degrees.max(), 1)
        network.max_pagerank = max(network.pagerank.max(), 1e-6)
        network.max_betweenness = max(network.betweenness.max(), 1e-6)
        
        # 归一化
        network.out_degrees_norm = network.out_degrees / network.max_out_degree
        network.in_degrees_norm = network.in_degrees / network.max_in_degree
        network.pagerank_norm = network.pagerank / network.max_pagerank
        network.betweenness_norm = network.betweenness / network.max_betweenness
