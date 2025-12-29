"""
工具模块 - 配置加载、日志、种子管理
"""

import os
import yaml
import random
import logging
import numpy as np
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List


# =============================================================================
# 配置管理
# =============================================================================

def load_config(config_path: str = None) -> Dict[str, Any]:
    """
    加载配置文件
    
    Args:
        config_path: 配置文件路径，如果为None则加载默认配置
    
    Returns:
        配置字典
    """
    if config_path is None:
        # 默认配置路径
        config_path = Path(__file__).parent.parent.parent / "configs" / "default.yaml"
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    return config


def merge_configs(base_config: Dict, override_config: Dict) -> Dict:
    """
    合并配置，override_config中的值会覆盖base_config
    """
    result = base_config.copy()
    
    for key, value in override_config.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = value
    
    return result


class Config:
    """
    配置类，支持点号访问
    """
    def __init__(self, config_dict: Dict[str, Any]):
        for key, value in config_dict.items():
            if isinstance(value, dict):
                setattr(self, key, Config(value))
            else:
                setattr(self, key, value)
    
    def to_dict(self) -> Dict[str, Any]:
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, Config):
                result[key] = value.to_dict()
            else:
                result[key] = value
        return result
    
    def __repr__(self):
        return f"Config({self.to_dict()})"


# =============================================================================
# 随机种子管理
# =============================================================================

def set_seed(seed: int):
    """
    设置全局随机种子
    
    Args:
        seed: 随机种子
    """
    random.seed(seed)
    np.random.seed(seed)
    
    # PyTorch种子（如果可用）
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


# =============================================================================
# 日志管理
# =============================================================================

def setup_logger(
    name: str,
    log_dir: str = None,
    level: int = logging.INFO,
    console: bool = True
) -> logging.Logger:
    """
    设置日志器
    
    Args:
        name: 日志器名称
        log_dir: 日志目录
        level: 日志级别
        console: 是否输出到控制台
    
    Returns:
        日志器
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers = []  # 清除已有handler
    
    formatter = logging.Formatter(
        '%(asctime)s | %(name)s | %(levelname)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # 控制台输出
    if console:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # 文件输出
    if log_dir is not None:
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = os.path.join(log_dir, f'{name}_{timestamp}.log')
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


# =============================================================================
# 结果保存
# =============================================================================

class ResultSaver:
    """
    结果保存器 - 支持增量保存，防止程序崩溃丢失结果
    """
    
    def __init__(self, save_dir: str, experiment_name: str = None):
        """
        Args:
            save_dir: 保存目录
            experiment_name: 实验名称
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        if experiment_name is None:
            experiment_name = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.experiment_name = experiment_name
        
        # 创建实验子目录
        self.exp_dir = self.save_dir / experiment_name
        self.exp_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建子目录
        self.metrics_dir = self.exp_dir / "metrics"
        self.plots_dir = self.exp_dir / "plots"
        self.checkpoints_dir = self.exp_dir / "checkpoints"
        
        for d in [self.metrics_dir, self.plots_dir, self.checkpoints_dir]:
            d.mkdir(parents=True, exist_ok=True)
        
        # 存储累积数据
        self.metrics_history = []
    
    def save_metrics(self, metrics: Dict[str, Any], step: int = None, prefix: str = ""):
        """
        保存指标
        
        Args:
            metrics: 指标字典
            step: 步数
            prefix: 文件前缀
        """
        import json
        
        # 添加时间戳
        metrics['timestamp'] = datetime.now().isoformat()
        if step is not None:
            metrics['step'] = step
        
        # 追加到历史
        self.metrics_history.append(metrics)
        
        # 保存到文件
        filename = f"{prefix}_metrics.json" if prefix else "metrics.json"
        filepath = self.metrics_dir / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.metrics_history, f, indent=2, ensure_ascii=False, default=str)
    
    def save_table(self, data: Dict[str, List], filename: str):
        """
        保存表格数据为CSV
        
        Args:
            data: 表格数据，key为列名
            filename: 文件名
        """
        import pandas as pd
        
        df = pd.DataFrame(data)
        filepath = self.metrics_dir / f"{filename}.csv"
        df.to_csv(filepath, index=False)
        
        return filepath
    
    def save_plot(self, fig, filename: str, dpi: int = 150):
        """
        保存图表
        
        Args:
            fig: matplotlib figure
            filename: 文件名（不含扩展名）
            dpi: 分辨率
        """
        filepath = self.plots_dir / f"{filename}.png"
        fig.savefig(filepath, dpi=dpi, bbox_inches='tight')
        return filepath
    
    def save_config(self, config: Dict[str, Any]):
        """
        保存配置
        """
        filepath = self.exp_dir / "config.yaml"
        with open(filepath, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)


# =============================================================================
# 数据结构
# =============================================================================

@dataclass
class EpisodeMetrics:
    """
    单个Episode的指标
    """
    episode_id: int
    seed: int
    
    # 主要指标
    final_infection_rate: float = 0.0
    intervention_effectiveness: float = 0.0
    infection_reduction_rate: float = 0.0
    
    # 辅助指标
    peak_infection_rate: float = 0.0
    debunk_coverage: float = 0.0
    total_cost: float = 0.0
    budget_utilization: float = 0.0
    
    # 分类型有效率
    throttle_effectiveness: float = 0.0
    ban_effectiveness: float = 0.0
    debunk_effectiveness: float = 0.0
    
    # 动作统计
    total_throttle_actions: int = 0
    total_ban_actions: int = 0
    total_debunk_actions: int = 0
    
    # 奖励
    total_reward: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'episode_id': self.episode_id,
            'seed': self.seed,
            'final_infection_rate': self.final_infection_rate,
            'intervention_effectiveness': self.intervention_effectiveness,
            'infection_reduction_rate': self.infection_reduction_rate,
            'peak_infection_rate': self.peak_infection_rate,
            'debunk_coverage': self.debunk_coverage,
            'total_cost': self.total_cost,
            'budget_utilization': self.budget_utilization,
            'throttle_effectiveness': self.throttle_effectiveness,
            'ban_effectiveness': self.ban_effectiveness,
            'debunk_effectiveness': self.debunk_effectiveness,
            'total_throttle_actions': self.total_throttle_actions,
            'total_ban_actions': self.total_ban_actions,
            'total_debunk_actions': self.total_debunk_actions,
            'total_reward': self.total_reward
        }


@dataclass
class TrainingMetrics:
    """
    训练过程指标
    """
    step: int
    episode: int
    
    # 损失
    policy_loss: float = 0.0
    value_loss: float = 0.0
    entropy: float = 0.0
    
    # 评估
    eval_reward_mean: float = 0.0
    eval_reward_std: float = 0.0
    eval_effectiveness_mean: float = 0.0
    eval_infection_rate_mean: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'step': self.step,
            'episode': self.episode,
            'policy_loss': self.policy_loss,
            'value_loss': self.value_loss,
            'entropy': self.entropy,
            'eval_reward_mean': self.eval_reward_mean,
            'eval_reward_std': self.eval_reward_std,
            'eval_effectiveness_mean': self.eval_effectiveness_mean,
            'eval_infection_rate_mean': self.eval_infection_rate_mean
        }


# =============================================================================
# 辅助函数
# =============================================================================

def ensure_dir(path: str) -> Path:
    """确保目录存在"""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def get_device():
    """获取计算设备"""
    try:
        import torch
        if torch.cuda.is_available():
            return torch.device('cuda')
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return torch.device('mps')
        else:
            return torch.device('cpu')
    except ImportError:
        return 'cpu'


def normalize(x: np.ndarray, min_val: float = None, max_val: float = None) -> np.ndarray:
    """
    MinMax归一化
    """
    if min_val is None:
        min_val = x.min()
    if max_val is None:
        max_val = x.max()
    
    if max_val - min_val < 1e-8:
        return np.zeros_like(x)
    
    return (x - min_val) / (max_val - min_val)


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Sigmoid函数"""
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Softmax函数"""
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)
