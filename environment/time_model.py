"""
时间模型 v3.1 - 时间-群体活跃度
"""

import numpy as np
from typing import Dict, Any, List
from dataclasses import dataclass

from .node import GroupType


@dataclass
class TimeContext:
    """时间上下文"""
    step: int
    hour: int
    day_type: str
    group_activity: Dict[str, float]
    time_heterogeneity: float
    
    def get_activity(self, group: str) -> float:
        return self.group_activity.get(group, 0.5)


class TimeModel:
    """时间-群体活跃度模型"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        time_config = config.get('time', {})
        
        self.episode_length = time_config.get('episode_length', 48)
        self.step_hours = time_config.get('step_hours', 1)
        self.start_hour = time_config.get('start_hour', 8)
        self.start_day = time_config.get('start_day', 'weekday')
        
        # 活跃度曲线
        activity_config = config.get('activity', {})
        self.hourly_activity = {}
        self.weekday_mod = {}
        self.weekend_mod = {}
        
        for group in GroupType:
            g = group.value
            g_config = activity_config.get(g, {})
            
            hourly = g_config.get('hourly', [0.5] * 24)
            if len(hourly) < 24:
                hourly = hourly + [0.5] * (24 - len(hourly))
            self.hourly_activity[g] = np.array(hourly[:24])
            
            self.weekday_mod[g] = g_config.get('weekday_mod', 1.0)
            self.weekend_mod[g] = g_config.get('weekend_mod', 1.0)
    
    def get_time_context(self, step: int) -> TimeContext:
        hour = (self.start_hour + step * self.step_hours) % 24
        
        # 日期类型
        days_passed = (self.start_hour + step * self.step_hours) // 24
        if self.start_day == 'weekday':
            is_weekend = days_passed % 7 >= 5
        else:
            is_weekend = days_passed % 7 < 2
        day_type = 'weekend' if is_weekend else 'weekday'
        
        # 群体活跃度
        group_activity = {}
        for group in GroupType:
            g = group.value
            base = self.hourly_activity[g][hour]
            mod = self.weekend_mod[g] if is_weekend else self.weekday_mod[g]
            group_activity[g] = np.clip(base * mod, 0.01, 1.0)
        
        # 时间异质性
        activities = list(group_activity.values())
        mean_act = np.mean(activities)
        std_act = np.std(activities)
        heterogeneity = std_act / (mean_act + 0.01)
        
        return TimeContext(
            step=step,
            hour=hour,
            day_type=day_type,
            group_activity=group_activity,
            time_heterogeneity=heterogeneity
        )
    
    def get_future_activities(self, current_step: int, horizon: int) -> List[Dict[str, float]]:
        future = []
        for i in range(1, horizon + 1):
            ctx = self.get_time_context(min(current_step + i, self.episode_length - 1))
            future.append(ctx.group_activity)
        return future
