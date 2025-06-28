"""
策略模块

实现具体的交易策略逻辑，决策何时买入和卖出
"""

from .base_strategy import BaseStrategy
from .mtf_divergence_strategy import MultiTimeframeDivergenceStrategy, create_mtf_strategy
from .config import StrategyConfig, create_strategy_config

__all__ = [
    'BaseStrategy',
    'MultiTimeframeDivergenceStrategy',
    'create_mtf_strategy',
    'StrategyConfig',
    'create_strategy_config'
] 