"""
分析引擎模块

该模块负责从数据中识别特定的"分析模式"，如背离、金叉等。
它不关心指标如何计算，只负责分析已计算的指标数据。
"""

from .peak_trough_finder import PeakTroughFinder
from .divergence import DivergenceAnalyzer
from .pattern_detector import PatternDetector

__all__ = ['PeakTroughFinder', 'DivergenceAnalyzer', 'PatternDetector'] 