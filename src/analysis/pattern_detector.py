"""
技术分析模式检测模块

检测各种技术分析模式，如金叉死叉、突破等
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any


class PatternDetector:
    """技术分析模式检测器"""
    
    def __init__(self):
        pass
    
    def detect_golden_cross(self, fast_line: pd.Series, 
                           slow_line: pd.Series) -> List[Dict[str, Any]]:
        """
        检测金叉模式
        
        Args:
            fast_line: 快线数据
            slow_line: 慢线数据
            
        Returns:
            金叉点信息列表
        """
        golden_crosses = []
        
        for i in range(1, len(fast_line)):
            # 金叉条件：前一个快线<=慢线，当前快线>慢线
            if (fast_line.iloc[i-1] <= slow_line.iloc[i-1] and 
                fast_line.iloc[i] > slow_line.iloc[i]):
                
                golden_crosses.append({
                    'index': i,
                    'fast_value': fast_line.iloc[i],
                    'slow_value': slow_line.iloc[i],
                    'cross_strength': abs(fast_line.iloc[i] - slow_line.iloc[i])
                })
        
        return golden_crosses
    
    def detect_death_cross(self, fast_line: pd.Series, 
                          slow_line: pd.Series) -> List[Dict[str, Any]]:
        """
        检测死叉模式
        
        Args:
            fast_line: 快线数据
            slow_line: 慢线数据
            
        Returns:
            死叉点信息列表
        """
        death_crosses = []
        
        for i in range(1, len(fast_line)):
            # 死叉条件：前一个快线>=慢线，当前快线<慢线
            if (fast_line.iloc[i-1] >= slow_line.iloc[i-1] and 
                fast_line.iloc[i] < slow_line.iloc[i]):
                
                death_crosses.append({
                    'index': i,
                    'fast_value': fast_line.iloc[i],
                    'slow_value': slow_line.iloc[i],
                    'cross_strength': abs(fast_line.iloc[i] - slow_line.iloc[i])
                })
        
        return death_crosses
    
    def detect_macd_signals(self, df: pd.DataFrame) -> Dict[str, List]:
        """
        检测MACD信号
        
        Args:
            df: 包含MACD数据的DataFrame
            
        Returns:
            MACD信号字典
        """
        if 'macd' not in df.columns or 'macd_signal' not in df.columns:
            raise ValueError("DataFrame必须包含'macd'和'macd_signal'列")
        
        golden_crosses = self.detect_golden_cross(df['macd'], df['macd_signal'])
        death_crosses = self.detect_death_cross(df['macd'], df['macd_signal'])
        
        return {
            'golden_crosses': golden_crosses,
            'death_crosses': death_crosses
        }
    
    def detect_zero_line_cross(self, indicator: pd.Series, 
                              zero_level: float = 0) -> Dict[str, List]:
        """
        检测零轴穿越
        
        Args:
            indicator: 指标数据
            zero_level: 零轴水平，默认0
            
        Returns:
            零轴穿越信息
        """
        up_crosses = []  # 上穿零轴
        down_crosses = []  # 下穿零轴
        
        for i in range(1, len(indicator)):
            # 上穿零轴
            if indicator.iloc[i-1] <= zero_level < indicator.iloc[i]:
                up_crosses.append({
                    'index': i,
                    'value': indicator.iloc[i]
                })
            
            # 下穿零轴
            elif indicator.iloc[i-1] >= zero_level > indicator.iloc[i]:
                down_crosses.append({
                    'index': i,
                    'value': indicator.iloc[i]
                })
        
        return {
            'up_crosses': up_crosses,
            'down_crosses': down_crosses
        }
    
    def detect_overbought_oversold(self, indicator: pd.Series,
                                  overbought_level: float = 80,
                                  oversold_level: float = 20) -> Dict[str, List]:
        """
        检测超买超卖区域
        
        Args:
            indicator: 指标数据
            overbought_level: 超买水平
            oversold_level: 超卖水平
            
        Returns:
            超买超卖信息
        """
        overbought_points = []
        oversold_points = []
        
        for i, value in enumerate(indicator):
            if value >= overbought_level:
                overbought_points.append({
                    'index': i,
                    'value': value
                })
            elif value <= oversold_level:
                oversold_points.append({
                    'index': i,
                    'value': value
                })
        
        return {
            'overbought': overbought_points,
            'oversold': oversold_points
        }
    
    def detect_kdj_signals(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        检测KDJ信号
        
        Args:
            df: 包含KDJ数据的DataFrame
            
        Returns:
            KDJ信号字典
        """
        required_columns = ['kdj_k', 'kdj_d', 'kdj_j']
        missing = [col for col in required_columns if col not in df.columns]
        if missing:
            raise ValueError(f"DataFrame必须包含{missing}列")
        
        results = {}
        
        # K线与D线交叉
        results['kd_crosses'] = {
            'golden': self.detect_golden_cross(df['kdj_k'], df['kdj_d']),
            'death': self.detect_death_cross(df['kdj_k'], df['kdj_d'])
        }
        
        # 超买超卖
        results['overbought_oversold'] = self.detect_overbought_oversold(
            df['kdj_j'], overbought_level=90, oversold_level=10
        )
        
        # J值零轴穿越
        if 'kdj_j' in df.columns:
            results['j_zero_cross'] = self.detect_zero_line_cross(df['kdj_j'])
        
        return results
    
    def detect_trend_strength(self, price_data: pd.Series, 
                             window: int = 20) -> pd.Series:
        """
        检测趋势强度
        
        Args:
            price_data: 价格数据
            window: 计算窗口
            
        Returns:
            趋势强度序列
        """
        # 计算移动平均
        ma = price_data.rolling(window=window).mean()
        
        # 计算趋势强度（价格变化率的移动平均）
        price_change = price_data.pct_change()
        trend_strength = price_change.rolling(window=window).mean()
        
        return trend_strength
    
    def detect_support_resistance(self, high_data: pd.Series, 
                                 low_data: pd.Series,
                                 window: int = 20,
                                 touch_threshold: float = 0.02) -> Dict[str, List]:
        """
        检测支撑阻力位
        
        Args:
            high_data: 最高价数据
            low_data: 最低价数据
            window: 计算窗口
            touch_threshold: 触及阈值
            
        Returns:
            支撑阻力位信息
        """
        resistance_levels = []
        support_levels = []
        
        # 寻找阻力位（高点聚集区域）
        for i in range(window, len(high_data) - window):
            level = high_data.iloc[i]
            touch_count = 0
            
            # 统计该水平附近的触及次数
            for j in range(i - window, i + window):
                if abs(high_data.iloc[j] - level) / level < touch_threshold:
                    touch_count += 1
            
            if touch_count >= 3:  # 至少3次触及
                resistance_levels.append({
                    'level': level,
                    'index': i,
                    'touch_count': touch_count
                })
        
        # 寻找支撑位（低点聚集区域）
        for i in range(window, len(low_data) - window):
            level = low_data.iloc[i]
            touch_count = 0
            
            # 统计该水平附近的触及次数
            for j in range(i - window, i + window):
                if abs(low_data.iloc[j] - level) / level < touch_threshold:
                    touch_count += 1
            
            if touch_count >= 3:  # 至少3次触及
                support_levels.append({
                    'level': level,
                    'index': i,
                    'touch_count': touch_count
                })
        
        return {
            'resistance': resistance_levels,
            'support': support_levels
        }