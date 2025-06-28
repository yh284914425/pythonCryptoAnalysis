"""
MACD指标计算模块

MACD (Moving Average Convergence Divergence) 是一个趋势跟踪动量指标
"""

import pandas as pd
from .base import IndicatorBase


class MACDIndicator(IndicatorBase):
    """MACD指标计算器"""
    
    def calculate(self, df: pd.DataFrame, 
                  fast_period: int = 12, 
                  slow_period: int = 26, 
                  signal_period: int = 9) -> pd.DataFrame:
        """
        计算MACD指标
        
        Args:
            df: K线数据DataFrame
            fast_period: 快线周期，默认12
            slow_period: 慢线周期，默认26
            signal_period: 信号线周期，默认9
            
        Returns:
            添加了MACD相关列的DataFrame
        """
        self.validate_dataframe(df, min_rows=max(fast_period, slow_period) + signal_period)
        
        # 复制DataFrame避免修改原数据
        result_df = df.copy()
        
        # 计算快线和慢线EMA
        close_price = df['收盘价'].astype(float)
        ema_fast = self.ema(close_price, fast_period)
        ema_slow = self.ema(close_price, slow_period)
        
        # 计算MACD线
        macd_line = ema_fast - ema_slow
        
        # 计算信号线（MACD的EMA）
        signal_line = self.ema(macd_line, signal_period)
        
        # 计算直方图
        histogram = macd_line - signal_line
        
        # 添加到结果DataFrame
        result_df['macd'] = macd_line
        result_df['macd_signal'] = signal_line
        result_df['macd_histogram'] = histogram
        
        return result_df


def calculate_macd(df: pd.DataFrame, 
                  fast_period: int = 12, 
                  slow_period: int = 26, 
                  signal_period: int = 9) -> pd.DataFrame:
    """
    便捷函数：计算MACD指标
    
    Args:
        df: K线数据DataFrame，需包含'收盘价'列
        fast_period: 快线周期，默认12
        slow_period: 慢线周期，默认26  
        signal_period: 信号线周期，默认9
        
    Returns:
        添加了macd, macd_signal, macd_histogram列的DataFrame
    """
    indicator = MACDIndicator()
    return indicator.calculate(df, fast_period, slow_period, signal_period)