"""
KDJ指标计算模块

KDJ是在KD指标基础上发展起来的随机指标，属于超买超卖类指标
"""

import pandas as pd
import numpy as np
from .base import IndicatorBase


class KDJIndicator(IndicatorBase):
    """KDJ指标计算器"""
    
    def calculate(self, df: pd.DataFrame,
                  n: int = 34,
                  m1: int = 3,
                  m2: int = 8,
                  m3: int = 1,
                  m4: int = 6,
                  m5: int = 1,
                  j_period: int = 3) -> pd.DataFrame:
        """
        计算KDJ指标
        
        Args:
            df: K线数据DataFrame
            n: RSV计算周期，默认34
            m1: RSV平滑周期，默认3
            m2: K值计算周期，默认8
            m3: K值权重，默认1
            m4: D值计算周期，默认6
            m5: D值权重，默认1
            j_period: J1计算周期，默认3
            
        Returns:
            添加了KDJ相关列的DataFrame
        """
        self.validate_dataframe(df, min_rows=n)
        
        # 复制DataFrame避免修改原数据
        result_df = df.copy()
        
        # 提取价格数据
        high = df['最高价'].astype(float)
        low = df['最低价'].astype(float)
        close = df['收盘价'].astype(float)
        
        # 计算LLV和HHV
        llv = self.lowest(low, n)
        hhv = self.highest(high, n)
        
        # 计算平滑后的最高价和最低价
        lowv = self.ema(llv, m1)
        highv = self.ema(hhv, m1)
        
        # 计算RSV
        rsv = np.where(
            highv == lowv,
            50,
            100 * (close - lowv) / (highv - lowv)
        )
        
        # 计算RSV的EMA平滑
        rsv_series = pd.Series(rsv, index=df.index)
        rsv_ema = self.ema(rsv_series, m1)
        
        # 使用自定义SMA计算K值和D值
        k_values = self.custom_sma(rsv_ema.tolist(), m2, m3)
        d_values = self.custom_sma(k_values, m4, m5)
        
        # 计算J值
        k_series = pd.Series(k_values, index=df.index)
        d_series = pd.Series(d_values, index=df.index)
        j_values = 3 * k_series - 2 * d_series
        
        # 计算J1（J的移动平均）
        j1_values = self.sma(j_values, j_period)
        
        # 添加到结果DataFrame
        result_df['kdj_k'] = k_series
        result_df['kdj_d'] = d_series
        result_df['kdj_j'] = j_values
        result_df['kdj_j1'] = j1_values
        
        return result_df


def calculate_kdj(df: pd.DataFrame,
                  n: int = 34,
                  m1: int = 3,
                  m2: int = 8,
                  m3: int = 1,
                  m4: int = 6,
                  m5: int = 1,
                  j_period: int = 3) -> pd.DataFrame:
    """
    便捷函数：计算KDJ指标
    
    Args:
        df: K线数据DataFrame，需包含'最高价', '最低价', '收盘价'列
        n: RSV计算周期，默认34
        m1: RSV平滑周期，默认3
        m2: K值计算周期，默认8
        m3: K值权重，默认1
        m4: D值计算周期，默认6
        m5: D值权重，默认1
        j_period: J1计算周期，默认3
        
    Returns:
        添加了kdj_k, kdj_d, kdj_j, kdj_j1列的DataFrame
    """
    indicator = KDJIndicator()
    return indicator.calculate(df, n, m1, m2, m3, m4, m5, j_period)