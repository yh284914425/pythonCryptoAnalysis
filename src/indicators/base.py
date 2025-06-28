"""
指标计算基础类

提供技术指标计算的通用接口和工具函数
"""

import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional


class IndicatorBase(ABC):
    """技术指标计算基础类"""
    
    @abstractmethod
    def calculate(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        计算技术指标
        
        Args:
            df: K线数据DataFrame，需包含中文列名：开盘价, 最高价, 最低价, 收盘价, 成交量
            **kwargs: 指标参数
            
        Returns:
            添加了指标列的DataFrame
        """
        pass
    
    @staticmethod
    def validate_dataframe(df: pd.DataFrame, min_rows: int = 2) -> None:
        """
        验证输入DataFrame格式
        
        Args:
            df: 输入的DataFrame
            min_rows: 最小行数要求
            
        Raises:
            ValueError: 数据格式不符合要求
        """
        if not isinstance(df, pd.DataFrame):
            raise ValueError("输入必须是pandas DataFrame")
        
        if len(df) < min_rows:
            raise ValueError(f"数据量不足，需要至少{min_rows}行数据")
        
        required_columns = ['开盘价', '最高价', '最低价', '收盘价', '成交量']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"缺少必要的列: {missing_columns}")
    
    @staticmethod
    def sma(data: pd.Series, period: int) -> pd.Series:
        """简单移动平均"""
        return data.rolling(window=period, min_periods=1).mean()
    
    @staticmethod
    def ema(data: pd.Series, period: int) -> pd.Series:
        """指数移动平均"""
        return data.ewm(span=period, adjust=False).mean()
    
    @staticmethod
    def custom_sma(data: list, n: int, m: int) -> list:
        """
        自定义平滑移动平均 (用于KDJ等指标)
        
        Args:
            data: 数据列表
            n: 周期参数
            m: 权重参数
            
        Returns:
            平滑后的数据列表
        """
        if not data:
            return []
        
        result = [data[0]]
        for i in range(1, len(data)):
            sma_value = (m * data[i] + (n - m) * result[i-1]) / n
            result.append(sma_value)
        
        return result
    
    @staticmethod
    def highest(data: pd.Series, period: int) -> pd.Series:
        """最高值"""
        return data.rolling(window=period, min_periods=1).max()
    
    @staticmethod
    def lowest(data: pd.Series, period: int) -> pd.Series:
        """最低值"""
        return data.rolling(window=period, min_periods=1).min()