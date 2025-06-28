"""
高点低点检测模块

提供通用功能来寻找数据序列（价格或指标）中的显著高点和低点
"""

import pandas as pd
import numpy as np
from typing import List, Tuple, Optional
from scipy.signal import find_peaks


class PeakTroughFinder:
    """
    高点低点检测器
    
    注意：本类提供多种高低点检测方法，但当前背离分析主要使用 find_pivot_points 方法。
    其他方法作为备用方案保留。
    """
    
    def __init__(self, min_distance: int = 5, prominence: float = 0.01):
        """
        初始化检测器
        
        Args:
            min_distance: 相邻峰值间最小距离
            prominence: 峰值的最小显著性
        """
        self.min_distance = min_distance
        self.prominence = prominence
    
    def find_peaks_and_troughs(self, data: pd.Series) -> Tuple[List[int], List[int]]:
        """
        [备用方法] 基于scipy的高低点检测
        
        注意：此方法当前未被主要策略使用，保留作为备用方案。
        
        Args:
            data: 数据序列
            
        Returns:
            (高点索引列表, 低点索引列表)
        """
        if len(data) < self.min_distance * 2:
            return [], []
        
        # 寻找高点
        peaks, _ = find_peaks(
            data.values,
            distance=self.min_distance,
            prominence=self.prominence * (data.max() - data.min())
        )
        
        # 寻找低点（反转数据寻找峰值）
        troughs, _ = find_peaks(
            -data.values,
            distance=self.min_distance,
            prominence=self.prominence * (data.max() - data.min())
        )
        
        return peaks.tolist(), troughs.tolist()
    
    def find_significant_peaks(self, price_data: pd.Series, 
                              window: int = 10, 
                              threshold: float = 0.02) -> List[int]:
        """
        [备用方法] 寻找价格数据中的显著高点
        
        注意：此方法当前未被主要策略使用，保留作为备用方案。
        
        Args:
            price_data: 价格数据
            window: 查看窗口大小
            threshold: 显著性阈值（百分比）
            
        Returns:
            显著高点的索引列表
        """
        peaks = []
        
        for i in range(window, len(price_data) - window):
            current_price = price_data.iloc[i]
            
            # 检查是否为局部最高点
            left_max = price_data.iloc[i-window:i].max()
            right_max = price_data.iloc[i+1:i+window+1].max()
            
            if current_price >= left_max and current_price >= right_max:
                # 检查显著性
                local_range = price_data.iloc[i-window:i+window+1]
                if (current_price - local_range.min()) / local_range.mean() > threshold:
                    peaks.append(i)
        
        return peaks
    
    def find_significant_troughs(self, price_data: pd.Series, 
                                window: int = 10, 
                                threshold: float = 0.02) -> List[int]:
        """
        [备用方法] 寻找价格数据中的显著低点
        
        注意：此方法当前未被主要策略使用，保留作为备用方案。
        
        Args:
            price_data: 价格数据
            window: 查看窗口大小
            threshold: 显著性阈值（百分比）
            
        Returns:
            显著低点的索引列表
        """
        troughs = []
        
        for i in range(window, len(price_data) - window):
            current_price = price_data.iloc[i]
            
            # 检查是否为局部最低点
            left_min = price_data.iloc[i-window:i].min()
            right_min = price_data.iloc[i+1:i+window+1].min()
            
            if current_price <= left_min and current_price <= right_min:
                # 检查显著性
                local_range = price_data.iloc[i-window:i+window+1]
                if (local_range.max() - current_price) / local_range.mean() > threshold:
                    troughs.append(i)
        
        return troughs
    
    def find_pivot_points(self, high_data: pd.Series, 
                         low_data: pd.Series, 
                         left_bars: int = 4, 
                         right_bars: int = 2) -> Tuple[List[int], List[int]]:
        """
        [主要方法] 使用经典的Pivot Point方法寻找转折点
        
        注意：这是当前背离分析策略推荐使用的高低点检测方法。
        
        Args:
            high_data: 最高价数据
            low_data: 最低价数据
            left_bars: 左侧比较柱数
            right_bars: 右侧比较柱数
            
        Returns:
            (高点转折索引, 低点转折索引)
        """
        high_pivots = []
        low_pivots = []
        
        for i in range(left_bars, len(high_data) - right_bars):
            # 检查高点转折
            is_high_pivot = True
            current_high = high_data.iloc[i]
            
            # 检查左侧
            for j in range(i - left_bars, i):
                if high_data.iloc[j] > current_high:
                    is_high_pivot = False
                    break
            
            # 检查右侧
            if is_high_pivot:
                for j in range(i + 1, i + right_bars + 1):
                    if high_data.iloc[j] > current_high:
                        is_high_pivot = False
                        break
            
            if is_high_pivot:
                high_pivots.append(i)
            
            # 检查低点转折
            is_low_pivot = True
            current_low = low_data.iloc[i]
            
            # 检查左侧
            for j in range(i - left_bars, i):
                if low_data.iloc[j] < current_low:
                    is_low_pivot = False
                    break
            
            # 检查右侧
            if is_low_pivot:
                for j in range(i + 1, i + right_bars + 1):
                    if low_data.iloc[j] < current_low:
                        is_low_pivot = False
                        break
            
            if is_low_pivot:
                low_pivots.append(i)
        
        return high_pivots, low_pivots
    
    def filter_peaks_by_distance(self, peaks: List[int], 
                                min_distance: int) -> List[int]:
        """
        根据最小距离过滤峰值
        
        Args:
            peaks: 峰值索引列表
            min_distance: 最小距离
            
        Returns:
            过滤后的峰值索引列表
        """
        if not peaks:
            return []
        
        filtered_peaks = [peaks[0]]
        
        for peak in peaks[1:]:
            if peak - filtered_peaks[-1] >= min_distance:
                filtered_peaks.append(peak)
        
        return filtered_peaks