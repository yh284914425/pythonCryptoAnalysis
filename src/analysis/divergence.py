"""
背离分析模块

通用背离分析器，可用于任何指标与价格的背离检测
"""

import pandas as pd
import numpy as np
from typing import List, Tuple, Dict, Optional, Any
from .peak_trough_finder import PeakTroughFinder


class DivergenceAnalyzer:
    """通用背离分析器"""
    
    def __init__(self, peak_finder: Optional[PeakTroughFinder] = None):
        """
        初始化背离分析器
        
        Args:
            peak_finder: 高点低点查找器，如果不提供则使用默认配置
        """
        self.peak_finder = peak_finder or PeakTroughFinder()
    
    def detect_bullish_divergence(self, 
                                 price_data: pd.Series, 
                                 indicator_data: pd.Series,
                                 min_distance: int = 10) -> List[Dict[str, Any]]:
        """
        [备用方法] 通用的看涨背离检测
        
        注意：此方法为通用背离检测逻辑，当前策略使用专门的KDJ和MACD背离检测方法。
        保留作为扩展其他指标背离分析的备用方案。
        
        Args:
            price_data: 价格数据
            indicator_data: 指标数据
            min_distance: 背离点间最小距离
            
        Returns:
            背离信息列表，每个包含{'index': int, 'prev_index': int, 'strength': float}
        """
        bullish_divergences = []
        
        # 寻找价格和指标的低点
        price_troughs = self.peak_finder.find_significant_troughs(price_data)
        indicator_troughs = self.peak_finder.find_significant_troughs(indicator_data)
        
        # 合并并排序所有低点
        all_troughs = sorted(set(price_troughs + indicator_troughs))
        
        # 检测背离
        for i in range(1, len(all_troughs)):
            curr_idx = all_troughs[i]
            prev_idx = all_troughs[i-1]
            
            # 检查距离
            if curr_idx - prev_idx < min_distance:
                continue
            
            # 背离条件：价格创新低，指标不创新低
            price_curr = price_data.iloc[curr_idx]
            price_prev = price_data.iloc[prev_idx]
            indicator_curr = indicator_data.iloc[curr_idx]
            indicator_prev = indicator_data.iloc[prev_idx]
            
            if price_curr < price_prev and indicator_curr > indicator_prev:
                # 计算背离强度
                price_change = (price_prev - price_curr) / price_prev
                indicator_change = (indicator_curr - indicator_prev) / abs(indicator_prev) if indicator_prev != 0 else 0
                strength = (price_change + indicator_change) / 2
                
                bullish_divergences.append({
                    'index': curr_idx,
                    'prev_index': prev_idx,
                    'strength': strength,
                    'price_change': price_change,
                    'indicator_change': indicator_change
                })
        
        return bullish_divergences
    
    def detect_bearish_divergence(self, 
                                 price_data: pd.Series, 
                                 indicator_data: pd.Series,
                                 min_distance: int = 10) -> List[Dict[str, Any]]:
        """
        [备用方法] 通用的看跌背离检测
        
        注意：此方法为通用背离检测逻辑，当前策略使用专门的KDJ和MACD背离检测方法。
        保留作为扩展其他指标背离分析的备用方案。
        
        Args:
            price_data: 价格数据
            indicator_data: 指标数据
            min_distance: 背离点间最小距离
            
        Returns:
            背离信息列表，每个包含{'index': int, 'prev_index': int, 'strength': float}
        """
        bearish_divergences = []
        
        # 寻找价格和指标的高点
        price_peaks = self.peak_finder.find_significant_peaks(price_data)
        indicator_peaks = self.peak_finder.find_significant_peaks(indicator_data)
        
        # 合并并排序所有高点
        all_peaks = sorted(set(price_peaks + indicator_peaks))
        
        # 检测背离
        for i in range(1, len(all_peaks)):
            curr_idx = all_peaks[i]
            prev_idx = all_peaks[i-1]
            
            # 检查距离
            if curr_idx - prev_idx < min_distance:
                continue
            
            # 背离条件：价格创新高，指标不创新高
            price_curr = price_data.iloc[curr_idx]
            price_prev = price_data.iloc[prev_idx]
            indicator_curr = indicator_data.iloc[curr_idx]
            indicator_prev = indicator_data.iloc[prev_idx]
            
            if price_curr > price_prev and indicator_curr < indicator_prev:
                # 计算背离强度
                price_change = (price_curr - price_prev) / price_prev
                indicator_change = (indicator_prev - indicator_curr) / abs(indicator_prev) if indicator_prev != 0 else 0
                strength = (price_change + indicator_change) / 2
                
                bearish_divergences.append({
                    'index': curr_idx,
                    'prev_index': prev_idx,
                    'strength': strength,
                    'price_change': price_change,
                    'indicator_change': indicator_change
                })
        
        return bearish_divergences
    
    def CROSS(self, a1, b1, a2, b2):
        """交叉判断：前一根a1<=b1，当前a2>b2"""
        return a1 <= b1 and a2 > b2
    
    def detect_kdj_cross_divergences(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        [主要方法] 使用原有的精确KDJ背离检测方法
        
        注意：这是当前策略的核心背离检测算法，完全保留了原有的精确逻辑。
        
        Args:
            df: 包含价格和KDJ数据的DataFrame
            
        Returns:
            背离信息列表，包含顶背离和底背离
        """
        # 防御性检查：DataFrame有效性
        if df is None or df.empty:
            return []
        
        required_columns = ['kdj_j', 'kdj_j1', '收盘价']
        missing = [col for col in required_columns if col not in df.columns]
        if missing:
            raise ValueError(f"DataFrame必须包含{missing}列")
        
        # 防御性检查：数据长度
        if len(df) < 35:  # 需要至少35个数据点进行分析
            return []
        
        # 转换为列表，便于索引操作，并处理NaN值
        try:
            j = df['kdj_j'].fillna(0).tolist()
            j1 = df['kdj_j1'].fillna(0).tolist()
            close = df['收盘价'].fillna(method='ffill').tolist()
        except Exception:
            return []
        
        divergences = []
        n = 34  # 最小分析周期
        
        # 检测背离
        for i in range(n, len(j)):
            # J上穿J1 (底背离检测)
            j_cross_up_j1 = self.CROSS(j[i-1], j1[i-1], j[i], j1[i])
            # J1上穿J (顶背离检测)  
            j1_cross_up_j = self.CROSS(j1[i-1], j[i-1], j1[i], j[i])
            
            # 底部背离检测
            if j_cross_up_j1:
                # 寻找上一个J上穿J1的位置
                last_cross_index = -1
                for k_idx in range(i - 1, n - 1, -1):
                    if self.CROSS(j[k_idx-1], j1[k_idx-1], j[k_idx], j1[k_idx]):
                        last_cross_index = k_idx
                        break
                
                if last_cross_index != -1:
                    # 判断底部背离条件
                    if (close[last_cross_index] > close[i] and 
                        j[i] > j[last_cross_index] and 
                        j[i] < 20):
                        divergences.append({
                            'type': 'bullish',
                            'index': i,
                            'prev_index': last_cross_index,
                            'j_current': j[i],
                            'j_previous': j[last_cross_index],
                            'price_current': close[i],
                            'price_previous': close[last_cross_index]
                        })
            
            # 顶部背离检测
            if j1_cross_up_j:
                # 寻找上一个J1上穿J的位置
                last_cross_index = -1
                for k_idx in range(i - 1, n - 1, -1):
                    if self.CROSS(j1[k_idx-1], j[k_idx-1], j1[k_idx], j[k_idx]):
                        last_cross_index = k_idx
                        break
                
                if last_cross_index != -1:
                    # 判断顶部背离条件
                    if (close[last_cross_index] < close[i] and 
                        j1[last_cross_index] > j1[i] and 
                        j[i] > 90):
                        divergences.append({
                            'type': 'bearish',
                            'index': i,
                            'prev_index': last_cross_index,
                            'j1_current': j1[i],
                            'j1_previous': j1[last_cross_index],
                            'price_current': close[i],
                            'price_previous': close[last_cross_index]
                        })
        
        return divergences
    
    def analyze_macd_divergence(self, df: pd.DataFrame) -> Dict[str, List]:
        """
        分析MACD背离（使用原有的金叉死叉方法）
        
        Args:
            df: 包含价格和MACD数据的DataFrame
            
        Returns:
            {'bullish': [...], 'bearish': [...]}
        """
        required_columns = ['macd', 'macd_signal', '最高价', '最低价']
        missing = [col for col in required_columns if col not in df.columns]
        if missing:
            raise ValueError(f"DataFrame必须包含{missing}列")
        
        # 使用原有的基于金叉死叉的背离检测方法
        return self._detect_macd_cross_divergences(df)
    
    def _detect_macd_cross_divergences(self, df: pd.DataFrame) -> Dict[str, List]:
        """
        [主要方法] 使用原有的MACD金叉死叉背离检测方法
        
        注意：这是当前策略的MACD背离检测算法，基于金叉死叉的精确逻辑。
        
        Args:
            df: 包含MACD数据的DataFrame
            
        Returns:
            背离结果字典
        """
        # 防御性检查：DataFrame有效性
        if df is None or df.empty:
            return {'bullish': [], 'bearish': []}
        
        # 防御性检查：数据长度
        if len(df) < 10:  # 需要足够的数据点
            return {'bullish': [], 'bearish': []}
        
        try:
            # 标记金叉和死叉，并处理NaN值
            df_copy = df.copy()
            df_copy = df_copy.fillna(method='ffill').fillna(0)  # 填充NaN值
            
            df_copy['golden_cross'] = (df_copy['macd'].shift(1) < df_copy['macd_signal'].shift(1)) & (df_copy['macd'] > df_copy['macd_signal'])
            df_copy['death_cross'] = (df_copy['macd'].shift(1) > df_copy['macd_signal'].shift(1)) & (df_copy['macd'] < df_copy['macd_signal'])
        except Exception:
            return {'bullish': [], 'bearish': []}
        
        bullish_divergences = []
        bearish_divergences = []
        
        # 找出所有金叉和死叉的索引
        golden_cross_idx = df_copy.index[df_copy['golden_cross']].tolist()
        death_cross_idx = df_copy.index[df_copy['death_cross']].tolist()
        
        # 检查底背离（金叉）
        for i in range(1, len(golden_cross_idx)):
            prev_idx = golden_cross_idx[i-1]
            curr_idx = golden_cross_idx[i]
            prev_pos = df_copy.index.get_loc(prev_idx)
            curr_pos = df_copy.index.get_loc(curr_idx)
            
            # 当前价格创新低，MACD未创新低
            if (df_copy['最低价'].iloc[curr_pos] < df_copy['最低价'].iloc[prev_pos] and 
                df_copy['macd'].iloc[curr_pos] > df_copy['macd'].iloc[prev_pos]):
                
                bullish_divergences.append({
                    'type': 'bullish',
                    'index': curr_pos,
                    'prev_index': prev_pos,
                    'price_current': df_copy['最低价'].iloc[curr_pos],
                    'price_previous': df_copy['最低价'].iloc[prev_pos],
                    'macd_current': df_copy['macd'].iloc[curr_pos],
                    'macd_previous': df_copy['macd'].iloc[prev_pos]
                })
        
        # 检查顶背离（死叉）
        for i in range(1, len(death_cross_idx)):
            prev_idx = death_cross_idx[i-1]
            curr_idx = death_cross_idx[i]
            prev_pos = df_copy.index.get_loc(prev_idx)
            curr_pos = df_copy.index.get_loc(curr_idx)
            
            # 当前价格创新高，MACD未创新高
            if (df_copy['最高价'].iloc[curr_pos] > df_copy['最高价'].iloc[prev_pos] and 
                df_copy['macd'].iloc[curr_pos] < df_copy['macd'].iloc[prev_pos]):
                
                bearish_divergences.append({
                    'type': 'bearish',
                    'index': curr_pos,
                    'prev_index': prev_pos,
                    'price_current': df_copy['最高价'].iloc[curr_pos],
                    'price_previous': df_copy['最高价'].iloc[prev_pos],
                    'macd_current': df_copy['macd'].iloc[curr_pos],
                    'macd_previous': df_copy['macd'].iloc[prev_pos]
                })
        
        return {'bullish': bullish_divergences, 'bearish': bearish_divergences}
    
    def analyze_kdj_divergence(self, df: pd.DataFrame) -> Dict[str, List]:
        """
        分析KDJ背离（使用原有的精确方法）
        
        Args:
            df: 包含价格和KDJ数据的DataFrame
            
        Returns:
            {'bullish': [...], 'bearish': [...]}
        """
        # 使用原有的精确KDJ背离检测方法
        all_divergences = self.detect_kdj_cross_divergences(df)
        
        # 分离看涨和看跌背离
        bullish = [d for d in all_divergences if d['type'] == 'bullish']
        bearish = [d for d in all_divergences if d['type'] == 'bearish']
        
        return {'bullish': bullish, 'bearish': bearish}
    
    def get_recent_divergences(self, divergences: List[Dict], 
                              window: int = 50) -> List[Dict]:
        """
        获取最近的背离信号
        
        Args:
            divergences: 背离列表
            window: 时间窗口
            
        Returns:
            最近的背离信号
        """
        if not divergences:
            return []
        
        max_index = max(d['index'] for d in divergences)
        threshold = max_index - window
        
        return [d for d in divergences if d['index'] > threshold]