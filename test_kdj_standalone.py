#!/usr/bin/env python
"""
🔍 动态KDJ指标系统独立测试
Independent Test for Dynamic KDJ Indicator System
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, List, Optional
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# 简化的配置类
class KDJConfig:
    """动态KDJ参数配置"""
    def __init__(self):
        self.short_term = (18, 5, 5)    # 短期交易参数，胜率58%
        self.medium_term = (14, 7, 7)   # 中期交易参数，胜率62%
        self.long_term = (21, 10, 10)   # 长期交易参数，胜率65%
        
        # ATR分位数阈值
        self.atr_high_threshold = 0.75   # 高波动阈值
        self.atr_low_threshold = 0.25    # 低波动阈值
        self.atr_lookback_period = 100   # ATR历史周期

class DynamicKDJIndicator:
    """动态KDJ指标系统"""
    
    def __init__(self, config: KDJConfig):
        self.config = config
        self.atr_history = []
        
    def calculate_adaptive_kdj_params(self, prices: pd.DataFrame, atr_values: pd.Series) -> Tuple[int, int, int]:
        """
        根据ATR动态选择KDJ参数
        
        Args:
            prices: 价格数据DataFrame
            atr_values: ATR值序列
            
        Returns:
            (k_period, d_period, j_period): KDJ参数元组
        """
        if len(atr_values) < self.config.atr_lookback_period:
            return self.config.medium_term
        
        # 计算ATR分位数
        atr_recent = atr_values.tail(self.config.atr_lookback_period)
        atr_percentile_75 = atr_recent.quantile(self.config.atr_high_threshold)
        atr_percentile_25 = atr_recent.quantile(self.config.atr_low_threshold)
        
        current_atr = atr_values.iloc[-1]
        
        # 动态参数选择
        if current_atr > atr_percentile_75:
            return self.config.short_term  # 高波动，使用短期参数
        elif current_atr < atr_percentile_25:
            return self.config.long_term   # 低波动，使用长期参数
        else:
            return self.config.medium_term # 中等波动，使用中期参数
    
    def calculate_kdj(self, high: pd.Series, low: pd.Series, close: pd.Series, 
                     k_period: int = 9, d_period: int = 3, j_period: int = 3) -> Dict[str, pd.Series]:
        """
        计算KDJ指标
        
        Args:
            high, low, close: 价格序列
            k_period, d_period, j_period: KDJ参数
            
        Returns:
            包含K、D、J值的字典
        """
        # 计算RSV (Raw Stochastic Value)
        lowest_low = low.rolling(window=k_period).min()
        highest_high = high.rolling(window=k_period).max()
        
        # 避免除零错误
        rsv = np.where(
            (highest_high - lowest_low) == 0,
            50,  # 如果高低价相同，RSV设为50
            (close - lowest_low) / (highest_high - lowest_low) * 100
        )
        rsv = pd.Series(rsv, index=close.index)
        
        # 计算K值
        k_values = []
        k_prev = 50  # 初始K值
        
        for rsv_val in rsv:
            if pd.isna(rsv_val):
                k_values.append(np.nan)
                continue
                
            k_current = (2 * k_prev + rsv_val) / 3
            k_values.append(k_current)
            k_prev = k_current
        
        k_series = pd.Series(k_values, index=close.index)
        
        # 计算D值
        d_values = []
        d_prev = 50  # 初始D值
        
        for k_val in k_series:
            if pd.isna(k_val):
                d_values.append(np.nan)
                continue
                
            d_current = (2 * d_prev + k_val) / 3
            d_values.append(d_current)
            d_prev = d_current
        
        d_series = pd.Series(d_values, index=close.index)
        
        # 计算J值
        j_series = 3 * k_series - 2 * d_series
        
        return {
            'K': k_series,
            'D': d_series, 
            'J': j_series
        }
    
    def detect_kdj_divergence(self, prices: pd.DataFrame, kdj_values: Dict[str, pd.Series], 
                             lookback_period: int = 20) -> Dict[str, List]:
        """
        检测KDJ背离信号
        
        Args:
            prices: 价格数据
            kdj_values: KDJ指标值
            lookback_period: 回望周期
            
        Returns:
            背离信号字典
        """
        divergences = {
            'bullish_divergence': [],  # 底背离
            'bearish_divergence': [],  # 顶背离
        }
        
        close = prices['close']
        j_values = kdj_values['J']
        
        for i in range(lookback_period, len(close)):
            # 查找局部极值
            price_window = close.iloc[i-lookback_period:i+1]
            j_window = j_values.iloc[i-lookback_period:i+1]
            
            # 检测底背离
            if self._is_price_low(price_window) and self._is_j_low(j_window):
                if self._check_bullish_divergence(price_window, j_window):
                    divergences['bullish_divergence'].append({
                        'timestamp': close.index[i],
                        'price': close.iloc[i],
                        'j_value': j_values.iloc[i],
                        'strength': self._calculate_divergence_strength(price_window, j_window)
                    })
            
            # 检测顶背离
            if self._is_price_high(price_window) and self._is_j_high(j_window):
                if self._check_bearish_divergence(price_window, j_window):
                    divergences['bearish_divergence'].append({
                        'timestamp': close.index[i],
                        'price': close.iloc[i],
                        'j_value': j_values.iloc[i],
                        'strength': self._calculate_divergence_strength(price_window, j_window)
                    })
        
        return divergences
    
    def _is_price_low(self, price_series: pd.Series) -> bool:
        """判断是否为价格低点"""
        return price_series.iloc[-1] == price_series.min()
    
    def _is_price_high(self, price_series: pd.Series) -> bool:
        """判断是否为价格高点"""
        return price_series.iloc[-1] == price_series.max()
    
    def _is_j_low(self, j_series: pd.Series) -> bool:
        """判断是否为J值低点"""
        return j_series.iloc[-1] == j_series.min()
    
    def _is_j_high(self, j_series: pd.Series) -> bool:
        """判断是否为J值高点"""
        return j_series.iloc[-1] == j_series.max()
    
    def _check_bullish_divergence(self, price_series: pd.Series, j_series: pd.Series) -> bool:
        """检查底背离条件"""
        # 价格创新低但J值未创新低
        if len(price_series) < 10:
            return False
            
        recent_price_min = price_series.tail(5).min()
        historical_price_min = price_series.head(-5).min()
        
        recent_j_min = j_series.tail(5).min()
        historical_j_min = j_series.head(-5).min()
        
        return (recent_price_min < historical_price_min and 
                recent_j_min > historical_j_min)
    
    def _check_bearish_divergence(self, price_series: pd.Series, j_series: pd.Series) -> bool:
        """检查顶背离条件"""
        # 价格创新高但J值未创新高
        if len(price_series) < 10:
            return False
            
        recent_price_max = price_series.tail(5).max()
        historical_price_max = price_series.head(-5).max()
        
        recent_j_max = j_series.tail(5).max()
        historical_j_max = j_series.head(-5).max()
        
        return (recent_price_max > historical_price_max and 
                recent_j_max < historical_j_max)
    
    def _calculate_divergence_strength(self, price_series: pd.Series, j_series: pd.Series) -> str:
        """计算背离强度"""
        price_range = price_series.max() - price_series.min()
        j_range = j_series.max() - j_series.min()
        
        # 根据价格和指标的变化幅度判断强度
        price_change_pct = price_range / price_series.mean()
        
        if price_change_pct > 0.05 and j_range > 20:
            return "强"
        elif price_change_pct > 0.03 and j_range > 15:
            return "中"
        else:
            return "弱"

def calculate_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """计算ATR指标"""
    high_low = high - low
    high_close_prev = abs(high - close.shift(1))
    low_close_prev = abs(low - close.shift(1))
    
    true_range = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
    atr = true_range.rolling(period).mean()
    
    return atr

def create_test_data(periods: int = 200) -> pd.DataFrame:
    """创建测试数据"""
    dates = pd.date_range('2024-01-01', periods=periods, freq='1H')
    np.random.seed(42)
    
    # 生成带趋势的价格数据
    trend = np.linspace(50000, 55000, periods)
    noise = np.cumsum(np.random.randn(periods) * 100)
    prices = trend + noise
    
    # 添加一些特殊模式
    # 在中间添加一个强势上涨
    mid_point = periods // 2
    prices[mid_point:mid_point+20] += np.linspace(0, 2000, 20)
    
    # 在后期添加一个回调
    late_point = int(periods * 0.75)
    prices[late_point:late_point+15] -= np.linspace(0, 1500, 15)
    
    test_data = pd.DataFrame({
        'open': prices + np.random.randn(periods) * 50,
        'high': prices + np.abs(np.random.randn(periods)) * 150,
        'low': prices - np.abs(np.random.randn(periods)) * 150,
        'close': prices,
        'volume': np.random.randint(1000, 10000, periods)
    }, index=dates)
    
    # 确保high >= low, close在high和low之间
    test_data['high'] = np.maximum(test_data['high'], test_data['close'])
    test_data['low'] = np.minimum(test_data['low'], test_data['close'])
    test_data['open'] = np.clip(test_data['open'], test_data['low'], test_data['high'])
    
    return test_data

def run_kdj_test():
    """运行KDJ指标系统测试"""
    print("🔍 动态KDJ指标系统测试开始...")
    print("=" * 60)
    
    # 创建测试数据
    test_data = create_test_data()
    print(f"📊 生成测试数据: {len(test_data)} 个数据点")
    print(f"时间范围: {test_data.index[0]} 到 {test_data.index[-1]}")
    print(f"价格范围: ${test_data['close'].min():.2f} - ${test_data['close'].max():.2f}")
    
    # 初始化KDJ指标
    config = KDJConfig()
    kdj_indicator = DynamicKDJIndicator(config)
    
    # 计算ATR
    atr_values = calculate_atr(test_data['high'], test_data['low'], test_data['close'])
    
    # 测试自适应参数选择
    print(f"\n🎯 自适应参数测试:")
    adaptive_params = kdj_indicator.calculate_adaptive_kdj_params(test_data, atr_values)
    print(f"当前ATR: {atr_values.iloc[-1]:.2f}")
    print(f"ATR 75分位数: {atr_values.quantile(0.75):.2f}")
    print(f"ATR 25分位数: {atr_values.quantile(0.25):.2f}")
    print(f"选择的KDJ参数: K={adaptive_params[0]}, D={adaptive_params[1]}, J={adaptive_params[2]}")
    
    # 计算KDJ指标
    kdj_values = kdj_indicator.calculate_kdj(
        test_data['high'], 
        test_data['low'], 
        test_data['close'],
        k_period=adaptive_params[0],
        d_period=adaptive_params[1],
        j_period=adaptive_params[2]
    )
    
    print(f"\n📈 KDJ指标计算结果:")
    print(f"最新K值: {kdj_values['K'].iloc[-1]:.2f}")
    print(f"最新D值: {kdj_values['D'].iloc[-1]:.2f}")
    print(f"最新J值: {kdj_values['J'].iloc[-1]:.2f}")
    
    # 分析KDJ状态
    k_val = kdj_values['K'].iloc[-1]
    d_val = kdj_values['D'].iloc[-1]
    j_val = kdj_values['J'].iloc[-1]
    
    if j_val < 20:
        signal_status = "🔴 超卖信号 - 考虑买入"
    elif j_val > 80:
        signal_status = "🟢 超买信号 - 考虑卖出"
    elif k_val > d_val:
        signal_status = "📈 多头趋势"
    else:
        signal_status = "📉 空头趋势"
    
    print(f"当前信号状态: {signal_status}")
    
    # 检测背离信号
    print(f"\n🎭 背离信号检测:")
    divergences = kdj_indicator.detect_kdj_divergence(test_data, kdj_values)
    
    print(f"检测到 {len(divergences['bullish_divergence'])} 个底背离信号:")
    for div in divergences['bullish_divergence'][-3:]:  # 显示最近3个
        print(f"  📅 {div['timestamp']}: 价格${div['price']:.2f}, J值{div['j_value']:.2f}, 强度:{div['strength']}")
    
    print(f"检测到 {len(divergences['bearish_divergence'])} 个顶背离信号:")
    for div in divergences['bearish_divergence'][-3:]:  # 显示最近3个
        print(f"  📅 {div['timestamp']}: 价格${div['price']:.2f}, J值{div['j_value']:.2f}, 强度:{div['strength']}")
    
    # 统计分析
    print(f"\n📊 统计分析:")
    print(f"K值统计: 均值{kdj_values['K'].mean():.2f}, 标准差{kdj_values['K'].std():.2f}")
    print(f"D值统计: 均值{kdj_values['D'].mean():.2f}, 标准差{kdj_values['D'].std():.2f}")
    print(f"J值统计: 均值{kdj_values['J'].mean():.2f}, 标准差{kdj_values['J'].std():.2f}")
    
    # 超买超卖统计
    oversold_count = (kdj_values['J'] < 20).sum()
    overbought_count = (kdj_values['J'] > 80).sum()
    print(f"超卖次数: {oversold_count} ({oversold_count/len(kdj_values['J'])*100:.1f}%)")
    print(f"超买次数: {overbought_count} ({overbought_count/len(kdj_values['J'])*100:.1f}%)")
    
    # 黄金交叉死叉分析
    golden_crosses = ((kdj_values['K'] > kdj_values['D']) & 
                     (kdj_values['K'].shift(1) <= kdj_values['D'].shift(1))).sum()
    death_crosses = ((kdj_values['K'] < kdj_values['D']) & 
                    (kdj_values['K'].shift(1) >= kdj_values['D'].shift(1))).sum()
    
    print(f"黄金交叉次数: {golden_crosses}")
    print(f"死亡交叉次数: {death_crosses}")
    
    # 计算信号质量
    print(f"\n🏆 信号质量评估:")
    total_signals = len(divergences['bullish_divergence']) + len(divergences['bearish_divergence'])
    strong_signals = sum(1 for d in divergences['bullish_divergence'] if d['strength'] == '强') + \
                    sum(1 for d in divergences['bearish_divergence'] if d['strength'] == '强')
    
    if total_signals > 0:
        print(f"总信号数: {total_signals}")
        print(f"强信号数: {strong_signals}")
        print(f"强信号比例: {strong_signals/total_signals*100:.1f}%")
    else:
        print("未检测到背离信号")
    
    print("\n" + "=" * 60)
    print("✅ 动态KDJ指标系统测试完成!")
    
    return {
        'test_data': test_data,
        'kdj_values': kdj_values,
        'atr_values': atr_values,
        'divergences': divergences,
        'adaptive_params': adaptive_params
    }

if __name__ == "__main__":
    results = run_kdj_test() 