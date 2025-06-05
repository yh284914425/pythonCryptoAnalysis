"""
📊 技术指标模块
Technical Indicators Module

包含KDJ、MACD、ADX、Volume Profile等核心技术指标的计算和分析
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, List, Optional
import talib
from .config import KDJConfig, TechnicalIndicatorsConfig

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
        rsv = (close - lowest_low) / (highest_high - lowest_low) * 100
        
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
        recent_price_min = price_series.tail(5).min()
        historical_price_min = price_series.head(-5).min()
        
        recent_j_min = j_series.tail(5).min()
        historical_j_min = j_series.head(-5).min()
        
        return (recent_price_min < historical_price_min and 
                recent_j_min > historical_j_min)
    
    def _check_bearish_divergence(self, price_series: pd.Series, j_series: pd.Series) -> bool:
        """检查顶背离条件"""
        # 价格创新高但J值未创新高
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
        if price_range > price_series.mean() * 0.05 and j_range > 20:
            return "强"
        elif price_range > price_series.mean() * 0.03 and j_range > 15:
            return "中"
        else:
            return "弱"

class TechnicalIndicators:
    """技术指标集合类"""
    
    def __init__(self, config: TechnicalIndicatorsConfig):
        self.config = config
        self.kdj_indicator = DynamicKDJIndicator(KDJConfig())
    
    def calculate_macd(self, close: pd.Series) -> Dict[str, pd.Series]:
        """计算MACD指标"""
        exp1 = close.ewm(span=self.config.macd_fast).mean()
        exp2 = close.ewm(span=self.config.macd_slow).mean()
        macd_line = exp1 - exp2
        signal_line = macd_line.ewm(span=self.config.macd_signal).mean()
        histogram = macd_line - signal_line
        
        return {
            'macd': macd_line,
            'signal': signal_line,
            'histogram': histogram
        }
    
    def calculate_rsi(self, close: pd.Series) -> pd.Series:
        """计算RSI指标"""
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.config.rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.config.rsi_period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def calculate_adx(self, high: pd.Series, low: pd.Series, close: pd.Series) -> Dict[str, pd.Series]:
        """计算ADX指标"""
        adx = talib.ADX(high.values, low.values, close.values, timeperiod=self.config.adx_period)
        plus_di = talib.PLUS_DI(high.values, low.values, close.values, timeperiod=self.config.adx_period)
        minus_di = talib.MINUS_DI(high.values, low.values, close.values, timeperiod=self.config.adx_period)
        
        return {
            'adx': pd.Series(adx, index=close.index),
            'plus_di': pd.Series(plus_di, index=close.index),
            'minus_di': pd.Series(minus_di, index=close.index)
        }
    
    def calculate_volume_profile(self, close: pd.Series, volume: pd.Series) -> Dict[str, float]:
        """计算成交量指标"""
        # 成交量移动平均
        volume_ma = volume.rolling(window=self.config.volume_period).mean()
        
        # 成交量比率
        volume_ratio = volume.iloc[-1] / volume_ma.iloc[-1] if volume_ma.iloc[-1] > 0 else 0
        
        # 价格成交量趋势
        price_volume_trend = ((close.pct_change() * volume).cumsum())
        
        return {
            'volume_ratio': volume_ratio,
            'volume_ma': volume_ma.iloc[-1],
            'pvt': price_volume_trend.iloc[-1],
            'volume_surge': volume_ratio > self.config.volume_threshold
        }
    
    def calculate_bollinger_bands(self, close: pd.Series, period: int = 20, std_dev: float = 2) -> Dict[str, pd.Series]:
        """计算布林带指标"""
        sma = close.rolling(window=period).mean()
        std = close.rolling(window=period).std()
        
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        
        # 布林带位置
        bb_position = (close - lower_band) / (upper_band - lower_band)
        
        return {
            'upper': upper_band,
            'middle': sma,
            'lower': lower_band,
            'position': bb_position
        }
    
    def calculate_atr(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """计算ATR指标"""
        atr = talib.ATR(high.values, low.values, close.values, timeperiod=period)
        return pd.Series(atr, index=close.index)

class MultiTimeframeAnalysis:
    """多时间框架分析"""
    
    def __init__(self, technical_indicators: TechnicalIndicators):
        self.indicators = technical_indicators
    
    def analyze_timeframe_confluence(self, data_dict: Dict[str, pd.DataFrame]) -> Dict[str, Dict]:
        """
        分析多时间框架共振
        
        Args:
            data_dict: 不同时间框架的数据 {'1h': df, '4h': df, '1d': df}
            
        Returns:
            各时间框架的信号强度分析
        """
        confluence_analysis = {}
        
        for timeframe, data in data_dict.items():
            # 计算各项技术指标
            kdj = self.indicators.kdj_indicator.calculate_kdj(
                data['high'], data['low'], data['close']
            )
            macd = self.indicators.calculate_macd(data['close'])
            rsi = self.indicators.calculate_rsi(data['close'])
            adx = self.indicators.calculate_adx(data['high'], data['low'], data['close'])
            
            # 检测背离信号
            divergences = self.indicators.kdj_indicator.detect_kdj_divergence(data, kdj)
            
            # 计算信号强度
            signal_strength = self._calculate_signal_strength(kdj, macd, rsi, adx, divergences)
            
            confluence_analysis[timeframe] = {
                'kdj': kdj,
                'macd': macd,
                'rsi': rsi.iloc[-1],
                'adx': adx['adx'].iloc[-1],
                'divergences': divergences,
                'signal_strength': signal_strength,
                'timestamp': data.index[-1]
            }
        
        return confluence_analysis
    
    def _calculate_signal_strength(self, kdj: Dict, macd: Dict, rsi: float, 
                                  adx: float, divergences: Dict) -> Dict[str, int]:
        """计算综合信号强度"""
        bullish_score = 0
        bearish_score = 0
        
        # KDJ信号
        j_value = kdj['J'].iloc[-1]
        if j_value < 20:
            bullish_score += 2
        elif j_value > 80:
            bearish_score += 2
        
        # MACD信号
        if macd['histogram'].iloc[-1] > macd['histogram'].iloc[-2]:
            bullish_score += 1
        else:
            bearish_score += 1
        
        # RSI信号
        if rsi < 30:
            bullish_score += 1
        elif rsi > 70:
            bearish_score += 1
        
        # ADX趋势强度
        if adx > 25:
            # 强趋势，增加当前方向的权重
            if macd['macd'].iloc[-1] > 0:
                bullish_score += 1
            else:
                bearish_score += 1
        
        # 背离信号
        if divergences['bullish_divergence']:
            bullish_score += 3  # 背离信号权重较高
        if divergences['bearish_divergence']:
            bearish_score += 3
        
        return {
            'bullish_score': bullish_score,
            'bearish_score': bearish_score,
            'net_score': bullish_score - bearish_score
        }

if __name__ == "__main__":
    # 测试技术指标模块
    from .config import TechnicalIndicatorsConfig
    
    # 创建测试数据
    dates = pd.date_range('2024-01-01', periods=100, freq='1H')
    np.random.seed(42)
    prices = 50000 + np.cumsum(np.random.randn(100) * 100)
    
    test_data = pd.DataFrame({
        'open': prices + np.random.randn(100) * 50,
        'high': prices + np.abs(np.random.randn(100)) * 100,
        'low': prices - np.abs(np.random.randn(100)) * 100,
        'close': prices,
        'volume': np.random.randint(1000, 10000, 100)
    }, index=dates)
    
    # 测试技术指标
    config = TechnicalIndicatorsConfig()
    indicators = TechnicalIndicators(config)
    
    # 测试KDJ
    kdj = indicators.kdj_indicator.calculate_kdj(
        test_data['high'], test_data['low'], test_data['close']
    )
    print("🔍 KDJ指标测试:")
    print(f"K值: {kdj['K'].iloc[-1]:.2f}")
    print(f"D值: {kdj['D'].iloc[-1]:.2f}")
    print(f"J值: {kdj['J'].iloc[-1]:.2f}")
    
    # 测试背离检测
    divergences = indicators.kdj_indicator.detect_kdj_divergence(test_data, kdj)
    print(f"\n📊 背离信号:")
    print(f"底背离数量: {len(divergences['bullish_divergence'])}")
    print(f"顶背离数量: {len(divergences['bearish_divergence'])}")
    
    print("\n✅ 技术指标模块测试完成") 