import pandas as pd
import numpy as np
import sys
import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec
import datetime
from typing import Dict, List, Any, Optional
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
import warnings

# 添加项目根目录到路径，以便导入其他模块
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

try:
    from src.strategies.divergence_analyzer import DivergenceAnalyzer
except ImportError:
    # 如果直接运行当前文件，尝试相对导入
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from divergence_analyzer import DivergenceAnalyzer


class BaseIndicator(ABC):
    """所有指标的基类"""
    
    def __init__(self, name: str, category: str, params: Dict[str, Any] = None):
        self.name = name
        self.category = category
        self.params = params or {}
        self.cache = {}
    
    @abstractmethod
    def calculate(self, df: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """计算指标值"""
        pass
    
    @abstractmethod
    def get_signal(self, values: Dict[str, Any]) -> str:
        """获取信号类型: buy/sell/neutral"""
        pass
    
    @abstractmethod
    def get_strength(self, values: Dict[str, Any]) -> float:
        """获取信号强度: 0-1"""
        pass
    
    def get_confidence(self, values: Dict[str, Any]) -> float:
        """获取信号置信度: 0-1"""
        return 0.5  # 默认实现


class RSIIndicator(BaseIndicator):
    """RSI指标实现"""
    
    def __init__(self, period: int = 14):
        super().__init__("RSI", "momentum", {"period": period})
    
    def calculate(self, df: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """计算RSI"""
        closes = df['收盘价'].astype(float)
        period = self.params['period']
        
        delta = closes.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return {
            'values': rsi.values,
            'current': rsi.iloc[-1] if len(rsi) > 0 else 50,
            'previous': rsi.iloc[-2] if len(rsi) > 1 else 50
        }
    
    def get_signal(self, values: Dict[str, Any]) -> str:
        """获取RSI信号"""
        current = values['current']
        if current > 70:
            return 'sell'
        elif current < 30:
            return 'buy'
        else:
            return 'neutral'
    
    def get_strength(self, values: Dict[str, Any]) -> float:
        """获取RSI信号强度"""
        current = values['current']
        if current > 80:
            return min((current - 80) / 20, 1.0)
        elif current < 20:
            return min((20 - current) / 20, 1.0)
        elif current > 70:
            return (current - 70) / 10 * 0.6
        elif current < 30:
            return (30 - current) / 10 * 0.6
        else:
            return 0.0


class MACDIndicator(BaseIndicator):
    """MACD指标实现"""
    
    def __init__(self, fast: int = 12, slow: int = 26, signal: int = 9):
        super().__init__("MACD", "momentum", {"fast": fast, "slow": slow, "signal": signal})
    
    def calculate(self, df: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """计算MACD"""
        closes = df['收盘价'].astype(float)
        
        ema_fast = closes.ewm(span=self.params['fast']).mean()
        ema_slow = closes.ewm(span=self.params['slow']).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=self.params['signal']).mean()
        histogram = macd_line - signal_line
        
        return {
            'macd': macd_line.values,
            'signal': signal_line.values,
            'histogram': histogram.values,
            'current_macd': macd_line.iloc[-1] if len(macd_line) > 0 else 0,
            'current_signal': signal_line.iloc[-1] if len(signal_line) > 0 else 0,
            'current_histogram': histogram.iloc[-1] if len(histogram) > 0 else 0
        }
    
    def get_signal(self, values: Dict[str, Any]) -> str:
        """获取MACD信号"""
        macd = values['current_macd']
        signal = values['current_signal']
        histogram = values['current_histogram']
        
        if macd > signal and histogram > 0:
            return 'buy'
        elif macd < signal and histogram < 0:
            return 'sell'
        else:
            return 'neutral'
    
    def get_strength(self, values: Dict[str, Any]) -> float:
        """获取MACD信号强度"""
        histogram = abs(values['current_histogram'])
        # 简单的强度计算，实际应用中可以根据历史数据标准化
        return min(histogram * 1000, 1.0)


class EMAIndicator(BaseIndicator):
    """EMA指标实现"""
    
    def __init__(self, periods: List[int] = [20, 50, 200]):
        super().__init__("EMA", "trend", {"periods": periods})
    
    def calculate(self, df: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """计算多周期EMA"""
        closes = df['收盘价'].astype(float)
        result = {}
        
        for period in self.params['periods']:
            ema = closes.ewm(span=period).mean()
            result[f'ema_{period}'] = ema.values
            result[f'current_ema_{period}'] = ema.iloc[-1] if len(ema) > 0 else closes.iloc[-1]
        
        return result
    
    def get_signal(self, values: Dict[str, Any]) -> str:
        """获取EMA信号"""
        # 使用20, 50, 200周期的EMA排列判断趋势
        ema_20 = values.get('current_ema_20', 0)
        ema_50 = values.get('current_ema_50', 0)
        ema_200 = values.get('current_ema_200', 0)
        
        if ema_20 > ema_50 > ema_200:
            return 'buy'
        elif ema_20 < ema_50 < ema_200:
            return 'sell'
        else:
            return 'neutral'
    
    def get_strength(self, values: Dict[str, Any]) -> float:
        """获取EMA信号强度"""
        ema_20 = values.get('current_ema_20', 0)
        ema_50 = values.get('current_ema_50', 0)
        
        if ema_20 == 0 or ema_50 == 0:
            return 0.0
        
        # 计算EMA间的距离作为强度指标
        strength = abs(ema_20 - ema_50) / ema_50
        return min(strength * 10, 1.0)


# 时间框架层级定义
TIMEFRAME_HIERARCHY = {
    'primary': ['1h', '4h', '1d'],      # 主要分析框架
    'secondary': ['30m', '2h'],         # 次要确认框架
    'reference': ['15m', '1w']          # 参考框架
}

# 每个时间框架计算的指标
TIMEFRAME_INDICATORS = {
    '15m': ['RSI', 'Volume'],           # 只计算快速指标
    '30m': ['RSI', 'MACD', 'Volume'], 
    '1h': ['ALL'],                      # 计算所有指标
    '2h': ['KDJ', 'RSI', 'MACD', 'ADX'],
    '4h': ['ALL'],                      # 计算所有指标
    '1d': ['ALL'],                      # 计算所有指标
    '1w': ['Support_Resistance', 'Trend'] # 只计算长期指标
}

# 指标分类结构
INDICATOR_CATEGORIES = {
    'momentum': ['RSI', 'MACD', 'Stochastic', 'ROC', 'KDJ'],
    'trend': ['EMA', 'SMA', 'ADX', 'Ichimoku'],
    'volatility': ['Bollinger', 'ATR', 'Keltner', 'DonchianChannel'],
    'volume': ['OBV', 'VolumeProfile', 'MFI', 'VWAP'],
    'custom': ['DynamicKDJ', 'DivergenceDetector']
}


class IndicatorManager:
    """管理所有指标的计算和缓存"""
    
    def __init__(self):
        self.indicators = {}
        self.cache = {}
        self._register_default_indicators()
    
    def _register_default_indicators(self):
        """注册默认指标"""
        self.register_indicator(RSIIndicator())
        self.register_indicator(MACDIndicator())
        self.register_indicator(EMAIndicator())
    
    def register_indicator(self, indicator: BaseIndicator):
        """注册指标"""
        self.indicators[indicator.name] = indicator
    
    def calculate_indicator(self, indicator_name: str, df: pd.DataFrame, timeframe: str) -> Optional[Dict[str, Any]]:
        """计算单个指标"""
        if indicator_name not in self.indicators:
            warnings.warn(f"指标 {indicator_name} 未注册")
            return None
        
        cache_key = f"{indicator_name}_{timeframe}_{len(df)}"
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        try:
            indicator = self.indicators[indicator_name]
            # 传递额外参数给自定义指标
            kwargs = {'timeframe': timeframe}
            result = indicator.calculate(df, **kwargs)
            result['signal'] = indicator.get_signal(result)
            result['strength'] = indicator.get_strength(result)
            result['confidence'] = indicator.get_confidence(result)
            
            self.cache[cache_key] = result
            return result
        except Exception as e:
            warnings.warn(f"计算指标 {indicator_name} 时出错: {str(e)}")
            return None
    
    def calculate_all(self, df: pd.DataFrame, timeframe: str) -> Dict[str, Any]:
        """计算所有适用的指标"""
        results = {}
        
        # 确定该时间框架需要计算的指标
        required_indicators = TIMEFRAME_INDICATORS.get(timeframe, [])
        if 'ALL' in required_indicators:
            required_indicators = list(self.indicators.keys())
        
        # 并行计算指标
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {}
            for indicator_name in required_indicators:
                if indicator_name in self.indicators:
                    future = executor.submit(self.calculate_indicator, indicator_name, df, timeframe)
                    futures[indicator_name] = future
            
            for indicator_name, future in futures.items():
                try:
                    result = future.result(timeout=30)
                    if result:
                        results[indicator_name] = result
                except Exception as e:
                    warnings.warn(f"计算指标 {indicator_name} 超时或出错: {str(e)}")
        
        return results
    
    def get_indicator_by_category(self, category: str) -> List[str]:
        """根据类别获取指标列表"""
        return INDICATOR_CATEGORIES.get(category, [])
    
    def clear_cache(self):
        """清空缓存"""
        self.cache.clear()


class SignalScorer:
    """统一的信号评分系统"""
    
    def __init__(self):
        # 各指标类别的基础权重
        self.category_weights = {
            'momentum': 0.3,
            'trend': 0.4,
            'volatility': 0.15,
            'volume': 0.1,
            'custom': 0.05
        }
        
        # 时间框架权重
        self.timeframe_weights = {
            '15m': 0.05,
            '30m': 0.1,
            '1h': 0.25,
            '2h': 0.15,
            '4h': 0.35,
            '1d': 0.4,
            '1w': 0.2
        }
    
    def score_single_indicator(self, indicator_result: Dict[str, Any], indicator_name: str) -> float:
        """为单个指标评分"""
        if not indicator_result:
            return 0.0
        
        signal = indicator_result.get('signal', 'neutral')
        strength = indicator_result.get('strength', 0.0)
        confidence = indicator_result.get('confidence', 0.5)
        
        # 基础分数
        if signal == 'buy':
            base_score = strength * confidence
        elif signal == 'sell':
            base_score = -strength * confidence
        else:
            base_score = 0.0
        
        return base_score
    
    def combine_scores(self, indicator_scores: Dict[str, float], indicator_manager: IndicatorManager) -> float:
        """合并指标分数"""
        total_score = 0.0
        total_weight = 0.0
        
        for indicator_name, score in indicator_scores.items():
            # 获取指标类别
            category = None
            for cat, indicators in INDICATOR_CATEGORIES.items():
                if indicator_name in indicators:
                    category = cat
                    break
            
            if category:
                weight = self.category_weights.get(category, 0.1)
                total_score += score * weight
                total_weight += weight
        
        return total_score / total_weight if total_weight > 0 else 0.0
    
    def apply_timeframe_weights(self, timeframe_scores: Dict[str, float]) -> float:
        """应用时间框架权重"""
        total_score = 0.0
        total_weight = 0.0
        
        for timeframe, score in timeframe_scores.items():
            weight = self.timeframe_weights.get(timeframe, 0.1)
            total_score += score * weight
            total_weight += weight
        
        return total_score / total_weight if total_weight > 0 else 0.0
    
    def get_final_decision(self, combined_score: float, threshold: float = 0.3) -> Dict[str, Any]:
        """获取最终决策"""
        if combined_score > threshold:
            return {
                'direction': 'buy',
                'strength': abs(combined_score),
                'confidence': min(abs(combined_score) / threshold, 1.0)
            }
        elif combined_score < -threshold:
            return {
                'direction': 'sell',
                'strength': abs(combined_score),
                'confidence': min(abs(combined_score) / threshold, 1.0)
            }
        else:
            return {
                'direction': 'neutral',
                'strength': 0.0,
                'confidence': 0.5
            }


class MultiTimeframeCoordinator:
    """协调不同时间框架的指标计算"""
    
    def __init__(self, timeframes: List[str] = None):
        self.timeframes = timeframes or TIMEFRAME_HIERARCHY['primary']
        self.indicator_manager = IndicatorManager()
        self.signal_scorer = SignalScorer()
        self.data_cache = {}
    
    def load_data(self, symbol: str, data_dict: Dict[str, pd.DataFrame]) -> bool:
        """加载多时间框架数据"""
        try:
            self.data_cache[symbol] = data_dict
            return True
        except Exception as e:
            warnings.warn(f"加载数据时出错: {str(e)}")
            return False
    
    def calculate_indicators(self, symbol: str) -> Dict[str, Dict[str, Any]]:
        """计算所有时间框架的指标"""
        if symbol not in self.data_cache:
            raise ValueError(f"未找到 {symbol} 的数据")
        
        results = {}
        
        for timeframe in self.timeframes:
            if timeframe in self.data_cache[symbol]:
                df = self.data_cache[symbol][timeframe]
                indicators = self.indicator_manager.calculate_all(df, timeframe)
                results[timeframe] = indicators
        
        return results
    
    def align_signals(self, symbol: str) -> Dict[str, Any]:
        """对齐不同时间框架的信号"""
        indicator_results = self.calculate_indicators(symbol)
        
        # 计算每个时间框架的综合分数
        timeframe_scores = {}
        
        for timeframe, indicators in indicator_results.items():
            indicator_scores = {}
            for indicator_name, result in indicators.items():
                score = self.signal_scorer.score_single_indicator(result, indicator_name)
                indicator_scores[indicator_name] = score
            
            # 合并该时间框架的所有指标分数
            combined_score = self.signal_scorer.combine_scores(indicator_scores, self.indicator_manager)
            timeframe_scores[timeframe] = combined_score
        
        # 应用时间框架权重，得到最终分数
        final_score = self.signal_scorer.apply_timeframe_weights(timeframe_scores)
        final_decision = self.signal_scorer.get_final_decision(final_score)
        
        return {
            'symbol': symbol,
            'timeframe_scores': timeframe_scores,
            'final_score': final_score,
            'decision': final_decision,
            'indicator_details': indicator_results
        }


class DynamicKDJ:
    """
    动态KDJ参数系统，根据市场波动性自动调整KDJ参数
    """
    def __init__(self, lookback_period=252):
        """
        初始化动态KDJ系统
        :param lookback_period: 历史回溯周期，默认252个交易日(约一年)
        """
        self.lookback = lookback_period
        self.atr_percentiles = {}  # 存储各币种的ATR分位数
        self.current_params = {}   # 当前使用的参数
        self.analyzer = DivergenceAnalyzer()  # 使用现有的背离分析器
    
    def calculate_atr(self, df, period=14):
        """
        计算ATR指标
        :param df: DataFrame，包含high, low, close列
        :param period: ATR周期
        :return: ATR值列表
        """
        high = df['最高价'].astype(float).values
        low = df['最低价'].astype(float).values
        close = df['收盘价'].astype(float).values
        
        # 计算True Range
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        
        # 第一个值不可用，设为0
        tr2[0] = 0
        tr3[0] = 0
        
        # 计算最大值
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        
        # 计算ATR
        atr = np.zeros_like(tr)
        atr[0] = tr[0]
        for i in range(1, len(tr)):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
        
        return atr
    
    def update_atr_percentiles(self, symbol, df):
        """
        更新ATR分位数
        :param symbol: 交易对符号
        :param df: DataFrame，包含价格数据
        """
        # 确保数据量足够
        if len(df) < self.lookback:
            print(f"警告: {symbol}数据量不足，需要至少{self.lookback}条记录")
            lookback = len(df)
        else:
            lookback = self.lookback
        
        # 计算ATR
        atr = self.calculate_atr(df.tail(lookback))
        
        # 计算分位数
        self.atr_percentiles[symbol] = {
            "25%": np.percentile(atr, 25),
            "50%": np.percentile(atr, 50),
            "75%": np.percentile(atr, 75),
            "current": atr[-1]
        }
    
    def determine_market_volatility(self, symbol):
        """
        确定市场波动状态
        :param symbol: 交易对符号
        :return: 波动状态，可能值为 "high", "medium", "low"
        """
        if symbol not in self.atr_percentiles:
            return "medium"  # 默认为中等波动
        
        percentiles = self.atr_percentiles[symbol]
        current_atr = percentiles["current"]
        
        if current_atr > percentiles["75%"]:
            return "high"
        elif current_atr < percentiles["25%"]:
            return "low"
        else:
            return "medium"
    
    def get_optimal_kdj_params(self, symbol, config):
        """
        获取最优KDJ参数
        :param symbol: 交易对符号
        :param config: 策略配置对象
        :return: KDJ参数字典
        """
        volatility = self.determine_market_volatility(symbol)
        params = config.get_kdj_params(volatility)
        
        # 更新当前参数
        self.current_params[symbol] = {
            "volatility": volatility,
            "params": params
        }
        
        return params
    
    def calculate_adaptive_kdj(self, df, symbol, config):
        """
        计算自适应KDJ指标和背离
        :param df: DataFrame，包含价格数据
        :param symbol: 交易对符号
        :param config: 策略配置对象
        :return: 包含KDJ和背离信息的字典
        """
        # 更新ATR分位数
        self.update_atr_percentiles(symbol, df)
        
        # 获取最优参数
        params = self.get_optimal_kdj_params(symbol, config)
        
        # 转换为列表格式，以便使用DivergenceAnalyzer
        klines_data = df.to_dict('records')
        
        # 使用背离分析器计算KDJ和背离，传入动态参数
        result = self.analyzer.calculate_kdj_indicators(klines_data, params)
        
        # 添加当前使用的参数信息
        if result:
            result['current_params'] = self.current_params[symbol]
        
        return result


class ADXFilter:
    """
    ADX市场状态过滤器，用于判断市场趋势状态并调整信号强度
    """
    def __init__(self, period=14):
        """
        初始化ADX过滤器
        :param period: ADX计算周期
        """
        self.period = period
        self.trending_threshold = 25  # 趋势市场阈值
        self.sideways_threshold = 20  # 震荡市场阈值
    
    def calculate_adx(self, df):
        """
        计算ADX指标
        :param df: DataFrame，包含high, low, close列
        :return: ADX值列表
        """
        high = df['最高价'].astype(float).values
        low = df['最低价'].astype(float).values
        close = df['收盘价'].astype(float).values
        
        # 计算+DI和-DI
        up_move = high - np.roll(high, 1)
        down_move = np.roll(low, 1) - low
        
        # 第一个值不可用
        up_move[0] = 0
        down_move[0] = 0
        
        # 计算方向指标
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
        
        # 计算TR
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        tr2[0] = 0
        tr3[0] = 0
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        
        # 计算平滑值
        period = self.period
        tr_smooth = np.zeros_like(tr)
        plus_dm_smooth = np.zeros_like(plus_dm)
        minus_dm_smooth = np.zeros_like(minus_dm)
        
        # 初始值
        tr_smooth[period-1] = np.sum(tr[:period])
        plus_dm_smooth[period-1] = np.sum(plus_dm[:period])
        minus_dm_smooth[period-1] = np.sum(minus_dm[:period])
        
        # 计算平滑值
        for i in range(period, len(tr)):
            tr_smooth[i] = tr_smooth[i-1] - (tr_smooth[i-1] / period) + tr[i]
            plus_dm_smooth[i] = plus_dm_smooth[i-1] - (plus_dm_smooth[i-1] / period) + plus_dm[i]
            minus_dm_smooth[i] = minus_dm_smooth[i-1] - (minus_dm_smooth[i-1] / period) + minus_dm[i]
        
        # 计算DI
        plus_di = 100 * plus_dm_smooth / tr_smooth
        minus_di = 100 * minus_dm_smooth / tr_smooth
        
        # 计算DX
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        dx = np.where(np.isnan(dx), 0, dx)  # 处理除零情况
        
        # 计算ADX
        adx = np.zeros_like(dx)
        adx[2*period-2] = np.mean(dx[period-1:2*period-1])
        
        for i in range(2*period-1, len(dx)):
            adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
        
        return adx
    
    def get_market_regime(self, adx_value):
        """
        判断市场状态
        :param adx_value: ADX值
        :return: 市场状态，可能值为 "trending", "sideways", "transition"
        """
        if adx_value > self.trending_threshold:
            return "trending"
        elif adx_value < self.sideways_threshold:
            return "sideways"
        else:
            return "transition"
    
    def adjust_signal_strength(self, base_signal, market_regime):
        """
        根据市场状态调整信号强度
        :param base_signal: 基础信号强度 (0-1)
        :param market_regime: 市场状态
        :return: 调整后的信号强度
        """
        if market_regime == "trending":
            return min(base_signal * 1.5, 1.0)  # 趋势市场增强信号，但不超过1
        elif market_regime == "sideways":
            return base_signal * 0.5  # 震荡市场减弱信号
        else:
            return base_signal  # 过渡状态保持不变
    
    def should_trade(self, signal_strength, threshold=0.4):
        """
        判断是否应该交易
        :param signal_strength: 信号强度
        :param threshold: 交易阈值
        :return: 布尔值，表示是否应该交易
        """
        return signal_strength >= threshold


class TechnicalAnalyzer:
    """
    技术分析器，整合多个技术指标并生成交易信号
    支持多时间框架分析
    """
    def __init__(self, config, timeframes: List[str] = None):
        """
        初始化技术分析器
        :param config: 策略配置对象
        :param timeframes: 要分析的时间框架列表
        """
        self.config = config
        self.timeframes = timeframes or TIMEFRAME_HIERARCHY['primary']
        
        # 初始化组件
        self.coordinator = MultiTimeframeCoordinator(self.timeframes)
        self.dynamic_kdj = DynamicKDJ(lookback_period=config.technical["atr"]["lookback"])
        self.adx_filter = ADXFilter(period=config.technical["adx"]["period"])
        
        # 注册自定义指标
        self._register_custom_indicators()
    
    def _register_custom_indicators(self):
        """注册自定义指标到协调器"""
        # 注册动态KDJ为自定义指标
        class DynamicKDJIndicator(BaseIndicator):
            def __init__(self, analyzer_instance):
                super().__init__("DynamicKDJ", "custom")
                self.analyzer = analyzer_instance
            
            def calculate(self, df: pd.DataFrame, **kwargs) -> Dict[str, Any]:
                symbol = kwargs.get('symbol', 'UNKNOWN')
                config = kwargs.get('config')
                if config is None:
                    config = self.analyzer.config
                result = self.analyzer.calculate_adaptive_kdj(df, symbol, config)
                if result:
                    return {
                        'k': result.get('k', []),
                        'd': result.get('d', []),
                        'j': result.get('j', []),
                        'current_j': result['j'][-1] if result.get('j') else 50,
                        'top_divergence': result.get('top_divergence', [False])[-1],
                        'bottom_divergence': result.get('bottom_divergence', [False])[-1]
                    }
                return {'current_j': 50, 'top_divergence': False, 'bottom_divergence': False}
            
            def get_signal(self, values: Dict[str, Any]) -> str:
                if values.get('top_divergence'):
                    return 'sell'
                elif values.get('bottom_divergence'):
                    return 'buy'
                else:
                    j_current = values.get('current_j', 50)
                    if j_current > 80:
                        return 'sell'
                    elif j_current < 20:
                        return 'buy'
                    else:
                        return 'neutral'
            
            def get_strength(self, values: Dict[str, Any]) -> float:
                if values.get('top_divergence') or values.get('bottom_divergence'):
                    return 0.9  # 背离信号强度高
                
                j_current = values.get('current_j', 50)
                if j_current > 80:
                    return min((j_current - 80) / 20, 1.0)
                elif j_current < 20:
                    return min((20 - j_current) / 20, 1.0)
                else:
                    return 0.0
        
        # 注册ADX指标
        class ADXIndicator(BaseIndicator):
            def __init__(self, adx_filter_instance):
                super().__init__("ADX", "trend")
                self.adx_filter = adx_filter_instance
            
            def calculate(self, df: pd.DataFrame, **kwargs) -> Dict[str, Any]:
                adx_values = self.adx_filter.calculate_adx(df)
                current_adx = adx_values[-1] if len(adx_values) > 0 else 25
                market_regime = self.adx_filter.get_market_regime(current_adx)
                
                return {
                    'values': adx_values,
                    'current': current_adx,
                    'market_regime': market_regime
                }
            
            def get_signal(self, values: Dict[str, Any]) -> str:
                regime = values.get('market_regime', 'transition')
                if regime == 'trending':
                    return 'neutral'  # ADX本身不提供方向，只确认趋势强度
                else:
                    return 'neutral'
            
            def get_strength(self, values: Dict[str, Any]) -> float:
                current_adx = values.get('current', 25)
                if current_adx > 25:
                    return min((current_adx - 25) / 50, 1.0)  # 趋势强度
                else:
                    return 0.0
        
        # 注册到协调器
        self.coordinator.indicator_manager.register_indicator(DynamicKDJIndicator(self.dynamic_kdj))
        self.coordinator.indicator_manager.register_indicator(ADXIndicator(self.adx_filter))
    
    def analyze_market_multitimeframe(self, data_dict: Dict[str, pd.DataFrame], symbol: str) -> Dict[str, Any]:
        """
        多时间框架市场分析
        :param data_dict: 包含不同时间框架数据的字典 {'1h': df, '4h': df, '1d': df}
        :param symbol: 交易对符号
        :return: 多时间框架分析结果
        """
        # 加载数据到协调器
        success = self.coordinator.load_data(symbol, data_dict)
        if not success:
            raise ValueError(f"加载 {symbol} 数据失败")
        
        # 执行多时间框架分析
        analysis_result = self.coordinator.align_signals(symbol)
        
        # 添加额外的分析信息
        analysis_result['timestamp'] = datetime.datetime.now()
        
        # 获取主要时间框架的价格信息
        main_timeframe = '4h' if '4h' in data_dict else list(data_dict.keys())[0]
        if main_timeframe in data_dict:
            df = data_dict[main_timeframe]
            analysis_result['close_price'] = df['收盘价'].iloc[-1]
            analysis_result['main_timeframe'] = main_timeframe
        
        # 添加风险评估
        analysis_result['risk_assessment'] = self._assess_risk(analysis_result)
        
        return analysis_result
    
    def _assess_risk(self, analysis_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        评估交易风险
        :param analysis_result: 分析结果
        :return: 风险评估结果
        """
        decision = analysis_result.get('decision', {})
        timeframe_scores = analysis_result.get('timeframe_scores', {})
        
        # 计算时间框架一致性
        positive_scores = sum(1 for score in timeframe_scores.values() if score > 0.1)
        negative_scores = sum(1 for score in timeframe_scores.values() if score < -0.1)
        total_scores = len(timeframe_scores)
        
        if total_scores == 0:
            consistency = 0.0
        else:
            consistency = max(positive_scores, negative_scores) / total_scores
        
        # 计算信号强度分布
        score_variance = np.var(list(timeframe_scores.values())) if timeframe_scores else 0
        
        # 风险等级评估
        if consistency > 0.8 and decision.get('confidence', 0) > 0.7:
            risk_level = 'low'
        elif consistency > 0.6 and decision.get('confidence', 0) > 0.5:
            risk_level = 'medium'
        else:
            risk_level = 'high'
        
        return {
            'level': risk_level,
            'consistency': consistency,
            'score_variance': score_variance,
            'recommendation': self._get_risk_recommendation(risk_level, decision)
        }
    
    def _get_risk_recommendation(self, risk_level: str, decision: Dict[str, Any]) -> str:
        """
        获取风险建议
        :param risk_level: 风险等级
        :param decision: 交易决策
        :return: 风险建议
        """
        direction = decision.get('direction', 'neutral')
        
        if risk_level == 'low':
            if direction != 'neutral':
                return f"风险较低，可以考虑{direction}操作，建议正常仓位"
            else:
                return "风险较低，但信号不明确，建议观望"
        elif risk_level == 'medium':
            if direction != 'neutral':
                return f"风险中等，可以考虑{direction}操作，建议减少仓位"
            else:
                return "风险中等，信号不明确，建议观望"
        else:  # high risk
            return "风险较高，建议观望或使用小仓位试探"
    
    def analyze_market(self, df, symbol):
        """
        分析市场并生成交易信号
        :param df: DataFrame，包含价格数据
        :param symbol: 交易对符号
        :return: 分析结果字典
        """
        # 计算自适应KDJ和背离
        kdj_result = self.dynamic_kdj.calculate_adaptive_kdj(df, symbol, self.config)
        
        # 计算ADX
        adx = self.adx_filter.calculate_adx(df)
        current_adx = adx[-1]
        
        # 判断市场状态
        market_regime = self.adx_filter.get_market_regime(current_adx)
        
        # 提取最新的背离信号
        latest_top_divergence = kdj_result['top_divergence'][-1] if kdj_result else False
        latest_bottom_divergence = kdj_result['bottom_divergence'][-1] if kdj_result else False
        
        # 计算基础信号强度 (0-1)
        base_signal = 0
        signal_type = "neutral"
        
        if latest_top_divergence:
            base_signal = 0.8  # 顶部背离，卖出信号
            signal_type = "sell"
        elif latest_bottom_divergence:
            base_signal = 0.8  # 底部背离，买入信号
            signal_type = "buy"
        else:
            # 增加基于价格和KDJ指标的额外信号
            if kdj_result and len(kdj_result['j']) > 1:
                j_values = kdj_result['j']
                j_current = j_values[-1]
                j_prev = j_values[-2]
                
                # 超买区域的卖出信号
                if j_current > 80 and j_prev > j_current:
                    base_signal = 0.6
                    signal_type = "sell"
                # 超卖区域的买入信号
                elif j_current < 20 and j_current > j_prev:
                    base_signal = 0.6
                    signal_type = "buy"
                # J线上穿50的买入信号
                elif j_prev < 50 and j_current > 50:
                    base_signal = 0.5
                    signal_type = "buy"
                # J线下穿50的卖出信号
                elif j_prev > 50 and j_current < 50:
                    base_signal = 0.5
                    signal_type = "sell"
        
        # 根据市场状态调整信号强度
        adjusted_signal = self.adx_filter.adjust_signal_strength(base_signal, market_regime)
        
        # 判断是否应该交易
        should_trade = self.adx_filter.should_trade(adjusted_signal)
        
        # 获取时间戳
        timestamp = None
        if '开盘时间' in df.columns:
            timestamp = df['开盘时间'].iloc[-1]
        
        return {
            "symbol": symbol,
            "timestamp": timestamp,
            "close_price": df['收盘价'].iloc[-1],
            "market_regime": market_regime,
            "adx": current_adx,
            "kdj_params": self.dynamic_kdj.current_params.get(symbol, {}),
            "top_divergence": latest_top_divergence,
            "bottom_divergence": latest_bottom_divergence,
            "signal_type": signal_type,
            "signal_strength": adjusted_signal,
            "should_trade": should_trade
        }
        
    def analyze_historical_data(self, df, symbol, min_lookback=30):
        """
        分析历史数据并输出每一天的分析结果
        :param df: DataFrame，包含价格数据
        :param symbol: 交易对符号
        :param min_lookback: 最小回溯天数，确保有足够数据计算指标
        :return: 包含每日分析结果的DataFrame
        """
        results = []
        
        # 确保有足够的初始数据来计算指标
        for i in range(min_lookback, len(df)):
            # 使用截止到当前日期的数据
            current_df = df.iloc[:i+1]
            
            try:
                # 分析当前日期的市场状况
                result = self.analyze_market(current_df, symbol)
                results.append(result)
            except Exception as e:
                print(f"分析第{i}天数据时出错: {str(e)}")
        
        # 转换为DataFrame便于查看
        results_df = pd.DataFrame(results)
        return results_df

    def visualize_results(self, df, results_df, last_n_days=120, save_path=None):
        """
        可视化分析结果，将K线图与交易信号结合展示
        :param df: 原始K线数据DataFrame
        :param results_df: 分析结果DataFrame
        :param last_n_days: 展示最近的天数
        :param save_path: 保存图片的路径，如果为None则显示图片
        :return: None
        """
        # 确保数据量足够
        if len(results_df) < last_n_days:
            last_n_days = len(results_df)
            print(f"数据量不足，只展示全部 {last_n_days} 天数据")
        
        # 获取最近N天的数据
        recent_df = df.iloc[-last_n_days:].copy()
        recent_results = results_df.iloc[-last_n_days:].copy()
        
        # 将时间列转换为datetime类型
        if '开盘时间' in recent_df.columns:
            recent_df['日期'] = pd.to_datetime(recent_df['开盘时间'])
        
        if 'timestamp' in recent_results.columns:
            recent_results['日期'] = pd.to_datetime(recent_results['timestamp'])
        
        # 创建图表
        plt.style.use('seaborn-v0_8-darkgrid')
        fig = plt.figure(figsize=(18, 12))
        gs = GridSpec(4, 1, height_ratios=[3, 1, 1, 1])
        
        # 绘制K线图
        ax1 = plt.subplot(gs[0])
        ax1.set_title(f'比特币技术分析 - 最近{last_n_days}天', fontsize=16)
        
        # 绘制价格
        ax1.plot(recent_df['日期'], recent_df['收盘价'], label='收盘价', color='#1f77b4', linewidth=2)
        
        # 标记买入信号
        buy_signals = recent_results[(recent_results['signal_type'] == 'buy') & (recent_results['should_trade'] == True)]
        if not buy_signals.empty:
            ax1.scatter(buy_signals['日期'], buy_signals['close_price'], 
                       marker='^', color='green', s=150, label='买入信号')
            
            # 添加买入信号注释
            for i, signal in buy_signals.iterrows():
                ax1.annotate(f"买入\n强度:{signal['signal_strength']:.2f}", 
                           (mdates.date2num(signal['日期']), signal['close_price']),
                           xytext=(0, 30), textcoords='offset points',
                           arrowprops=dict(arrowstyle='->', color='green'),
                           ha='center', fontsize=9)
        
        # 标记卖出信号
        sell_signals = recent_results[(recent_results['signal_type'] == 'sell') & (recent_results['should_trade'] == True)]
        if not sell_signals.empty:
            ax1.scatter(sell_signals['日期'], sell_signals['close_price'], 
                       marker='v', color='red', s=150, label='卖出信号')
            
            # 添加卖出信号注释
            for i, signal in sell_signals.iterrows():
                ax1.annotate(f"卖出\n强度:{signal['signal_strength']:.2f}", 
                           (mdates.date2num(signal['日期']), signal['close_price']),
                           xytext=(0, -30), textcoords='offset points',
                           arrowprops=dict(arrowstyle='->', color='red'),
                           ha='center', fontsize=9)
        
        # 标记背离
        top_divergence = recent_results[recent_results['top_divergence'] == True]
        if not top_divergence.empty:
            ax1.scatter(top_divergence['日期'], top_divergence['close_price'], 
                       marker='X', color='purple', s=120, label='顶部背离')
        
        bottom_divergence = recent_results[recent_results['bottom_divergence'] == True]
        if not bottom_divergence.empty:
            ax1.scatter(bottom_divergence['日期'], bottom_divergence['close_price'], 
                       marker='X', color='blue', s=120, label='底部背离')
        
        # 设置x轴格式
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax1.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
        
        # 添加网格和图例
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='upper left')
        
        # 绘制ADX指标
        ax2 = plt.subplot(gs[1], sharex=ax1)
        ax2.set_title('ADX指标与市场状态', fontsize=12)
        ax2.plot(recent_results['日期'], recent_results['adx'], label='ADX', color='purple', linewidth=1.5)
        
        # 添加市场状态背景色
        for i, row in recent_results.iterrows():
            if row['market_regime'] == 'trending':
                ax2.axvspan(row['日期'], row['日期'] + pd.Timedelta(days=1), 
                           alpha=0.2, color='green', label='_trending')
            elif row['market_regime'] == 'sideways':
                ax2.axvspan(row['日期'], row['日期'] + pd.Timedelta(days=1), 
                           alpha=0.2, color='red', label='_sideways')
            else:  # transition
                ax2.axvspan(row['日期'], row['日期'] + pd.Timedelta(days=1), 
                           alpha=0.2, color='yellow', label='_transition')
        
        # 添加趋势阈值线
        ax2.axhline(y=25, color='green', linestyle='--', alpha=0.7, label='趋势阈值(25)')
        ax2.axhline(y=20, color='red', linestyle='--', alpha=0.7, label='震荡阈值(20)')
        
        # 设置y轴范围
        ax2.set_ylim(0, max(recent_results['adx']) * 1.1)
        ax2.legend(loc='upper left')
        
        # 绘制信号强度
        ax3 = plt.subplot(gs[2], sharex=ax1)
        ax3.set_title('信号强度和交易决策', fontsize=12)
        
        # 绘制信号强度柱状图
        bars = ax3.bar(recent_results['日期'], recent_results['signal_strength'], 
                      color=recent_results['signal_type'].map({'buy': 'green', 'sell': 'red', 'neutral': 'gray'}),
                      alpha=0.7, width=0.8)
        
        # 添加交易阈值线
        ax3.axhline(y=0.4, color='black', linestyle='--', alpha=0.7, label='交易阈值(0.4)')
        
        # 设置y轴范围
        ax3.set_ylim(0, 1.1)
        ax3.legend(loc='upper left')
        
        # 绘制KDJ参数变化
        ax4 = plt.subplot(gs[3], sharex=ax1)
        ax4.set_title('KDJ参数动态调整', fontsize=12)
        
        # 提取KDJ参数
        k_values = []
        d_values = []
        j_values = []
        volatility = []
        
        for i, row in recent_results.iterrows():
            if isinstance(row['kdj_params'], dict) and 'params' in row['kdj_params']:
                k_values.append(row['kdj_params']['params'].get('k', 0))
                d_values.append(row['kdj_params']['params'].get('d', 0))
                j_values.append(row['kdj_params']['params'].get('j', 0))
                volatility.append(row['kdj_params'].get('volatility', 'unknown'))
            else:
                k_values.append(0)
                d_values.append(0)
                j_values.append(0)
                volatility.append('unknown')
        
        recent_results['k_param'] = k_values
        recent_results['d_param'] = d_values
        recent_results['j_param'] = j_values
        recent_results['volatility'] = volatility
        
        # 绘制KDJ参数
        ax4.plot(recent_results['日期'], recent_results['k_param'], label='K周期', color='blue')
        ax4.plot(recent_results['日期'], recent_results['d_param'], label='D周期', color='orange')
        ax4.plot(recent_results['日期'], recent_results['j_param'], label='J周期', color='green')
        
        # 添加波动性背景色
        for i, row in recent_results.iterrows():
            if row['volatility'] == 'high':
                ax4.axvspan(row['日期'], row['日期'] + pd.Timedelta(days=1), 
                           alpha=0.2, color='red', label='_high')
            elif row['volatility'] == 'medium':
                ax4.axvspan(row['日期'], row['日期'] + pd.Timedelta(days=1), 
                           alpha=0.2, color='yellow', label='_medium')
            elif row['volatility'] == 'low':
                ax4.axvspan(row['日期'], row['日期'] + pd.Timedelta(days=1), 
                           alpha=0.2, color='green', label='_low')
        
        ax4.legend(loc='upper left')
        
        # 添加图例说明
        fig.text(0.02, 0.02, "市场状态: 绿色=趋势 黄色=过渡 红色=震荡\n"
                           "波动性: 红色=高 黄色=中 绿色=低\n"
                           "信号: 绿色=买入 红色=卖出 灰色=中性", fontsize=10)
        
        # 调整布局
        plt.tight_layout()
        
        # 保存或显示图片
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"图表已保存至: {save_path}")
        else:
            plt.show()
        
        plt.close()
        
        # 返回带有参数的结果DataFrame，方便进一步分析
        return recent_results


if __name__ == "__main__":
    # 测试代码
    try:
        from src.strategies.config import create_strategy_config
        from src.strategies.divergence_analyzer import load_bitcoin_data
    except ImportError:
        from config import create_strategy_config
        from divergence_analyzer import load_bitcoin_data
    import pandas as pd
    
    # 加载数据
    print("加载测试数据...")
    klines_data = load_bitcoin_data()
    if klines_data:
        df = pd.DataFrame(klines_data)
        
        # 创建配置和分析器
        config = create_strategy_config("standard")
        analyzer = TechnicalAnalyzer(config)
        
        print("=" * 80)
        print("📊 多时间框架技术分析系统测试")
        print("=" * 80)
        
        # 模拟多时间框架数据（实际应用中需要从交易所获取不同时间框架的数据）
        print("\n1. 准备多时间框架数据...")
        
        # 模拟不同时间框架的数据 (这里用同一份数据模拟，实际应用中应该是不同时间框架的真实数据)
        data_dict = {
            '1h': df.copy(),    # 1小时数据
            '4h': df.copy(),    # 4小时数据  
            '1d': df.copy()     # 日线数据
        }
        
        # 为了模拟效果，对不同时间框架的数据进行一些处理
        data_dict['1h'] = data_dict['1h'].tail(200)  # 1小时用最近200个数据点
        data_dict['4h'] = data_dict['4h'].tail(100)  # 4小时用最近100个数据点
        data_dict['1d'] = data_dict['1d'].tail(50)   # 日线用最近50个数据点
        
        print(f"✓ 1小时数据: {len(data_dict['1h'])} 条记录")
        print(f"✓ 4小时数据: {len(data_dict['4h'])} 条记录") 
        print(f"✓ 日线数据: {len(data_dict['1d'])} 条记录")
        
        # 执行多时间框架分析
        print("\n2. 执行多时间框架分析...")
        try:
            multitf_result = analyzer.analyze_market_multitimeframe(data_dict, "BTCUSDT")
            
            print("\n" + "=" * 60)
            print("📈 多时间框架分析结果")
            print("=" * 60)
            
            # 显示基本信息
            print(f"交易对: {multitf_result['symbol']}")
            print(f"分析时间: {multitf_result['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"当前价格: {multitf_result.get('close_price', 'N/A')}")
            print(f"主要时间框架: {multitf_result.get('main_timeframe', 'N/A')}")
            
            # 显示各时间框架得分
            print(f"\n📊 各时间框架综合得分:")
            timeframe_scores = multitf_result.get('timeframe_scores', {})
            for tf, score in timeframe_scores.items():
                direction = "📈 看涨" if score > 0.1 else "📉 看跌" if score < -0.1 else "🔄 中性"
                print(f"  {tf:>4}: {score:>8.3f} ({direction})")
            
            # 显示最终决策
            print(f"\n🎯 最终决策:")
            decision = multitf_result.get('decision', {})
            print(f"  方向: {decision.get('direction', 'neutral').upper()}")
            print(f"  强度: {decision.get('strength', 0):.3f}")
            print(f"  置信度: {decision.get('confidence', 0):.3f}")
            print(f"  综合得分: {multitf_result.get('final_score', 0):.3f}")
            
            # 显示风险评估
            print(f"\n⚠️ 风险评估:")
            risk = multitf_result.get('risk_assessment', {})
            print(f"  风险等级: {risk.get('level', 'unknown').upper()}")
            print(f"  一致性: {risk.get('consistency', 0):.3f}")
            print(f"  得分方差: {risk.get('score_variance', 0):.3f}")
            print(f"  建议: {risk.get('recommendation', '无建议')}")
            
            # 显示各指标详细结果
            print(f"\n📋 各时间框架指标详情:")
            indicator_details = multitf_result.get('indicator_details', {})
            for tf, indicators in indicator_details.items():
                print(f"\n  {tf} 时间框架:")
                for indicator_name, result in indicators.items():
                    signal = result.get('signal', 'neutral')
                    strength = result.get('strength', 0)
                    confidence = result.get('confidence', 0)
                    print(f"    {indicator_name:>12}: {signal:>8} (强度:{strength:.2f}, 置信:{confidence:.2f})")
            
        except Exception as e:
            print(f"❌ 多时间框架分析出错: {str(e)}")
            import traceback
            traceback.print_exc()
        
        print("\n" + "=" * 80)
        print("🔄 单时间框架传统分析对比")
        print("=" * 80)
        
        # 执行传统单时间框架分析作为对比
        print("\n3. 执行传统单时间框架分析...")
        result = analyzer.analyze_market(df, "BTCUSDT")
        
        # 打印传统分析结果
        print("\n📊 传统分析结果:")
        print(f"交易对: {result['symbol']}")
        print(f"收盘价: {result['close_price']}")
        print(f"市场状态: {result['market_regime']}")
        print(f"ADX值: {result['adx']:.2f}")
        print(f"顶部背离: {'是' if result['top_divergence'] else '否'}")
        print(f"底部背离: {'是' if result['bottom_divergence'] else '否'}")
        print(f"信号类型: {result['signal_type']}")
        print(f"信号强度: {result['signal_strength']:.2f}")
        print(f"建议交易: {'是' if result['should_trade'] else '否'}")
        
        # 保存结果
        print("\n4. 保存分析结果...")
        try:
            # 保存多时间框架分析结果
            import json
            with open("btc_multitimeframe_analysis.json", "w", encoding='utf-8') as f:
                # 处理datetime对象
                result_copy = multitf_result.copy()
                result_copy['timestamp'] = result_copy['timestamp'].isoformat()
                json.dump(result_copy, f, ensure_ascii=False, indent=2)
            print("✓ 多时间框架分析结果已保存至: btc_multitimeframe_analysis.json")
            
            # 保存传统分析结果到CSV
            historical_results = analyzer.analyze_historical_data(df, "BTCUSDT")
            output_file = "btc_traditional_analysis_results.csv"
            historical_results.to_csv(output_file, index=False)
            print(f"✓ 传统分析结果已保存至: {output_file}")
            
        except Exception as e:
            print(f"❌ 保存结果时出错: {str(e)}")
        
        print("\n" + "=" * 80)
        print("✅ 测试完成!")
        print("=" * 80)
        print("💡 新系统特点:")
        print("  • 支持多时间框架协同分析")
        print("  • 智能权重分配和信号融合")
        print("  • 全面的风险评估体系")
        print("  • 并行计算提升性能")
        print("  • 可扩展的指标体系架构")
        
    else:
        print("❌ 无法加载测试数据") 