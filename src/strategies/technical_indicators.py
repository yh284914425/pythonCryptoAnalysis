import pandas as pd
import numpy as np
import sys
import os

# 添加项目根目录到路径，以便导入其他模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.strategies.divergence_analyzer import DivergenceAnalyzer


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
    
    def should_trade(self, signal_strength, threshold=0.7):
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
    """
    def __init__(self, config):
        """
        初始化技术分析器
        :param config: 策略配置对象
        """
        self.config = config
        self.dynamic_kdj = DynamicKDJ(lookback_period=config.technical["atr"]["lookback"])
        self.adx_filter = ADXFilter(period=config.technical["adx"]["period"])
    
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
        
        # 根据市场状态调整信号强度
        adjusted_signal = self.adx_filter.adjust_signal_strength(base_signal, market_regime)
        
        # 判断是否应该交易
        should_trade = self.adx_filter.should_trade(adjusted_signal)
        
        return {
            "symbol": symbol,
            "timestamp": df.index[-1] if isinstance(df.index, pd.DatetimeIndex) else None,
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


if __name__ == "__main__":
    # 测试代码
    from src.strategies.config import create_strategy_config
    import pandas as pd
    from src.strategies.divergence_analyzer import load_bitcoin_data
    
    # 加载数据
    print("加载测试数据...")
    klines_data = load_bitcoin_data()
    if klines_data:
        df = pd.DataFrame(klines_data)
        
        # 创建配置和分析器
        config = create_strategy_config("standard")
        analyzer = TechnicalAnalyzer(config)
        
        # 分析市场
        print("\n执行市场分析...")
        result = analyzer.analyze_market(df, "BTCUSDT")
        
        # 打印结果
        print("\n分析结果:")
        print(f"交易对: {result['symbol']}")
        print(f"收盘价: {result['close_price']}")
        print(f"市场状态: {result['market_regime']}")
        print(f"ADX值: {result['adx']:.2f}")
        print(f"KDJ参数: {result['kdj_params']}")
        print(f"顶部背离: {'是' if result['top_divergence'] else '否'}")
        print(f"底部背离: {'是' if result['bottom_divergence'] else '否'}")
        print(f"信号类型: {result['signal_type']}")
        print(f"信号强度: {result['signal_strength']:.2f}")
        print(f"建议交易: {'是' if result['should_trade'] else '否'}")
    else:
        print("无法加载测试数据") 