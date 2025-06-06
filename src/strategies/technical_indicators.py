import pandas as pd
import numpy as np
import sys
import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec
import datetime

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
        
        # 分析最新的市场状况
        print("\n执行最新市场分析...")
        result = analyzer.analyze_market(df, "BTCUSDT")
        
        # 打印结果
        print("\n最新分析结果:")
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
        
        # 分析历史数据并输出每一天的结果
        print("\n分析历史数据...")
        historical_results = analyzer.analyze_historical_data(df, "BTCUSDT")
        
        # 显示历史分析结果
        pd.set_option('display.max_rows', 20)  # 设置显示行数
        pd.set_option('display.width', 1000)   # 设置显示宽度
        
        # 选择要显示的关键列
        display_columns = ['timestamp', 'close_price', 'market_regime', 'adx', 
                          'signal_type', 'signal_strength', 'should_trade',
                          'top_divergence', 'bottom_divergence']
        
        # 显示历史分析结果
        print("\n历史分析结果:")
        print(historical_results[display_columns].tail(10))
        
        # 保存结果到CSV文件
        try:
            output_file = "btc_technical_analysis_results.csv"
            historical_results.to_csv(output_file, index=False)
            print(f"\n分析结果已保存到文件: {output_file}")
        except Exception as e:
            print(f"\n保存结果到CSV文件时出错: {str(e)}")
        
        # 可视化分析结果
        try:
            print("\n生成可视化分析图表...")
            # 可视化最近60天的数据
            analyzer.visualize_results(df, historical_results, last_n_days=60, 
                                      save_path="btc_technical_analysis_chart.png")
            print("可视化图表已保存至: btc_technical_analysis_chart.png")
        except Exception as e:
            print(f"\n生成可视化图表时出错: {str(e)}")
    else:
        print("无法加载测试数据") 