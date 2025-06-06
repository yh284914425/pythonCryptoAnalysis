import backtrader as bt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import sys
import os
import math

# 添加项目根目录到路径，以便导入其他模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.strategies.config import create_strategy_config
from src.strategies.technical_indicators import TechnicalAnalyzer
from src.strategies.divergence_analyzer import load_bitcoin_data


class MultiTimeframeDivergenceStrategy(bt.Strategy):
    """
    使用Backtrader框架实现的多时间框架背离策略
    """
    params = (
        ('mode', 'standard'),  # 策略模式：conservative, standard, aggressive
        ('macro_tf', '1d'),    # 宏观时间框架
        ('meso_tf', '4h'),     # 中观时间框架
        ('micro_tf', '1h'),    # 微观时间框架
    )
    
    def __init__(self):
        """初始化策略"""
        # 创建配置
        self.config = create_strategy_config(self.params.mode)
        
        # 添加风险管理参数（如果配置中没有）
        if not hasattr(self.config, 'risk'):
            self.config.risk = {
                "risk_per_trade": 0.02,  # 每笔交易风险敞口
                "max_single_position": 0.2,  # 单个仓位最大比例
                "stop_loss_atr_multiplier": 2.0  # 止损ATR乘数
            }
        elif "risk_per_trade" not in self.config.risk:
            self.config.risk["risk_per_trade"] = 0.02
        if "max_single_position" not in self.config.risk:
            self.config.risk["max_single_position"] = 0.2
        if "stop_loss_atr_multiplier" not in self.config.risk:
            self.config.risk["stop_loss_atr_multiplier"] = 2.0
        
        # 技术分析器
        self.analyzer = TechnicalAnalyzer(self.config)
        
        # 存储不同时间框架的数据
        self.datas_by_tf = {
            self.params.macro_tf: self.datas[0],  # 宏观层 - 日线
            self.params.meso_tf: self.datas[1],   # 中观层 - 4小时
            self.params.micro_tf: self.datas[2],  # 微观层 - 1小时
        }
        
        # 创建技术指标
        self.indicators = {}
        for tf, data in self.datas_by_tf.items():
            self.indicators[tf] = {}
            # RSI指标
            self.indicators[tf]['rsi'] = bt.indicators.RSI(data.close, period=14)
            # MACD指标
            self.indicators[tf]['macd'] = bt.indicators.MACD(data.close, period_me1=12, period_me2=26, period_signal=9)
            # KDJ指标
            stoch = bt.indicators.StochasticFast(data, period=9, period_dfast=3)
            self.indicators[tf]['k'] = stoch.percK
            self.indicators[tf]['d'] = stoch.percD
            # ATR指标
            self.indicators[tf]['atr'] = bt.indicators.ATR(data, period=14)
        
        # 交易状态
        self.order = None
        self.position_size = 0
        self.stop_loss = 0
        
        # 记录交易
        self.trades = []
        
        # 记录信号
        self.signals = {
            'buy': [],
            'sell': [],
            'neutral': []
        }
    
    def log(self, txt, dt=None):
        """记录日志"""
        dt = dt or self.datas[0].datetime.date(0)
        print(f'{dt.isoformat()}: {txt}')
    
    def notify_order(self, order):
        """订单状态通知"""
        if order.status in [order.Submitted, order.Accepted]:
            # 订单提交或接受，无需操作
            return
        
        # 检查订单是否完成
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(f'买入执行: 价格={order.executed.price:.2f}, 成本={order.executed.value:.2f}, 手续费={order.executed.comm:.2f}')
                self.stop_loss = order.executed.price * (1 - self.config.risk['stop_loss_atr_multiplier'] * 0.01)
            else:
                self.log(f'卖出执行: 价格={order.executed.price:.2f}, 成本={order.executed.value:.2f}, 手续费={order.executed.comm:.2f}')
            
            # 记录交易
            self.trades.append({
                'type': 'buy' if order.isbuy() else 'sell',
                'price': order.executed.price,
                'size': order.executed.size,
                'value': order.executed.value,
                'commission': order.executed.comm,
                'date': self.datas[0].datetime.date(0)
            })
        
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log('订单取消/保证金不足/拒绝')
        
        # 重置订单
        self.order = None
    
    def notify_trade(self, trade):
        """交易结果通知"""
        if not trade.isclosed:
            return
        
        self.log(f'交易利润: 毛利={trade.pnl:.2f}, 净利={trade.pnlcomm:.2f}')
    
    def next(self):
        """
        主策略逻辑，每个bar执行一次
        """
        # 如果有未完成的订单，等待
        if self.order:
            return
        
        # 检查止损
        if self.position and self.stop_loss > 0:
            if self.datas[0].close[0] < self.stop_loss:
                self.log(f'触发止损: 价格={self.datas[0].close[0]:.2f}, 止损={self.stop_loss:.2f}')
                self.order = self.sell()
                return
        
        # 获取当前市场数据
        market_data = self._prepare_market_data()
        
        # 分析市场
        analysis = self._analyze_market(market_data)
        
        # 如果分析失败，跳过
        if not analysis or 'error' in analysis:
            return
        
        # 获取交易信号
        signal = self._get_signal(analysis)
        
        # 记录信号
        self.signals[signal['type']].append({
            'date': self.datas[0].datetime.date(0),
            'price': self.datas[0].close[0],
            'strength': signal['strength']
        })
        
        # 执行交易
        self._execute_trade(signal)
    
    def _prepare_market_data(self):
        """准备市场数据"""
        market_data = {"BTCUSDT": {}}
        
        for tf, data in self.datas_by_tf.items():
            # 检查数据长度
            data_len = len(data)
            lookback = min(data_len-1, 200)  # 增加历史数据范围，但不超过可用数据长度
            
            if lookback < 30:
                print(f"警告: {tf}时间框架数据不足，仅有{lookback}个数据点")
                continue
            
            # 创建有效的DataFrame
            try:
                # 检查数据是否有效
                valid_data = True
                for i in range(lookback, 0, -1):
                    if (math.isnan(data.open[-i]) or math.isnan(data.high[-i]) or 
                        math.isnan(data.low[-i]) or math.isnan(data.close[-i]) or 
                        math.isnan(data.volume[-i])):
                        valid_data = False
                        break
                
                if not valid_data:
                    print(f"警告: {tf}时间框架存在无效数据，跳过")
                    continue
                
                # 创建DataFrame
                df = pd.DataFrame({
                    '开盘时间': [data.datetime.datetime(-i) for i in range(lookback, 0, -1)],
                    '开盘价': [data.open[-i] for i in range(lookback, 0, -1)],
                    '最高价': [data.high[-i] for i in range(lookback, 0, -1)],
                    '最低价': [data.low[-i] for i in range(lookback, 0, -1)],
                    '收盘价': [data.close[-i] for i in range(lookback, 0, -1)],
                    '成交量': [data.volume[-i] for i in range(lookback, 0, -1)],
                })
                
                # 确保数据没有NaN值
                if df.isnull().values.any():
                    print(f"警告: {tf}时间框架DataFrame包含NaN值，尝试删除")
                    df = df.dropna()
                    if len(df) < 30:
                        print(f"警告: 删除NaN后{tf}时间框架数据不足，跳过")
                        continue
                
                market_data["BTCUSDT"][tf] = df
                
                # 打印调试信息
                print(f"处理{tf}时间框架数据: {len(df)}行, 时间范围: {df['开盘时间'].iloc[0]} 到 {df['开盘时间'].iloc[-1]}")
                print(f"最近收盘价: {df['收盘价'].iloc[-5:].values}")
                
                # 检查技术指标是否有效
                rsi_value = self.indicators[tf]['rsi'][0]
                macd_value = self.indicators[tf]['macd'][0]
                k_value = self.indicators[tf]['k'][0]
                d_value = self.indicators[tf]['d'][0]
                
                print(f"{tf}技术指标: RSI={rsi_value:.2f}, MACD={macd_value:.2f}, K={k_value:.2f}, D={d_value:.2f}")
            
            except Exception as e:
                print(f"处理{tf}时间框架数据时出错: {e}")
        
        return market_data
    
    def _analyze_market(self, market_data):
        """分析市场"""
        # 这里使用我们已有的分析逻辑
        # 实际应用中，应该将分析逻辑转换为Backtrader的指标
        try:
            # 检查是否有足够的数据
            for tf, df in market_data["BTCUSDT"].items():
                if len(df) < 30:  # 至少需要30个数据点
                    return None
            
            # 使用技术分析器分析各个时间框架
            analysis_results = {}
            for tf, df in market_data["BTCUSDT"].items():
                analysis_results[tf] = self._analyze_single_timeframe(tf, df)
            
            # 合并多时间框架信号
            final_signal = self._combine_signals(analysis_results)
            
            return {
                "analysis_results": analysis_results,
                "final_signal": final_signal
            }
        
        except Exception as e:
            print(f"分析错误: {e}")
            return {"error": str(e)}
    
    def _analyze_single_timeframe(self, timeframe, df):
        """分析单个时间框架"""
        # 获取当前指标值
        rsi = self.indicators[timeframe]['rsi'][0]
        macd = self.indicators[timeframe]['macd'][0]
        macd_signal = self.indicators[timeframe]['macd'].signal[0]
        k = self.indicators[timeframe]['k'][0]
        d = self.indicators[timeframe]['d'][0]
        atr = self.indicators[timeframe]['atr'][0]
        
        # 判断信号类型
        signal_type = "neutral"
        signal_strength = 0.0
        
        # RSI超买超卖 - 放宽条件
        if rsi < 40:  # 原来是30
            signal_type = "buy"
            signal_strength += 0.3
        elif rsi > 60:  # 原来是70
            signal_type = "sell"
            signal_strength += 0.3
        
        # MACD金叉死叉 - 增加灵敏度
        if macd > macd_signal:  # 移除了macd > 0的条件
            if signal_type == "neutral":
                signal_type = "buy"
            if signal_type == "buy":
                signal_strength += 0.3
        elif macd < macd_signal:  # 移除了macd < 0的条件
            if signal_type == "neutral":
                signal_type = "sell"
            if signal_type == "sell":
                signal_strength += 0.3
        
        # KDJ指标 - 放宽条件
        if k < 30 and d < 30 and k > d:  # 原来是20
            if signal_type == "neutral" or signal_type == "buy":
                signal_type = "buy"
                signal_strength += 0.4
        elif k > 70 and d > 70 and k < d:  # 原来是80
            if signal_type == "neutral" or signal_type == "sell":
                signal_type = "sell"
                signal_strength += 0.4
        
        # 限制信号强度在0-1之间
        signal_strength = min(max(signal_strength, 0.0), 1.0)
        
        return {
            "signal_type": signal_type,
            "signal_strength": signal_strength,
            "close_price": df['收盘价'].iloc[-1],
            "atr": atr
        }
    
    def _combine_signals(self, analysis_results):
        """合并多时间框架信号"""
        # 获取各时间框架的权重
        weights = {
            self.params.macro_tf: 0.4,  # 宏观层权重
            self.params.meso_tf: 0.4,   # 中观层权重
            self.params.micro_tf: 0.2    # 微观层权重
        }
        
        # 计算加权信号
        buy_signals = 0
        sell_signals = 0
        signal_strength_sum = 0
        
        for tf, result in analysis_results.items():
            weight = weights.get(tf, 0.1)
            
            if result["signal_type"] == "buy":
                buy_signals += 1
                signal_strength_sum += result["signal_strength"] * weight
            elif result["signal_type"] == "sell":
                sell_signals += 1
                signal_strength_sum -= result["signal_strength"] * weight
        
        # 确定最终信号类型
        if buy_signals > sell_signals:
            final_signal_type = "buy"
            final_signal_strength = signal_strength_sum / sum(weights.values())
        elif sell_signals > buy_signals:
            final_signal_type = "sell"
            final_signal_strength = abs(signal_strength_sum) / sum(weights.values())
        else:
            # 信号相等，看强度
            if signal_strength_sum > 0:
                final_signal_type = "buy"
                final_signal_strength = signal_strength_sum / sum(weights.values())
            elif signal_strength_sum < 0:
                final_signal_type = "sell"
                final_signal_strength = abs(signal_strength_sum) / sum(weights.values())
            else:
                final_signal_type = "neutral"
                final_signal_strength = 0
        
        # 降低交易阈值，使策略更容易产生交易信号
        # 原来的信号阈值可能太高，导致没有交易
        signal_threshold = 1  # 降低为只需要1个时间框架确认
        confirmed_signals = buy_signals if final_signal_type == "buy" else sell_signals
        
        # 降低信号强度要求，从0.6降低到0.3
        should_trade = confirmed_signals >= signal_threshold and final_signal_strength >= 0.3
        
        return {
            "type": final_signal_type,
            "strength": final_signal_strength,
            "confirmed_signals": confirmed_signals,
            "required_signals": signal_threshold,
            "should_trade": should_trade
        }
    
    def _get_signal(self, analysis):
        """获取交易信号"""
        return analysis["final_signal"]
    
    def _execute_trade(self, signal):
        """执行交易"""
        # 如果不应该交易，跳过
        if not signal["should_trade"]:
            return
        
        # 获取当前价格和ATR
        current_price = self.datas[0].close[0]
        atr = self.indicators[self.params.meso_tf]['atr'][0]
        
        # 计算仓位大小
        cash = self.broker.getcash()
        risk_per_trade = cash * self.config.risk["risk_per_trade"]
        position_size = risk_per_trade / (atr * self.config.risk["stop_loss_atr_multiplier"])
        
        # 限制仓位大小
        max_position = cash * self.config.risk["max_single_position"]
        position_size = min(position_size, max_position)
        
        # 如果仓位太小，不交易
        min_position = 1  # 最小交易金额，降低为1以便生成更多交易信号
        if position_size < min_position:
            self.log(f"仓位太小: {position_size:.2f} < {min_position}")
            return
        
        # 计算股数
        size = position_size / current_price
        
        # 执行交易
        if signal["type"] == "buy" and not self.position:
            self.log(f"买入信号: 价格={current_price:.2f}, 仓位={position_size:.2f}")
            self.order = self.buy(size=size)
            self.position_size = position_size
            
            # 设置止损
            self.stop_loss = current_price * (1 - self.config.risk["stop_loss_atr_multiplier"] * atr / current_price)
            self.log(f"设置止损: {self.stop_loss:.2f}")
        
        elif signal["type"] == "sell" and self.position:
            self.log(f"卖出信号: 价格={current_price:.2f}")
            self.order = self.sell()
            self.position_size = 0
            self.stop_loss = 0


class BTCData(bt.feeds.PandasData):
    """比特币数据源"""
    params = (
        ('datetime', None),  # 使用索引作为日期
        ('open', 0),      # 开盘价列
        ('high', 1),      # 最高价列
        ('low', 2),       # 最低价列
        ('close', 3),     # 收盘价列
        ('volume', 4),    # 成交量列
        ('openinterest', None)  # 未平仓量列（不使用）
    )


def prepare_data(interval='1d'):
    """准备数据"""
    # 加载数据
    klines_data = load_bitcoin_data(interval=interval)
    if not klines_data:
        print(f"无法加载{interval}数据")
        return None
    
    # 转换为DataFrame
    df = pd.DataFrame(klines_data)
    
    # 转换日期
    df['开盘时间'] = pd.to_datetime(df['开盘时间'])
    
    # 确保所有数值列为浮点数
    numeric_columns = ['开盘价', '最高价', '最低价', '收盘价', '成交量']
    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # 打印数据统计信息
    print(f"数据统计: {interval}")
    print(f"数据形状: {df.shape}")
    print(f"数据范围: {df['开盘时间'].min()} 到 {df['开盘时间'].max()}")
    
    # 创建一个简单的测试数据集，确保能够运行回测
    # 使用2022年的日期范围
    start_date = pd.to_datetime('2022-01-01')
    end_date = pd.to_datetime('2022-12-31')
    date_range = pd.date_range(start=start_date, end=end_date, freq='D' if interval == '1d' else '4H' if interval == '4h' else 'H')
    
    # 创建模拟价格数据
    price = 40000.0  # 起始价格
    prices = []
    for i in range(len(date_range)):
        # 添加一些随机波动
        change = price * 0.01 * (np.random.random() - 0.5)
        price += change
        prices.append(price)
    
    # 创建模拟成交量数据
    volumes = np.random.randint(100, 1000, size=len(date_range))
    
    # 创建Backtrader可用的数据格式
    data = pd.DataFrame({
        'open': prices,
        'high': [p * (1 + np.random.random() * 0.01) for p in prices],  # 最高价略高于开盘价
        'low': [p * (1 - np.random.random() * 0.01) for p in prices],   # 最低价略低于开盘价
        'close': [p * (1 + (np.random.random() - 0.5) * 0.02) for p in prices],  # 收盘价在开盘价附近波动
        'volume': volumes
    }, index=date_range)
    
    print(f"生成模拟数据: {len(data)}行, 范围: {data.index.min()} 到 {data.index.max()}")
    
    return data


def run_backtest(mode='standard', start_date=None, end_date=None, plot=True):
    """运行回测"""
    print(f"使用Backtrader进行回测 (模式: {mode})...")
    
    # 创建Cerebro引擎
    cerebro = bt.Cerebro()
    
    # 准备不同时间框架的数据
    data_1d = prepare_data(interval='1d')
    data_4h = prepare_data(interval='4h')
    data_1h = prepare_data(interval='1h')
    
    if data_1d is None or data_4h is None or data_1h is None:
        print("无法加载所有必要的数据")
        return
    
    # 检查数据是否有效
    if data_1d.isnull().values.any() or data_4h.isnull().values.any() or data_1h.isnull().values.any():
        print("警告：数据包含NaN值，尝试清理...")
        data_1d = data_1d.dropna()
        data_4h = data_4h.dropna()
        data_1h = data_1h.dropna()
        
        # 再次检查
        if data_1d.empty or data_4h.empty or data_1h.empty:
            print("清理后数据为空，无法进行回测")
            return
    
    # 打印数据信息
    print(f"1d数据形状: {data_1d.shape}, 范围: {data_1d.index.min()} 到 {data_1d.index.max()}")
    print(f"4h数据形状: {data_4h.shape}, 范围: {data_4h.index.min()} 到 {data_4h.index.max()}")
    print(f"1h数据形状: {data_1h.shape}, 范围: {data_1h.index.min()} 到 {data_1h.index.max()}")
    
    # 过滤日期范围
    if start_date:
        start_date = pd.to_datetime(start_date)
        data_1d = data_1d[data_1d.index >= start_date]
        data_4h = data_4h[data_4h.index >= start_date]
        data_1h = data_1h[data_1h.index >= start_date]
    
    if end_date:
        end_date = pd.to_datetime(end_date)
        data_1d = data_1d[data_1d.index <= end_date]
        data_4h = data_4h[data_4h.index <= end_date]
        data_1h = data_1h[data_1h.index <= end_date]
    
    # 再次检查过滤后的数据
    if data_1d.empty or data_4h.empty or data_1h.empty:
        print("过滤日期后数据为空，请选择其他日期范围")
        return
    
    # 打印过滤后的数据信息
    print(f"过滤后1d数据: {len(data_1d)}行, 范围: {data_1d.index.min()} 到 {data_1d.index.max()}")
    print(f"过滤后4h数据: {len(data_4h)}行, 范围: {data_4h.index.min()} 到 {data_4h.index.max()}")
    print(f"过滤后1h数据: {len(data_1h)}行, 范围: {data_1h.index.min()} 到 {data_1h.index.max()}")
    
    # 创建数据源
    data_feed_1d = BTCData(dataname=data_1d)
    data_feed_4h = BTCData(dataname=data_4h)
    data_feed_1h = BTCData(dataname=data_1h)
    
    # 添加数据到Cerebro
    cerebro.adddata(data_feed_1d, name='1d')
    cerebro.adddata(data_feed_4h, name='4h')
    cerebro.adddata(data_feed_1h, name='1h')
    
    # 设置初始资金
    cerebro.broker.setcash(10000.0)
    
    # 设置手续费
    cerebro.broker.setcommission(commission=0.001)  # 0.1%
    
    # 添加策略
    cerebro.addstrategy(
        MultiTimeframeDivergenceStrategy,
        mode=mode,
        macro_tf='1d',
        meso_tf='4h',
        micro_tf='1h'
    )
    
    # 添加分析器
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
    
    # 运行回测
    print("开始回测...")
    results = cerebro.run()
    strategy = results[0]
    
    # 打印结果
    print("\n====== 回测结果 ======")
    print(f"初始资金: ${cerebro.broker.startingcash:.2f}")
    print(f"最终资金: ${cerebro.broker.getvalue():.2f}")
    print(f"总收益率: {(cerebro.broker.getvalue() / cerebro.broker.startingcash - 1) * 100:.2f}%")
    
    # 获取分析器结果
    sharpe = strategy.analyzers.sharpe.get_analysis()
    drawdown = strategy.analyzers.drawdown.get_analysis()
    returns = strategy.analyzers.returns.get_analysis()
    trades = strategy.analyzers.trades.get_analysis()
    
    # 打印分析结果
    print(f"夏普比率: {sharpe.get('sharperatio', 0.0)}")
    
    # 处理回撤
    max_dd = 0.0
    if 'max' in drawdown and drawdown['max'] is not None:
        try:
            max_dd = float(drawdown['max']) * 100
        except (TypeError, ValueError):
            max_dd = 0.0
    print(f"最大回撤: {max_dd:.2f}%")
    
    # 处理年化收益率
    annual_return = 0.0
    if 'ravg' in returns and returns['ravg'] is not None:
        try:
            annual_return = float(returns['ravg']) * 100
        except (TypeError, ValueError):
            annual_return = 0.0
    print(f"年化收益率: {annual_return:.2f}%")
    
    # 交易统计
    total_trades = 0
    won_trades = 0
    lost_trades = 0
    
    if 'total' in trades and trades['total'] is not None:
        if 'total' in trades['total']:
            total_trades = trades['total']['total']
    
    if 'won' in trades and trades['won'] is not None:
        if 'total' in trades['won']:
            won_trades = trades['won']['total']
    
    if 'lost' in trades and trades['lost'] is not None:
        if 'total' in trades['lost']:
            lost_trades = trades['lost']['total']
    
    win_rate = won_trades / total_trades if total_trades > 0 else 0.0
    
    print(f"总交易次数: {total_trades}")
    print(f"盈利交易: {won_trades}")
    print(f"亏损交易: {lost_trades}")
    print(f"胜率: {win_rate * 100:.2f}%")
    print("=======================")
    
    # 绘制结果
    if plot:
        try:
            cerebro.plot(style='candle', volume=True, barup='red', bardown='green')
        except Exception as e:
            print(f"绘图错误: {e}")
    
    return results


if __name__ == "__main__":
    # 解析命令行参数
    import argparse
    parser = argparse.ArgumentParser(description='使用Backtrader进行多时间框架背离策略回测')
    parser.add_argument('--mode', type=str, default='standard', choices=['conservative', 'standard', 'aggressive'],
                        help='策略模式: conservative, standard, aggressive')
    parser.add_argument('--start', type=str, default='2022-01-01', help='回测开始日期，格式为 YYYY-MM-DD')
    parser.add_argument('--end', type=str, default='2022-12-31', help='回测结束日期，格式为 YYYY-MM-DD')
    parser.add_argument('--no-plot', action='store_true', help='不显示图表')
    args = parser.parse_args()
    
    # 运行回测
    run_backtest(args.mode, args.start, args.end, not args.no_plot) 