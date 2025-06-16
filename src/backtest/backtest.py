import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import glob
from datetime import datetime
sys.path.append('src/analysis')
from macd_kdj_divergence import (
    load_data, 
    calculate_macd, 
    detect_divergence, 
    calculate_kdj_indicators_for_df
)

class BacktestEngine:
    def __init__(self, initial_capital=10000.0, position_size=0.2, risk_free_rate=0.02):
        """
        初始化回测引擎
        :param initial_capital: 初始资金
        :param position_size: 每次交易仓位比例
        :param risk_free_rate: 无风险利率(年化)，用于计算夏普比率
        """
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.position_size = position_size
        self.risk_free_rate = risk_free_rate
        self.positions = {}  # 持仓情况
        self.trade_history = []  # 交易历史
        self.equity_curve = []  # 资金曲线
        self.daily_returns = []  # 日收益率
        
    def reset(self):
        """重置回测状态"""
        self.capital = self.initial_capital
        self.positions = {}
        self.trade_history = []
        self.equity_curve = []
        self.daily_returns = []
        
    def prepare_data(self, symbol, file_path=None, interval='1d', years=2, use_api=False):
        """
        准备回测数据
        :param symbol: 交易对名称
        :param file_path: CSV文件路径
        :param interval: K线周期
        :param years: 回测年数
        :param use_api: 是否使用API获取数据
        :return: 准备好的DataFrame
        """
        # 加载数据
        if use_api:
            from macd_kdj_divergence import load_data_from_api
            df = load_data_from_api(symbol=symbol, interval=interval, years=years)
        else:
            if not file_path:
                raise ValueError("必须提供file_path参数")
            df = load_data(file_path, years=years)
            
        # 计算MACD指标
        df = calculate_macd(df)
        
        # 计算KDJ指标并检测背离
        df, kdj_top_info, kdj_bottom_info = calculate_kdj_indicators_for_df(df)
        
        # 检测MACD背离
        df, macd_top_info, macd_bottom_info = detect_divergence(df)
        
        # 添加策略信号
        df['macd_top_signal'] = df['top_divergence']
        df['macd_bottom_signal'] = df['bottom_divergence']
        df['kdj_top_signal'] = df['kdj_top_divergence']
        df['kdj_bottom_signal'] = df['kdj_bottom_divergence']
        
        # 添加信号组合标志
        df['buy_signal'] = df['macd_bottom_signal'] | df['kdj_bottom_signal']
        df['sell_signal'] = df['macd_top_signal'] | df['kdj_top_signal']
        
        # 信号确认：同时出现MACD和KDJ底背离则信号更强
        df['strong_buy_signal'] = df['macd_bottom_signal'] & df['kdj_bottom_signal']
        df['strong_sell_signal'] = df['macd_top_signal'] & df['kdj_top_signal']
        
        return df

    def run_backtest(self, df, symbol, strategy='combined', holding_period=5):
        """
        运行回测
        :param df: 数据DataFrame
        :param symbol: 交易对名称
        :param strategy: 策略类型，可选：'macd', 'kdj', 'combined', 'strong'
        :param holding_period: 最大持仓周期
        """
        self.reset()
        self.equity_curve = [self.initial_capital]
        daily_capital = [self.initial_capital]  # 每日资金
        
        # 选择策略信号
        if strategy == 'macd':
            buy_signal = 'macd_bottom_signal'
            sell_signal = 'macd_top_signal'
        elif strategy == 'kdj':
            buy_signal = 'kdj_bottom_signal'
            sell_signal = 'kdj_top_signal'
        elif strategy == 'combined':
            buy_signal = 'buy_signal'
            sell_signal = 'sell_signal'
        elif strategy == 'strong':
            buy_signal = 'strong_buy_signal'
            sell_signal = 'strong_sell_signal'
        else:
            raise ValueError(f"不支持的策略类型: {strategy}")
        
        in_position = False
        entry_price = 0
        entry_date = None
        holding_days = 0
        
        # 修复：使用索引列表记录对应日期
        date_index = [df.index[0]]  # 初始日期
        
        for idx, row in df.iterrows():
            current_capital = self.capital
            
            # 更新持仓时间
            if in_position:
                holding_days += 1
            
            # 卖出信号处理
            if in_position and (row[sell_signal] or holding_days >= holding_period):
                self.capital = self.capital * (1 + self.position_size * (row['close'] / entry_price - 1))
                trade_profit = (row['close'] - entry_price) / entry_price * 100
                self.trade_history.append({
                    'symbol': symbol,
                    'entry_date': entry_date,
                    'entry_price': entry_price,
                    'exit_date': idx,
                    'exit_price': row['close'],
                    'profit_pct': trade_profit,
                    'holding_period': holding_days,
                    'exit_reason': 'signal' if row[sell_signal] else 'timeout',
                    'strategy': strategy
                })
                in_position = False
                entry_price = 0
                holding_days = 0
            
            # 买入信号处理
            elif not in_position and row[buy_signal]:
                entry_price = row['close']
                entry_date = idx
                in_position = True
                
            # 记录资金曲线
            self.equity_curve.append(self.capital)
            daily_capital.append(self.capital)
            date_index.append(idx)  # 记录对应日期
            
            # 计算每日收益率
            daily_return = (self.capital - current_capital) / current_capital if current_capital > 0 else 0
            self.daily_returns.append(daily_return)
        
        # 平仓最后的持仓
        if in_position:
            last_price = df['close'].iloc[-1]
            self.capital = self.capital * (1 + self.position_size * (last_price / entry_price - 1))
            trade_profit = (last_price - entry_price) / entry_price * 100
            self.trade_history.append({
                'symbol': symbol,
                'entry_date': entry_date,
                'entry_price': entry_price,
                'exit_date': df.index[-1],
                'exit_price': last_price,
                'profit_pct': trade_profit,
                'holding_period': holding_days,
                'exit_reason': 'end_of_test',
                'strategy': strategy
            })
            # 更新最终资金
            self.equity_curve[-1] = self.capital
            daily_capital[-1] = self.capital
        
        # 转换交易历史为DataFrame
        self.trade_df = pd.DataFrame(self.trade_history)
        
        # 转换每日资金为DataFrame以便计算最大回撤
        # 修复：确保日期索引和资金数据长度一致
        self.daily_equity = pd.Series(daily_capital, index=date_index)
        
        # 计算统计数据
        self.calculate_statistics()
        
        return {
            'final_capital': self.capital,
            'return_pct': (self.capital / self.initial_capital - 1) * 100,
            'trade_history': self.trade_df,
            'statistics': self.stats
        }
    
    def calculate_statistics(self):
        """计算回测统计数据"""
        self.stats = {}
        
        # 总收益率
        self.stats['total_return_pct'] = (self.capital / self.initial_capital - 1) * 100
        
        # 如果没有交易，返回基本统计信息
        if len(self.trade_history) == 0:
            self.stats.update({
                'total_trades': 0,
                'win_rate': 0,
                'avg_profit': 0,
                'max_profit': 0,
                'max_loss': 0,
                'profit_factor': 0,
                'avg_holding_period': 0,
                'max_drawdown_pct': 0,
                'sharpe_ratio': 0,
                'sortino_ratio': 0
            })
            return
        
        # 总交易次数
        self.stats['total_trades'] = len(self.trade_df)
        
        # 胜率
        win_trades = len(self.trade_df[self.trade_df['profit_pct'] > 0])
        self.stats['win_rate'] = win_trades / len(self.trade_df) * 100 if len(self.trade_df) > 0 else 0
        
        # 平均盈利
        self.stats['avg_profit'] = self.trade_df['profit_pct'].mean()
        
        # 最大盈利和亏损
        self.stats['max_profit'] = self.trade_df['profit_pct'].max()
        self.stats['max_loss'] = self.trade_df['profit_pct'].min()
        
        # 盈亏比
        profits = self.trade_df[self.trade_df['profit_pct'] > 0]['profit_pct'].sum()
        losses = abs(self.trade_df[self.trade_df['profit_pct'] < 0]['profit_pct'].sum())
        self.stats['profit_factor'] = profits / losses if losses != 0 else float('inf')
        
        # 平均持仓周期
        self.stats['avg_holding_period'] = self.trade_df['holding_period'].mean()
        
        # 最大回撤（百分比）
        if len(self.daily_equity) > 1:
            # 计算每个时点的回撤百分比
            rolling_max = self.daily_equity.cummax()
            drawdown = (self.daily_equity - rolling_max) / rolling_max * 100
            self.stats['max_drawdown_pct'] = abs(drawdown.min())
        else:
            self.stats['max_drawdown_pct'] = 0
            
        # 计算夏普比率（年化）
        if self.daily_returns and np.std(self.daily_returns) > 0:
            # 假设252个交易日/年
            trading_days_per_year = 252
            avg_daily_return = np.mean(self.daily_returns)
            daily_std = np.std(self.daily_returns)
            
            # 年化收益率
            annual_return = (1 + avg_daily_return) ** trading_days_per_year - 1
            # 年化风险
            annual_std = daily_std * np.sqrt(trading_days_per_year)
            # 年化无风险利率
            daily_risk_free = (1 + self.risk_free_rate) ** (1/trading_days_per_year) - 1
            
            # 夏普比率
            self.stats['sharpe_ratio'] = (annual_return - self.risk_free_rate) / annual_std if annual_std > 0 else 0
            
            # 索提诺比率（只考虑下行风险）
            downside_returns = [r for r in self.daily_returns if r < daily_risk_free]
            if downside_returns:
                downside_std = np.std(downside_returns)
                annual_downside_std = downside_std * np.sqrt(trading_days_per_year)
                self.stats['sortino_ratio'] = (annual_return - self.risk_free_rate) / annual_downside_std if annual_downside_std > 0 else 0
            else:
                self.stats['sortino_ratio'] = float('inf')
        else:
            self.stats['sharpe_ratio'] = 0
            self.stats['sortino_ratio'] = 0
    
    def plot_equity_curve(self):
        """绘制资金曲线和回撤曲线"""
        if len(self.daily_equity) < 2:
            print("资金曲线数据不足，无法绘图")
            return
            
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), gridspec_kw={'height_ratios': [3, 1]})
        
        # 资金曲线
        ax1.plot(self.daily_equity.index, self.daily_equity.values)
        ax1.set_title('回测资金曲线')
        ax1.set_ylabel('资金')
        ax1.grid(True)
        
        # 回撤曲线
        rolling_max = self.daily_equity.cummax()
        drawdown = (self.daily_equity - rolling_max) / rolling_max * 100
        ax2.fill_between(drawdown.index, drawdown.values, 0, color='red', alpha=0.3)
        ax2.set_title('回撤百分比')
        ax2.set_ylabel('回撤(%)')
        ax2.set_xlabel('日期')
        ax2.grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def plot_trade_distribution(self):
        """绘制交易分布"""
        if len(self.trade_df) == 0:
            print("没有交易记录可以展示")
            return
            
        plt.figure(figsize=(15, 10))
        
        # 盈亏分布
        plt.subplot(221)
        self.trade_df['profit_pct'].hist(bins=20)
        plt.title('盈亏分布')
        plt.xlabel('收益率(%)')
        plt.ylabel('交易次数')
        
        # 持仓周期分布
        plt.subplot(222)
        self.trade_df['holding_period'].hist(bins=20)
        plt.title('持仓周期分布')
        plt.xlabel('持仓天数')
        plt.ylabel('交易次数')
        
        # 退出原因分布
        plt.subplot(223)
        self.trade_df['exit_reason'].value_counts().plot(kind='bar')
        plt.title('退出原因分布')
        plt.xlabel('退出原因')
        plt.ylabel('交易次数')
        
        # 按月份的胜率
        plt.subplot(224)
        if 'entry_date' in self.trade_df.columns and len(self.trade_df) > 0:
            monthly_win_rate = []
            for month in range(1, 13):
                monthly_trades = self.trade_df[pd.DatetimeIndex(self.trade_df['entry_date']).month == month]
                if len(monthly_trades) > 0:
                    win_rate = len(monthly_trades[monthly_trades['profit_pct'] > 0]) / len(monthly_trades) * 100
                    monthly_win_rate.append(win_rate)
                else:
                    monthly_win_rate.append(0)
            plt.bar(range(1, 13), monthly_win_rate)
            plt.title('月度胜率')
            plt.xlabel('月份')
            plt.ylabel('胜率(%)')
            
        plt.tight_layout()
        plt.show()

    def print_statistics(self):
        """打印统计数据"""
        print("\n===== 回测统计 =====")
        print(f"初始资金: {self.initial_capital:.2f}")
        print(f"最终资金: {self.capital:.2f}")
        print(f"总收益率: {self.stats['total_return_pct']:.2f}%")
        print(f"总交易次数: {self.stats['total_trades']}")
        print(f"胜率: {self.stats['win_rate']:.2f}%")
        print(f"平均盈利: {self.stats['avg_profit']:.2f}%")
        print(f"最大盈利: {self.stats['max_profit']:.2f}%")
        print(f"最大亏损: {self.stats['max_loss']:.2f}%")
        print(f"盈亏比: {self.stats['profit_factor']:.2f}")
        print(f"平均持仓周期: {self.stats['avg_holding_period']:.2f} 天")
        print(f"最大回撤率: {self.stats['max_drawdown_pct']:.2f}%")
        print(f"夏普比率: {self.stats['sharpe_ratio']:.2f}")
        print(f"索提诺比率: {self.stats['sortino_ratio']:.2f}")
        print("=====================\n")

def run_multiple_backtests(symbols, timeframes, strategies, holding_periods, data_dir='crypto_data'):
    """
    运行多种参数组合的回测
    :param symbols: 交易对列表
    :param timeframes: 时间周期列表
    :param strategies: 策略列表
    :param holding_periods: 持仓周期列表
    :param data_dir: 数据目录
    :return: 回测结果DataFrame
    """
    results = []
    backtest = BacktestEngine()
    
    for symbol in symbols:
        for timeframe in timeframes:
            file_path = f"{data_dir}/{symbol}/{timeframe}.csv"
            
            # 检查文件是否存在
            if not os.path.exists(file_path):
                print(f"警告: 文件不存在 {file_path}")
                continue
                
            try:
                df = backtest.prepare_data(symbol, file_path=file_path)
                
                for strategy in strategies:
                    for holding_period in holding_periods:
                        result = backtest.run_backtest(
                            df, 
                            symbol=symbol,
                            strategy=strategy,
                            holding_period=holding_period
                        )
                        
                        # 整合回测参数和结果
                        results.append({
                            'symbol': symbol,
                            'timeframe': timeframe,
                            'strategy': strategy,
                            'holding_period': holding_period,
                            'return_pct': result['return_pct'],
                            'win_rate': backtest.stats['win_rate'],
                            'total_trades': backtest.stats['total_trades'],
                            'avg_profit': backtest.stats['avg_profit'],
                            'profit_factor': backtest.stats['profit_factor'],
                            'max_drawdown_pct': backtest.stats['max_drawdown_pct'],
                            'sharpe_ratio': backtest.stats['sharpe_ratio'],
                            'avg_holding_period': backtest.stats['avg_holding_period']
                        })
                        
                        print(f"完成 {symbol} {timeframe} 策略:{strategy} 持仓期:{holding_period} "
                              f"收益率:{result['return_pct']:.2f}% 胜率:{backtest.stats['win_rate']:.2f}% "
                              f"交易数:{backtest.stats['total_trades']} 最大回撤:{backtest.stats['max_drawdown_pct']:.2f}% "
                              f"夏普比率:{backtest.stats['sharpe_ratio']:.2f}")
                
            except Exception as e:
                print(f"处理 {file_path} 时出错: {str(e)}")
    
    # 将结果转换为DataFrame并排序
    results_df = pd.DataFrame(results)
    if not results_df.empty:
        results_df = results_df.sort_values(by='return_pct', ascending=False)
    
    return results_df

def plot_comparative_results(results_df):
    """绘制比较性结果"""
    if results_df.empty:
        print("没有结果可以展示")
        return
    
    plt.figure(figsize=(20, 20))
    
    # 增加子图数量，包含更多指标
    
    # 按策略的回报率对比
    plt.subplot(331)
    results_df.groupby('strategy')['return_pct'].mean().plot(kind='bar')
    plt.title('平均收益率（按策略）')
    plt.ylabel('收益率(%)')
    
    # 按时间周期的回报率对比
    plt.subplot(332)
    results_df.groupby('timeframe')['return_pct'].mean().plot(kind='bar')
    plt.title('平均收益率（按时间周期）')
    plt.ylabel('收益率(%)')
    
    # 按持仓周期的回报率对比
    plt.subplot(333)
    results_df.groupby('holding_period')['return_pct'].mean().plot(kind='bar')
    plt.title('平均收益率（按持仓周期）')
    plt.xlabel('持仓周期')
    plt.ylabel('收益率(%)')
    
    # 按策略的胜率对比
    plt.subplot(334)
    results_df.groupby('strategy')['win_rate'].mean().plot(kind='bar')
    plt.title('平均胜率（按策略）')
    plt.ylabel('胜率(%)')
    
    # 按时间周期的胜率对比
    plt.subplot(335)
    results_df.groupby('timeframe')['win_rate'].mean().plot(kind='bar')
    plt.title('平均胜率（按时间周期）')
    plt.ylabel('胜率(%)')
    
    # 按持仓周期的胜率对比
    plt.subplot(336)
    results_df.groupby('holding_period')['win_rate'].mean().plot(kind='bar')
    plt.title('平均胜率（按持仓周期）')
    plt.ylabel('胜率(%)')
    
    # 按策略的最大回撤对比
    plt.subplot(337)
    results_df.groupby('strategy')['max_drawdown_pct'].mean().plot(kind='bar')
    plt.title('平均最大回撤（按策略）')
    plt.ylabel('回撤(%)')
    
    # 按策略的夏普比率对比
    plt.subplot(338)
    results_df.groupby('strategy')['sharpe_ratio'].mean().plot(kind='bar')
    plt.title('平均夏普比率（按策略）')
    plt.ylabel('夏普比率')
    
    # 按交易对的总交易次数对比
    plt.subplot(339)
    results_df.groupby('symbol')['total_trades'].sum().plot(kind='bar')
    plt.title('总交易次数（按交易对）')
    plt.ylabel('交易次数')
    
    plt.tight_layout()
    plt.show()

def find_best_parameters(symbols=None, min_timeframe='1h', max_combinations=None):
    """
    寻找最优参数组合
    :param symbols: 指定要测试的交易对列表，如果为None则测试默认列表
    :param min_timeframe: 最小时间周期，低于此周期的数据将被忽略
    :param max_combinations: 最大组合数，用于限制测试规模
    :return: 回测结果DataFrame
    """
    # 待测试的交易对
    if symbols is None:
        symbols = ['BTC', 'ETH', 'PEPE']
    elif isinstance(symbols, str):
        symbols = [symbols]  # 如果传入单个字符串，转换为列表
    
    # 检查可用的时间周期文件
    timeframes = []
    for symbol in symbols:
        symbol_dir = f"crypto_data/{symbol}"
        if os.path.exists(symbol_dir):
            # 获取目录中所有CSV文件，并提取时间周期
            for file in glob.glob(f"{symbol_dir}/*.csv"):
                tf = os.path.basename(file).replace('.csv', '')
                if tf not in timeframes:
                    timeframes.append(tf)
    
    # 过滤掉小于min_timeframe的时间周期
    # 时间周期排序映射表
    timeframe_order = {
        '1m': 1, '3m': 3, '5m': 5, '15m': 15, '30m': 30,
        '1h': 60, '2h': 120, '4h': 240, '6h': 360, '8h': 480, '12h': 720,
        '1d': 1440, '3d': 4320, '1w': 10080, '1M': 43200
    }
    
    # 获取min_timeframe的值
    min_minutes = timeframe_order.get(min_timeframe, 60)  # 默认为1小时
    
    # 过滤时间周期
    filtered_timeframes = []
    for tf in timeframes:
        # 如果时间周期在映射表中且大于等于min_minutes，则保留
        tf_minutes = timeframe_order.get(tf)
        if tf_minutes is not None and tf_minutes >= min_minutes:
            filtered_timeframes.append(tf)
        else:
            # 尝试解析其他格式的时间周期
            try:
                if tf.endswith('h') and tf[:-1].isdigit():
                    hours = int(tf[:-1])
                    minutes = hours * 60
                    if minutes >= min_minutes:
                        filtered_timeframes.append(tf)
                elif tf.endswith('d') and tf[:-1].isdigit():
                    days = int(tf[:-1])
                    minutes = days * 1440
                    if minutes >= min_minutes:
                        filtered_timeframes.append(tf)
            except:
                # 无法解析的时间周期格式，忽略
                pass
    
    timeframes = filtered_timeframes
    
    # 待测试的策略
    strategies = ['macd', 'kdj', 'combined', 'strong']
    
    # 待测试的持仓周期
    holding_periods = [1, 3, 5, 10, 15, 20]
    
    # 计算组合总数
    total_combinations = len(symbols) * len(timeframes) * len(strategies) * len(holding_periods)
    
    # 如果超出最大组合数，进行随机抽样或策略减少
    if max_combinations and total_combinations > max_combinations:
        print(f"组合总数 {total_combinations} 超过最大限制 {max_combinations}，将减少测试规模")
        # 这里可以实现抽样策略，例如减少持仓周期或者策略种类
        # 为简单起见，我们减少持仓周期的数量
        holding_periods = [5, 10, 20]  # 减少持仓周期选项
        total_combinations = len(symbols) * len(timeframes) * len(strategies) * len(holding_periods)
    
    print(f"开始测试 {len(symbols)} 个交易对, {len(timeframes)} 个时间周期, "
          f"{len(strategies)} 个策略, {len(holding_periods)} 个持仓周期")
    print(f"总共 {total_combinations} 种组合")
    
    print(f"测试的交易对: {symbols}")
    print(f"测试的时间周期: {timeframes}")
    
    # 运行回测
    results = run_multiple_backtests(symbols, timeframes, strategies, holding_periods)
    
    # 保存结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_filename = f"backtest_results_{'-'.join(symbols)}_{timestamp}.csv"
    results.to_csv(results_filename, index=False)
    print(f"结果已保存到: {results_filename}")
    
    # 打印最佳结果（按不同指标排序）
    print("\n===== 按回报率的总体最佳参数组合 =====")
    print(results.sort_values(by='return_pct', ascending=False).head(10))
    
    print("\n===== 按夏普比率的总体最佳参数组合 =====")
    print(results.sort_values(by='sharpe_ratio', ascending=False).head(10))
    
    print("\n===== 按回撤率的总体最佳参数组合 =====")
    print(results.sort_values(by='max_drawdown_pct', ascending=True).head(10))
    
    # 筛选交易次数足够多的结果
    min_trades = 5
    filtered_results = results[results['total_trades'] >= min_trades]
    if not filtered_results.empty:
        print(f"\n===== 交易次数大于{min_trades}次的最佳参数组合 =====")
        print(filtered_results.sort_values(by='return_pct', ascending=False).head(10))
    
    # 按不同维度分析最佳结果
    print("\n===== 按策略的最佳参数 =====")
    for strategy in strategies:
        strategy_results = results[results['strategy'] == strategy].head(3)
        print(f"\n策略 {strategy} 的最佳表现:")
        print(strategy_results)
    
    print("\n===== 按时间周期的最佳参数 =====")
    for timeframe in timeframes:
        tf_results = results[results['timeframe'] == timeframe].head(3)
        print(f"\n时间周期 {timeframe} 的最佳表现:")
        print(tf_results)
    
    print("\n===== 按交易对的最佳参数 =====")
    for symbol in symbols:
        symbol_results = results[results['symbol'] == symbol].head(3)
        print(f"\n交易对 {symbol} 的最佳表现:")
        print(symbol_results)
    
    # 绘制比较图
    plot_comparative_results(results)
    
    return results

def backtest_single_configuration(symbol, timeframe, strategy='combined', holding_period=5, data_dir='crypto_data', plot=True):
    """
    运行单个配置的回测
    :param symbol: 交易对
    :param timeframe: 时间周期
    :param strategy: 策略类型
    :param holding_period: 持仓周期
    :param data_dir: 数据目录
    :param plot: 是否绘图
    :return: 回测结果
    """
    file_path = f"{data_dir}/{symbol}/{timeframe}.csv"
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"数据文件不存在: {file_path}")
        
    backtest = BacktestEngine()
    df = backtest.prepare_data(symbol, file_path=file_path)
    result = backtest.run_backtest(df, symbol=symbol, strategy=strategy, holding_period=holding_period)
    
    backtest.print_statistics()
    
    if plot:
        backtest.plot_equity_curve()
        backtest.plot_trade_distribution()
        
    return result

def compare_risk_adjusted_metrics():
    """对比风险调整后的指标，生成更详细的分析报告"""
    # 可以在这里实现更高级的风险度量和报告功能
    pass

if __name__ == "__main__":
    # 单一配置回测示例
    # backtest_single_configuration('BTC', '1d', strategy='combined', holding_period=5)
    
    # 寻找最优参数组合 - 只针对PEPE一个币种进行分析
    find_best_parameters(symbols='PEPE', min_timeframe='1h')
