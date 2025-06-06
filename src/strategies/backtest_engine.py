import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import os
import sys

# 添加项目根目录到路径，以便导入其他模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


class BacktestEngine:
    """
    回测引擎，用于测试和验证策略性能
    """
    def __init__(self, start_date=None, end_date=None, initial_capital=10000):
        """
        初始化回测引擎
        :param start_date: 回测开始日期，格式 'YYYY-MM-DD'
        :param end_date: 回测结束日期，格式 'YYYY-MM-DD'
        :param initial_capital: 初始资金
        """
        self.start_date = start_date
        self.end_date = end_date
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.trades = []
        self.positions = {}
        self.daily_balance = []
        self.fee_rate = 0.001  # 交易手续费率 (0.1%)
        self.slippage = 0.001  # 滑点 (0.1%)
    
    def load_data(self, symbol, timeframe="1d", data_dir="crypto_data"):
        """
        加载历史数据
        :param symbol: 交易对符号
        :param timeframe: 时间框架
        :param data_dir: 数据目录
        :return: DataFrame
        """
        filename = f"{symbol}_{timeframe}.csv"
        filepath = os.path.join(data_dir, filename)
        
        if not os.path.exists(filepath):
            print(f"数据文件不存在: {filepath}")
            return None
        
        try:
            df = pd.read_csv(filepath, encoding='utf-8-sig')
            print(f"成功加载数据: {len(df)} 条记录")
            
            # 转换日期列
            df['开盘时间'] = pd.to_datetime(df['开盘时间'])
            
            # 过滤日期范围
            if self.start_date:
                df = df[df['开盘时间'] >= self.start_date]
            if self.end_date:
                df = df[df['开盘时间'] <= self.end_date]
            
            print(f"过滤后数据范围: {df['开盘时间'].iloc[0]} 到 {df['开盘时间'].iloc[-1]}, {len(df)} 条记录")
            return df
        except Exception as e:
            print(f"加载数据失败: {e}")
            return None
    
    def run_backtest(self, strategy, data):
        """
        执行回测
        :param strategy: 策略对象
        :param data: 数据字典，键为交易对符号
        :return: 回测结果
        """
        print(f"开始回测...")
        print(f"初始资金: ${self.initial_capital:,.2f}")
        
        # 重置回测状态
        self.current_capital = self.initial_capital
        self.trades = []
        self.positions = {}
        self.daily_balance = []
        
        # 记录初始余额
        self.daily_balance.append({
            'date': data[list(data.keys())[0]]['开盘时间'].iloc[0],
            'balance': self.initial_capital,
            'equity': self.initial_capital
        })
        
        # 遍历每个交易日
        for i in range(len(data[list(data.keys())[0]])):
            current_date = data[list(data.keys())[0]]['开盘时间'].iloc[i]
            
            # 构建当前时点的数据切片
            current_data = {}
            current_prices = {}
            
            for symbol, df in data.items():
                if i < len(df):
                    current_data[symbol] = df.iloc[:i+1]
                    current_prices[symbol] = float(df['收盘价'].iloc[i])
            
            # 检查止损
            self._check_stop_losses(current_prices)
            
            # 生成交易信号
            signals = self._generate_signals(strategy, current_data)
            
            # 执行交易
            for signal in signals:
                self._execute_trade(signal, current_date)
            
            # 更新每日余额
            equity = self._calculate_equity(current_prices)
            self.daily_balance.append({
                'date': current_date,
                'balance': self.current_capital,
                'equity': equity
            })
        
        # 平掉所有剩余仓位
        final_prices = {symbol: float(df['收盘价'].iloc[-1]) for symbol, df in data.items()}
        self._close_all_positions(final_prices)
        
        # 计算回测结果
        results = self._calculate_results()
        
        return results
    
    def _generate_signals(self, strategy, data):
        """
        生成交易信号
        :param strategy: 策略对象
        :param data: 当前数据
        :return: 信号列表
        """
        signals = []
        
        for symbol, df in data.items():
            # 使用策略分析市场
            analysis = strategy.analyze_market(df, symbol)
            
            # 如果有交易信号
            if analysis["should_trade"]:
                signals.append({
                    "symbol": symbol,
                    "type": analysis["signal_type"],
                    "strength": analysis["signal_strength"],
                    "price": float(df['收盘价'].iloc[-1]),
                    "atr": strategy.technical_analyzer.dynamic_kdj.calculate_atr(df)[-1]
                })
        
        return signals
    
    def _execute_trade(self, signal, date):
        """
        执行交易
        :param signal: 交易信号
        :param date: 交易日期
        """
        symbol = signal["symbol"]
        signal_type = signal["type"]
        price = signal["price"]
        atr = signal["atr"]
        
        # 计算交易价格（考虑滑点）
        if signal_type == "buy":
            trade_price = price * (1 + self.slippage)
        else:  # sell
            trade_price = price * (1 - self.slippage)
        
        # 买入信号
        if signal_type == "buy":
            # 检查是否已有相同方向的仓位
            if symbol in self.positions and self.positions[symbol]["type"] == "long":
                return
            
            # 平掉反向仓位
            if symbol in self.positions and self.positions[symbol]["type"] == "short":
                self._close_position(symbol, trade_price, "signal_reverse")
            
            # 计算仓位大小
            position_size = self._calculate_position_size(signal)
            
            # 如果仓位大小为0，不交易
            if position_size <= 0:
                return
            
            # 计算可买入数量
            quantity = position_size / trade_price
            
            # 计算交易费用
            fee = position_size * self.fee_rate
            
            # 计算止损价格
            stop_loss = trade_price - (atr * 2.0)
            
            # 更新资金
            self.current_capital -= (position_size + fee)
            
            # 记录交易
            trade_id = f"{symbol}_long_{date.strftime('%Y%m%d%H%M%S')}"
            self.trades.append({
                "id": trade_id,
                "date": date,
                "symbol": symbol,
                "type": "buy",
                "price": trade_price,
                "quantity": quantity,
                "position_size": position_size,
                "fee": fee,
                "stop_loss": stop_loss
            })
            
            # 添加持仓
            self.positions[symbol] = {
                "type": "long",
                "entry_price": trade_price,
                "quantity": quantity,
                "position_size": position_size,
                "stop_loss": stop_loss,
                "entry_date": date,
                "trade_id": trade_id
            }
        
        # 卖出信号
        elif signal_type == "sell":
            # 检查是否已有相同方向的仓位
            if symbol in self.positions and self.positions[symbol]["type"] == "short":
                return
            
            # 平掉反向仓位
            if symbol in self.positions and self.positions[symbol]["type"] == "long":
                self._close_position(symbol, trade_price, "signal_reverse")
            
            # 计算仓位大小
            position_size = self._calculate_position_size(signal)
            
            # 如果仓位大小为0，不交易
            if position_size <= 0:
                return
            
            # 计算可卖出数量
            quantity = position_size / trade_price
            
            # 计算交易费用
            fee = position_size * self.fee_rate
            
            # 计算止损价格
            stop_loss = trade_price + (atr * 2.0)
            
            # 更新资金
            self.current_capital -= fee
            
            # 记录交易
            trade_id = f"{symbol}_short_{date.strftime('%Y%m%d%H%M%S')}"
            self.trades.append({
                "id": trade_id,
                "date": date,
                "symbol": symbol,
                "type": "sell",
                "price": trade_price,
                "quantity": quantity,
                "position_size": position_size,
                "fee": fee,
                "stop_loss": stop_loss
            })
            
            # 添加持仓
            self.positions[symbol] = {
                "type": "short",
                "entry_price": trade_price,
                "quantity": quantity,
                "position_size": position_size,
                "stop_loss": stop_loss,
                "entry_date": date,
                "trade_id": trade_id
            }
    
    def _close_position(self, symbol, price, reason="manual"):
        """
        平仓
        :param symbol: 交易对符号
        :param price: 平仓价格
        :param reason: 平仓原因
        """
        if symbol not in self.positions:
            return
        
        position = self.positions[symbol]
        position_type = position["type"]
        entry_price = position["entry_price"]
        quantity = position["quantity"]
        
        # 计算平仓价格（考虑滑点）
        if position_type == "long":
            close_price = price * (1 - self.slippage)
        else:  # short
            close_price = price * (1 + self.slippage)
        
        # 计算盈亏
        if position_type == "long":
            profit_loss = (close_price - entry_price) * quantity
        else:  # short
            profit_loss = (entry_price - close_price) * quantity
        
        # 计算交易费用
        fee = close_price * quantity * self.fee_rate
        
        # 更新资金
        self.current_capital += position["position_size"] + profit_loss - fee
        
        # 记录平仓
        self.trades.append({
            "id": f"close_{position['trade_id']}",
            "date": datetime.now(),
            "symbol": symbol,
            "type": "close_" + position_type,
            "price": close_price,
            "quantity": quantity,
            "profit_loss": profit_loss,
            "fee": fee,
            "reason": reason
        })
        
        # 移除持仓
        del self.positions[symbol]
    
    def _close_all_positions(self, prices):
        """
        平掉所有仓位
        :param prices: 价格字典
        """
        for symbol in list(self.positions.keys()):
            if symbol in prices:
                self._close_position(symbol, prices[symbol], "backtest_end")
    
    def _check_stop_losses(self, prices):
        """
        检查止损
        :param prices: 当前价格字典
        """
        for symbol, position in list(self.positions.items()):
            if symbol not in prices:
                continue
            
            current_price = prices[symbol]
            
            # 检查是否触发止损
            if position["type"] == "long" and current_price <= position["stop_loss"]:
                self._close_position(symbol, current_price, "stop_loss")
            elif position["type"] == "short" and current_price >= position["stop_loss"]:
                self._close_position(symbol, current_price, "stop_loss")
    
    def _calculate_position_size(self, signal):
        """
        计算仓位大小
        :param signal: 交易信号
        :return: 仓位大小
        """
        # 简单版本：使用固定比例
        return self.current_capital * 0.1 * signal["strength"]
    
    def _calculate_equity(self, prices):
        """
        计算当前权益
        :param prices: 当前价格字典
        :return: 权益
        """
        equity = self.current_capital
        
        for symbol, position in self.positions.items():
            if symbol not in prices:
                continue
            
            current_price = prices[symbol]
            
            if position["type"] == "long":
                profit_loss = (current_price - position["entry_price"]) * position["quantity"]
            else:  # short
                profit_loss = (position["entry_price"] - current_price) * position["quantity"]
            
            equity += profit_loss
        
        return equity
    
    def _calculate_results(self):
        """
        计算回测结果
        :return: 结果字典
        """
        # 转换为DataFrame
        daily_balance_df = pd.DataFrame(self.daily_balance)
        trades_df = pd.DataFrame(self.trades)
        
        # 计算收益
        final_equity = daily_balance_df['equity'].iloc[-1]
        total_return = (final_equity - self.initial_capital) / self.initial_capital
        
        # 计算年化收益率
        days = (daily_balance_df['date'].iloc[-1] - daily_balance_df['date'].iloc[0]).days
        annual_return = (1 + total_return) ** (365 / max(1, days)) - 1
        
        # 计算最大回撤
        daily_balance_df['cummax'] = daily_balance_df['equity'].cummax()
        daily_balance_df['drawdown'] = (daily_balance_df['cummax'] - daily_balance_df['equity']) / daily_balance_df['cummax']
        max_drawdown = daily_balance_df['drawdown'].max()
        
        # 计算交易统计
        total_trades = len(trades_df[trades_df['type'].isin(['buy', 'sell'])])
        
        # 计算胜率
        if total_trades > 0:
            winning_trades = len(trades_df[trades_df['profit_loss'] > 0])
            win_rate = winning_trades / total_trades
        else:
            win_rate = 0
        
        # 计算盈亏比
        if len(trades_df[trades_df['profit_loss'] < 0]) > 0:
            avg_win = trades_df[trades_df['profit_loss'] > 0]['profit_loss'].mean() if len(trades_df[trades_df['profit_loss'] > 0]) > 0 else 0
            avg_loss = abs(trades_df[trades_df['profit_loss'] < 0]['profit_loss'].mean()) if len(trades_df[trades_df['profit_loss'] < 0]) > 0 else 1
            profit_loss_ratio = avg_win / avg_loss if avg_loss > 0 else 0
        else:
            profit_loss_ratio = 0
        
        # 计算夏普比率
        if len(daily_balance_df) > 1:
            daily_returns = daily_balance_df['equity'].pct_change().dropna()
            sharpe_ratio = (daily_returns.mean() / daily_returns.std()) * (252 ** 0.5) if daily_returns.std() > 0 else 0
        else:
            sharpe_ratio = 0
        
        return {
            "initial_capital": self.initial_capital,
            "final_equity": final_equity,
            "total_return": total_return,
            "annual_return": annual_return,
            "max_drawdown": max_drawdown,
            "total_trades": total_trades,
            "win_rate": win_rate,
            "profit_loss_ratio": profit_loss_ratio,
            "sharpe_ratio": sharpe_ratio,
            "daily_balance": daily_balance_df,
            "trades": trades_df
        }
    
    def plot_results(self, results):
        """
        绘制回测结果
        :param results: 回测结果
        """
        daily_balance = results["daily_balance"]
        
        plt.figure(figsize=(12, 8))
        
        # 绘制权益曲线
        plt.subplot(2, 1, 1)
        plt.plot(daily_balance['date'], daily_balance['equity'])
        plt.title('回测权益曲线')
        plt.xlabel('日期')
        plt.ylabel('权益')
        plt.grid(True)
        
        # 绘制回撤
        plt.subplot(2, 1, 2)
        plt.fill_between(daily_balance['date'], daily_balance['drawdown'] * 100)
        plt.title('回撤百分比')
        plt.xlabel('日期')
        plt.ylabel('回撤 (%)')
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def print_results(self, results):
        """
        打印回测结果
        :param results: 回测结果
        """
        print("\n====== 回测结果 ======")
        print(f"初始资金: ${results['initial_capital']:,.2f}")
        print(f"最终权益: ${results['final_equity']:,.2f}")
        print(f"总收益率: {results['total_return']*100:.2f}%")
        print(f"年化收益率: {results['annual_return']*100:.2f}%")
        print(f"最大回撤: {results['max_drawdown']*100:.2f}%")
        print(f"总交易次数: {results['total_trades']}")
        print(f"胜率: {results['win_rate']*100:.2f}%")
        print(f"盈亏比: {results['profit_loss_ratio']:.2f}")
        print(f"夏普比率: {results['sharpe_ratio']:.2f}")


if __name__ == "__main__":
    # 测试代码
    from src.strategies.config import create_strategy_config
    from src.analysis.divergence_analysis import load_bitcoin_data
    
    class DummyStrategy:
        def __init__(self):
            pass
        
        def analyze_market(self, df, symbol):
            # 简单的移动平均策略
            if len(df) < 20:
                return {"should_trade": False}
            
            ma5 = df['收盘价'].astype(float).rolling(5).mean()
            ma20 = df['收盘价'].astype(float).rolling(20).mean()
            
            if ma5.iloc[-1] > ma20.iloc[-1] and ma5.iloc[-2] <= ma20.iloc[-2]:
                return {
                    "symbol": symbol,
                    "signal_type": "buy",
                    "strength": 0.8,
                    "should_trade": True
                }
            elif ma5.iloc[-1] < ma20.iloc[-1] and ma5.iloc[-2] >= ma20.iloc[-2]:
                return {
                    "symbol": symbol,
                    "signal_type": "sell",
                    "strength": 0.8,
                    "should_trade": True
                }
            else:
                return {"should_trade": False}
    
    # 加载数据
    print("加载测试数据...")
    klines_data = load_bitcoin_data()
    if klines_data:
        df = pd.DataFrame(klines_data)
        
        # 创建回测引擎
        backtest = BacktestEngine(
            start_date="2022-01-01",
            end_date="2022-12-31",
            initial_capital=10000
        )
        
        # 创建策略
        strategy = DummyStrategy()
        
        # 运行回测
        results = backtest.run_backtest(strategy, {"BTCUSDT": df})
        
        # 打印结果
        backtest.print_results(results)
        
        # 绘制结果
        # backtest.plot_results(results)
    else:
        print("无法加载测试数据") 