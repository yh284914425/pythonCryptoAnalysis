"""
回测引擎

通用回测引擎，接收交易信号和价格数据，模拟交易过程并输出绩效报告
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime, timedelta
import warnings

from .portfolio import Portfolio
from .performance import PerformanceAnalyzer


class BacktestEngine:
    """通用回测引擎"""
    
    def __init__(self, 
                 strategy: Any = None,
                 portfolio: Portfolio = None,
                 commission: float = 0.001,
                 slippage: float = 0.0005,
                 optimize_timeframes: bool = True):
        """
        初始化回测引擎
        
        Args:
            strategy: 交易策略对象
            portfolio: 投资组合对象
            commission: 手续费率
            slippage: 滑点
            optimize_timeframes: 是否优化时间框架处理（仅在关键时间点调用策略）
        """
        self.strategy = strategy
        self.portfolio = portfolio or Portfolio(100000.0)
        self.commission = commission
        self.slippage = slippage
        self.optimize_timeframes = optimize_timeframes
        
        self.initial_capital = self.portfolio.initial_capital
        self.performance_analyzer = PerformanceAnalyzer()
        
        self.trades = []  # 交易记录（开仓和平仓的完整记录）
        self.open_trades = {}  # 当前开仓交易记录 {symbol: trade_record}
        self.completed_trades = []  # 已完成的交易记录（含盈亏）
        self.equity_curve = []  # 权益曲线
        self.daily_returns = []  # 日收益率
        
        # 时间框架优化相关
        self.last_analysis_times = {}  # 记录各时间框架最后分析时间
        self.analysis_count = 0  # 分析调用计数
        self.skipped_analysis_count = 0  # 跳过的分析计数
        
        # 时间框架映射（分钟）
        self.timeframe_minutes = {
            '1m': 1, '5m': 5, '15m': 15, '30m': 30,
            '1h': 60, '2h': 120, '4h': 240, '6h': 360, '8h': 480, '12h': 720,
            '1d': 1440, '3d': 4320, '1w': 10080
        }
        
    def run_backtest(self, 
                    market_data: Dict[str, pd.DataFrame],
                    symbol: str = "BTCUSDT",
                    start_date: Optional[str] = None,
                    end_date: Optional[str] = None) -> Dict[str, Any]:
        """
        运行回测
        
        Args:
            market_data: 市场数据字典
            symbol: 交易对符号
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            回测结果
        """
        # 重置状态
        self._reset_backtest()
        
        # 获取时间索引
        time_index = self._get_time_index(market_data, start_date, end_date)
        
        if len(time_index) == 0:
            return {'error': '没有有效的时间数据'}
        
        print(f"开始回测: {time_index[0]} 到 {time_index[-1]}")
        print(f"总共 {len(time_index)} 个时间点")
        
        # 逐时间点执行回测
        for i, timestamp in enumerate(time_index):
            try:
                # 获取当前时间点的市场数据
                current_data = self._get_current_market_data(market_data, timestamp, i)
                
                if not current_data:
                    continue
                
                # 更新投资组合价值
                self._update_portfolio_value(current_data, timestamp)
                
                # 检查止损
                self._check_stop_losses(current_data, timestamp)
                
                # 生成交易信号（使用时间框架优化）
                if i >= 50:  # 等待足够的历史数据
                    # 检查是否需要重新分析策略
                    should_analyze = self._should_analyze_strategy(timestamp, market_data)
                    
                    if should_analyze:
                        self.analysis_count += 1
                        historical_data = self._get_historical_data(market_data, timestamp, i)
                        
                        if historical_data:
                            # 分析市场
                            analysis_result = self.strategy.analyze_market(historical_data, symbol)
                            
                            # 生成交易信号
                            trading_signal = self.strategy.generate_trading_signal(analysis_result)
                            
                            # 执行交易
                            if trading_signal.get('action') in ['buy', 'sell']:
                                self._execute_trade(trading_signal, current_data, timestamp)
                    else:
                        self.skipped_analysis_count += 1
                
                # 记录权益曲线
                self.equity_curve.append({
                    'timestamp': timestamp,
                    'equity': self.portfolio.get_total_value(),
                    'cash': self.portfolio.cash,
                    'positions_value': self.portfolio.get_positions_value()
                })
                
                if i % 100 == 0:
                    print(f"进度: {i}/{len(time_index)} ({i/len(time_index)*100:.1f}%)")
            
            except Exception as e:
                print(f"回测过程中出错 (时间点 {i}): {e}")
                continue
        
        # 显示优化效果
        if self.optimize_timeframes:
            total_possible = len(time_index) - 50  # 减去等待期
            efficiency = (1 - self.skipped_analysis_count / total_possible) * 100 if total_possible > 0 else 0
            print(f"\\n时间框架优化统计:")
            print(f"  总时间点: {len(time_index)}")
            print(f"  策略分析次数: {self.analysis_count}")
            print(f"  跳过次数: {self.skipped_analysis_count}")
            print(f"  计算效率提升: {self.skipped_analysis_count / total_possible * 100:.1f}% 减少")
        
        print("回测完成，生成报告...")
        return self._generate_backtest_report()
    
    def _reset_backtest(self):
        """重置回测状态"""
        self.portfolio = Portfolio(self.initial_capital)
        self.trades = []
        self.open_trades = {}
        self.completed_trades = []
        self.equity_curve = []
        self.daily_returns = []
        self.last_analysis_times = {}
        self.analysis_count = 0
        self.skipped_analysis_count = 0
    
    def _get_time_index(self, market_data: Dict[str, pd.DataFrame],
                       start_date: Optional[str] = None,
                       end_date: Optional[str] = None) -> List:
        """
        获取时间索引
        
        Args:
            market_data: 市场数据
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            时间索引列表
        """
        # 使用最小时间框架的数据作为基准
        timeframes = ['1h', '4h', '1d']
        base_timeframe = None
        
        for tf in timeframes:
            if tf in market_data:
                base_timeframe = tf
                break
        
        if not base_timeframe:
            return []
        
        df = market_data[base_timeframe]
        time_index = df.index.tolist()
        
        # 应用日期过滤
        if start_date:
            start_dt = pd.to_datetime(start_date)
            time_index = [t for t in time_index if t >= start_dt]
        
        if end_date:
            end_dt = pd.to_datetime(end_date)
            time_index = [t for t in time_index if t <= end_dt]
        
        return time_index
    
    def _should_analyze_strategy(self, timestamp, market_data: Dict[str, pd.DataFrame]) -> bool:
        """
        判断是否需要重新调用策略分析（时间框架优化）
        
        Args:
            timestamp: 当前时间戳
            market_data: 市场数据
            
        Returns:
            是否需要分析
        """
        if not self.optimize_timeframes:
            return True  # 如果未启用优化，总是分析
        
        # 获取策略使用的时间框架
        strategy_timeframes = []
        if hasattr(self.strategy, 'config') and hasattr(self.strategy.config, 'technical'):
            tf_config = self.strategy.config.technical.get('timeframes', {})
            strategy_timeframes = [
                tf_config.get('macro', '1d'),
                tf_config.get('meso', '4h'), 
                tf_config.get('micro', '1h')
            ]
        else:
            strategy_timeframes = list(market_data.keys())
        
        # 检查是否有任何关键时间框架需要更新
        for timeframe in strategy_timeframes:
            if timeframe not in market_data:
                continue
                
            # 获取该时间框架的分钟数
            tf_minutes = self.timeframe_minutes.get(timeframe, 60)
            
            # 检查是否到了该时间框架的收盘时间
            if self._is_timeframe_close(timestamp, tf_minutes):
                # 检查距离上次分析是否超过了该时间框架的间隔
                if timeframe not in self.last_analysis_times:
                    self.last_analysis_times[timeframe] = timestamp
                    return True
                
                time_diff = (timestamp - self.last_analysis_times[timeframe]).total_seconds() / 60
                if time_diff >= tf_minutes * 0.9:  # 允许10%的误差
                    self.last_analysis_times[timeframe] = timestamp
                    return True
        
        return False
    
    def _is_timeframe_close(self, timestamp, tf_minutes: int) -> bool:
        """
        判断当前时间是否为指定时间框架的收盘时间
        
        Args:
            timestamp: 当前时间戳
            tf_minutes: 时间框架分钟数
            
        Returns:
            是否为收盘时间
        """
        # 转换为分钟级别的时间戳
        total_minutes = timestamp.hour * 60 + timestamp.minute
        
        # 检查是否为该时间框架的整数倍时间点
        return total_minutes % tf_minutes == 0
    
    def _get_current_market_data(self, market_data: Dict[str, pd.DataFrame],
                                timestamp, index: int) -> Dict[str, Any]:
        """
        获取当前时间点的市场数据
        
        Args:
            market_data: 完整市场数据
            timestamp: 当前时间戳
            index: 当前索引
            
        Returns:
            当前市场数据
        """
        current_data = {}
        
        try:
            for timeframe, df in market_data.items():
                # 找到对应的时间点
                if timestamp in df.index:
                    row = df.loc[timestamp]
                    current_data[timeframe] = {
                        'open': float(row['开盘价']),
                        'high': float(row['最高价']),
                        'low': float(row['最低价']),
                        'close': float(row['收盘价']),
                        'volume': float(row['成交量']),
                        'timestamp': timestamp
                    }
                else:
                    # 使用最近的数据
                    valid_data = df[df.index <= timestamp]
                    if len(valid_data) > 0:
                        row = valid_data.iloc[-1]
                        current_data[timeframe] = {
                            'open': float(row['开盘价']),
                            'high': float(row['最高价']),
                            'low': float(row['最低价']),
                            'close': float(row['收盘价']),
                            'volume': float(row['成交量']),
                            'timestamp': timestamp
                        }
            
            return current_data
            
        except Exception as e:
            print(f"获取当前市场数据失败: {e}")
            return {}
    
    def _get_historical_data(self, market_data: Dict[str, pd.DataFrame],
                            timestamp, index: int,
                            lookback: int = 100) -> Dict[str, pd.DataFrame]:
        """
        获取历史数据用于策略分析
        
        Args:
            market_data: 完整市场数据
            timestamp: 当前时间戳
            index: 当前索引
            lookback: 回望期数
            
        Returns:
            历史数据
        """
        historical_data = {}
        
        try:
            for timeframe, df in market_data.items():
                # 获取当前时间点之前的数据
                historical_df = df[df.index <= timestamp].tail(lookback)
                
                if len(historical_df) >= 50:  # 确保有足够的数据
                    historical_data[timeframe] = historical_df
            
            return historical_data
            
        except Exception as e:
            print(f"获取历史数据失败: {e}")
            return {}
    
    def _update_portfolio_value(self, current_data: Dict[str, Any], timestamp):
        """
        更新投资组合价值
        
        Args:
            current_data: 当前市场数据
            timestamp: 时间戳
        """
        # 使用中等时间框架的价格
        price_data = None
        for tf in ['4h', '1h', '1d']:
            if tf in current_data:
                price_data = current_data[tf]
                break
        
        if price_data:
            current_price = price_data['close']
            self.portfolio.update_positions_value({'BTCUSDT': current_price})
    
    def _check_stop_losses(self, current_data: Dict[str, Any], timestamp):
        """
        检查止损
        
        Args:
            current_data: 当前市场数据
            timestamp: 时间戳
        """
        # 获取当前价格
        price_data = None
        for tf in ['4h', '1h', '1d']:
            if tf in current_data:
                price_data = current_data[tf]
                break
        
        if not price_data:
            return
        
        current_price = price_data['close']
        
        # 检查所有持仓的止损
        positions_to_close = []
        
        for symbol, position in self.portfolio.positions.items():
            if position['stop_loss']:
                if position['side'] == 'long':
                    if current_price <= position['stop_loss']:
                        positions_to_close.append((symbol, 'stop_loss'))
                elif position['side'] == 'short':
                    if current_price >= position['stop_loss']:
                        positions_to_close.append((symbol, 'stop_loss'))
        
        # 执行止损
        for symbol, reason in positions_to_close:
            self._close_position(symbol, current_price, timestamp, reason)
    
    def _execute_trade(self, trading_signal: Dict[str, Any], 
                      current_data: Dict[str, Any], timestamp):
        """
        执行交易
        
        Args:
            trading_signal: 交易信号
            current_data: 当前市场数据
            timestamp: 时间戳
        """
        try:
            symbol = trading_signal.get('symbol', 'BTCUSDT')
            action = trading_signal['action']
            entry_price = trading_signal['entry_price']
            position_size = trading_signal['position_size']
            stop_loss = trading_signal.get('stop_loss')
            
            # 应用滑点
            if action == 'buy':
                execution_price = entry_price * (1 + self.slippage)
            else:
                execution_price = entry_price * (1 - self.slippage)
            
            # 计算手续费
            commission_cost = position_size * self.commission
            
            # 检查资金是否足够
            if action == 'buy':
                total_cost = position_size + commission_cost
                if self.portfolio.cash < total_cost:
                    print(f"资金不足，跳过交易 (需要: {total_cost:.2f}, 现有: {self.portfolio.cash:.2f})")
                    return
            
            # 执行交易
            if action == 'buy':
                success = self.portfolio.open_position(
                    symbol=symbol,
                    side='long',
                    size=position_size,
                    entry_price=execution_price,
                    stop_loss=stop_loss
                )
            else:  # sell
                success = self.portfolio.open_position(
                    symbol=symbol,
                    side='short',
                    size=position_size,
                    entry_price=execution_price,
                    stop_loss=stop_loss
                )
            
            if success:
                # 记录开仓交易到开仓记录中
                trade_record = {
                    'trade_id': f"{symbol}_{timestamp.strftime('%Y%m%d_%H%M%S')}",
                    'open_timestamp': timestamp,
                    'symbol': symbol,
                    'action': action,  # 'buy' or 'sell'
                    'side': 'long' if action == 'buy' else 'short',
                    'size': position_size,
                    'entry_price': entry_price,
                    'execution_price': execution_price,
                    'stop_loss': stop_loss,
                    'commission': commission_cost,
                    'signal_strength': trading_signal.get('signal_strength', 0),
                    'confidence': trading_signal.get('confidence', 0),
                    'status': 'open'
                }
                
                # 存储开仓记录
                self.open_trades[symbol] = trade_record
                
                # 同时记录到总交易列表（用于历史查看）
                self.trades.append(trade_record.copy())
                
                print(f"执行交易: {action} {symbol} @ {execution_price:.2f}, 大小: {position_size:.2f}")
            
        except Exception as e:
            print(f"执行交易失败: {e}")
    
    def _close_position(self, symbol: str, exit_price: float, 
                       timestamp, reason: str = 'manual'):
        """
        平仓
        
        Args:
            symbol: 交易对
            exit_price: 平仓价格
            timestamp: 时间戳
            reason: 平仓原因
        """
        try:
            # 检查是否有对应的开仓记录
            if symbol not in self.open_trades:
                print(f"警告: 没有找到 {symbol} 的开仓记录")
                return
            
            open_trade = self.open_trades[symbol]
            
            # 先从投资组合中平仓
            success, portfolio_pnl = self.portfolio.close_position(symbol, exit_price)
            
            if success:
                # 计算交易盈亏
                entry_price = open_trade['execution_price']
                side = open_trade['side']
                size = open_trade['size']
                
                # 计算价格变化的盈亏
                if side == 'long':
                    price_pnl = (exit_price - entry_price) * (size / entry_price)  # 按数量计算
                    price_pnl_pct = (exit_price - entry_price) / entry_price * 100
                else:  # short
                    price_pnl = (entry_price - exit_price) * (size / entry_price)  # 按数量计算  
                    price_pnl_pct = (entry_price - exit_price) / entry_price * 100
                
                # 计算持仓时间
                hold_duration = (timestamp - open_trade['open_timestamp']).total_seconds() / 3600  # 小时
                
                # 创建完整的交易记录
                completed_trade = open_trade.copy()
                completed_trade.update({
                    'close_timestamp': timestamp,
                    'exit_price': exit_price,
                    'close_reason': reason,
                    'price_pnl': price_pnl,
                    'price_pnl_pct': price_pnl_pct,
                    'portfolio_pnl': portfolio_pnl,  # 投资组合层面的盈亏
                    'hold_duration_hours': hold_duration,
                    'status': 'closed'
                })
                
                # 存储到已完成交易记录
                self.completed_trades.append(completed_trade)
                
                # 更新总交易列表中对应的记录
                for trade in self.trades:
                    if (trade['symbol'] == symbol and 
                        trade['open_timestamp'] == open_trade['open_timestamp']):
                        trade.update(completed_trade)
                        break
                
                # 从开仓记录中移除
                del self.open_trades[symbol]
                
                print(f"平仓: {symbol} @ {exit_price:.2f}, "
                      f"盈亏: {price_pnl:.2f} ({price_pnl_pct:+.2f}%), "
                      f"原因: {reason}, 持仓: {hold_duration:.1f}h")
            else:
                print(f"平仓失败: {symbol}")
                
        except Exception as e:
            print(f"平仓失败: {e}")
            import traceback
            print(traceback.format_exc())
    
    def _generate_backtest_report(self) -> Dict[str, Any]:
        """
        生成回测报告
        
        Returns:
            回测报告
        """
        try:
            # 基础统计
            final_equity = self.portfolio.get_total_value()
            total_return = (final_equity - self.initial_capital) / self.initial_capital
            
            # 计算详细性能指标
            equity_df = pd.DataFrame(self.equity_curve)
            if len(equity_df) > 1:
                equity_df['returns'] = equity_df['equity'].pct_change()
                performance_metrics = self.performance_analyzer.calculate_metrics(
                    equity_df['returns'].dropna()
                )
            else:
                performance_metrics = {}
            
            # 交易统计
            trade_stats = self._calculate_trade_statistics()
            
            return {
                'summary': {
                    'initial_capital': self.initial_capital,
                    'final_equity': final_equity,
                    'total_return': total_return,
                    'total_return_pct': total_return * 100,
                    'total_trades_opened': len(self.trades),
                    'total_trades_completed': len(self.completed_trades),
                    'open_positions': len(self.open_trades)
                },
                'optimization_stats': {
                    'timeframe_optimization_enabled': self.optimize_timeframes,
                    'total_analysis_calls': self.analysis_count,
                    'skipped_analysis_calls': self.skipped_analysis_count,
                    'efficiency_improvement_pct': (self.skipped_analysis_count / (self.analysis_count + self.skipped_analysis_count) * 100) if (self.analysis_count + self.skipped_analysis_count) > 0 else 0
                },
                'performance_metrics': performance_metrics,
                'trade_statistics': trade_stats,
                'equity_curve': equity_df.to_dict('records') if len(equity_df) > 0 else [],
                'all_trades': self.trades,  # 所有交易记录（包括开仓和已完成）
                'completed_trades': self.completed_trades,  # 仅已完成的交易
                'open_trades': list(self.open_trades.values())  # 当前开仓的交易
            }
            
        except Exception as e:
            return {
                'error': f'生成报告失败: {e}',
                'summary': {
                    'initial_capital': self.initial_capital,
                    'final_equity': self.portfolio.get_total_value(),
                    'total_trades': len(self.trades)
                }
            }
    
    def _calculate_trade_statistics(self) -> Dict[str, Any]:
        """
        计算交易统计（基于引擎自身的精确交易记录）
        
        Returns:
            交易统计
        """
        if not self.completed_trades:
            return {
                'total_trades': 0,
                'open_trades': len(self.open_trades),
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0.0,
                'avg_profit': 0.0,
                'avg_loss': 0.0,
                'profit_factor': 0.0,
                'max_profit': 0.0,
                'max_loss': 0.0,
                'avg_hold_duration_hours': 0.0,
                'total_commission': 0.0
            }
        
        # 从已完成的交易记录中提取盈亏百分比
        trade_pnl_pcts = [trade['price_pnl_pct'] for trade in self.completed_trades]
        trade_pnl_amounts = [trade['price_pnl'] for trade in self.completed_trades]
        
        # 分类盈利和亏损交易
        winning_trades = [pnl for pnl in trade_pnl_pcts if pnl > 0]
        losing_trades = [pnl for pnl in trade_pnl_pcts if pnl < 0]
        break_even_trades = [pnl for pnl in trade_pnl_pcts if pnl == 0]
        
        total_trades = len(self.completed_trades)
        num_winning = len(winning_trades)
        num_losing = len(losing_trades)
        num_break_even = len(break_even_trades)
        
        # 计算统计指标
        win_rate = num_winning / total_trades * 100 if total_trades > 0 else 0
        avg_profit = sum(winning_trades) / num_winning if num_winning > 0 else 0
        avg_loss = sum(losing_trades) / num_losing if num_losing > 0 else 0
        
        profit_factor = abs(sum(winning_trades) / sum(losing_trades)) if losing_trades else float('inf') if winning_trades else 0
        
        max_profit = max(trade_pnl_pcts) if trade_pnl_pcts else 0
        max_loss = min(trade_pnl_pcts) if trade_pnl_pcts else 0
        
        # 计算平均持仓时间
        avg_hold_duration = sum(trade['hold_duration_hours'] for trade in self.completed_trades) / total_trades
        
        # 计算总手续费
        total_commission = sum(trade['commission'] for trade in self.completed_trades)
        
        return {
            'total_trades': total_trades,
            'open_trades': len(self.open_trades),
            'winning_trades': num_winning,
            'losing_trades': num_losing,
            'break_even_trades': num_break_even,
            'win_rate': win_rate,
            'avg_profit': avg_profit,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'max_profit': max_profit,
            'max_loss': max_loss,
            'avg_hold_duration_hours': avg_hold_duration,
            'total_commission': total_commission,
            'gross_profit': sum(winning_trades),
            'gross_loss': sum(losing_trades),
            'net_profit': sum(trade_pnl_pcts),
            'total_pnl_amount': sum(trade_pnl_amounts)
        }