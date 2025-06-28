#!/usr/bin/env python3
"""
模拟交易系统 (Paper Trading)

按照BACKTESTING_AND_SIMULATION_PLAN.md的要求，实现实时模拟交易功能
"""

import os
import sys
import json
import time
import schedule
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional
import logging
from threading import Thread
import signal
import argparse

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from src.strategies import create_mtf_strategy, StrategyConfig
from src.backtest import Portfolio, PerformanceAnalyzer
from src.data_collection.downData import get_binance_klines


class PaperTradingEngine:
    """模拟交易引擎"""
    
    def __init__(self, 
                 strategy_mode: str = "standard",
                 initial_capital: float = 100000.0,
                 symbol: str = "BTCUSDT",
                 log_dir: str = "paper_trading_logs"):
        """
        初始化模拟交易引擎
        
        Args:
            strategy_mode: 策略模式
            initial_capital: 初始资金
            symbol: 交易对
            log_dir: 日志目录
        """
        self.strategy_mode = strategy_mode
        self.symbol = symbol
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        # 创建策略和投资组合
        self.strategy = create_mtf_strategy(strategy_mode)
        self.portfolio = Portfolio(initial_cash=initial_capital)
        self.performance_analyzer = PerformanceAnalyzer()
        
        # 交易记录
        self.signals_history = []
        self.trades_history = []
        self.portfolio_history = []
        
        # 运行状态
        self.is_running = False
        self.start_time = None
        
        # 设置日志
        self._setup_logging()
        
    def _setup_logging(self):
        """设置日志"""
        log_file = self.log_dir / f"paper_trading_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file, encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"模拟交易引擎初始化完成")
        self.logger.info(f"策略模式: {self.strategy_mode}")
        self.logger.info(f"交易对: {self.symbol}")
        self.logger.info(f"初始资金: ${self.portfolio.initial_cash:,.2f}")
    
    def fetch_realtime_data(self) -> Dict[str, pd.DataFrame]:
        """
        获取实时市场数据
        
        Returns:
            实时市场数据字典
        """
        timeframes = ['1d', '4h', '1h']
        market_data = {}
        
        try:
            for tf in timeframes:
                # 获取最新的K线数据
                df = get_binance_klines(
                    symbol=self.symbol,
                    interval=tf,
                    limit=200  # 获取足够的历史数据用于指标计算
                )
                
                if df is not None and len(df) > 0:
                    # 设置时间索引
                    df = df.set_index('开盘时间')
                    market_data[tf] = df
                    self.logger.debug(f"获取 {tf} 数据: {len(df)} 条")
                else:
                    self.logger.warning(f"无法获取 {tf} 数据")
                    
                # 避免请求过于频繁
                time.sleep(0.5)
            
            if market_data:
                self.logger.info(f"成功获取 {len(market_data)} 个时间框架的实时数据")
            else:
                self.logger.error("未能获取任何实时数据")
                
            return market_data
            
        except Exception as e:
            self.logger.error(f"获取实时数据失败: {e}")
            return {}
    
    def analyze_market_and_trade(self):
        """分析市场并执行交易逻辑"""
        try:
            self.logger.info("开始市场分析...")
            
            # 获取实时数据
            market_data = self.fetch_realtime_data()
            
            if not market_data:
                self.logger.warning("无可用市场数据，跳过本次分析")
                return
            
            # 更新投资组合价值（使用4小时线的收盘价）
            if '4h' in market_data:
                current_price = float(market_data['4h']['收盘价'].iloc[-1])
                self.portfolio.update_positions_value({self.symbol: current_price})
                self.logger.debug(f"当前价格: ${current_price:,.2f}")
            
            # 分析市场
            analysis_result = self.strategy.analyze_market(market_data, self.symbol)
            
            # 记录分析结果
            signal_record = {
                'timestamp': datetime.now(),
                'symbol': self.symbol,
                'signal_type': analysis_result.get('signal_type', 'neutral'),
                'signal_strength': analysis_result.get('signal_strength', 0.0),
                'confidence': analysis_result.get('confidence', 0.0),
                'confirmed_signals': analysis_result.get('confirmed_signals', 0),
                'required_signals': analysis_result.get('required_signals', 3),
                'current_price': current_price if '4h' in market_data else 0
            }
            
            self.signals_history.append(signal_record)
            
            self.logger.info(f"市场分析完成 - 信号: {signal_record['signal_type']}, "
                           f"强度: {signal_record['signal_strength']:.3f}, "
                           f"置信度: {signal_record['confidence']:.3f}")
            
            # 生成交易信号
            trading_signal = self.strategy.generate_trading_signal(analysis_result)
            
            if trading_signal.get('action') in ['buy', 'sell']:
                self.logger.info(f"生成交易信号: {trading_signal['action']} "
                               f"${trading_signal.get('position_size', 0):,.2f}")
                
                # 执行模拟交易
                self._execute_paper_trade(trading_signal, current_price if '4h' in market_data else 0)
            
            elif trading_signal.get('action') == 'hold':
                reason = trading_signal.get('reason', '无交易信号')
                self.logger.info(f"保持观望: {reason}")
            
            # 记录投资组合状态
            portfolio_record = {
                'timestamp': datetime.now(),
                'total_value': self.portfolio.get_total_value(),
                'cash': self.portfolio.cash,
                'positions_value': self.portfolio.get_positions_value(),
                'num_positions': len(self.portfolio.positions),
                'current_price': current_price if '4h' in market_data else 0
            }
            
            self.portfolio_history.append(portfolio_record)
            
            # 每次分析后保存状态
            self._save_trading_state()
            
        except Exception as e:
            self.logger.error(f"市场分析过程中出错: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
    
    def _execute_paper_trade(self, trading_signal: Dict[str, Any], current_price: float):
        """
        执行模拟交易
        
        Args:
            trading_signal: 交易信号
            current_price: 当前价格
        """
        try:
            action = trading_signal['action']
            symbol = trading_signal.get('symbol', self.symbol)
            position_size = trading_signal.get('position_size', 0)
            stop_loss = trading_signal.get('stop_loss')
            
            # 检查是否已有持仓
            existing_position = self.portfolio.get_position(symbol)
            
            if action == 'buy' and not existing_position:
                # 开多头仓位
                success = self.portfolio.open_position(
                    symbol=symbol,
                    side='long',
                    size=position_size,
                    entry_price=current_price,
                    stop_loss=stop_loss
                )
                
                if success:
                    trade_record = {
                        'timestamp': datetime.now(),
                        'action': 'buy',
                        'symbol': symbol,
                        'price': current_price,
                        'size': position_size,
                        'stop_loss': stop_loss,
                        'signal_strength': trading_signal.get('signal_strength', 0),
                        'confidence': trading_signal.get('confidence', 0)
                    }
                    
                    self.trades_history.append(trade_record)
                    
                    self.logger.info(f"✅ 开仓成功: 买入 {symbol} @ ${current_price:,.2f}, "
                                   f"仓位大小: ${position_size:,.2f}")
                else:
                    self.logger.warning(f"❌ 开仓失败: 资金不足或其他原因")
            
            elif action == 'sell' and existing_position:
                # 平多头仓位
                success, pnl = self.portfolio.close_position(symbol, current_price)
                
                if success:
                    trade_record = {
                        'timestamp': datetime.now(),
                        'action': 'sell',
                        'symbol': symbol,
                        'price': current_price,
                        'pnl': pnl,
                        'pnl_pct': (pnl / existing_position['initial_value'] * 100) if existing_position['initial_value'] != 0 else 0,
                        'hold_duration': (datetime.now() - existing_position['entry_time']).total_seconds() / 3600  # 小时
                    }
                    
                    self.trades_history.append(trade_record)
                    
                    self.logger.info(f"✅ 平仓成功: 卖出 {symbol} @ ${current_price:,.2f}, "
                                   f"盈亏: ${pnl:,.2f} ({trade_record['pnl_pct']:.2f}%)")
                else:
                    self.logger.warning(f"❌ 平仓失败")
            
            else:
                if action == 'buy' and existing_position:
                    self.logger.info(f"⚠️  已有 {symbol} 持仓，跳过买入信号")
                elif action == 'sell' and not existing_position:
                    self.logger.info(f"⚠️  无 {symbol} 持仓，跳过卖出信号")
                    
        except Exception as e:
            self.logger.error(f"执行模拟交易失败: {e}")
    
    def _save_trading_state(self):
        """保存交易状态到文件"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # 保存信号历史
            if self.signals_history:
                signals_file = self.log_dir / f"signals_history_{timestamp}.json"
                signals_data = []
                for signal in self.signals_history[-10:]:  # 只保存最近10条
                    signal_copy = signal.copy()
                    signal_copy['timestamp'] = signal_copy['timestamp'].isoformat()
                    signals_data.append(signal_copy)
                
                with open(signals_file, 'w', encoding='utf-8') as f:
                    json.dump(signals_data, f, ensure_ascii=False, indent=2)
            
            # 保存交易历史
            if self.trades_history:
                trades_file = self.log_dir / f"trades_history_{timestamp}.json"
                trades_data = []
                for trade in self.trades_history:
                    trade_copy = trade.copy()
                    trade_copy['timestamp'] = trade_copy['timestamp'].isoformat()
                    trades_data.append(trade_copy)
                
                with open(trades_file, 'w', encoding='utf-8') as f:
                    json.dump(trades_data, f, ensure_ascii=False, indent=2)
            
            # 保存投资组合历史
            if self.portfolio_history:
                portfolio_file = self.log_dir / f"portfolio_history_{timestamp}.json"
                portfolio_data = []
                for record in self.portfolio_history[-100:]:  # 只保存最近100条
                    record_copy = record.copy()
                    record_copy['timestamp'] = record_copy['timestamp'].isoformat()
                    portfolio_data.append(record_copy)
                
                with open(portfolio_file, 'w', encoding='utf-8') as f:
                    json.dump(portfolio_data, f, ensure_ascii=False, indent=2)
            
            # 保存当前投资组合状态
            portfolio_summary = self.portfolio.get_portfolio_summary()
            summary_file = self.log_dir / f"portfolio_summary_{timestamp}.json"
            
            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump(portfolio_summary, f, ensure_ascii=False, indent=2)
                
        except Exception as e:
            self.logger.error(f"保存交易状态失败: {e}")
    
    def print_status_report(self):
        """打印状态报告"""
        try:
            portfolio_summary = self.portfolio.get_portfolio_summary()
            
            print("\\n" + "="*60)
            print("模拟交易状态报告")
            print("="*60)
            
            # 基本信息
            runtime = (datetime.now() - self.start_time) if self.start_time else timedelta(0)
            print(f"运行时间: {runtime}")
            print(f"策略模式: {self.strategy_mode}")
            print(f"交易对: {self.symbol}")
            
            # 投资组合状态
            print(f"\\n💰 投资组合状态:")
            print(f"  总价值: ${portfolio_summary['total_value']:,.2f}")
            print(f"  现金: ${portfolio_summary['cash']:,.2f}")
            print(f"  持仓价值: ${portfolio_summary['positions_value']:,.2f}")
            print(f"  总收益率: {portfolio_summary['total_return_pct']:.2f}%")
            print(f"  现金使用率: {portfolio_summary['cash_utilization']:.1f}%")
            
            # 持仓详情
            if portfolio_summary['position_details']:
                print(f"\\n📈 当前持仓:")
                for pos in portfolio_summary['position_details']:
                    print(f"  {pos['symbol']}: {pos['side']} "
                          f"${pos['current_value']:,.2f} "
                          f"({pos['pnl_pct']:+.2f}%)")
            
            # 交易统计
            completed_trades = [t for t in self.trades_history if 'pnl' in t]
            if completed_trades:
                profits = [t['pnl_pct'] for t in completed_trades]
                winning_trades = [p for p in profits if p > 0]
                
                print(f"\\n📊 交易统计:")
                print(f"  总交易数: {len(completed_trades)}")
                print(f"  盈利交易: {len(winning_trades)}")
                print(f"  胜率: {len(winning_trades)/len(completed_trades)*100:.1f}%")
                if profits:
                    print(f"  平均收益: {np.mean(profits):.2f}%")
                    print(f"  最大盈利: {max(profits):.2f}%")
                    print(f"  最大亏损: {min(profits):.2f}%")
            
            # 最近信号
            if self.signals_history:
                latest_signal = self.signals_history[-1]
                print(f"\\n🎯 最新信号 ({latest_signal['timestamp'].strftime('%H:%M:%S')}):")
                print(f"  类型: {latest_signal['signal_type']}")
                print(f"  强度: {latest_signal['signal_strength']:.3f}")
                print(f"  置信度: {latest_signal['confidence']:.3f}")
                print(f"  确认指标: {latest_signal['confirmed_signals']}/{latest_signal['required_signals']}")
            
            print("="*60)
            
        except Exception as e:
            self.logger.error(f"生成状态报告失败: {e}")
    
    def start_paper_trading(self, check_interval_minutes: int = 60):
        """
        开始模拟交易
        
        Args:
            check_interval_minutes: 检查间隔（分钟）
        """
        self.logger.info(f"启动模拟交易，检查间隔: {check_interval_minutes} 分钟")
        self.is_running = True
        self.start_time = datetime.now()
        
        # 设置定时任务
        schedule.every(check_interval_minutes).minutes.do(self.analyze_market_and_trade)
        
        # 设置状态报告（每小时）
        schedule.every().hour.do(self.print_status_report)
        
        # 立即执行一次分析
        self.analyze_market_and_trade()
        self.print_status_report()
        
        self.logger.info("模拟交易已启动，按Ctrl+C停止")
        
        try:
            while self.is_running:
                schedule.run_pending()
                time.sleep(60)  # 每分钟检查一次任务
                
        except KeyboardInterrupt:
            self.logger.info("收到停止信号，正在关闭...")
            self.stop_paper_trading()
    
    def stop_paper_trading(self):
        """停止模拟交易"""
        self.is_running = False
        
        # 生成最终报告
        self.generate_final_report()
        
        self.logger.info("模拟交易已停止")
    
    def generate_final_report(self):
        """生成最终报告"""
        try:
            self.logger.info("生成最终报告...")
            
            # 计算性能指标
            if len(self.portfolio_history) > 1:
                portfolio_df = pd.DataFrame(self.portfolio_history)
                portfolio_df['timestamp'] = pd.to_datetime(portfolio_df['timestamp'])
                portfolio_df = portfolio_df.set_index('timestamp')
                
                returns = portfolio_df['total_value'].pct_change().dropna()
                performance_metrics = self.performance_analyzer.calculate_metrics(returns)
            else:
                performance_metrics = {}
            
            # 组装最终报告
            final_report = {
                'summary': {
                    'start_time': self.start_time.isoformat() if self.start_time else None,
                    'end_time': datetime.now().isoformat(),
                    'duration_hours': (datetime.now() - self.start_time).total_seconds() / 3600 if self.start_time else 0,
                    'strategy_mode': self.strategy_mode,
                    'symbol': self.symbol,
                    'initial_capital': self.portfolio.initial_cash,
                    'final_value': self.portfolio.get_total_value(),
                    'total_return': (self.portfolio.get_total_value() - self.portfolio.initial_cash) / self.portfolio.initial_cash
                },
                'performance_metrics': performance_metrics,
                'portfolio_summary': self.portfolio.get_portfolio_summary(),
                'trading_statistics': {
                    'total_signals': len(self.signals_history),
                    'total_trades': len([t for t in self.trades_history if 'pnl' in t]),
                    'buy_signals': len([s for s in self.signals_history if s['signal_type'] == 'buy']),
                    'sell_signals': len([s for s in self.signals_history if s['signal_type'] == 'sell']),
                    'neutral_signals': len([s for s in self.signals_history if s['signal_type'] == 'neutral'])
                }
            }
            
            # 保存最终报告
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_file = self.log_dir / f"final_report_{timestamp}.json"
            
            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump(final_report, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"最终报告已保存: {report_file}")
            
            # 打印摘要
            print("\\n" + "="*60)
            print("🏁 模拟交易最终报告")
            print("="*60)
            print(f"运行时长: {final_report['summary']['duration_hours']:.1f} 小时")
            print(f"总收益率: {final_report['summary']['total_return']:.2%}")
            print(f"最终价值: ${final_report['summary']['final_value']:,.2f}")
            print(f"总信号数: {final_report['trading_statistics']['total_signals']}")
            print(f"总交易数: {final_report['trading_statistics']['total_trades']}")
            
            if performance_metrics:
                print(f"夏普比率: {performance_metrics['sharpe_ratio']:.3f}")
                print(f"最大回撤: {performance_metrics['max_drawdown']:.2%}")
            
            print("="*60)
            
        except Exception as e:
            self.logger.error(f"生成最终报告失败: {e}")


def signal_handler(signum, frame):
    """信号处理器"""
    print("\\n收到停止信号...")
    sys.exit(0)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='加密货币模拟交易系统')
    parser.add_argument('--mode', choices=['conservative', 'standard', 'aggressive'], 
                       default='standard', help='策略模式')
    parser.add_argument('--symbol', default='BTCUSDT', help='交易对')
    parser.add_argument('--capital', type=float, default=100000.0, help='初始资金')
    parser.add_argument('--interval', type=int, default=60, help='检查间隔（分钟）')
    parser.add_argument('--duration', type=int, help='运行时长（小时），不指定则持续运行')
    
    args = parser.parse_args()
    
    # 设置信号处理
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    print("="*80)
    print("🚀 加密货币模拟交易系统启动")
    print(f"策略模式: {args.mode}")
    print(f"交易对: {args.symbol}")
    print(f"初始资金: ${args.capital:,.2f}")
    print(f"检查间隔: {args.interval} 分钟")
    if args.duration:
        print(f"运行时长: {args.duration} 小时")
    else:
        print("运行模式: 持续运行")
    print("="*80)
    
    # 创建和启动模拟交易引擎
    engine = PaperTradingEngine(
        strategy_mode=args.mode,
        initial_capital=args.capital,
        symbol=args.symbol
    )
    
    try:
        if args.duration:
            # 定时运行
            def stop_after_duration():
                time.sleep(args.duration * 3600)
                engine.stop_paper_trading()
            
            Thread(target=stop_after_duration, daemon=True).start()
        
        engine.start_paper_trading(check_interval_minutes=args.interval)
        
    except Exception as e:
        print(f"\\n❌ 模拟交易过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        print("\\n模拟交易系统已退出")


if __name__ == "__main__":
    main()