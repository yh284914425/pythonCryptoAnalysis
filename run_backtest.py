#!/usr/bin/env python3
"""
主回测脚本

按照BACKTESTING_AND_SIMULATION_PLAN.md的要求，提供完整的策略回测和分析功能
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import warnings
import argparse

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from src.strategies import create_mtf_strategy, StrategyConfig
from src.backtest import BacktestEngine, Portfolio, PerformanceAnalyzer
from src.data_collection.downData import load_existing_data

warnings.filterwarnings('ignore')


class BacktestRunner:
    """回测运行器"""
    
    def __init__(self, data_dir: str = "crypto_data", results_dir: str = "backtest_results"):
        """
        初始化回测运行器
        
        Args:
            data_dir: 数据目录
            results_dir: 结果输出目录
        """
        self.data_dir = Path(data_dir)
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        
        self.performance_analyzer = PerformanceAnalyzer()
        
    def load_market_data(self, symbol: str, timeframes: List[str]) -> Dict[str, pd.DataFrame]:
        """
        加载市场数据
        
        Args:
            symbol: 交易对符号
            timeframes: 时间框架列表
            
        Returns:
            市场数据字典
        """
        coin_name = symbol.replace('USDT', '')
        coin_data_dir = self.data_dir / coin_name
        
        market_data = {}
        
        for tf in timeframes:
            data_file = coin_data_dir / f"{tf}.csv"
            if data_file.exists():
                df = load_existing_data(str(data_file))
                if df is not None and len(df) > 0:
                    market_data[tf] = df
                    print(f"加载 {symbol} {tf} 数据: {len(df)} 条记录")
                else:
                    print(f"警告: {symbol} {tf} 数据文件为空")
            else:
                print(f"警告: 未找到 {symbol} {tf} 数据文件: {data_file}")
        
        return market_data
    
    def prepare_data_for_timeframe(self, df: pd.DataFrame, 
                                  start_date: Optional[str] = None,
                                  end_date: Optional[str] = None) -> pd.DataFrame:
        """
        为指定时间段准备数据
        
        Args:
            df: 原始数据
            start_date: 开始日期 (YYYY-MM-DD)
            end_date: 结束日期 (YYYY-MM-DD)
            
        Returns:
            处理后的数据
        """
        if df is None or len(df) == 0:
            return df
        
        # 确保时间列为datetime类型
        if '开盘时间' in df.columns:
            df['开盘时间'] = pd.to_datetime(df['开盘时间'])
            df = df.set_index('开盘时间').sort_index()
        
        # 过滤时间范围
        if start_date:
            df = df[df.index >= start_date]
        if end_date:
            df = df[df.index <= end_date]
        
        # 确保数据完整性
        required_columns = ['开盘价', '最高价', '最低价', '收盘价', '成交量']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"数据缺少必要列: {missing_columns}")
        
        # 处理缺失值
        df = df.dropna(subset=required_columns)
        
        return df
    
    def run_single_backtest(self, config: StrategyConfig, 
                           symbol: str,
                           start_date: Optional[str] = None,
                           end_date: Optional[str] = None,
                           initial_capital: float = 100000.0) -> Dict[str, Any]:
        """
        运行单次回测
        
        Args:
            config: 策略配置
            symbol: 交易对
            start_date: 开始日期
            end_date: 结束日期
            initial_capital: 初始资金
            
        Returns:
            回测结果
        """
        print(f"\\n开始回测 {symbol} ({config.mode} 模式)")
        print(f"时间范围: {start_date or '最早'} 到 {end_date or '最新'}")
        print(f"初始资金: ${initial_capital:,.2f}")
        
        try:
            # 加载数据
            timeframes = [
                config.technical["timeframes"]["macro"],
                config.technical["timeframes"]["meso"], 
                config.technical["timeframes"]["micro"]
            ]
            
            market_data = self.load_market_data(symbol, timeframes)
            
            if not market_data:
                return {"error": f"无法加载 {symbol} 的市场数据"}
            
            # 准备数据
            prepared_data = {}
            for tf, df in market_data.items():
                prepared_df = self.prepare_data_for_timeframe(df, start_date, end_date)
                if len(prepared_df) == 0:
                    return {"error": f"{symbol} {tf} 在指定时间范围内无数据"}
                prepared_data[tf] = prepared_df
                print(f"  {tf}: {len(prepared_df)} 条记录")
            
            # 创建策略和回测引擎
            strategy = create_mtf_strategy(config.mode)
            portfolio = Portfolio(initial_cash=initial_capital)
            engine = BacktestEngine(strategy, portfolio)
            
            # 运行回测
            print("\\n执行回测...")
            results = engine.run_backtest(prepared_data, symbol)
            
            # 计算性能指标
            if 'equity_curve' in results and len(results['equity_curve']) > 0:
                equity_curve = pd.DataFrame(results['equity_curve'])
                equity_curve['timestamp'] = pd.to_datetime(equity_curve['timestamp'])
                equity_curve = equity_curve.set_index('timestamp')
                
                # 计算收益率
                returns = equity_curve['equity'].pct_change().dropna()
                
                # 性能分析
                performance_metrics = self.performance_analyzer.calculate_metrics(returns)
                
                # 交易统计
                trade_stats = self._calculate_trade_statistics(results.get('trades', []))
                
                # 组装结果
                backtest_result = {
                    'config': {
                        'mode': config.mode,
                        'symbol': symbol,
                        'start_date': start_date,
                        'end_date': end_date,
                        'initial_capital': initial_capital
                    },
                    'performance_metrics': performance_metrics,
                    'trade_statistics': trade_stats,
                    'portfolio_history': equity_curve.to_dict('records'),
                    'trades': results.get('trades', []),
                    'signals': results.get('signals', []),
                    'final_value': equity_curve['equity'].iloc[-1] if len(equity_curve) > 0 else initial_capital,
                    'total_return': (equity_curve['equity'].iloc[-1] / initial_capital - 1) if len(equity_curve) > 0 else 0.0
                }
                
                print(f"\\n回测完成!")
                print(f"最终资产: ${backtest_result['final_value']:,.2f}")
                print(f"总收益率: {backtest_result['total_return']:.2%}")
                print(f"夏普比率: {performance_metrics['sharpe_ratio']:.3f}")
                print(f"最大回撤: {performance_metrics['max_drawdown']:.2%}")
                print(f"交易次数: {trade_stats['total_trades']}")
                print(f"胜率: {trade_stats['win_rate']:.2%}")
                
                return backtest_result
            else:
                return {"error": "回测未产生有效的投资组合历史"}
                
        except Exception as e:
            error_msg = f"回测过程中出错: {str(e)}"
            print(f"错误: {error_msg}")
            return {"error": error_msg}
    
    def _calculate_trade_statistics(self, trades: List[Dict]) -> Dict[str, Any]:
        """计算交易统计"""
        if not trades:
            return {
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0.0,
                'avg_profit': 0.0,
                'avg_loss': 0.0,
                'profit_factor': 0.0,
                'max_profit': 0.0,
                'max_loss': 0.0
            }
        
        profits = [trade.get('profit_pct', 0) for trade in trades if 'profit_pct' in trade]
        
        if not profits:
            profits = [0]
        
        winning_trades = [p for p in profits if p > 0]
        losing_trades = [p for p in profits if p < 0]
        
        return {
            'total_trades': len(profits),
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': len(winning_trades) / len(profits) * 100 if profits else 0,
            'avg_profit': np.mean(winning_trades) if winning_trades else 0,
            'avg_loss': np.mean(losing_trades) if losing_trades else 0,
            'profit_factor': sum(winning_trades) / abs(sum(losing_trades)) if losing_trades else float('inf') if winning_trades else 0,
            'max_profit': max(profits) if profits else 0,
            'max_loss': min(profits) if profits else 0
        }
    
    def run_baseline_test(self, symbol: str = "BTCUSDT",
                         initial_capital: float = 100000.0) -> Dict[str, Any]:
        """
        运行基准测试
        
        Args:
            symbol: 交易对
            initial_capital: 初始资金
            
        Returns:
            基准测试结果
        """
        print("\\n" + "="*60)
        print("阶段二: 基准测试 (Baseline Test)")
        print("="*60)
        
        # 使用标准配置
        config = StrategyConfig(mode="standard")
        
        # 运行完整历史数据回测
        result = self.run_single_backtest(
            config=config,
            symbol=symbol,
            initial_capital=initial_capital
        )
        
        # 保存基准结果
        if 'error' not in result:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            baseline_file = self.results_dir / f"baseline_test_{symbol}_{timestamp}.json"
            
            # 为了JSON序列化，处理特殊类型
            result_for_json = self._prepare_result_for_json(result)
            
            with open(baseline_file, 'w', encoding='utf-8') as f:
                json.dump(result_for_json, f, ensure_ascii=False, indent=2)
            
            print(f"\\n基准测试结果已保存到: {baseline_file}")
        
        return result
    
    def run_parameter_sensitivity_analysis(self, symbol: str = "BTCUSDT",
                                         initial_capital: float = 100000.0) -> Dict[str, List[Dict]]:
        """
        运行参数敏感性分析
        
        Args:
            symbol: 交易对
            initial_capital: 初始资金
            
        Returns:
            敏感性分析结果
        """
        print("\\n" + "="*60)
        print("阶段二: 参数敏感性分析 (Parameter Sensitivity)")
        print("="*60)
        
        results = {
            'stop_loss_sensitivity': [],
            'signal_threshold_sensitivity': []
        }
        
        # 1. 止损倍率敏感性分析
        print("\\n1. 止损倍率敏感性分析...")
        base_config = StrategyConfig(mode="standard")
        
        stop_loss_multipliers = [1.5, 2.0, 2.5, 3.0]
        for multiplier in stop_loss_multipliers:
            print(f"\\n测试止损倍率: {multiplier}")
            
            # 创建修改后的配置
            config = StrategyConfig(mode="standard")
            for asset_type in config.risk['stop_loss_multiplier']:
                config.risk['stop_loss_multiplier'][asset_type] = multiplier
            
            result = self.run_single_backtest(
                config=config,
                symbol=symbol,
                initial_capital=initial_capital
            )
            
            if 'error' not in result:
                result['parameter_tested'] = f"stop_loss_multiplier_{multiplier}"
                results['stop_loss_sensitivity'].append(result)
        
        # 2. 信号强度阈值敏感性分析
        print("\\n2. 信号强度阈值敏感性分析...")
        # 注意：这需要修改策略代码中的阈值，这里我们通过不同的模式来模拟
        modes = ["conservative", "standard", "aggressive"]
        for mode in modes:
            print(f"\\n测试模式: {mode}")
            
            config = StrategyConfig(mode=mode)
            result = self.run_single_backtest(
                config=config,
                symbol=symbol,
                initial_capital=initial_capital
            )
            
            if 'error' not in result:
                result['parameter_tested'] = f"mode_{mode}"
                results['signal_threshold_sensitivity'].append(result)
        
        # 保存敏感性分析结果
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        sensitivity_file = self.results_dir / f"sensitivity_analysis_{symbol}_{timestamp}.json"
        
        results_for_json = {}
        for key, value_list in results.items():
            results_for_json[key] = [self._prepare_result_for_json(result) for result in value_list]
        
        with open(sensitivity_file, 'w', encoding='utf-8') as f:
            json.dump(results_for_json, f, ensure_ascii=False, indent=2)
        
        print(f"\\n敏感性分析结果已保存到: {sensitivity_file}")
        
        # 打印总结
        self._print_sensitivity_summary(results)
        
        return results
    
    def run_robustness_check(self, symbols: List[str] = ["BTCUSDT", "ETHUSDT", "PEPEUSDT"],
                           initial_capital: float = 100000.0) -> Dict[str, List[Dict]]:
        """
        运行鲁棒性检验
        
        Args:
            symbols: 交易对列表
            initial_capital: 初始资金
            
        Returns:
            鲁棒性检验结果
        """
        print("\\n" + "="*60)
        print("阶段二: 鲁棒性检验 (Robustness Check)")
        print("="*60)
        
        results = {
            'cross_asset_results': [],
            'market_period_results': []
        }
        
        config = StrategyConfig(mode="standard")
        
        # 1. 跨资产测试
        print("\\n1. 跨资产测试...")
        for symbol in symbols:
            print(f"\\n测试交易对: {symbol}")
            
            result = self.run_single_backtest(
                config=config,
                symbol=symbol,
                initial_capital=initial_capital
            )
            
            if 'error' not in result:
                result['test_type'] = f"cross_asset_{symbol}"
                results['cross_asset_results'].append(result)
        
        # 2. 不同市场周期测试 (以BTCUSDT为例)
        print("\\n2. 不同市场周期测试...")
        
        # 定义不同的市场周期（需要根据实际数据调整）
        market_periods = [
            ("2021-01-01", "2021-12-31", "bull_market_2021"),
            ("2022-01-01", "2022-12-31", "bear_market_2022"), 
            ("2023-01-01", "2023-12-31", "recovery_2023"),
            ("2024-01-01", None, "recent_2024")
        ]
        
        for start_date, end_date, period_name in market_periods:
            print(f"\\n测试市场周期: {period_name} ({start_date} 到 {end_date or '最新'})")
            
            result = self.run_single_backtest(
                config=config,
                symbol="BTCUSDT",
                start_date=start_date,
                end_date=end_date,
                initial_capital=initial_capital
            )
            
            if 'error' not in result:
                result['test_type'] = f"market_period_{period_name}"
                results['market_period_results'].append(result)
        
        # 保存鲁棒性检验结果
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        robustness_file = self.results_dir / f"robustness_check_{timestamp}.json"
        
        results_for_json = {}
        for key, value_list in results.items():
            results_for_json[key] = [self._prepare_result_for_json(result) for result in value_list]
        
        with open(robustness_file, 'w', encoding='utf-8') as f:
            json.dump(results_for_json, f, ensure_ascii=False, indent=2)
        
        print(f"\\n鲁棒性检验结果已保存到: {robustness_file}")
        
        # 打印总结
        self._print_robustness_summary(results)
        
        return results
    
    def _prepare_result_for_json(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """准备结果用于JSON序列化"""
        result_copy = result.copy()
        
        # 处理可能包含非JSON可序列化对象的字段
        if 'portfolio_history' in result_copy:
            # portfolio_history 已经是字典列表，但可能包含numpy类型
            portfolio_history = result_copy['portfolio_history']
            for item in portfolio_history:
                for key, value in item.items():
                    if isinstance(value, (np.integer, np.floating)):
                        item[key] = float(value)
                    elif isinstance(value, np.ndarray):
                        item[key] = value.tolist()
        
        # 处理numpy类型的指标
        for key in ['performance_metrics', 'trade_statistics']:
            if key in result_copy and isinstance(result_copy[key], dict):
                for metric_key, metric_value in result_copy[key].items():
                    if isinstance(metric_value, (np.integer, np.floating)):
                        result_copy[key][metric_key] = float(metric_value)
                    elif isinstance(metric_value, np.ndarray):
                        result_copy[key][metric_key] = metric_value.tolist()
        
        return result_copy
    
    def _print_sensitivity_summary(self, results: Dict[str, List[Dict]]):
        """打印敏感性分析总结"""
        print("\\n" + "="*50)
        print("敏感性分析总结")
        print("="*50)
        
        # 止损倍率敏感性
        if results['stop_loss_sensitivity']:
            print("\\n止损倍率敏感性:")
            print("倍率\\t总收益率\\t夏普比率\\t最大回撤\\t交易次数")
            print("-" * 60)
            
            for result in results['stop_loss_sensitivity']:
                param = result['parameter_tested'].split('_')[-1]
                total_return = result['total_return']
                sharpe = result['performance_metrics']['sharpe_ratio']
                max_dd = result['performance_metrics']['max_drawdown']
                trades = result['trade_statistics']['total_trades']
                
                print(f"{param}\\t{total_return:.2%}\\t\\t{sharpe:.3f}\\t\\t{max_dd:.2%}\\t\\t{trades}")
        
        # 模式敏感性
        if results['signal_threshold_sensitivity']:
            print("\\n模式敏感性:")
            print("模式\\t\\t总收益率\\t夏普比率\\t最大回撤\\t交易次数")
            print("-" * 70)
            
            for result in results['signal_threshold_sensitivity']:
                mode = result['parameter_tested'].split('_')[-1]
                total_return = result['total_return']
                sharpe = result['performance_metrics']['sharpe_ratio']
                max_dd = result['performance_metrics']['max_drawdown']
                trades = result['trade_statistics']['total_trades']
                
                print(f"{mode}\\t\\t{total_return:.2%}\\t\\t{sharpe:.3f}\\t\\t{max_dd:.2%}\\t\\t{trades}")
    
    def _print_robustness_summary(self, results: Dict[str, List[Dict]]):
        """打印鲁棒性检验总结"""
        print("\\n" + "="*50)
        print("鲁棒性检验总结")
        print("="*50)
        
        # 跨资产结果
        if results['cross_asset_results']:
            print("\\n跨资产表现:")
            print("交易对\\t\\t总收益率\\t夏普比率\\t最大回撤\\t胜率")
            print("-" * 60)
            
            for result in results['cross_asset_results']:
                symbol = result['config']['symbol']
                total_return = result['total_return']
                sharpe = result['performance_metrics']['sharpe_ratio']
                max_dd = result['performance_metrics']['max_drawdown']
                win_rate = result['trade_statistics']['win_rate']
                
                print(f"{symbol}\\t\\t{total_return:.2%}\\t\\t{sharpe:.3f}\\t\\t{max_dd:.2%}\\t\\t{win_rate:.1f}%")
        
        # 市场周期结果
        if results['market_period_results']:
            print("\\n市场周期表现:")
            print("周期\\t\\t\\t总收益率\\t夏普比率\\t最大回撤\\t胜率")
            print("-" * 70)
            
            for result in results['market_period_results']:
                period = result['test_type'].split('_', 2)[-1]
                total_return = result['total_return']
                sharpe = result['performance_metrics']['sharpe_ratio']
                max_dd = result['performance_metrics']['max_drawdown']
                win_rate = result['trade_statistics']['win_rate']
                
                print(f"{period:<15}\\t{total_return:.2%}\\t\\t{sharpe:.3f}\\t\\t{max_dd:.2%}\\t\\t{win_rate:.1f}%")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='加密货币策略回测工具')
    parser.add_argument('--test-type', choices=['baseline', 'sensitivity', 'robustness', 'all'], 
                       default='baseline', help='测试类型')
    parser.add_argument('--symbol', default='BTCUSDT', help='交易对')
    parser.add_argument('--capital', type=float, default=100000.0, help='初始资金')
    parser.add_argument('--data-dir', default='crypto_data', help='数据目录')
    parser.add_argument('--results-dir', default='backtest_results', help='结果目录')
    
    args = parser.parse_args()
    
    # 创建回测运行器
    runner = BacktestRunner(data_dir=args.data_dir, results_dir=args.results_dir)
    
    print("="*80)
    print("加密货币多时间框架背离策略回测系统")
    print(f"策略: MultiTimeframeDivergenceStrategy")
    print(f"数据目录: {args.data_dir}")
    print(f"结果目录: {args.results_dir}")
    print("="*80)
    
    try:
        if args.test_type == 'baseline' or args.test_type == 'all':
            runner.run_baseline_test(symbol=args.symbol, initial_capital=args.capital)
        
        if args.test_type == 'sensitivity' or args.test_type == 'all':
            runner.run_parameter_sensitivity_analysis(symbol=args.symbol, initial_capital=args.capital)
        
        if args.test_type == 'robustness' or args.test_type == 'all':
            runner.run_robustness_check(initial_capital=args.capital)
        
        print("\\n" + "="*80)
        print("所有测试完成!")
        print(f"结果文件保存在: {args.results_dir}")
        print("="*80)
        
    except KeyboardInterrupt:
        print("\\n测试被用户中断")
    except Exception as e:
        print(f"\\n测试过程中发生错误: {e}")
        raise


if __name__ == "__main__":
    main()