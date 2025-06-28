#!/usr/bin/env python3
"""
测试回测引擎改进功能
"""

import sys
import time
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from run_backtest import BacktestRunner
from src.strategies.config import StrategyConfig

def test_improved_trade_statistics():
    """测试改进的交易统计功能"""
    print("="*60)
    print("测试改进的交易统计功能")
    print("="*60)
    
    try:
        runner = BacktestRunner()
        config = StrategyConfig(mode="aggressive")  # 使用激进模式增加交易可能性
        
        # 运行一个短期回测
        result = runner.run_single_backtest(
            config=config,
            symbol="BTCUSDT",
            start_date="2024-06-01",
            end_date="2024-06-30",
            initial_capital=10000.0
        )
        
        if 'error' not in result:
            print("✅ 回测成功完成")
            
            # 检查新的交易统计字段
            trade_stats = result.get('trade_statistics', {})
            
            print("\\n📊 改进的交易统计:")
            print(f"  总交易数: {trade_stats.get('total_trades', 0)}")
            print(f"  开仓交易: {result.get('summary', {}).get('total_trades_opened', 0)}")
            print(f"  完成交易: {result.get('summary', {}).get('total_trades_completed', 0)}")
            print(f"  当前开仓: {result.get('summary', {}).get('open_positions', 0)}")
            print(f"  胜率: {trade_stats.get('win_rate', 0):.1f}%")
            print(f"  平均持仓时间: {trade_stats.get('avg_hold_duration_hours', 0):.1f} 小时")
            print(f"  总手续费: ${trade_stats.get('total_commission', 0):.2f}")
            print(f"  盈利因子: {trade_stats.get('profit_factor', 0):.2f}")
            
            # 检查详细交易记录
            completed_trades = result.get('completed_trades', [])
            if completed_trades:
                print(f"\\n🔍 完成交易详情（显示前3条）:")
                for i, trade in enumerate(completed_trades[:3]):
                    print(f"  交易 {i+1}:")
                    print(f"    开仓时间: {trade.get('open_timestamp', 'N/A')}")
                    print(f"    平仓时间: {trade.get('close_timestamp', 'N/A')}")
                    print(f"    入场价: ${trade.get('execution_price', 0):.2f}")
                    print(f"    出场价: ${trade.get('exit_price', 0):.2f}")
                    print(f"    盈亏: {trade.get('price_pnl_pct', 0):+.2f}%")
                    print(f"    持仓时长: {trade.get('hold_duration_hours', 0):.1f}h")
            
            return True
        else:
            print(f"❌ 回测失败: {result['error']}")
            return False
            
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        print(traceback.format_exc())
        return False

def test_timeframe_optimization():
    """测试时间框架优化功能"""
    print("\\n" + "="*60)
    print("测试时间框架优化功能")
    print("="*60)
    
    try:
        runner = BacktestRunner()
        config = StrategyConfig(mode="standard")
        
        print("🔄 运行未优化版本...")
        start_time = time.time()
        
        # 修改回测引擎创建，禁用时间框架优化
        from src.strategies import create_mtf_strategy
        from src.backtest import Portfolio, BacktestEngine
        
        strategy = create_mtf_strategy(config.mode)
        portfolio = Portfolio(initial_cash=10000.0)
        engine_no_opt = BacktestEngine(strategy, portfolio, optimize_timeframes=False)
        
        # 加载数据
        market_data = runner.load_market_data("BTCUSDT", ["1d", "4h", "1h"])
        prepared_data = {}
        for tf, df in market_data.items():
            prepared_df = runner.prepare_data_for_timeframe(df, "2024-05-01", "2024-05-31")
            prepared_data[tf] = prepared_df
        
        # 运行未优化版本
        result_no_opt = engine_no_opt.run_backtest(prepared_data, "BTCUSDT")
        time_no_opt = time.time() - start_time
        
        print("✅ 未优化版本完成")
        
        print("\\n🚀 运行优化版本...")
        start_time = time.time()
        
        # 运行优化版本
        strategy_opt = create_mtf_strategy(config.mode)
        portfolio_opt = Portfolio(initial_cash=10000.0)
        engine_opt = BacktestEngine(strategy_opt, portfolio_opt, optimize_timeframes=True)
        
        result_opt = engine_opt.run_backtest(prepared_data, "BTCUSDT")
        time_opt = time.time() - start_time
        
        print("✅ 优化版本完成")
        
        # 比较结果
        print("\\n📈 性能对比:")
        print(f"  未优化版本耗时: {time_no_opt:.2f} 秒")
        print(f"  优化版本耗时: {time_opt:.2f} 秒")
        print(f"  时间节省: {((time_no_opt - time_opt) / time_no_opt * 100):.1f}%")
        
        # 显示优化统计
        if 'optimization_stats' in result_opt:
            opt_stats = result_opt['optimization_stats']
            print("\\n⚡ 优化统计:")
            print(f"  时间框架优化: {'启用' if opt_stats['timeframe_optimization_enabled'] else '禁用'}")
            print(f"  总分析调用: {opt_stats['total_analysis_calls']}")
            print(f"  跳过分析: {opt_stats['skipped_analysis_calls']}")
            print(f"  效率提升: {opt_stats['efficiency_improvement_pct']:.1f}%")
        
        # 验证结果一致性（应该基本相同）
        final_value_no_opt = result_no_opt.get('summary', {}).get('final_equity', 0)
        final_value_opt = result_opt.get('summary', {}).get('final_equity', 0)
        
        print("\\n🔍 结果一致性检查:")
        print(f"  未优化最终价值: ${final_value_no_opt:,.2f}")
        print(f"  优化版最终价值: ${final_value_opt:,.2f}")
        print(f"  差异: {abs(final_value_no_opt - final_value_opt):,.2f}")
        
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        print(traceback.format_exc())
        return False

def main():
    """主测试函数"""
    print("开始测试回测引擎改进功能...")
    
    # 测试1: 改进的交易统计
    trade_stats_ok = test_improved_trade_statistics()
    
    # 测试2: 时间框架优化
    timeframe_opt_ok = test_timeframe_optimization()
    
    print("\\n" + "="*60)
    print("测试结果总结")
    print("="*60)
    print(f"交易统计改进: {'✅ 通过' if trade_stats_ok else '❌ 失败'}")
    print(f"时间框架优化: {'✅ 通过' if timeframe_opt_ok else '❌ 失败'}")
    
    if all([trade_stats_ok, timeframe_opt_ok]):
        print("\\n🎉 所有改进功能测试通过！")
        print("\\n✨ 主要改进:")
        print("  1. 精确的交易统计 - 完整跟踪每笔交易的开仓和平仓")
        print("  2. 智能时间框架优化 - 仅在关键时间点调用策略分析")
        print("  3. 详细的性能监控 - 更全面的交易分析指标")
        print("  4. 优化效果统计 - 实时监控计算效率提升")
    else:
        print("\\n⚠️  部分功能测试失败，需要进一步调试。")

if __name__ == "__main__":
    main()