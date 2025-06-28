#!/usr/bin/env python3
"""
简化的回测测试脚本
"""

import sys
import os
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from run_backtest import BacktestRunner
from src.strategies.config import StrategyConfig

def test_data_loading():
    """测试数据加载功能"""
    print("="*60)
    print("测试数据加载功能")
    print("="*60)
    
    runner = BacktestRunner(data_dir="crypto_data", results_dir="backtest_results")
    
    # 测试数据加载
    market_data = runner.load_market_data("BTCUSDT", ["1d", "4h", "1h"])
    
    print(f"加载的时间框架: {list(market_data.keys())}")
    for tf, df in market_data.items():
        print(f"  {tf}: {len(df)} 条记录")
        if len(df) > 0:
            print(f"    时间范围: {df.index[0]} 到 {df.index[-1]}")
            print(f"    列: {list(df.columns)}")
        break  # 只显示第一个作为示例
    
    return len(market_data) > 0

def test_strategy_creation():
    """测试策略创建"""
    print("\\n" + "="*60)
    print("测试策略创建")
    print("="*60)
    
    try:
        from src.strategies import create_mtf_strategy
        strategy = create_mtf_strategy("standard")
        print("✅ 策略创建成功")
        print(f"策略配置: {strategy.config.mode}")
        return True
    except Exception as e:
        print(f"❌ 策略创建失败: {e}")
        return False

def test_simple_backtest():
    """测试简单回测"""
    print("\\n" + "="*60)
    print("测试简单回测（少量数据）")
    print("="*60)
    
    try:
        runner = BacktestRunner()
        config = StrategyConfig(mode="standard")
        
        # 使用较小的时间范围进行测试
        result = runner.run_single_backtest(
            config=config,
            symbol="BTCUSDT",
            start_date="2024-01-01",
            end_date="2024-06-30",
            initial_capital=10000.0
        )
        
        if 'error' in result:
            print(f"❌ 回测失败: {result['error']}")
            return False
        else:
            print("✅ 回测成功完成")
            print(f"最终价值: ${result['final_value']:,.2f}")
            print(f"总收益率: {result['total_return']:.2%}")
            print(f"夏普比率: {result['performance_metrics']['sharpe_ratio']:.3f}")
            return True
            
    except Exception as e:
        print(f"❌ 回测过程中出错: {e}")
        import traceback
        print(traceback.format_exc())
        return False

def main():
    """主测试函数"""
    print("开始回测系统测试...")
    
    # 测试1: 数据加载
    data_ok = test_data_loading()
    
    # 测试2: 策略创建 
    strategy_ok = test_strategy_creation()
    
    # 测试3: 简单回测（只有在前两个测试通过时才运行）
    backtest_ok = False
    if data_ok and strategy_ok:
        backtest_ok = test_simple_backtest()
    
    print("\\n" + "="*60)
    print("测试结果总结")
    print("="*60)
    print(f"数据加载: {'✅ 通过' if data_ok else '❌ 失败'}")
    print(f"策略创建: {'✅ 通过' if strategy_ok else '❌ 失败'}")
    print(f"简单回测: {'✅ 通过' if backtest_ok else '❌ 失败'}")
    
    if data_ok and strategy_ok and backtest_ok:
        print("\\n🎉 所有测试通过！回测系统准备就绪。")
    else:
        print("\\n⚠️  部分测试失败，需要进一步调试。")

if __name__ == "__main__":
    main()