#!/usr/bin/env python3
"""
模拟交易系统测试脚本
"""

import sys
import time
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from paper_trading import PaperTradingEngine

def test_paper_trading_initialization():
    """测试模拟交易引擎初始化"""
    print("="*60)
    print("测试模拟交易引擎初始化")
    print("="*60)
    
    try:
        engine = PaperTradingEngine(
            strategy_mode="standard",
            initial_capital=10000.0,
            symbol="BTCUSDT"
        )
        
        print("✅ 模拟交易引擎初始化成功")
        print(f"策略模式: {engine.strategy_mode}")
        print(f"初始资金: ${engine.portfolio.initial_cash:,.2f}")
        print(f"交易对: {engine.symbol}")
        
        return True
        
    except Exception as e:
        print(f"❌ 模拟交易引擎初始化失败: {e}")
        import traceback
        print(traceback.format_exc())
        return False

def test_realtime_data_fetch():
    """测试实时数据获取"""
    print("\\n" + "="*60)
    print("测试实时数据获取")
    print("="*60)
    
    try:
        engine = PaperTradingEngine(initial_capital=10000.0)
        
        print("正在获取实时数据...")
        market_data = engine.fetch_realtime_data()
        
        if market_data:
            print("✅ 实时数据获取成功")
            for tf, df in market_data.items():
                print(f"  {tf}: {len(df)} 条记录")
                if len(df) > 0:
                    latest_price = float(df['收盘价'].iloc[-1])
                    print(f"    最新价格: ${latest_price:,.2f}")
            return True
        else:
            print("❌ 未获取到实时数据")
            return False
            
    except Exception as e:
        print(f"❌ 实时数据获取失败: {e}")
        import traceback
        print(traceback.format_exc())
        return False

def test_single_market_analysis():
    """测试单次市场分析"""
    print("\\n" + "="*60)
    print("测试单次市场分析")
    print("="*60)
    
    try:
        engine = PaperTradingEngine(initial_capital=10000.0)
        
        print("执行单次市场分析...")
        engine.analyze_market_and_trade()
        
        print("✅ 市场分析完成")
        
        # 检查是否生成了信号
        if engine.signals_history:
            latest_signal = engine.signals_history[-1]
            print(f"最新信号: {latest_signal['signal_type']}")
            print(f"信号强度: {latest_signal['signal_strength']:.3f}")
            print(f"置信度: {latest_signal['confidence']:.3f}")
        
        # 检查投资组合状态
        if engine.portfolio_history:
            latest_portfolio = engine.portfolio_history[-1]
            print(f"投资组合价值: ${latest_portfolio['total_value']:,.2f}")
        
        return True
        
    except Exception as e:
        print(f"❌ 市场分析失败: {e}")
        import traceback
        print(traceback.format_exc())
        return False

def test_short_run():
    """测试短时间运行"""
    print("\\n" + "="*60)
    print("测试短时间运行（30秒）")
    print("="*60)
    
    try:
        engine = PaperTradingEngine(
            strategy_mode="aggressive",  # 使用更激进的模式增加信号生成可能性
            initial_capital=10000.0
        )
        
        print("开始短时间运行测试...")
        
        # 执行几次分析
        for i in range(3):
            print(f"\\n--- 第 {i+1} 次分析 ---")
            engine.analyze_market_and_trade()
            
            if i < 2:  # 最后一次不等待
                print("等待10秒...")
                time.sleep(10)
        
        # 生成状态报告
        print("\\n--- 状态报告 ---")
        engine.print_status_report()
        
        # 生成最终报告
        print("\\n--- 最终报告 ---")
        engine.generate_final_report()
        
        print("✅ 短时间运行测试完成")
        return True
        
    except Exception as e:
        print(f"❌ 短时间运行测试失败: {e}")
        import traceback
        print(traceback.format_exc())
        return False

def main():
    """主测试函数"""
    print("开始模拟交易系统测试...")
    
    # 测试1: 初始化
    init_ok = test_paper_trading_initialization()
    
    # 测试2: 实时数据获取
    data_ok = test_realtime_data_fetch()
    
    # 测试3: 市场分析
    analysis_ok = test_single_market_analysis()
    
    # 测试4: 短时间运行（只有前面测试通过才运行）
    run_ok = False
    if init_ok and data_ok and analysis_ok:
        run_ok = test_short_run()
    
    print("\\n" + "="*60)
    print("测试结果总结")
    print("="*60)
    print(f"引擎初始化: {'✅ 通过' if init_ok else '❌ 失败'}")
    print(f"实时数据获取: {'✅ 通过' if data_ok else '❌ 失败'}")
    print(f"市场分析: {'✅ 通过' if analysis_ok else '❌ 失败'}")
    print(f"短时间运行: {'✅ 通过' if run_ok else '❌ 失败'}")
    
    if all([init_ok, data_ok, analysis_ok, run_ok]):
        print("\\n🎉 所有测试通过！模拟交易系统准备就绪。")
        print("\\n💡 使用方法:")
        print("python paper_trading.py --mode standard --capital 10000 --interval 60")
    else:
        print("\\n⚠️  部分测试失败，需要进一步调试。")

if __name__ == "__main__":
    main()