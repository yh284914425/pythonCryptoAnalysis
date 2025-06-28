"""
重构后代码使用示例

展示如何使用新的模块化架构进行加密货币交易分析
"""

import pandas as pd
import numpy as np
import sys
sys.path.append('/Users/sheng/Desktop/code/crypto')

from src.indicators import calculate_macd, calculate_kdj
from src.analysis import DivergenceAnalyzer, PatternDetector
from src.strategies import create_mtf_strategy
from src.backtest import BacktestEngine

def load_sample_data():
    """加载示例数据"""
    # 创建模拟的多时间框架数据
    np.random.seed(42)
    
    # 基础价格走势
    base_price = 100
    trend = np.cumsum(np.random.randn(1000) * 0.01) + base_price
    
    def create_timeframe_data(periods, sample_rate=1):
        indices = np.arange(0, len(trend), sample_rate)[:periods]
        prices = trend[indices]
        
        return pd.DataFrame({
            '开盘价': prices + np.random.randn(len(prices)) * 0.1,
            '最高价': prices + np.random.randn(len(prices)) * 0.1 + 0.5,
            '最低价': prices + np.random.randn(len(prices)) * 0.1 - 0.5,
            '收盘价': prices,
            '成交量': np.random.randint(1000, 5000, len(prices))
        })
    
    # 生成多时间框架数据
    data = {
        '1h': create_timeframe_data(500, 1),     # 1小时数据
        '4h': create_timeframe_data(125, 4),     # 4小时数据
        '1d': create_timeframe_data(50, 20)      # 日线数据
    }
    
    return data

def demonstrate_indicators():
    """演示指标计算模块"""
    print("=== 1. 指标计算模块演示 ===")
    
    # 加载数据
    data = load_sample_data()
    df = data['1h'].head(100)  # 使用1小时数据的前100条
    
    # 计算MACD
    print("计算MACD指标...")
    df_with_macd = calculate_macd(df)
    print(f"MACD列: {[col for col in df_with_macd.columns if 'macd' in col]}")
    
    # 计算KDJ
    print("计算KDJ指标...")
    df_with_kdj = calculate_kdj(df_with_macd)
    print(f"KDJ列: {[col for col in df_with_kdj.columns if 'kdj' in col]}")
    
    return df_with_kdj

def demonstrate_analysis():
    """演示分析模块"""
    print("\n=== 2. 分析模块演示 ===")
    
    # 获取包含指标的数据
    df = demonstrate_indicators()
    
    # 背离分析
    print("执行背离分析...")
    divergence_analyzer = DivergenceAnalyzer()
    
    # MACD背离
    macd_divergences = divergence_analyzer.analyze_macd_divergence(df)
    print(f"MACD背离 - 看涨: {len(macd_divergences['bullish'])}, 看跌: {len(macd_divergences['bearish'])}")
    
    # KDJ背离（使用原有精确方法）
    kdj_divergences = divergence_analyzer.analyze_kdj_divergence(df)
    print(f"KDJ背离 - 看涨: {len(kdj_divergences['bullish'])}, 看跌: {len(kdj_divergences['bearish'])}")
    
    # 模式检测
    print("执行模式检测...")
    pattern_detector = PatternDetector()
    
    # MACD金叉死叉
    macd_signals = pattern_detector.detect_macd_signals(df)
    print(f"MACD信号 - 金叉: {len(macd_signals['golden_crosses'])}, 死叉: {len(macd_signals['death_crosses'])}")
    
    # KDJ信号
    kdj_signals = pattern_detector.detect_kdj_signals(df)
    print(f"KDJ K/D金叉: {len(kdj_signals['kd_crosses']['golden'])}")
    print(f"KDJ超买: {len(kdj_signals['overbought_oversold']['overbought'])}")
    print(f"KDJ超卖: {len(kdj_signals['overbought_oversold']['oversold'])}")

def demonstrate_strategy():
    """演示策略模块"""
    print("\n=== 3. 策略模块演示 ===")
    
    # 加载多时间框架数据
    market_data = load_sample_data()
    
    # 创建策略
    print("创建多时间框架背离策略...")
    strategy = create_mtf_strategy("standard")
    print(f"策略模式: {strategy.config.mode}")
    print(f"信号阈值: {strategy.config.get_signal_threshold()} 个指标")
    
    # 市场分析
    print("执行市场分析...")
    analysis_result = strategy.analyze_market(market_data)
    
    print(f"分析结果:")
    print(f"  信号类型: {analysis_result.get('signal_type', 'N/A')}")
    print(f"  信号强度: {analysis_result.get('signal_strength', 0):.3f}")
    print(f"  置信度: {analysis_result.get('confidence', 0):.3f}")
    
    # 生成交易信号
    trading_signal = strategy.generate_trading_signal(analysis_result)
    print(f"交易信号: {trading_signal.get('action', 'hold')}")
    if trading_signal.get('reason'):
        print(f"原因: {trading_signal['reason']}")

def demonstrate_backtest():
    """演示回测模块"""
    print("\n=== 4. 回测模块演示 ===")
    
    # 创建回测引擎
    print("创建回测引擎...")
    backtest_engine = BacktestEngine(
        initial_capital=100000.0,
        commission=0.001,
        slippage=0.0005
    )
    
    # 创建策略
    strategy = create_mtf_strategy("conservative")  # 使用保守模式
    
    # 加载数据
    market_data = load_sample_data()
    
    print("执行快速回测...")
    try:
        # 运行短期回测
        results = backtest_engine.run_backtest(
            strategy=strategy,
            market_data=market_data
        )
        
        if 'error' in results:
            print(f"回测出错: {results['error']}")
        else:
            summary = results['summary']
            print(f"回测结果:")
            print(f"  初始资金: ${summary['initial_capital']:,.2f}")
            print(f"  最终权益: ${summary['final_equity']:,.2f}")
            print(f"  总收益率: {summary['total_return_pct']:.2f}%")
            print(f"  总交易数: {summary['total_trades']}")
            
            # 性能指标
            if 'performance_metrics' in results:
                metrics = results['performance_metrics']
                print(f"  夏普比率: {metrics.get('sharpe_ratio', 0):.3f}")
                print(f"  最大回撤: {metrics.get('max_drawdown', 0)*100:.2f}%")
                
    except Exception as e:
        print(f"回测失败: {e}")

def demonstrate_modular_usage():
    """演示模块化使用"""
    print("\n=== 5. 模块化使用演示 ===")
    
    # 独立使用指标模块
    print("独立使用指标计算...")
    data = load_sample_data()['1h'].head(50)
    
    # 只计算MACD
    macd_data = calculate_macd(data)
    current_macd = macd_data['macd'].iloc[-1]
    current_signal = macd_data['macd_signal'].iloc[-1]
    
    print(f"当前MACD: {current_macd:.4f}")
    print(f"当前信号线: {current_signal:.4f}")
    print(f"MACD状态: {'金叉' if current_macd > current_signal else '死叉'}")
    
    # 只使用分析模块
    print("独立使用分析模块...")
    kdj_data = calculate_kdj(macd_data)
    
    analyzer = DivergenceAnalyzer()
    recent_divergences = analyzer.get_recent_divergences(
        analyzer.analyze_kdj_divergence(kdj_data)['bullish'], 
        window=20
    )
    print(f"最近的KDJ看涨背离: {len(recent_divergences)} 个")

if __name__ == "__main__":
    print("🚀 重构后的加密货币交易策略系统演示")
    print("=" * 60)
    
    try:
        # 依次演示各个模块
        demonstrate_indicators()
        demonstrate_analysis() 
        demonstrate_strategy()
        demonstrate_backtest()
        demonstrate_modular_usage()
        
        print("\n" + "=" * 60)
        print("✅ 所有模块演示完成！")
        print("\n📊 重构成果:")
        print("  ✓ indicators - 纯指标计算层")
        print("  ✓ analysis - 分析引擎层") 
        print("  ✓ strategies - 策略决策层")
        print("  ✓ backtest - 回测与执行层")
        print("\n🎯 实现了完全的关注点分离和模块化设计")
        
    except Exception as e:
        print(f"❌ 演示过程中出错: {e}")
        import traceback
        traceback.print_exc()