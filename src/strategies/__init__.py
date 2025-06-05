"""
🚀 多周期背离策略模块
Multi-Timeframe Divergence Trading Strategy Module

包含基于背离分析的交易策略和风险管理工具
"""

# 配置模块
from .config import (
    StrategyConfig, 
    DEFAULT_CONFIG,
    KDJConfig,
    TechnicalIndicatorsConfig,
    OnChainConfig,
    AIConfig,
    SignalConfig,
    RiskManagementConfig,
    AssetSpecificConfig,
    PerformanceTargets,
    MonitoringConfig
)

# 技术指标模块
from .technical_indicators import (
    DynamicKDJIndicator,
    TechnicalIndicators,
    MultiTimeframeAnalysis
)

# 链上数据模块
from .onchain_indicators import (
    OnChainDataProvider,
    GlassnodeProvider,
    MVRVZScore,
    PuellMultiple,
    WhaleActivityMonitor,
    NetworkValueIndicators,
    OnChainSignalAggregator,
    HashRibbons
)

# AI增强模块
from .ai_enhanced import (
    TransformerPricePredictor,
    SentimentAnalyzer,
    ReinforcementLearningOptimizer,
    AISignalFilter
)

# 风险管理模块
from .risk_management import (
    KellyCalculator,
    DynamicPositionSizer,
    ATRStopLossCalculator,
    MultiLayerRiskControl,
    RiskManager
)

# 主策略模块
from .main_strategy import (
    MultiTimeframeDivergenceStrategy,
    create_strategy
)

# 原有模块保持兼容性
from .strategy_analysis import *
from .exit_strategy_analysis import *

__all__ = [
    # 配置类
    'StrategyConfig',
    'DEFAULT_CONFIG',
    'KDJConfig',
    'TechnicalIndicatorsConfig',
    'OnChainConfig',
    'AIConfig',
    'SignalConfig',
    'RiskManagementConfig',
    'AssetSpecificConfig',
    'PerformanceTargets',
    'MonitoringConfig',
    
    # 技术指标类
    'DynamicKDJIndicator',
    'TechnicalIndicators',
    'MultiTimeframeAnalysis',
    
    # 链上数据类
    'OnChainDataProvider',
    'GlassnodeProvider',
    'MVRVZScore',
    'PuellMultiple',
    'WhaleActivityMonitor',
    'NetworkValueIndicators',
    'OnChainSignalAggregator',
    'HashRibbons',
    
    # AI增强类
    'TransformerPricePredictor',
    'SentimentAnalyzer',
    'ReinforcementLearningOptimizer',
    'AISignalFilter',
    
    # 风险管理类
    'KellyCalculator',
    'DynamicPositionSizer',
    'ATRStopLossCalculator',
    'MultiLayerRiskControl',
    'RiskManager',
    
    # 主策略类
    'MultiTimeframeDivergenceStrategy',
    'create_strategy'
]

# 版本信息
__version__ = "1.0.0"
__author__ = "Crypto Strategy Team"
__description__ = "Advanced Multi-Timeframe Divergence Trading Strategy with AI Enhancement"

# 快速使用示例
def quick_start_example():
    """
    快速开始示例
    
    Returns:
        策略实例和示例配置
    """
    print("🚀 多周期背离策略快速开始")
    print("=" * 50)
    
    # 1. 创建策略实例
    strategy = create_strategy("conservative")
    
    # 2. 显示配置信息
    print(f"策略模式: {strategy.config.mode}")
    print(f"最大单仓位: {strategy.config.risk.max_single_position:.1%}")
    print(f"最大总仓位: {strategy.config.risk.max_total_position:.1%}")
    print(f"钻石信号阈值: {strategy.config.signals.diamond_signal_threshold}个指标确认")
    
    # 3. 显示各模块状态
    print("\n📊 已加载模块:")
    print("✅ 动态KDJ指标系统")
    print("✅ 多时间框架分析")
    print("✅ 链上数据聚合器")
    print("✅ AI信号过滤器")
    print("✅ 多层风险控制")
    
    # 4. 显示预期表现目标
    targets = strategy.config.targets
    print(f"\n🎯 预期表现目标:")
    print(f"保守胜率: {targets.conservative_win_rate:.1%}")
    print(f"保守月收益: {targets.conservative_monthly_return:.1%}")
    print(f"最大回撤: {targets.conservative_max_drawdown:.1%}")
    print(f"夏普比率: {targets.conservative_sharpe_ratio:.1f}")
    
    print("\n💡 使用提示:")
    print("1. strategy.analyze_market(market_data) - 执行市场分析")
    print("2. strategy.execute_trade(decision) - 执行交易")
    print("3. strategy.monitor_positions(market_data) - 监控持仓")
    print("4. strategy.get_strategy_status() - 获取策略状态")
    
    return strategy

if __name__ == "__main__":
    # 运行快速开始示例
    example_strategy = quick_start_example() 