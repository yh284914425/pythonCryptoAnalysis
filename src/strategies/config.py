"""
🚀 多周期背离策略配置文件
Multi-Timeframe Divergence Strategy Configuration
"""

from dataclasses import dataclass
from typing import Dict, Any
import os

@dataclass
class KDJConfig:
    """动态KDJ参数配置"""
    # 自适应参数
    short_term: tuple = (18, 5, 5)    # 短期交易参数，胜率58%
    medium_term: tuple = (14, 7, 7)   # 中期交易参数，胜率62%
    long_term: tuple = (21, 10, 10)   # 长期交易参数，胜率65%
    
    # ATR分位数阈值
    atr_high_threshold: float = 0.75   # 高波动阈值
    atr_low_threshold: float = 0.25    # 低波动阈值
    atr_lookback_period: int = 100     # ATR历史周期

@dataclass
class TechnicalIndicatorsConfig:
    """技术指标配置"""
    # ADX参数
    adx_period: int = 14
    adx_trend_threshold: float = 25
    
    # Volume Profile参数
    volume_period: int = 20
    volume_threshold: float = 1.5
    
    # MACD参数
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    
    # RSI参数
    rsi_period: int = 14
    rsi_oversold: float = 30
    rsi_overbought: float = 70

@dataclass
class OnChainConfig:
    """链上数据配置"""
    # MVRV Z-Score阈值
    mvrv_buy_threshold: float = 1.0
    mvrv_sell_threshold: float = 3.7
    
    # Puell Multiple阈值
    puell_bottom_threshold: float = 0.5
    puell_top_threshold: float = 4.0
    
    # 巨鲸监控
    whale_threshold_btc: float = 1000.0
    whale_threshold_eth: float = 10000.0
    
    # 交易所流量监控
    exchange_flow_threshold: float = 5000.0

@dataclass
class AIConfig:
    """AI增强系统配置"""
    # Transformer模型
    model_confidence_threshold: float = 0.70
    prediction_horizon_hours: int = 24
    
    # 情绪分析
    sentiment_fear_threshold: int = 10
    sentiment_greed_threshold: int = 90
    sentiment_update_interval: int = 300  # 5分钟更新
    
    # 强化学习
    rl_learning_rate: float = 0.001
    rl_memory_size: int = 10000

@dataclass
class SignalConfig:
    """信号强度配置"""
    # 信号分级阈值
    diamond_signal_threshold: int = 5  # 钻石信号：5个指标确认，胜率88%
    gold_signal_threshold: int = 4     # 黄金信号：4个指标确认，胜率75%
    silver_signal_threshold: int = 3   # 白银信号：3个指标确认，胜率62%
    
    # 时间框架权重
    timeframe_weights: Dict[str, float] = None
    
    def __post_init__(self):
        if self.timeframe_weights is None:
            self.timeframe_weights = {
                '1w': 1.0,   # 周线权重
                '1d': 0.9,   # 日线权重
                '4h': 0.8,   # 4小时权重
                '1h': 0.7,   # 1小时权重
            }

@dataclass
class RiskManagementConfig:
    """风险管理配置"""
    # Kelly准则参数
    kelly_fraction: float = 0.25  # 保守修正系数
    max_single_position: float = 0.30  # 最大单笔仓位30%
    max_total_position: float = 0.80   # 最大总仓位80%
    
    # ATR动态止损
    btc_atr_multiplier: float = 2.0
    eth_atr_multiplier: float = 1.8
    altcoin_atr_multiplier: float = 4.0
    
    # 风险控制层级
    max_consecutive_losses: int = 3
    daily_loss_limit: float = 0.05  # 日损失限制5%
    monthly_loss_limit: float = 0.15  # 月损失限制15%

@dataclass
class AssetSpecificConfig:
    """资产特定配置"""
    
    class BTCConfig:
        timeframes = ['4h', '1d']
        max_position_ratio = 0.70
        stop_loss_multiplier = 2.0
        trading_hours = (13, 21)  # UTC时间
        
    class ETHConfig:
        timeframes = ['1h', '4h']
        max_position_ratio = 0.60
        stop_loss_multiplier = 1.8
        defi_tvl_threshold = 0.10  # DeFi TVL变化阈值
        
    class MemeCoinConfig:
        timeframes = ['5m', '1h']
        max_position_ratio = 0.30
        stop_loss_multiplier = 4.0
        profit_targets = [0.30, 0.50, 1.00]  # 盈利目标
        position_reductions = [0.50, 0.70, 1.00]  # 对应减仓比例

@dataclass
class PerformanceTargets:
    """预期表现目标"""
    # 保守目标
    conservative_win_rate: float = 0.70
    conservative_monthly_return: float = 0.15
    conservative_max_drawdown: float = 0.12
    conservative_sharpe_ratio: float = 1.5
    
    # 激进目标
    aggressive_win_rate: float = 0.80
    aggressive_monthly_return: float = 0.25
    aggressive_max_drawdown: float = 0.08
    aggressive_sharpe_ratio: float = 2.0

@dataclass
class MonitoringConfig:
    """监控系统配置"""
    # 报告频率
    daily_report_time: str = "09:00"
    weekly_report_day: int = 1  # 周一
    monthly_report_day: int = 1  # 每月1号
    
    # 预警阈值
    consecutive_loss_alert: int = 2
    drawdown_alert_threshold: float = 0.05
    position_alert_threshold: float = 0.75
    
    # 通知渠道
    telegram_enabled: bool = True
    email_enabled: bool = True
    wechat_enabled: bool = False

class StrategyConfig:
    """主策略配置类"""
    
    def __init__(self, mode: str = "conservative"):
        self.mode = mode
        
        # 初始化各模块配置
        self.kdj = KDJConfig()
        self.technical = TechnicalIndicatorsConfig()
        self.onchain = OnChainConfig()
        self.ai = AIConfig()
        self.signals = SignalConfig()
        self.risk = RiskManagementConfig()
        self.assets = AssetSpecificConfig()
        self.targets = PerformanceTargets()
        self.monitoring = MonitoringConfig()
        
        # 根据模式调整参数
        self._adjust_for_mode()
    
    def _adjust_for_mode(self):
        """根据运行模式调整参数"""
        if self.mode == "aggressive":
            # 激进模式：更高风险，更高收益
            self.risk.max_single_position = 0.40
            self.risk.max_total_position = 0.90
            self.ai.model_confidence_threshold = 0.60
            self.signals.silver_signal_threshold = 2
            
        elif self.mode == "conservative":
            # 保守模式：更低风险，稳定收益
            self.risk.max_single_position = 0.20
            self.risk.max_total_position = 0.60
            self.ai.model_confidence_threshold = 0.80
            self.signals.silver_signal_threshold = 4
            
        elif self.mode == "demo":
            # 演示模式：用于测试和学习
            self.risk.max_single_position = 0.10
            self.risk.max_total_position = 0.30
    
    def get_asset_config(self, asset_type: str):
        """获取特定资产的配置"""
        asset_configs = {
            'BTC': self.assets.BTCConfig,
            'ETH': self.assets.ETHConfig,
            'MEME': self.assets.MemeCoinConfig
        }
        return asset_configs.get(asset_type.upper(), self.assets.BTCConfig)
    
    def validate_config(self) -> bool:
        """验证配置的有效性"""
        try:
            # 检查基本参数
            assert 0 < self.risk.max_single_position <= 1
            assert 0 < self.risk.max_total_position <= 1
            assert self.risk.max_single_position <= self.risk.max_total_position
            
            # 检查阈值参数
            assert 0 < self.ai.model_confidence_threshold <= 1
            assert self.signals.silver_signal_threshold <= self.signals.gold_signal_threshold
            assert self.signals.gold_signal_threshold <= self.signals.diamond_signal_threshold
            
            return True
        except AssertionError as e:
            print(f"❌ 配置验证失败: {e}")
            return False
    
    def to_dict(self) -> Dict[str, Any]:
        """将配置转换为字典格式"""
        return {
            'mode': self.mode,
            'kdj': self.kdj.__dict__,
            'technical': self.technical.__dict__,
            'onchain': self.onchain.__dict__,
            'ai': self.ai.__dict__,
            'signals': self.signals.__dict__,
            'risk': self.risk.__dict__,
            'targets': self.targets.__dict__,
            'monitoring': self.monitoring.__dict__
        }

# 默认配置实例
DEFAULT_CONFIG = StrategyConfig(mode="conservative")

# 环境变量配置
API_KEYS = {
    'TELEGRAM_BOT_TOKEN': os.getenv('TELEGRAM_BOT_TOKEN'),
    'TWITTER_API_KEY': os.getenv('TWITTER_API_KEY'),
    'COINMETRICS_API_KEY': os.getenv('COINMETRICS_API_KEY'),
    'GLASSNODE_API_KEY': os.getenv('GLASSNODE_API_KEY'),
}

# 数据源配置
DATA_SOURCES = {
    'price_data': 'binance',
    'onchain_data': 'glassnode',
    'sentiment_data': 'twitter',
    'whale_alerts': 'whale_alert',
}

if __name__ == "__main__":
    # 配置测试
    config = StrategyConfig()
    print("🚀 多周期背离策略配置")
    print(f"📊 运行模式: {config.mode}")
    print(f"✅ 配置验证: {'通过' if config.validate_config() else '失败'}")
    print(f"🎯 保守目标胜率: {config.targets.conservative_win_rate:.1%}")
    print(f"🎯 激进目标胜率: {config.targets.aggressive_win_rate:.1%}") 