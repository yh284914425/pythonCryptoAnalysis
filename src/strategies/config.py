class StrategyConfig:
    """
    策略配置管理类，用于管理不同交易模式的参数配置
    """
    def __init__(self, mode="standard"):
        """
        初始化策略配置
        :param mode: 策略模式，可选 "conservative"(保守), "standard"(标准), "aggressive"(激进)
        """
        self.mode = mode
        self.technical = self._init_technical_config()
        self.risk = self._init_risk_config()
        self.ai = self._init_ai_config()
    
    def _init_technical_config(self):
        """初始化技术分析相关配置"""
        technical_config = {
            # KDJ参数配置
            "kdj": {
                "high_volatility": {"k": 18, "d": 5, "j": 5},  # 高波动参数
                "medium_volatility": {"k": 14, "d": 7, "j": 7},  # 中波动参数
                "low_volatility": {"k": 21, "d": 10, "j": 10},  # 低波动参数
            },
            # ATR配置
            "atr": {
                "period": 14,  # ATR周期
                "high_threshold": 0.75,  # 高波动阈值(分位数)
                "low_threshold": 0.25,   # 低波动阈值(分位数)
                "lookback": 252,  # 回溯周期(约一年交易日)
            },
            # ADX配置
            "adx": {
                "period": 14,  # ADX周期
                "trending_threshold": 25,  # 趋势市场阈值
                "sideways_threshold": 20,  # 震荡市场阈值
            },
            # 时间框架配置
            "timeframes": {
                "macro": "1d",    # 宏观层(日线)
                "meso": "4h",     # 中观层(4小时)
                "micro": "1h",    # 微观层(1小时)
            },
            # 信号确认配置
            "signal_confirmation": {
                "conservative": 4,  # 保守模式需要的确认指标数
                "standard": 3,      # 标准模式需要的确认指标数
                "aggressive": 2,    # 激进模式需要的确认指标数
            }
        }
        
        return technical_config
    
    def _init_risk_config(self):
        """初始化风险管理相关配置"""
        # 基础配置
        base_config = {
            "stop_loss_multiplier": {
                "BTC": 2.0,   # BTC止损为ATR的2倍
                "ETH": 1.8,   # ETH止损为ATR的1.8倍
                "ALT": 4.0,   # 山寨币止损为ATR的4倍
            },
            "consecutive_loss_threshold": 3,  # 连续止损阈值
            "max_drawdown_threshold": 0.15,   # 最大回撤阈值(15%)
        }
        
        # 根据模式设置不同的仓位参数
        if self.mode == "conservative":
            base_config.update({
                "max_single_position": 0.20,  # 最大单仓20%
                "max_total_position": 0.60,   # 最大总仓60%
                "position_scaling": 0.8,      # 仓位缩放因子
            })
        elif self.mode == "standard":
            base_config.update({
                "max_single_position": 0.30,  # 最大单仓30%
                "max_total_position": 0.80,   # 最大总仓80%
                "position_scaling": 1.0,      # 仓位缩放因子
            })
        elif self.mode == "aggressive":
            base_config.update({
                "max_single_position": 0.40,  # 最大单仓40%
                "max_total_position": 0.90,   # 最大总仓90%
                "position_scaling": 1.2,      # 仓位缩放因子
            })
        else:  # 默认为演示模式
            base_config.update({
                "max_single_position": 0.10,  # 最大单仓10%
                "max_total_position": 0.30,   # 最大总仓30%
                "position_scaling": 0.5,      # 仓位缩放因子
            })
            
        return base_config
    
    def _init_ai_config(self):
        """初始化AI增强相关配置"""
        # 基础AI配置
        base_config = {
            "enabled": True,  # 是否启用AI
            "models": {
                "price_prediction": "transformer",  # 价格预测模型类型
                "sentiment": "bert",                # 情绪分析模型类型
            },
            "data_sources": {
                "twitter": True,   # 是否使用Twitter数据
                "reddit": True,    # 是否使用Reddit数据
                "news": True,      # 是否使用新闻数据
            },
            "update_frequency": 24,  # 模型更新频率(小时)
        }
        
        # 根据模式设置不同的AI确认阈值
        if self.mode == "conservative":
            base_config["model_confidence_threshold"] = 0.80  # 保守模式需要80%的置信度
        elif self.mode == "standard":
            base_config["model_confidence_threshold"] = 0.70  # 标准模式需要70%的置信度
        elif self.mode == "aggressive":
            base_config["model_confidence_threshold"] = 0.60  # 激进模式需要60%的置信度
        else:  # 默认为演示模式
            base_config["model_confidence_threshold"] = 0.90  # 演示模式需要90%的置信度
            
        return base_config
    
    def get_signal_threshold(self):
        """获取当前模式下的信号阈值"""
        return self.technical["signal_confirmation"][self.mode]
    
    def get_kdj_params(self, volatility):
        """
        根据波动性获取KDJ参数
        :param volatility: 波动性类型，可选 "high", "medium", "low"
        :return: KDJ参数字典
        """
        volatility_map = {
            "high": "high_volatility",
            "medium": "medium_volatility",
            "low": "low_volatility"
        }
        key = volatility_map.get(volatility, "medium_volatility")
        return self.technical["kdj"][key]
    
    def __str__(self):
        """打印配置信息"""
        return f"StrategyConfig(mode={self.mode})"


def create_strategy_config(mode="standard"):
    """
    创建策略配置的工厂函数
    :param mode: 策略模式
    :return: 配置对象
    """
    return StrategyConfig(mode=mode)


if __name__ == "__main__":
    # 测试不同模式的配置
    modes = ["conservative", "standard", "aggressive"]
    for mode in modes:
        config = create_strategy_config(mode)
        print(f"\n{mode.capitalize()} 模式配置:")
        print(f"  最大单仓: {config.risk['max_single_position']*100}%")
        print(f"  最大总仓: {config.risk['max_total_position']*100}%")
        print(f"  AI置信度: {config.ai['model_confidence_threshold']*100}%")
        print(f"  信号阈值: {config.get_signal_threshold()}个指标") 