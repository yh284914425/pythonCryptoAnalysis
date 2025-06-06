import pandas as pd
import numpy as np
from datetime import datetime, timedelta


class RiskManager:
    """
    风险管理器，负责止损设置、仓位管理和风险控制
    """
    def __init__(self, config):
        """
        初始化风险管理器
        :param config: 策略配置对象
        """
        self.config = config
        self.risk_config = config.risk
        self.consecutive_losses = 0
        self.last_trade_result = None
        self.trading_paused_until = None
        self.positions = {}  # 当前持仓
        self.total_exposure = 0  # 总仓位
    
    def calculate_stop_loss(self, entry_price, atr, asset_type):
        """
        计算动态止损位
        :param entry_price: 入场价格
        :param atr: ATR值
        :param asset_type: 资产类型，可选 "BTC", "ETH", "ALT"
        :return: 止损价格
        """
        # 获取对应资产的ATR乘数
        multiplier = self.risk_config["stop_loss_multiplier"].get(asset_type, 2.0)
        
        # 计算止损价格
        stop_loss_price = entry_price - (atr * multiplier)
        
        return stop_loss_price
    
    def calculate_position_size(self, signal_strength, atr_value, account_balance, asset_type="BTC"):
        """
        计算建议仓位大小
        :param signal_strength: 信号强度 (0-1)
        :param atr_value: 当前ATR值
        :param account_balance: 账户余额
        :param asset_type: 资产类型
        :return: 建议仓位大小 (占账户余额的百分比)
        """
        # 检查是否暂停交易
        if self.is_trading_paused():
            return 0
        
        # 基础仓位 = 最大单仓 × 信号强度
        base_position = self.risk_config["max_single_position"] * signal_strength
        
        # 波动性调整因子 (ATR越高，仓位越小)
        volatility_factor = 1.0
        if atr_value > 0:
            # 假设ATR的历史平均值为current_atr的一半
            avg_atr = atr_value * 0.5
            volatility_ratio = avg_atr / atr_value
            volatility_factor = min(max(volatility_ratio, 0.5), 1.5)
        
        # 连续止损调整
        loss_factor = 1.0
        if self.consecutive_losses > 0:
            loss_factor = max(0.5, 1 - (self.consecutive_losses * 0.25))
        
        # 计算最终仓位
        position_size = base_position * volatility_factor * loss_factor * self.risk_config["position_scaling"]
        
        # 确保不超过最大单仓限制
        position_size = min(position_size, self.risk_config["max_single_position"])
        
        # 确保总仓位不超过限制
        remaining_capacity = max(0, self.risk_config["max_total_position"] - self.total_exposure)
        position_size = min(position_size, remaining_capacity)
        
        # 转换为金额
        position_amount = account_balance * position_size
        
        return position_amount
    
    def update_risk_status(self, trade_result):
        """
        更新风险状态
        :param trade_result: 交易结果字典，包含 "profit_loss", "type" 等字段
        """
        self.last_trade_result = trade_result
        
        # 更新连续止损计数
        if trade_result["type"] == "stop_loss":
            self.consecutive_losses += 1
            
            # 检查是否需要暂停交易
            if self.consecutive_losses >= self.risk_config["consecutive_loss_threshold"]:
                self.pause_trading(days=7)  # 暂停交易7天
                print(f"⚠️ 警告: 连续{self.consecutive_losses}次止损，暂停交易7天")
        else:
            # 盈利交易，重置连续止损计数
            self.consecutive_losses = 0
    
    def pause_trading(self, days=7):
        """
        暂停交易一段时间
        :param days: 暂停天数
        """
        self.trading_paused_until = datetime.now() + timedelta(days=days)
    
    def is_trading_paused(self):
        """
        检查是否处于交易暂停状态
        :return: 布尔值
        """
        if self.trading_paused_until is None:
            return False
        
        return datetime.now() < self.trading_paused_until
    
    def add_position(self, symbol, amount, entry_price, stop_loss):
        """
        添加新仓位
        :param symbol: 交易对符号
        :param amount: 仓位金额
        :param entry_price: 入场价格
        :param stop_loss: 止损价格
        :return: 仓位ID
        """
        position_id = f"{symbol}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        self.positions[position_id] = {
            "symbol": symbol,
            "amount": amount,
            "entry_price": entry_price,
            "stop_loss": stop_loss,
            "entry_time": datetime.now(),
            "status": "active"
        }
        
        # 更新总仓位
        self.total_exposure += amount / self.get_account_balance()
        
        return position_id
    
    def update_position(self, position_id, current_price=None, new_stop_loss=None):
        """
        更新仓位信息
        :param position_id: 仓位ID
        :param current_price: 当前价格
        :param new_stop_loss: 新的止损价格
        :return: 更新后的仓位信息
        """
        if position_id not in self.positions:
            return None
        
        position = self.positions[position_id]
        
        if current_price is not None:
            position["current_price"] = current_price
            position["profit_loss"] = (current_price - position["entry_price"]) / position["entry_price"]
        
        if new_stop_loss is not None:
            position["stop_loss"] = new_stop_loss
        
        return position
    
    def close_position(self, position_id, exit_price, reason="manual"):
        """
        平仓
        :param position_id: 仓位ID
        :param exit_price: 出场价格
        :param reason: 平仓原因
        :return: 平仓结果
        """
        if position_id not in self.positions:
            return None
        
        position = self.positions[position_id]
        position["exit_price"] = exit_price
        position["exit_time"] = datetime.now()
        position["status"] = "closed"
        position["close_reason"] = reason
        position["profit_loss"] = (exit_price - position["entry_price"]) / position["entry_price"]
        
        # 更新总仓位
        self.total_exposure -= position["amount"] / self.get_account_balance()
        
        # 如果是止损，更新风险状态
        if reason == "stop_loss":
            self.update_risk_status({
                "type": "stop_loss",
                "profit_loss": position["profit_loss"]
            })
        
        return position
    
    def check_stop_losses(self, current_prices):
        """
        检查所有仓位的止损条件
        :param current_prices: 当前价格字典，键为交易对符号
        :return: 触发止损的仓位列表
        """
        triggered = []
        
        for position_id, position in self.positions.items():
            if position["status"] != "active":
                continue
            
            symbol = position["symbol"]
            if symbol not in current_prices:
                continue
            
            current_price = current_prices[symbol]
            
            # 检查是否触发止损
            if current_price <= position["stop_loss"]:
                triggered.append({
                    "position_id": position_id,
                    "symbol": symbol,
                    "entry_price": position["entry_price"],
                    "stop_loss": position["stop_loss"],
                    "current_price": current_price
                })
        
        return triggered
    
    def get_account_balance(self):
        """
        获取账户余额（示例方法，实际应从交易所API获取）
        :return: 账户余额
        """
        # 这里应该实现从交易所获取余额的逻辑
        # 示例返回固定值
        return 10000.0
    
    def get_risk_metrics(self):
        """
        获取风险指标
        :return: 风险指标字典
        """
        active_positions = sum(1 for p in self.positions.values() if p["status"] == "active")
        closed_positions = sum(1 for p in self.positions.values() if p["status"] == "closed")
        
        # 计算胜率
        if closed_positions > 0:
            winning_trades = sum(1 for p in self.positions.values() 
                               if p["status"] == "closed" and p.get("profit_loss", 0) > 0)
            win_rate = winning_trades / closed_positions
        else:
            win_rate = 0
        
        return {
            "total_exposure": self.total_exposure,
            "active_positions": active_positions,
            "consecutive_losses": self.consecutive_losses,
            "trading_paused": self.is_trading_paused(),
            "win_rate": win_rate
        }


if __name__ == "__main__":
    # 测试代码
    from src.strategies.config import create_strategy_config
    
    # 创建配置和风险管理器
    config = create_strategy_config("standard")
    risk_manager = RiskManager(config)
    
    # 测试仓位计算
    account_balance = 10000
    signal_strength = 0.8
    atr_value = 500  # BTC的ATR值示例
    
    position_size = risk_manager.calculate_position_size(
        signal_strength, atr_value, account_balance, "BTC"
    )
    
    print(f"账户余额: ${account_balance}")
    print(f"信号强度: {signal_strength}")
    print(f"ATR值: {atr_value}")
    print(f"建议仓位: ${position_size:.2f} ({position_size/account_balance*100:.1f}%)")
    
    # 测试止损计算
    entry_price = 50000  # BTC入场价格示例
    stop_loss = risk_manager.calculate_stop_loss(entry_price, atr_value, "BTC")
    print(f"\n入场价格: ${entry_price}")
    print(f"止损价格: ${stop_loss} ({(stop_loss-entry_price)/entry_price*100:.1f}%)")
    
    # 测试添加仓位
    position_id = risk_manager.add_position("BTCUSDT", position_size, entry_price, stop_loss)
    print(f"\n添加仓位: {position_id}")
    print(f"当前总仓位: {risk_manager.total_exposure*100:.1f}%")
    
    # 测试风险指标
    metrics = risk_manager.get_risk_metrics()
    print(f"\n风险指标:")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"  {key}: {value*100:.1f}%")
        else:
            print(f"  {key}: {value}") 