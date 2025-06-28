"""
策略基类

定义所有交易策略应遵循的抽象接口，规范策略的输入和输出格式
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
import pandas as pd
from datetime import datetime
from .config import StrategyConfig


class BaseStrategy(ABC):
    """交易策略抽象基类"""
    
    def __init__(self, config: Optional[StrategyConfig] = None):
        """
        初始化策略
        
        Args:
            config: 策略配置对象
        """
        self.config = config or StrategyConfig()
        self.positions = {}  # 当前持仓
        self.trade_history = []  # 交易历史
        self.last_analysis = {}  # 最后一次分析结果
        
    @abstractmethod
    def analyze_market(self, market_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        分析市场数据并生成交易决策
        
        Args:
            market_data: 市场数据字典，键为时间框架，值为K线DataFrame
            
        Returns:
            分析结果字典，必须包含以下键：
            - signal_type: 信号类型 ('buy', 'sell', 'neutral')
            - signal_strength: 信号强度 (0.0-1.0)
            - confidence: 置信度 (0.0-1.0)
            - analysis_details: 详细分析信息
        """
        pass
    
    @abstractmethod
    def generate_trading_signal(self, analysis_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        基于分析结果生成具体的交易信号
        
        Args:
            analysis_result: 市场分析结果
            
        Returns:
            交易信号字典，必须包含：
            - action: 交易动作 ('buy', 'sell', 'hold', 'close')
            - symbol: 交易对
            - entry_price: 入场价格
            - stop_loss: 止损价格 
            - take_profit: 止盈价格（可选）
            - position_size: 仓位大小
        """
        pass
    
    def validate_signal(self, signal: Dict[str, Any]) -> bool:
        """
        验证交易信号是否有效
        
        Args:
            signal: 交易信号
            
        Returns:
            信号是否有效
        """
        required_fields = ['action', 'symbol', 'entry_price', 'position_size']
        
        # 检查必需字段
        for field in required_fields:
            if field not in signal:
                return False
        
        # 检查数值有效性
        if signal['position_size'] <= 0:
            return False
            
        if signal['entry_price'] <= 0:
            return False
            
        return True
    
    def calculate_position_size(self, signal_strength: float, 
                               risk_amount: float, 
                               entry_price: float, 
                               stop_loss: float) -> float:
        """
        计算仓位大小
        
        Args:
            signal_strength: 信号强度
            risk_amount: 风险金额
            entry_price: 入场价格
            stop_loss: 止损价格
            
        Returns:
            建议仓位大小
        """
        # 防御性检查：输入参数验证
        if stop_loss <= 0 or entry_price <= 0 or risk_amount <= 0 or signal_strength <= 0:
            return 0.0
        
        # 计算风险百分比
        risk_pct = abs(entry_price - stop_loss) / entry_price
        
        # 防御性检查：避免除零错误
        if risk_pct == 0 or risk_pct < 1e-6:  # 防止极小的风险百分比
            return 0.0
        
        # 基础仓位大小 = 风险金额 / 单位风险
        try:
            base_size = risk_amount / risk_pct
        except ZeroDivisionError:
            return 0.0
        
        # 根据信号强度调整
        adjusted_size = base_size * signal_strength
        
        # 应用配置中的仓位限制
        account_balance = self.get_account_balance()
        if account_balance <= 0:
            return 0.0
        
        max_position = self.config.risk['max_single_position'] * account_balance
        
        return min(adjusted_size, max_position)
    
    def get_account_balance(self) -> float:
        """
        获取账户余额（需要在具体策略中实现）
        
        Returns:
            账户余额
        """
        return 100000.0  # 默认余额
    
    def add_position(self, signal: Dict[str, Any]) -> str:
        """
        添加新仓位
        
        Args:
            signal: 交易信号
            
        Returns:
            仓位ID
        """
        position_id = f"pos_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        position = {
            'id': position_id,
            'symbol': signal['symbol'],
            'action': signal['action'],
            'entry_price': signal['entry_price'],
            'stop_loss': signal.get('stop_loss'),
            'take_profit': signal.get('take_profit'),
            'position_size': signal['position_size'],
            'entry_time': datetime.now(),
            'status': 'active'
        }
        
        self.positions[position_id] = position
        return position_id
    
    def close_position(self, position_id: str, 
                      exit_price: float, 
                      reason: str = 'manual') -> Dict[str, Any]:
        """
        平仓
        
        Args:
            position_id: 仓位ID
            exit_price: 平仓价格
            reason: 平仓原因
            
        Returns:
            平仓结果
        """
        if position_id not in self.positions:
            return {'status': 'error', 'message': '仓位不存在'}
        
        position = self.positions[position_id]
        
        # 计算盈亏
        if position['action'] == 'buy':
            profit_loss = (exit_price - position['entry_price']) / position['entry_price']
        else:  # sell
            profit_loss = (position['entry_price'] - exit_price) / position['entry_price']
        
        # 更新仓位信息
        position.update({
            'exit_price': exit_price,
            'exit_time': datetime.now(),
            'status': 'closed',
            'close_reason': reason,
            'profit_loss': profit_loss
        })
        
        # 移到历史记录
        self.trade_history.append(position)
        del self.positions[position_id]
        
        return {
            'status': 'success',
            'position': position,
            'profit_loss': profit_loss
        }
    
    def get_active_positions(self) -> Dict[str, Dict]:
        """获取活跃仓位"""
        return self.positions.copy()
    
    def get_strategy_performance(self) -> Dict[str, Any]:
        """
        获取策略表现统计
        
        Returns:
            策略表现字典
        """
        if not self.trade_history:
            return {
                'total_trades': 0,
                'win_rate': 0.0,
                'avg_profit': 0.0,
                'total_return': 0.0
            }
        
        total_trades = len(self.trade_history)
        winning_trades = sum(1 for trade in self.trade_history if trade.get('profit_loss', 0) > 0)
        
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        avg_profit = sum(trade.get('profit_loss', 0) for trade in self.trade_history) / total_trades
        total_return = sum(trade.get('profit_loss', 0) for trade in self.trade_history)
        
        return {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': total_trades - winning_trades,
            'win_rate': win_rate,
            'avg_profit': avg_profit,
            'total_return': total_return,
            'active_positions': len(self.positions)
        }
    
    def should_trade(self, signal: Dict[str, Any]) -> bool:
        """
        判断是否应该执行交易
        
        Args:
            signal: 交易信号
            
        Returns:
            是否应该交易
        """
        # 基础验证
        if not self.validate_signal(signal):
            return False
        
        # 检查账户余额
        if self.get_account_balance() < signal['position_size']:
            return False
        
        # 检查最大仓位限制
        total_exposure = sum(pos['position_size'] for pos in self.positions.values())
        max_total = self.config.risk['max_total_position'] * self.get_account_balance()
        
        if total_exposure + signal['position_size'] > max_total:
            return False
        
        return True