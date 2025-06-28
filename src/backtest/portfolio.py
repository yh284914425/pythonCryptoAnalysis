"""
投资组合管理

管理现金、持仓和交易记录
"""

import pandas as pd
from typing import Dict, Any, Tuple, Optional, List
from datetime import datetime


class Portfolio:
    """投资组合管理器"""
    
    def __init__(self, initial_cash: float = 100000.0):
        """
        初始化投资组合
        
        Args:
            initial_cash: 初始现金
        """
        self.initial_cash = initial_cash
        self.initial_capital = initial_cash  # 为了兼容回测引擎
        self.cash = initial_cash
        self.positions = {}  # 持仓字典
        self.trade_history = []  # 交易历史
        self.current_prices = {}  # 当前价格
        
    def get_total_value(self) -> float:
        """
        获取投资组合总价值
        
        Returns:
            总价值（现金 + 持仓价值）
        """
        positions_value = self.get_positions_value()
        return self.cash + positions_value
    
    def get_positions_value(self) -> float:
        """
        获取所有持仓的总价值
        
        Returns:
            持仓总价值
        """
        total_value = 0.0
        
        for symbol, position in self.positions.items():
            current_price = self.current_prices.get(symbol, position['entry_price'])
            
            if position['side'] == 'long':
                # 多头: 当前价值 = 持仓数量 * 当前价格
                position_value = position['quantity'] * current_price
            else:  # short
                # 空头: 盈亏 = (入场价格 - 当前价格) * 数量
                pnl = (position['entry_price'] - current_price) * position['quantity']
                position_value = position['initial_value'] + pnl
            
            total_value += position_value
        
        return total_value
    
    def open_position(self, 
                     symbol: str,
                     side: str,  # 'long' or 'short'
                     size: float,  # 资金大小
                     entry_price: float,
                     stop_loss: Optional[float] = None,
                     take_profit: Optional[float] = None) -> bool:
        """
        开仓
        
        Args:
            symbol: 交易对
            side: 方向 ('long' 或 'short')
            size: 仓位大小（资金金额）
            entry_price: 入场价格
            stop_loss: 止损价格
            take_profit: 止盈价格
            
        Returns:
            是否成功开仓
        """
        try:
            # 检查是否已有该symbol的持仓
            if symbol in self.positions:
                print(f"已存在 {symbol} 持仓，跳过开仓")
                return False
            
            # 检查资金是否足够
            if self.cash < size:
                print(f"资金不足: 需要 {size:.2f}, 可用 {self.cash:.2f}")
                return False
            
            # 计算数量
            if side == 'long':
                quantity = size / entry_price
            else:  # short
                quantity = size / entry_price
            
            # 扣除资金
            self.cash -= size
            
            # 创建持仓记录
            position = {
                'symbol': symbol,
                'side': side,
                'quantity': quantity,
                'entry_price': entry_price,
                'entry_time': datetime.now(),
                'initial_value': size,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'status': 'open'
            }
            
            self.positions[symbol] = position
            
            # 记录交易
            trade_record = {
                'timestamp': datetime.now(),
                'symbol': symbol,
                'action': 'open',
                'side': side,
                'quantity': quantity,
                'price': entry_price,
                'value': size,
                'type': 'market'
            }
            self.trade_history.append(trade_record)
            
            return True
            
        except Exception as e:
            print(f"开仓失败: {e}")
            return False
    
    def close_position(self, symbol: str, exit_price: float) -> Tuple[bool, float]:
        """
        平仓
        
        Args:
            symbol: 交易对
            exit_price: 平仓价格
            
        Returns:
            (是否成功, 盈亏金额)
        """
        try:
            if symbol not in self.positions:
                return False, 0.0
            
            position = self.positions[symbol]
            
            # 计算盈亏
            if position['side'] == 'long':
                # 多头盈亏 = (平仓价 - 开仓价) * 数量
                pnl = (exit_price - position['entry_price']) * position['quantity']
                return_value = position['quantity'] * exit_price
            else:  # short
                # 空头盈亏 = (开仓价 - 平仓价) * 数量
                pnl = (position['entry_price'] - exit_price) * position['quantity']
                return_value = position['initial_value'] + pnl
            
            # 返还资金
            self.cash += return_value
            
            # 记录交易
            trade_record = {
                'timestamp': datetime.now(),
                'symbol': symbol,
                'action': 'close',
                'side': position['side'],
                'quantity': position['quantity'],
                'price': exit_price,
                'value': return_value,
                'pnl': pnl,
                'type': 'market'
            }
            self.trade_history.append(trade_record)
            
            # 移除持仓
            del self.positions[symbol]
            
            return True, pnl
            
        except Exception as e:
            print(f"平仓失败: {e}")
            return False, 0.0
    
    def update_positions_value(self, current_prices: Dict[str, float]):
        """
        更新持仓的当前价值
        
        Args:
            current_prices: 当前价格字典
        """
        self.current_prices.update(current_prices)
    
    def get_position(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        获取特定持仓信息
        
        Args:
            symbol: 交易对
            
        Returns:
            持仓信息或None
        """
        return self.positions.get(symbol)
    
    def get_position_pnl(self, symbol: str) -> float:
        """
        获取特定持仓的未实现盈亏
        
        Args:
            symbol: 交易对
            
        Returns:
            未实现盈亏
        """
        if symbol not in self.positions:
            return 0.0
        
        position = self.positions[symbol]
        current_price = self.current_prices.get(symbol, position['entry_price'])
        
        if position['side'] == 'long':
            pnl = (current_price - position['entry_price']) * position['quantity']
        else:  # short
            pnl = (position['entry_price'] - current_price) * position['quantity']
        
        return pnl
    
    def get_portfolio_summary(self) -> Dict[str, Any]:
        """
        获取投资组合摘要
        
        Returns:
            投资组合摘要
        """
        total_value = self.get_total_value()
        positions_value = self.get_positions_value()
        
        # 计算总收益率
        total_return = (total_value - self.initial_cash) / self.initial_cash
        
        # 持仓详情
        position_details = []
        for symbol, position in self.positions.items():
            current_price = self.current_prices.get(symbol, position['entry_price'])
            pnl = self.get_position_pnl(symbol)
            pnl_pct = pnl / position['initial_value'] * 100
            
            position_details.append({
                'symbol': symbol,
                'side': position['side'],
                'quantity': position['quantity'],
                'entry_price': position['entry_price'],
                'current_price': current_price,
                'initial_value': position['initial_value'],
                'current_value': position['quantity'] * current_price if position['side'] == 'long' 
                               else position['initial_value'] + pnl,
                'pnl': pnl,
                'pnl_pct': pnl_pct
            })
        
        return {
            'total_value': total_value,
            'cash': self.cash,
            'positions_value': positions_value,
            'total_return': total_return,
            'total_return_pct': total_return * 100,
            'num_positions': len(self.positions),
            'position_details': position_details,
            'cash_utilization': (self.initial_cash - self.cash) / self.initial_cash * 100
        }
    
    def get_trade_history(self) -> List[Dict[str, Any]]:
        """
        获取交易历史
        
        Returns:
            交易历史列表
        """
        return self.trade_history.copy()
    
    def reset(self):
        """重置投资组合到初始状态"""
        self.cash = self.initial_cash
        self.positions = {}
        self.trade_history = []
        self.current_prices = {}