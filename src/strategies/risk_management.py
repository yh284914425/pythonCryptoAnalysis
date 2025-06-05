"""
🛡️ 风险管理模块
Risk Management Module

包含Kelly准则、动态仓位管理、多层风险控制等核心风险管理功能
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import logging
from .config import RiskManagementConfig, AssetSpecificConfig

class KellyCalculator:
    """Kelly准则计算器"""
    
    def __init__(self, config: RiskManagementConfig):
        self.config = config
        self.trade_history = []
        
    def calculate_kelly_fraction(self, win_rate: float, avg_win: float, avg_loss: float) -> float:
        """
        计算Kelly分数
        
        Args:
            win_rate: 胜率 (0-1)
            avg_win: 平均盈利
            avg_loss: 平均亏损
            
        Returns:
            Kelly分数
        """
        if avg_loss <= 0:
            return 0.0
            
        # Kelly公式: f = (bp - q) / b
        # 其中 b = 盈亏比, p = 胜率, q = 败率
        win_loss_ratio = avg_win / abs(avg_loss)
        loss_rate = 1 - win_rate
        
        kelly_fraction = (win_rate * win_loss_ratio - loss_rate) / win_loss_ratio
        
        # 应用保守修正系数
        conservative_kelly = kelly_fraction * self.config.kelly_fraction
        
        # 限制在合理范围内
        return max(0, min(conservative_kelly, self.config.max_single_position))
    
    def update_trade_history(self, trade_result: Dict):
        """更新交易历史"""
        self.trade_history.append({
            'timestamp': trade_result.get('timestamp', datetime.now()),
            'profit': trade_result.get('profit', 0),
            'is_win': trade_result.get('profit', 0) > 0,
            'asset': trade_result.get('asset', 'BTC')
        })
        
        # 保留最近100笔交易
        if len(self.trade_history) > 100:
            self.trade_history = self.trade_history[-100:]
    
    def get_current_stats(self, lookback_period: int = 50) -> Dict[str, float]:
        """获取当前交易统计"""
        if len(self.trade_history) < 10:
            # 数据不足时使用默认值
            return {
                'win_rate': 0.65,
                'avg_win': 0.08,
                'avg_loss': -0.04,
                'kelly_fraction': 0.1
            }
        
        recent_trades = self.trade_history[-lookback_period:]
        
        wins = [t['profit'] for t in recent_trades if t['is_win']]
        losses = [t['profit'] for t in recent_trades if not t['is_win']]
        
        win_rate = len(wins) / len(recent_trades)
        avg_win = np.mean(wins) if wins else 0.05
        avg_loss = np.mean(losses) if losses else -0.03
        
        kelly_fraction = self.calculate_kelly_fraction(win_rate, avg_win, avg_loss)
        
        return {
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'kelly_fraction': kelly_fraction,
            'total_trades': len(recent_trades)
        }

class DynamicPositionSizer:
    """动态仓位计算器"""
    
    def __init__(self, config: RiskManagementConfig, kelly_calculator: KellyCalculator):
        self.config = config
        self.kelly_calculator = kelly_calculator
        
    def calculate_position_size(self, 
                              signal_strength: float,
                              entry_price: float,
                              stop_loss_price: float,
                              account_balance: float,
                              asset_type: str = 'BTC') -> Dict[str, float]:
        """
        计算动态仓位大小
        
        Args:
            signal_strength: 信号强度 (0-1)
            entry_price: 入场价格
            stop_loss_price: 止损价格
            account_balance: 账户余额
            asset_type: 资产类型
            
        Returns:
            仓位计算结果
        """
        
        # 1. 获取Kelly分数
        kelly_stats = self.kelly_calculator.get_current_stats()
        base_kelly_fraction = kelly_stats['kelly_fraction']
        
        # 2. 根据信号强度调整
        signal_adjusted_fraction = base_kelly_fraction * signal_strength
        
        # 3. 根据风险调整 (价格距离止损的距离)
        risk_per_unit = abs(entry_price - stop_loss_price) / entry_price
        max_risk_amount = account_balance * self.config.max_single_position
        
        # 基于风险的最大仓位
        max_position_by_risk = max_risk_amount / (risk_per_unit * entry_price)
        
        # 基于Kelly的建议仓位
        kelly_position_value = signal_adjusted_fraction * account_balance
        kelly_position_size = kelly_position_value / entry_price
        
        # 取较小值
        recommended_position = min(kelly_position_size, max_position_by_risk)
        
        # 4. 应用资产特定限制
        asset_limits = self._get_asset_limits(asset_type)
        max_asset_position = account_balance * asset_limits['max_position_ratio'] / entry_price
        
        final_position = min(recommended_position, max_asset_position)
        
        # 5. 确保不超过总仓位限制
        final_position = min(final_position, 
                           account_balance * self.config.max_total_position / entry_price)
        
        return {
            'position_size': final_position,
            'position_value': final_position * entry_price,
            'position_ratio': (final_position * entry_price) / account_balance,
            'kelly_fraction': base_kelly_fraction,
            'signal_adjusted_fraction': signal_adjusted_fraction,
            'risk_per_unit': risk_per_unit,
            'max_loss': final_position * (entry_price - stop_loss_price)
        }
    
    def _get_asset_limits(self, asset_type: str) -> Dict[str, float]:
        """获取资产特定限制"""
        asset_configs = {
            'BTC': {'max_position_ratio': 0.70},
            'ETH': {'max_position_ratio': 0.60},
            'MEME': {'max_position_ratio': 0.30}
        }
        return asset_configs.get(asset_type.upper(), asset_configs['BTC'])

class ATRStopLossCalculator:
    """ATR动态止损计算器"""
    
    def __init__(self, config: RiskManagementConfig):
        self.config = config
        
    def calculate_stop_loss(self, 
                          current_price: float,
                          atr_value: float,
                          position_type: str,
                          asset_type: str = 'BTC') -> Dict[str, float]:
        """
        计算ATR动态止损
        
        Args:
            current_price: 当前价格
            atr_value: ATR值
            position_type: 'long' 或 'short'
            asset_type: 资产类型
            
        Returns:
            止损计算结果
        """
        
        # 获取资产特定的ATR倍数
        atr_multipliers = {
            'BTC': self.config.btc_atr_multiplier,
            'ETH': self.config.eth_atr_multiplier,
            'MEME': self.config.altcoin_atr_multiplier
        }
        
        multiplier = atr_multipliers.get(asset_type.upper(), self.config.btc_atr_multiplier)
        stop_distance = atr_value * multiplier
        
        if position_type.lower() == 'long':
            stop_loss_price = current_price - stop_distance
            profit_target = current_price + (stop_distance * 2)  # 2:1风险回报比
        else:  # short
            stop_loss_price = current_price + stop_distance
            profit_target = current_price - (stop_distance * 2)
        
        return {
            'stop_loss_price': stop_loss_price,
            'profit_target': profit_target,
            'stop_distance': stop_distance,
            'risk_reward_ratio': 2.0,
            'risk_percentage': (stop_distance / current_price) * 100
        }
    
    def update_trailing_stop(self, 
                           current_price: float,
                           entry_price: float,
                           current_stop: float,
                           atr_value: float,
                           position_type: str,
                           asset_type: str = 'BTC') -> float:
        """更新移动止损"""
        
        multiplier = {
            'BTC': self.config.btc_atr_multiplier,
            'ETH': self.config.eth_atr_multiplier,
            'MEME': self.config.altcoin_atr_multiplier
        }.get(asset_type.upper(), self.config.btc_atr_multiplier)
        
        stop_distance = atr_value * multiplier
        
        if position_type.lower() == 'long':
            new_stop = current_price - stop_distance
            # 只在新止损更高时更新（移动止损只能上移）
            return max(current_stop, new_stop)
        else:  # short
            new_stop = current_price + stop_distance
            # 只在新止损更低时更新（空头止损只能下移）
            return min(current_stop, new_stop)

class MultiLayerRiskControl:
    """多层风险控制系统"""
    
    def __init__(self, config: RiskManagementConfig):
        self.config = config
        self.daily_losses = []
        self.consecutive_losses = 0
        self.monthly_pnl = []
        self.emergency_stop = False
        
    def check_pre_trade_risk(self, 
                           proposed_trade: Dict,
                           current_portfolio: Dict,
                           account_balance: float) -> Dict[str, any]:
        """交易前风险检查"""
        
        risk_checks = {
            'approved': True,
            'warnings': [],
            'blocks': [],
            'adjustments': {}
        }
        
        # 1. 单笔仓位检查
        position_ratio = proposed_trade['position_value'] / account_balance
        if position_ratio > self.config.max_single_position:
            risk_checks['blocks'].append(f"单笔仓位超限: {position_ratio:.2%} > {self.config.max_single_position:.2%}")
            risk_checks['approved'] = False
        
        # 2. 总仓位检查
        current_exposure = sum(pos.get('value', 0) for pos in current_portfolio.values())
        total_exposure = (current_exposure + proposed_trade['position_value']) / account_balance
        
        if total_exposure > self.config.max_total_position:
            risk_checks['blocks'].append(f"总仓位超限: {total_exposure:.2%} > {self.config.max_total_position:.2%}")
            risk_checks['approved'] = False
        
        # 3. 连续亏损检查
        if self.consecutive_losses >= self.config.max_consecutive_losses:
            risk_checks['blocks'].append(f"连续亏损超限: {self.consecutive_losses} >= {self.config.max_consecutive_losses}")
            risk_checks['approved'] = False
        
        # 4. 日损失限制检查
        today_loss = self._get_today_loss()
        if today_loss < -self.config.daily_loss_limit:
            risk_checks['blocks'].append(f"日损失超限: {today_loss:.2%} < {-self.config.daily_loss_limit:.2%}")
            risk_checks['approved'] = False
        
        # 5. 月损失限制检查
        month_loss = self._get_month_loss()
        if month_loss < -self.config.monthly_loss_limit:
            risk_checks['blocks'].append(f"月损失超限: {month_loss:.2%} < {-self.config.monthly_loss_limit:.2%}")
            risk_checks['approved'] = False
        
        # 6. 紧急停止检查
        if self.emergency_stop:
            risk_checks['blocks'].append("紧急停止模式激活")
            risk_checks['approved'] = False
        
        return risk_checks
    
    def check_portfolio_risk(self, current_portfolio: Dict, account_balance: float) -> Dict[str, any]:
        """投资组合风险检查"""
        
        portfolio_analysis = {
            'total_exposure': 0,
            'asset_concentration': {},
            'correlation_risk': 'low',
            'liquidity_risk': 'low',
            'recommendations': []
        }
        
        # 计算总敞口
        total_value = sum(pos.get('value', 0) for pos in current_portfolio.values())
        portfolio_analysis['total_exposure'] = total_value / account_balance
        
        # 计算资产集中度
        for asset, position in current_portfolio.items():
            concentration = position.get('value', 0) / account_balance
            portfolio_analysis['asset_concentration'][asset] = concentration
            
            # 检查过度集中
            if concentration > 0.5:
                portfolio_analysis['recommendations'].append(f"{asset}仓位过于集中: {concentration:.2%}")
        
        # 相关性风险分析（简化）
        crypto_assets = [asset for asset in current_portfolio.keys() 
                        if asset in ['BTC', 'ETH', 'ADA', 'DOT']]
        if len(crypto_assets) > 1:
            crypto_exposure = sum(current_portfolio[asset].get('value', 0) 
                                for asset in crypto_assets)
            if crypto_exposure / account_balance > 0.8:
                portfolio_analysis['correlation_risk'] = 'high'
                portfolio_analysis['recommendations'].append("加密货币资产相关性过高，考虑分散投资")
        
        return portfolio_analysis
    
    def update_loss_tracking(self, trade_result: Dict):
        """更新亏损跟踪"""
        profit = trade_result.get('profit', 0)
        
        # 更新连续亏损计数
        if profit < 0:
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0
        
        # 记录日损失
        today = datetime.now().date()
        self.daily_losses.append({
            'date': today,
            'profit': profit
        })
        
        # 清理旧数据（保留30天）
        cutoff_date = today - timedelta(days=30)
        self.daily_losses = [loss for loss in self.daily_losses 
                           if loss['date'] >= cutoff_date]
    
    def _get_today_loss(self) -> float:
        """获取今日亏损"""
        today = datetime.now().date()
        today_losses = [loss['profit'] for loss in self.daily_losses 
                       if loss['date'] == today]
        return sum(today_losses)
    
    def _get_month_loss(self) -> float:
        """获取本月亏损"""
        today = datetime.now().date()
        month_start = today.replace(day=1)
        month_losses = [loss['profit'] for loss in self.daily_losses 
                       if loss['date'] >= month_start]
        return sum(month_losses)
    
    def trigger_emergency_stop(self, reason: str):
        """触发紧急停止"""
        self.emergency_stop = True
        logging.critical(f"触发紧急停止: {reason}")
    
    def clear_emergency_stop(self):
        """清除紧急停止"""
        self.emergency_stop = False
        logging.info("紧急停止已清除")

class RiskManager:
    """风险管理主控制器"""
    
    def __init__(self, config: RiskManagementConfig):
        self.config = config
        self.kelly_calculator = KellyCalculator(config)
        self.position_sizer = DynamicPositionSizer(config, self.kelly_calculator)
        self.stop_loss_calculator = ATRStopLossCalculator(config)
        self.multi_layer_control = MultiLayerRiskControl(config)
        
    def evaluate_trade_proposal(self, 
                              signal: Dict,
                              market_data: pd.DataFrame,
                              current_portfolio: Dict,
                              account_balance: float) -> Dict[str, any]:
        """评估交易提案"""
        
        entry_price = market_data['close'].iloc[-1]
        atr_value = self._calculate_atr(market_data)
        
        # 计算止损
        stop_loss_info = self.stop_loss_calculator.calculate_stop_loss(
            current_price=entry_price,
            atr_value=atr_value,
            position_type='long',  # 假设做多
            asset_type=signal.get('asset', 'BTC')
        )
        
        # 计算仓位大小
        position_info = self.position_sizer.calculate_position_size(
            signal_strength=signal.get('strength', 0.5),
            entry_price=entry_price,
            stop_loss_price=stop_loss_info['stop_loss_price'],
            account_balance=account_balance,
            asset_type=signal.get('asset', 'BTC')
        )
        
        # 风险检查
        trade_proposal = {
            'asset': signal.get('asset', 'BTC'),
            'position_size': position_info['position_size'],
            'position_value': position_info['position_value'],
            'entry_price': entry_price,
            'stop_loss_price': stop_loss_info['stop_loss_price'],
            'profit_target': stop_loss_info['profit_target']
        }
        
        risk_check = self.multi_layer_control.check_pre_trade_risk(
            trade_proposal, current_portfolio, account_balance
        )
        
        return {
            'trade_proposal': trade_proposal,
            'position_info': position_info,
            'stop_loss_info': stop_loss_info,
            'risk_check': risk_check,
            'kelly_stats': self.kelly_calculator.get_current_stats(),
            'recommendation': 'execute' if risk_check['approved'] else 'reject'
        }
    
    def monitor_open_positions(self, 
                             open_positions: Dict,
                             current_market_data: Dict,
                             account_balance: float) -> Dict[str, any]:
        """监控开仓位置"""
        
        position_updates = {}
        portfolio_alerts = []
        
        for asset, position in open_positions.items():
            current_price = current_market_data.get(asset, {}).get('close', 0)
            if current_price == 0:
                continue
                
            # 计算当前盈亏
            entry_price = position['entry_price']
            position_size = position['position_size']
            current_value = position_size * current_price
            pnl = current_value - (position_size * entry_price)
            pnl_percentage = pnl / (position_size * entry_price)
            
            # 更新移动止损
            atr_value = self._calculate_atr_from_dict(current_market_data.get(asset, {}))
            if atr_value > 0:
                new_stop = self.stop_loss_calculator.update_trailing_stop(
                    current_price=current_price,
                    entry_price=entry_price,
                    current_stop=position.get('stop_loss', 0),
                    atr_value=atr_value,
                    position_type=position.get('type', 'long'),
                    asset_type=asset
                )
                
                position_updates[asset] = {
                    'current_price': current_price,
                    'current_value': current_value,
                    'pnl': pnl,
                    'pnl_percentage': pnl_percentage,
                    'updated_stop_loss': new_stop,
                    'stop_updated': new_stop != position.get('stop_loss', 0)
                }
            
            # 检查风险警报
            if pnl_percentage < -0.15:  # 15%亏损警报
                portfolio_alerts.append(f"{asset}仓位亏损超过15%: {pnl_percentage:.2%}")
        
        # 检查投资组合整体风险
        portfolio_risk = self.multi_layer_control.check_portfolio_risk(
            open_positions, account_balance
        )
        
        return {
            'position_updates': position_updates,
            'portfolio_alerts': portfolio_alerts,
            'portfolio_risk': portfolio_risk
        }
    
    def _calculate_atr(self, market_data: pd.DataFrame, period: int = 14) -> float:
        """计算ATR值"""
        if len(market_data) < period + 1:
            return market_data['high'].max() - market_data['low'].min()
        
        high_low = market_data['high'] - market_data['low']
        high_close_prev = abs(market_data['high'] - market_data['close'].shift(1))
        low_close_prev = abs(market_data['low'] - market_data['close'].shift(1))
        
        true_range = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
        return true_range.rolling(period).mean().iloc[-1]
    
    def _calculate_atr_from_dict(self, price_dict: Dict) -> float:
        """从价格字典计算简化ATR"""
        if not price_dict:
            return 0
        high = price_dict.get('high', 0)
        low = price_dict.get('low', 0)
        return high - low if high > low else 0

if __name__ == "__main__":
    # 测试风险管理模块
    from .config import RiskManagementConfig
    
    print("🛡️ 风险管理模块测试")
    
    # 初始化配置
    config = RiskManagementConfig()
    risk_manager = RiskManager(config)
    
    # 创建测试数据
    dates = pd.date_range('2024-01-01', periods=100, freq='1H')
    np.random.seed(42)
    prices = 50000 + np.cumsum(np.random.randn(100) * 100)
    
    test_market_data = pd.DataFrame({
        'open': prices + np.random.randn(100) * 50,
        'high': prices + np.abs(np.random.randn(100)) * 100,
        'low': prices - np.abs(np.random.randn(100)) * 100,
        'close': prices,
        'volume': np.random.randint(1000, 10000, 100)
    }, index=dates)
    
    # 模拟交易信号
    test_signal = {
        'asset': 'BTC',
        'signal_type': 'bullish',
        'strength': 0.75
    }
    
    # 模拟当前投资组合
    current_portfolio = {
        'BTC': {'value': 25000, 'entry_price': 48000, 'position_size': 0.52}
    }
    
    account_balance = 100000
    
    # 测试交易评估
    evaluation = risk_manager.evaluate_trade_proposal(
        signal=test_signal,
        market_data=test_market_data,
        current_portfolio=current_portfolio,
        account_balance=account_balance
    )
    
    print(f"\n📊 交易评估结果:")
    print(f"推荐仓位: {evaluation['position_info']['position_size']:.4f} BTC")
    print(f"仓位价值: ${evaluation['position_info']['position_value']:.2f}")
    print(f"仓位比例: {evaluation['position_info']['position_ratio']:.2%}")
    print(f"止损价格: ${evaluation['stop_loss_info']['stop_loss_price']:.2f}")
    print(f"盈利目标: ${evaluation['stop_loss_info']['profit_target']:.2f}")
    print(f"风险检查: {'通过' if evaluation['risk_check']['approved'] else '不通过'}")
    print(f"最终推荐: {evaluation['recommendation']}")
    
    print("\n✅ 风险管理模块测试完成") 