"""
🚀 多周期背离策略主控制器
Multi-Timeframe Divergence Strategy Main Controller

整合技术指标、链上数据、AI增强、风险管理等所有模块
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import logging
import asyncio
from dataclasses import asdict

from .config import StrategyConfig, DEFAULT_CONFIG
from .technical_indicators import TechnicalIndicators, MultiTimeframeAnalysis
from .onchain_indicators import OnChainSignalAggregator
from .ai_enhanced import AISignalFilter
from .risk_management import RiskManager

class MultiTimeframeDivergenceStrategy:
    """多周期背离策略主类"""
    
    def __init__(self, config: StrategyConfig = None):
        self.config = config or DEFAULT_CONFIG
        
        # 初始化各个模块
        self.technical_indicators = TechnicalIndicators(self.config.technical)
        self.multi_timeframe_analyzer = MultiTimeframeAnalysis(self.technical_indicators)
        self.onchain_aggregator = OnChainSignalAggregator(self.config.onchain)
        self.ai_filter = AISignalFilter(self.config.ai)
        self.risk_manager = RiskManager(self.config.risk)
        
        # 策略状态
        self.current_positions = {}
        self.signal_history = []
        self.performance_metrics = {}
        self.account_balance = 100000  # 默认账户余额
        
        # 日志设置
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        self.logger.info(f"🚀 多周期背离策略已初始化，运行模式: {self.config.mode}")
    
    async def analyze_market(self, market_data: Dict[str, pd.DataFrame]) -> Dict[str, any]:
        """
        市场分析主函数
        
        Args:
            market_data: 不同时间框架的市场数据
            例: {'1h': df, '4h': df, '1d': df}
            
        Returns:
            综合市场分析结果
        """
        analysis_start_time = datetime.now()
        
        try:
            # 1. 多时间框架技术分析
            self.logger.info("📊 开始多时间框架技术分析...")
            technical_analysis = self.multi_timeframe_analyzer.analyze_timeframe_confluence(market_data)
            
            # 2. 链上数据分析  
            self.logger.info("⛓️ 开始链上数据分析...")
            onchain_analysis = self.onchain_aggregator.get_comprehensive_analysis('BTC')
            
            # 3. 生成综合信号
            self.logger.info("🔍 生成综合交易信号...")
            raw_signal = self._generate_raw_signal(technical_analysis, onchain_analysis)
            
            # 4. AI信号过滤和增强
            if raw_signal['signal_type'] != 'neutral':
                self.logger.info("🤖 AI信号过滤和增强...")
                enhanced_signal = self.ai_filter.filter_trading_signal(
                    raw_signal=raw_signal,
                    market_data=market_data.get('1h', pd.DataFrame()),
                    technical_indicators=technical_analysis.get('1h', {}),
                    onchain_data=onchain_analysis
                )
            else:
                enhanced_signal = raw_signal
            
            # 5. 风险管理评估
            if enhanced_signal.get('recommendation') == 'execute':
                self.logger.info("🛡️ 风险管理评估...")
                risk_evaluation = self.risk_manager.evaluate_trade_proposal(
                    signal=enhanced_signal,
                    market_data=market_data.get('1h', pd.DataFrame()),
                    current_portfolio=self.current_positions,
                    account_balance=self.account_balance
                )
            else:
                risk_evaluation = {'recommendation': 'hold'}
            
            # 6. 生成最终决策
            final_decision = self._make_final_decision(enhanced_signal, risk_evaluation)
            
            analysis_time = (datetime.now() - analysis_start_time).total_seconds()
            
            # 记录信号历史
            self._record_signal_history(final_decision, technical_analysis, onchain_analysis)
            
            return {
                'timestamp': datetime.now(),
                'analysis_time_seconds': analysis_time,
                'technical_analysis': technical_analysis,
                'onchain_analysis': onchain_analysis,
                'raw_signal': raw_signal,
                'enhanced_signal': enhanced_signal,
                'risk_evaluation': risk_evaluation,
                'final_decision': final_decision,
                'strategy_performance': self._calculate_current_performance()
            }
            
        except Exception as e:
            self.logger.error(f"❌ 市场分析出错: {e}")
            return {
                'timestamp': datetime.now(),
                'error': str(e),
                'final_decision': {'action': 'hold', 'reason': 'Analysis error'}
            }
    
    def _generate_raw_signal(self, technical_analysis: Dict, onchain_analysis: Dict) -> Dict[str, any]:
        """生成原始交易信号"""
        
        signal_scores = {
            'bullish': 0,
            'bearish': 0,
            'neutral': 0
        }
        
        contributing_factors = []
        
        # 1. 技术分析信号评分
        for timeframe, analysis in technical_analysis.items():
            signal_strength = analysis.get('signal_strength', {})
            net_score = signal_strength.get('net_score', 0)
            weight = self.config.signals.timeframe_weights.get(timeframe, 0.5)
            
            if net_score > 2:
                signal_scores['bullish'] += weight * 2
                contributing_factors.append(f"{timeframe}技术面看涨")
            elif net_score < -2:
                signal_scores['bearish'] += weight * 2
                contributing_factors.append(f"{timeframe}技术面看跌")
            else:
                signal_scores['neutral'] += weight
        
        # 2. 链上数据信号评分
        onchain_signal = onchain_analysis.get('combined_signal', {})
        onchain_type = onchain_signal.get('signal_type', 'neutral')
        onchain_strength = onchain_signal.get('strength', 'weak')
        
        onchain_weight = 1.5 if onchain_strength == 'strong' else 1.0
        
        if onchain_type == 'bullish':
            signal_scores['bullish'] += onchain_weight
            contributing_factors.append("链上数据看涨")
        elif onchain_type == 'bearish':
            signal_scores['bearish'] += onchain_weight
            contributing_factors.append("链上数据看跌")
        else:
            signal_scores['neutral'] += 0.5
        
        # 3. 背离信号特殊加权
        divergence_signals = []
        for timeframe, analysis in technical_analysis.items():
            divergences = analysis.get('divergences', {})
            if divergences.get('bullish_divergence'):
                signal_scores['bullish'] += 2.0  # 背离信号高权重
                divergence_signals.append(f"{timeframe}底背离")
            if divergences.get('bearish_divergence'):
                signal_scores['bearish'] += 2.0
                divergence_signals.append(f"{timeframe}顶背离")
        
        # 4. 确定主信号
        max_score = max(signal_scores.values())
        dominant_signal = [k for k, v in signal_scores.items() if v == max_score][0]
        
        # 5. 计算信号强度
        total_score = sum(signal_scores.values())
        signal_strength = (max_score / total_score) if total_score > 0 else 0
        
        # 6. 信号确认数量
        confirmed_indicators = len(contributing_factors) + len(divergence_signals)
        
        # 7. 确定信号等级
        if confirmed_indicators >= self.config.signals.diamond_signal_threshold:
            signal_grade = 'diamond'  # 钻石信号
        elif confirmed_indicators >= self.config.signals.gold_signal_threshold:
            signal_grade = 'gold'     # 黄金信号
        elif confirmed_indicators >= self.config.signals.silver_signal_threshold:
            signal_grade = 'silver'   # 白银信号
        else:
            signal_grade = 'bronze'   # 青铜信号
        
        return {
            'signal_type': dominant_signal,
            'signal_grade': signal_grade,
            'strength_score': signal_strength,
            'confirmed_indicators': confirmed_indicators,
            'signal_scores': signal_scores,
            'contributing_factors': contributing_factors,
            'divergence_signals': divergence_signals,
            'timestamp': datetime.now()
        }
    
    def _make_final_decision(self, enhanced_signal: Dict, risk_evaluation: Dict) -> Dict[str, any]:
        """做出最终交易决策"""
        
        # 获取信号基本信息
        signal_type = enhanced_signal.get('signal_type', 'neutral')
        signal_grade = enhanced_signal.get('signal_grade', 'bronze')
        ai_confirmation = enhanced_signal.get('ai_confirmation', False)
        risk_approved = risk_evaluation.get('recommendation') == 'execute'
        
        # 决策逻辑
        if signal_type == 'neutral':
            action = 'hold'
            reason = "无明确信号"
            confidence = 0.5
            
        elif not ai_confirmation:
            action = 'hold'
            reason = "AI模型未确认信号"
            confidence = 0.3
            
        elif not risk_approved:
            action = 'hold'
            reason = "风险管理不通过"
            confidence = 0.2
            
        elif signal_grade in ['diamond', 'gold']:
            action = 'execute'
            reason = f"{signal_grade}级信号，强烈{signal_type}"
            confidence = 0.9 if signal_grade == 'diamond' else 0.8
            
        elif signal_grade == 'silver':
            action = 'execute'
            reason = f"白银级信号，{signal_type}"
            confidence = 0.7
            
        else:  # bronze
            action = 'hold'
            reason = "信号强度不足"
            confidence = 0.4
        
        # 获取推荐仓位信息
        position_info = risk_evaluation.get('position_info', {})
        
        decision = {
            'action': action,
            'signal_type': signal_type,
            'signal_grade': signal_grade,
            'reason': reason,
            'confidence': confidence,
            'ai_confirmation': ai_confirmation,
            'risk_approved': risk_approved,
            'timestamp': datetime.now()
        }
        
        # 如果决定执行交易，添加具体交易信息
        if action == 'execute':
            trade_proposal = risk_evaluation.get('trade_proposal', {})
            decision.update({
                'recommended_position_size': position_info.get('position_size', 0),
                'recommended_position_value': position_info.get('position_value', 0),
                'entry_price': trade_proposal.get('entry_price', 0),
                'stop_loss_price': trade_proposal.get('stop_loss_price', 0),
                'profit_target': trade_proposal.get('profit_target', 0),
                'max_loss_amount': position_info.get('max_loss', 0),
                'position_ratio': position_info.get('position_ratio', 0)
            })
        
        return decision
    
    def execute_trade(self, decision: Dict) -> Dict[str, any]:
        """执行交易（模拟）"""
        
        if decision['action'] != 'execute':
            return {'status': 'no_trade', 'reason': decision['reason']}
        
        try:
            # 模拟交易执行
            trade_id = f"trade_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            trade_info = {
                'trade_id': trade_id,
                'timestamp': datetime.now(),
                'asset': 'BTC',  # 默认BTC
                'action': 'buy' if decision['signal_type'] == 'bullish' else 'sell',
                'position_size': decision.get('recommended_position_size', 0),
                'entry_price': decision.get('entry_price', 0),
                'stop_loss': decision.get('stop_loss_price', 0),
                'profit_target': decision.get('profit_target', 0),
                'position_value': decision.get('recommended_position_value', 0),
                'signal_grade': decision['signal_grade'],
                'confidence': decision['confidence']
            }
            
            # 更新当前持仓
            self.current_positions[trade_id] = trade_info
            
            # 更新账户余额（模拟）
            self.account_balance -= trade_info['position_value']
            
            self.logger.info(f"✅ 交易执行成功: {trade_id}")
            return {'status': 'executed', 'trade_info': trade_info}
            
        except Exception as e:
            self.logger.error(f"❌ 交易执行失败: {e}")
            return {'status': 'failed', 'error': str(e)}
    
    def monitor_positions(self, current_market_data: Dict) -> Dict[str, any]:
        """监控当前持仓"""
        
        if not self.current_positions:
            return {'status': 'no_positions'}
        
        monitoring_result = self.risk_manager.monitor_open_positions(
            self.current_positions, current_market_data, self.account_balance
        )
        
        # 检查是否需要平仓
        close_signals = []
        
        for trade_id, update in monitoring_result.get('position_updates', {}).items():
            position = self.current_positions[trade_id]
            current_price = update['current_price']
            
            # 检查止损
            if ((position['action'] == 'buy' and current_price <= position['stop_loss']) or
                (position['action'] == 'sell' and current_price >= position['stop_loss'])):
                close_signals.append({
                    'trade_id': trade_id,
                    'reason': 'stop_loss',
                    'current_price': current_price,
                    'pnl': update['pnl']
                })
            
            # 检查盈利目标
            elif ((position['action'] == 'buy' and current_price >= position['profit_target']) or
                  (position['action'] == 'sell' and current_price <= position['profit_target'])):
                close_signals.append({
                    'trade_id': trade_id,
                    'reason': 'profit_target',
                    'current_price': current_price,
                    'pnl': update['pnl']
                })
        
        return {
            'status': 'monitoring',
            'position_updates': monitoring_result,
            'close_signals': close_signals,
            'portfolio_alerts': monitoring_result.get('portfolio_alerts', [])
        }
    
    def close_position(self, trade_id: str, current_price: float, reason: str) -> Dict[str, any]:
        """平仓"""
        
        if trade_id not in self.current_positions:
            return {'status': 'error', 'message': 'Position not found'}
        
        position = self.current_positions[trade_id]
        
        # 计算盈亏
        entry_price = position['entry_price']
        position_size = position['position_size']
        
        if position['action'] == 'buy':
            pnl = (current_price - entry_price) * position_size
        else:  # sell
            pnl = (entry_price - current_price) * position_size
        
        pnl_percentage = pnl / position['position_value']
        
        # 更新账户余额
        self.account_balance += position['position_value'] + pnl
        
        # 记录交易结果
        trade_result = {
            'trade_id': trade_id,
            'close_timestamp': datetime.now(),
            'close_price': current_price,
            'close_reason': reason,
            'pnl': pnl,
            'pnl_percentage': pnl_percentage,
            'position_info': position
        }
        
        # 更新风险管理模块
        self.risk_manager.kelly_calculator.update_trade_history({
            'timestamp': datetime.now(),
            'profit': pnl_percentage,
            'asset': position['asset']
        })
        
        self.risk_manager.multi_layer_control.update_loss_tracking({
            'profit': pnl_percentage
        })
        
        # 从当前持仓中移除
        del self.current_positions[trade_id]
        
        self.logger.info(f"📊 平仓完成: {trade_id}, 盈亏: {pnl:.2f} ({pnl_percentage:.2%})")
        
        return {'status': 'closed', 'trade_result': trade_result}
    
    def _record_signal_history(self, decision: Dict, technical_analysis: Dict, onchain_analysis: Dict):
        """记录信号历史"""
        
        signal_record = {
            'timestamp': datetime.now(),
            'decision': decision,
            'technical_summary': {
                timeframe: {
                    'signal_strength': analysis.get('signal_strength', {}),
                    'divergence_count': len(analysis.get('divergences', {}).get('bullish_divergence', [])) +
                                      len(analysis.get('divergences', {}).get('bearish_divergence', []))
                }
                for timeframe, analysis in technical_analysis.items()
            },
            'onchain_summary': {
                'signal_type': onchain_analysis.get('combined_signal', {}).get('signal_type'),
                'strength': onchain_analysis.get('combined_signal', {}).get('strength'),
                'recommendation': onchain_analysis.get('recommendation')
            }
        }
        
        self.signal_history.append(signal_record)
        
        # 保留最近1000条记录
        if len(self.signal_history) > 1000:
            self.signal_history = self.signal_history[-1000:]
    
    def _calculate_current_performance(self) -> Dict[str, any]:
        """计算当前策略表现"""
        
        if len(self.signal_history) < 10:
            return {'status': 'insufficient_data'}
        
        # 统计信号分布
        signal_types = [record['decision']['signal_type'] for record in self.signal_history[-100:]]
        signal_grades = [record['decision']['signal_grade'] for record in self.signal_history[-100:]]
        actions = [record['decision']['action'] for record in self.signal_history[-100:]]
        
        # 统计Kelly计算器的交易记录
        kelly_stats = self.risk_manager.kelly_calculator.get_current_stats()
        
        return {
            'total_signals': len(self.signal_history),
            'recent_signal_distribution': {
                'bullish': signal_types.count('bullish'),
                'bearish': signal_types.count('bearish'),
                'neutral': signal_types.count('neutral')
            },
            'signal_grade_distribution': {
                'diamond': signal_grades.count('diamond'),
                'gold': signal_grades.count('gold'),
                'silver': signal_grades.count('silver'),
                'bronze': signal_grades.count('bronze')
            },
            'action_distribution': {
                'execute': actions.count('execute'),
                'hold': actions.count('hold')
            },
            'current_account_balance': self.account_balance,
            'active_positions': len(self.current_positions),
            'trading_stats': kelly_stats
        }
    
    def get_strategy_status(self) -> Dict[str, any]:
        """获取策略状态报告"""
        
        return {
            'timestamp': datetime.now(),
            'strategy_config': {
                'mode': self.config.mode,
                'max_single_position': self.config.risk.max_single_position,
                'max_total_position': self.config.risk.max_total_position,
                'signal_thresholds': {
                    'diamond': self.config.signals.diamond_signal_threshold,
                    'gold': self.config.signals.gold_signal_threshold,
                    'silver': self.config.signals.silver_signal_threshold
                }
            },
            'current_status': {
                'account_balance': self.account_balance,
                'active_positions': len(self.current_positions),
                'position_details': self.current_positions,
                'emergency_stop': self.risk_manager.multi_layer_control.emergency_stop,
                'consecutive_losses': self.risk_manager.multi_layer_control.consecutive_losses
            },
            'performance_metrics': self._calculate_current_performance()
        }

# 策略工厂函数
def create_strategy(mode: str = "conservative") -> MultiTimeframeDivergenceStrategy:
    """创建策略实例"""
    config = StrategyConfig(mode=mode)
    return MultiTimeframeDivergenceStrategy(config)

if __name__ == "__main__":
    # 测试主策略
    import asyncio
    
    async def test_strategy():
        print("🚀 多周期背离策略测试")
        
        # 创建策略实例
        strategy = create_strategy("conservative")
        
        # 创建测试市场数据
        dates_1h = pd.date_range('2024-01-01', periods=168, freq='1H')  # 7天1小时数据
        dates_4h = pd.date_range('2024-01-01', periods=42, freq='4H')   # 7天4小时数据
        dates_1d = pd.date_range('2024-01-01', periods=7, freq='1D')    # 7天日线数据
        
        np.random.seed(42)
        base_price = 50000
        
        def create_test_data(dates, base_volatility=100):
            prices = base_price + np.cumsum(np.random.randn(len(dates)) * base_volatility)
            return pd.DataFrame({
                'open': prices + np.random.randn(len(dates)) * 50,
                'high': prices + np.abs(np.random.randn(len(dates))) * 100,
                'low': prices - np.abs(np.random.randn(len(dates))) * 100,
                'close': prices,
                'volume': np.random.randint(1000, 10000, len(dates))
            }, index=dates)
        
        test_market_data = {
            '1h': create_test_data(dates_1h, 50),
            '4h': create_test_data(dates_4h, 200),
            '1d': create_test_data(dates_1d, 500)
        }
        
        # 执行市场分析
        print("\n📊 执行市场分析...")
        analysis_result = await strategy.analyze_market(test_market_data)
        
        # 显示分析结果
        print(f"\n📈 分析结果:")
        print(f"分析耗时: {analysis_result.get('analysis_time_seconds', 0):.2f}秒")
        
        final_decision = analysis_result.get('final_decision', {})
        print(f"最终决策: {final_decision.get('action')}")
        print(f"信号类型: {final_decision.get('signal_type')}")
        print(f"信号等级: {final_decision.get('signal_grade')}")
        print(f"置信度: {final_decision.get('confidence', 0):.2%}")
        print(f"决策原因: {final_decision.get('reason')}")
        
        # 如果有交易信号，执行交易
        if final_decision.get('action') == 'execute':
            print(f"\n💰 执行交易...")
            trade_result = strategy.execute_trade(final_decision)
            print(f"交易结果: {trade_result.get('status')}")
            
            if trade_result.get('status') == 'executed':
                trade_info = trade_result['trade_info']
                print(f"交易ID: {trade_info['trade_id']}")
                print(f"仓位大小: {trade_info['position_size']:.4f} BTC")
                print(f"入场价格: ${trade_info['entry_price']:.2f}")
                print(f"止损价格: ${trade_info['stop_loss']:.2f}")
                print(f"盈利目标: ${trade_info['profit_target']:.2f}")
        
        # 获取策略状态
        print(f"\n📊 策略状态:")
        status = strategy.get_strategy_status()
        print(f"账户余额: ${status['current_status']['account_balance']:.2f}")
        print(f"活跃持仓: {status['current_status']['active_positions']}")
        
        performance = status['performance_metrics']
        if performance.get('status') != 'insufficient_data':
            print(f"总信号数: {performance['total_signals']}")
            print(f"执行比例: {performance['action_distribution'].get('execute', 0)}/100")
        
        print("\n✅ 策略测试完成")
    
    # 运行测试
    asyncio.run(test_strategy()) 