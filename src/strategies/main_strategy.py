import pandas as pd
import numpy as np
from datetime import datetime
import sys
import os
import asyncio  # 添加asyncio导入
import matplotlib.pyplot as plt
from tqdm import tqdm

# 添加项目根目录到路径，以便导入其他模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.strategies.config import create_strategy_config
from src.strategies.technical_indicators import TechnicalAnalyzer
from src.strategies.risk_management import RiskManager
from src.strategies.divergence_analyzer import load_bitcoin_data


class MultiTimeframeDivergenceStrategy:
    """
    多周期背离策略主控制器
    """
    def __init__(self, config=None):
        """
        初始化策略
        :param config: 策略配置对象，如果为None则使用默认配置
        """
        self.config = config if config else create_strategy_config("standard")
        self.technical_analyzer = TechnicalAnalyzer(self.config)
        self.risk_manager = RiskManager(self.config)
        self.active_trades = {}
        self.trade_history = []
        self.last_analysis = {}
    
    async def analyze_market(self, market_data):
        """
        分析市场数据并生成交易决策
        :param market_data: 市场数据字典，键为交易对，值为包含时间框架的字典
        :return: 分析结果字典
        """
        # 确保数据包含所有必要的时间框架
        required_timeframes = [
            self.config.technical["timeframes"]["macro"],  # 宏观层
            self.config.technical["timeframes"]["meso"],   # 中观层
            self.config.technical["timeframes"]["micro"]   # 微观层
        ]
        
        # 分析各个交易对
        analysis_results = {}
        
        for symbol, data_dict in market_data.items():
            # 检查该交易对是否包含所有必要的时间框架
            for tf in required_timeframes:
                if tf not in data_dict:
                    print(f"错误: 交易对 {symbol} 缺少必要的时间框架数据 {tf}")
                    return {"error": f"缺少时间框架 {tf}"}
            
            # 初始化该交易对的分析结果
            analysis_results[symbol] = {}
            
            # 分析各个时间框架
            for timeframe, df in data_dict.items():
                # 使用技术分析器分析市场
                result = self.technical_analyzer.analyze_market(df, symbol)
                analysis_results[symbol][timeframe] = result
        
        # 合并多时间框架信号
        final_decisions = self._combine_timeframe_signals(analysis_results)
        
        # 应用风险管理
        risk_adjusted_decisions = self._apply_risk_management(final_decisions)
        
        # 保存分析结果
        self.last_analysis = {
            "timestamp": datetime.now(),
            "analysis_results": analysis_results,
            "final_decisions": final_decisions,
            "risk_adjusted_decisions": risk_adjusted_decisions
        }
        
        return {
            "analysis_results": analysis_results,
            "final_decisions": final_decisions,
            "risk_adjusted_decisions": risk_adjusted_decisions,
            "final_decision": risk_adjusted_decisions
        }
    
    def _combine_timeframe_signals(self, analysis_results):
        """
        合并多时间框架信号
        :param analysis_results: 各时间框架的分析结果
        :return: 合并后的决策
        """
        combined_decisions = {}
        
        for symbol, timeframe_results in analysis_results.items():
            # 初始化信号计数
            buy_signals = 0
            sell_signals = 0
            signal_strength_sum = 0
            
            # 获取各时间框架的权重
            weights = {
                self.config.technical["timeframes"]["macro"]: 0.4,  # 宏观层权重
                self.config.technical["timeframes"]["meso"]: 0.4,   # 中观层权重
                self.config.technical["timeframes"]["micro"]: 0.2    # 微观层权重
            }
            
            # 计算加权信号
            for timeframe, result in timeframe_results.items():
                weight = weights.get(timeframe, 0.1)
                
                if result["signal_type"] == "buy":
                    buy_signals += 1
                    signal_strength_sum += result["signal_strength"] * weight
                elif result["signal_type"] == "sell":
                    sell_signals += 1
                    signal_strength_sum -= result["signal_strength"] * weight
            
            # 确定最终信号类型
            if buy_signals > sell_signals:
                final_signal_type = "buy"
                final_signal_strength = signal_strength_sum / sum(weights.values())
            elif sell_signals > buy_signals:
                final_signal_type = "sell"
                final_signal_strength = abs(signal_strength_sum) / sum(weights.values())
            else:
                # 信号相等，看强度
                if signal_strength_sum > 0:
                    final_signal_type = "buy"
                    final_signal_strength = signal_strength_sum / sum(weights.values())
                elif signal_strength_sum < 0:
                    final_signal_type = "sell"
                    final_signal_strength = abs(signal_strength_sum) / sum(weights.values())
                else:
                    final_signal_type = "neutral"
                    final_signal_strength = 0
            
            # 判断是否达到交易阈值
            signal_threshold = self.config.get_signal_threshold()
            confirmed_signals = buy_signals if final_signal_type == "buy" else sell_signals
            
            should_trade = confirmed_signals >= signal_threshold and final_signal_strength >= 0.6
            
            # 构建决策
            combined_decisions[symbol] = {
                "signal_type": final_signal_type,
                "signal_strength": final_signal_strength,
                "confirmed_signals": confirmed_signals,
                "required_signals": signal_threshold,
                "should_trade": should_trade,
                "timeframe_results": timeframe_results
            }
        
        return combined_decisions
    
    def _apply_risk_management(self, decisions):
        """
        应用风险管理规则
        :param decisions: 交易决策
        :return: 风险调整后的决策
        """
        risk_adjusted = {}
        
        # 获取账户余额
        account_balance = self.risk_manager.get_account_balance()
        
        for symbol, decision in decisions.items():
            # 复制原始决策
            adjusted = decision.copy()
            
            # 如果不应该交易，跳过风险管理
            if not decision["should_trade"]:
                adjusted["action"] = "none"
                risk_adjusted[symbol] = adjusted
                continue
            
            # 获取当前价格和ATR值
            current_price = None
            atr_value = None
            
            # 从中观层获取价格和ATR
            meso_tf = self.config.technical["timeframes"]["meso"]
            if meso_tf in decision["timeframe_results"]:
                result = decision["timeframe_results"][meso_tf]
                current_price = result["close_price"]
                # 假设ATR已经在技术分析器中计算
                # 这里需要根据实际情况调整
                atr_value = 0.05 * current_price  # 示例：假设ATR为价格的5%
            
            # 如果无法获取价格或ATR，跳过
            if not current_price or not atr_value:
                adjusted["action"] = "none"
                adjusted["reason"] = "缺少价格或ATR数据"
                risk_adjusted[symbol] = adjusted
                continue
            
            # 计算建议仓位大小
            position_size = self.risk_manager.calculate_position_size(
                decision["signal_strength"], 
                atr_value, 
                account_balance, 
                "BTC" if "BTC" in symbol else "ALT"
            )
            
            # 如果仓位太小，不交易
            if position_size < 100:  # 假设最小交易金额为$100
                adjusted["action"] = "none"
                adjusted["reason"] = "仓位太小"
                risk_adjusted[symbol] = adjusted
                continue
            
            # 计算止损价格
            if decision["signal_type"] == "buy":
                stop_loss = self.risk_manager.calculate_stop_loss(
                    current_price, 
                    atr_value, 
                    "BTC" if "BTC" in symbol else "ALT"
                )
            else:  # sell
                # 对于做空，止损在价格之上
                stop_loss = current_price + (atr_value * 2.0)
            
            # 添加交易执行信息
            adjusted["action"] = "execute"
            adjusted["position_size"] = position_size
            adjusted["entry_price"] = current_price
            adjusted["stop_loss"] = stop_loss
            
            risk_adjusted[symbol] = adjusted
        
        return risk_adjusted
    
    def execute_trade(self, decision):
        """
        执行交易
        :param decision: 交易决策
        :return: 交易结果
        """
        if decision["action"] != "execute":
            return {"status": "skipped", "reason": decision.get("reason", "不满足交易条件")}
        
        symbol = decision["symbol"]
        signal_type = decision["signal_type"]
        position_size = decision["position_size"]
        entry_price = decision["entry_price"]
        stop_loss = decision["stop_loss"]
        
        # 添加仓位
        position_id = self.risk_manager.add_position(symbol, position_size, entry_price, stop_loss)
        
        # 记录交易
        trade = {
            "id": position_id,
            "symbol": symbol,
            "type": signal_type,
            "position_size": position_size,
            "entry_price": entry_price,
            "stop_loss": stop_loss,
            "entry_time": datetime.now(),
            "status": "active"
        }
        
        self.active_trades[position_id] = trade
        
        return {
            "status": "executed",
            "trade": trade
        }
    
    def close_position(self, position_id, current_price, reason="manual"):
        """
        平仓
        :param position_id: 仓位ID
        :param current_price: 当前价格
        :param reason: 平仓原因
        :return: 平仓结果
        """
        # 使用风险管理器平仓
        result = self.risk_manager.close_position(position_id, current_price, reason)
        
        if result:
            # 更新交易记录
            if position_id in self.active_trades:
                trade = self.active_trades[position_id]
                trade["exit_price"] = current_price
                trade["exit_time"] = datetime.now()
                trade["status"] = "closed"
                trade["close_reason"] = reason
                trade["profit_loss"] = (current_price - trade["entry_price"]) / trade["entry_price"] \
                    if trade["type"] == "buy" else (trade["entry_price"] - current_price) / trade["entry_price"]
                
                # 移动到历史记录
                self.trade_history.append(trade)
                del self.active_trades[position_id]
                
                return {
                    "status": "closed",
                    "trade": trade
                }
        
        return {
            "status": "failed",
            "reason": "无法找到仓位或平仓失败"
        }
    
    def monitor_positions(self, current_market_data):
        """
        监控持仓
        :param current_market_data: 当前市场数据
        :return: 监控结果
        """
        # 提取当前价格
        current_prices = {}
        for symbol, data_dict in current_market_data.items():
            for timeframe, df in data_dict.items():
                if timeframe == self.config.technical["timeframes"]["meso"]:
                    current_prices[symbol] = float(df['收盘价'].iloc[-1])
        
        # 检查止损
        stop_loss_triggers = self.risk_manager.check_stop_losses(current_prices)
        
        # 构建监控结果
        result = {
            "active_positions": len(self.active_trades),
            "current_prices": current_prices,
            "stop_loss_triggers": stop_loss_triggers,
            "close_signals": []
        }
        
        # 添加止损信号
        for trigger in stop_loss_triggers:
            result["close_signals"].append({
                "position_id": trigger["position_id"],
                "symbol": trigger["symbol"],
                "current_price": trigger["current_price"],
                "reason": "stop_loss"
            })
        
        return result
    
    def get_strategy_status(self):
        """
        获取策略状态
        :return: 状态字典
        """
        # 获取风险指标
        risk_metrics = self.risk_manager.get_risk_metrics()
        
        # 计算交易统计
        closed_trades = len(self.trade_history)
        winning_trades = sum(1 for t in self.trade_history if t.get("profit_loss", 0) > 0)
        
        win_rate = winning_trades / closed_trades if closed_trades > 0 else 0
        
        # 计算平均盈亏
        if closed_trades > 0:
            avg_profit = sum(t.get("profit_loss", 0) for t in self.trade_history) / closed_trades
        else:
            avg_profit = 0
        
        return {
            "current_status": {
                "active_positions": len(self.active_trades),
                "account_balance": self.risk_manager.get_account_balance(),
                "total_exposure": risk_metrics["total_exposure"],
                "trading_paused": risk_metrics["trading_paused"],
            },
            "performance_metrics": {
                "trading_stats": {
                    "total_trades": closed_trades + len(self.active_trades),
                    "closed_trades": closed_trades,
                    "win_rate": win_rate,
                    "avg_profit": avg_profit,
                },
                "risk_metrics": risk_metrics
            },
            "config": {
                "mode": self.config.mode,
                "signal_threshold": self.config.get_signal_threshold(),
                "max_position": self.config.risk["max_single_position"],
                "max_total": self.config.risk["max_total_position"]
            }
        }


def create_strategy(mode="standard"):
    """
    创建策略实例的工厂函数
    :param mode: 策略模式
    :return: 策略实例
    """
    config = create_strategy_config(mode)
    return MultiTimeframeDivergenceStrategy(config)


async def test_strategy():
    # 加载数据
    print("加载测试数据...")
    
    # 加载不同时间框架的数据
    klines_data_1d = load_bitcoin_data(interval='1d')
    klines_data_4h = load_bitcoin_data(interval='4h')
    klines_data_1h = load_bitcoin_data(interval='1h')
    
    # 检查是否所有时间框架的数据都加载成功
    if not klines_data_1d or not klines_data_4h or not klines_data_1h:
        print("无法加载所有必要的时间框架数据")
        return
    
    # 转换为DataFrame
    df_1d = pd.DataFrame(klines_data_1d)
    df_4h = pd.DataFrame(klines_data_4h)
    df_1h = pd.DataFrame(klines_data_1h)
    print("成功加载所有时间框架数据")
    
    # 打印数据框的列名，以便了解数据格式
    print("\n数据框列名:")
    print(f"1d数据列: {df_1d.columns.tolist()}")
    
    # 创建策略
    strategy = create_strategy("standard")
    
    # 获取配置中的时间框架名称
    macro_tf = strategy.config.technical["timeframes"]["macro"]  # 应该是 "1d"
    meso_tf = strategy.config.technical["timeframes"]["meso"]    # 应该是 "4h"
    micro_tf = strategy.config.technical["timeframes"]["micro"]  # 应该是 "1h"
    
    print(f"使用的时间框架: 宏观={macro_tf}, 中观={meso_tf}, 微观={micro_tf}")
    
    # 构建市场数据 - 直接使用时间框架作为键，而不是嵌套在交易对下面
    # 这是因为analyze_market函数期望的数据结构是market_data[timeframe] = df
    market_data = {
        "BTCUSDT": {  # 添加交易对层级
            macro_tf: df_1d,  # 宏观层 - 日线
            meso_tf: df_4h,   # 中观层 - 4小时
            micro_tf: df_1h   # 微观层 - 1小时
        }
    }
    
    # 分析市场
    print("\n执行市场分析...")
    try:
        result = await strategy.analyze_market(market_data)
        
        # 检查结果是否包含错误
        if isinstance(result, dict) and "error" in result:
            print(f"分析错误: {result['error']}")
            return
            
        # 打印分析结果
        print("\n分析结果:")
        if "final_decision" in result:
            for symbol, decision in result["final_decision"].items():
                print(f"交易对: {symbol}")
                print(f"信号类型: {decision['signal_type']}")
                print(f"信号强度: {decision['signal_strength']:.2f}")
                print(f"确认信号数: {decision['confirmed_signals']}/{decision['required_signals']}")
                print(f"建议交易: {'是' if decision['should_trade'] else '否'}")
                print(f"执行操作: {decision['action']}")
                
                if decision['action'] == 'execute':
                    print(f"仓位大小: ${decision['position_size']:.2f}")
                    print(f"入场价格: ${decision['entry_price']:.2f}")
                    print(f"止损价格: ${decision['stop_loss']:.2f}")
                
                print("-" * 40)
            
            # 如果有可执行的交易，执行它
            for symbol, decision in result["final_decision"].items():
                if decision["action"] == "execute":
                    decision["symbol"] = symbol  # 添加交易对信息
                    
                    print(f"\n执行交易: {symbol} {decision['signal_type']}")
                    trade_result = strategy.execute_trade(decision)
                    print(f"交易结果: {trade_result['status']}")
                    
                    if trade_result["status"] == "executed":
                        trade = trade_result["trade"]
                        print(f"交易ID: {trade['id']}")
                        print(f"仓位大小: ${trade['position_size']:.2f}")
                        print(f"入场价格: ${trade['entry_price']:.2f}")
                        print(f"止损价格: ${trade['stop_loss']:.2f}")
        else:
            print("结果中缺少最终决策数据")
            print(f"可用键: {list(result.keys())}")
        
        # 获取策略状态
        status = strategy.get_strategy_status()
        print("\n策略状态:")
        print(f"活跃仓位: {status['current_status']['active_positions']}")
        print(f"账户余额: ${status['current_status']['account_balance']:,.2f}")
        print(f"总仓位占比: {status['current_status']['total_exposure']*100:.1f}%")
        print(f"交易暂停: {'是' if status['current_status']['trading_paused'] else '否'}")
    except Exception as e:
        print(f"执行错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # 解析命令行参数
    import argparse
    parser = argparse.ArgumentParser(description='多时间框架背离策略')
    parser.add_argument('--mode', type=str, default='standard', choices=['conservative', 'standard', 'aggressive'],
                        help='策略模式: conservative, standard, aggressive')
    args = parser.parse_args()
    
    # 运行测试
    asyncio.run(test_strategy()) 