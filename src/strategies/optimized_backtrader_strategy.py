import backtrader as bt
import pandas as pd
import numpy as np
from datetime import datetime
import os

class AdvancedDivergenceStrategy(bt.Strategy):
    """
    高收益背离策略 - 目标20%年化收益
    KDJ背离 + MACD + RSI + 布林带 + 趋势跟踪
    """
    
    params = (
        ('target_timeframe', '1h'),          # 主要交易周期
        ('min_signal_score', 65),            # 提高信号门槛 - 质量胜于数量
        ('max_risk_per_trade', 0.012),       # 适中风险1.2%
        ('strong_signal_threshold', 75),     # 提高强信号门槛
        ('max_position_ratio', 0.15),        # 适中仓位15%
        ('atr_period', 10),                  # 适中ATR周期
        ('atr_stop_multiplier', 2.0),        # 适中止损倍数
        ('trailing_stop_ratio', 0.03),       # 适中移动止损3%
        ('profit_target_ratio', 2.2),        # 适中风险回报比2.2:1
        ('max_hold_days', 8),                # 适中持仓时间
        
        # MACD参数 - 平衡设置
        ('macd_fast', 7),                    # 
        ('macd_slow', 20),                   # 
        ('macd_signal', 5),                  # 
        
        # RSI参数
        ('rsi_period', 13),                  # 
        ('rsi_oversold', 32),                # 
        ('rsi_overbought', 68),              # 
        
        # 布林带参数
        ('bb_period', 18),                   # 
        ('bb_dev', 1.9),                     # 
        
        # 趋势跟踪参数
        ('trend_period', 45),                # 
        ('momentum_period', 12),             # 
    )

    def __init__(self):
        # 数据引用
        self.dataclose = self.datas[0].close
        self.dataopen = self.datas[0].open  
        self.datahigh = self.datas[0].high
        self.datalow = self.datas[0].low
        self.datavolume = self.datas[0].volume
        
        # 技术指标
        self.atr = bt.indicators.ATR(self.datas[0], period=self.p.atr_period)
        self.sma_20 = bt.indicators.SMA(self.datas[0], period=20)
        self.sma_60 = bt.indicators.SMA(self.datas[0], period=60)
        
        # 趋势跟踪指标
        self.sma_trend = bt.indicators.SMA(self.datas[0], period=self.p.trend_period)
        self.momentum = bt.indicators.Momentum(self.datas[0], period=self.p.momentum_period)
        self.ema_fast = bt.indicators.EMA(self.datas[0], period=12)
        self.ema_slow = bt.indicators.EMA(self.datas[0], period=26)
        
        # MACD指标 - 更敏感参数
        self.macd = bt.indicators.MACD(self.datas[0], 
                                      period_me1=self.p.macd_fast,
                                      period_me2=self.p.macd_slow,
                                      period_signal=self.p.macd_signal)
        self.macd_histogram = self.macd.macd - self.macd.signal
        
        # RSI指标 - 更敏感参数
        self.rsi = bt.indicators.RSI(self.datas[0], period=self.p.rsi_period)
        
        # 布林带指标 - 更紧参数
        self.bollinger = bt.indicators.BollingerBands(self.datas[0], 
                                                     period=self.p.bb_period,
                                                     devfactor=self.p.bb_dev)
        
        # 布林带位置计算
        self.bb_position = (self.dataclose - self.bollinger.lines.bot) / (self.bollinger.lines.top - self.bollinger.lines.bot)
        
        # 交易状态
        self.order = None
        self.position_entry_price = None
        self.position_entry_date = None
        self.stop_price = None
        self.profit_target = None
        self.trailing_stop = None
        
        # 风险控制 - 更宽松
        self.consecutive_losses = 0
        self.max_consecutive_losses = 0
        self.daily_trades = 0
        self.last_trade_date = None
        
        # 统计数据
        self.trade_count = 0
        self.win_count = 0
        self.total_profit = 0
        
        # 加载背离数据
        self.divergence_data = self.load_divergence_data()
        
        # 资金管理
        self.initial_cash = self.broker.get_cash()

    def load_divergence_data(self):
        """加载并处理背离数据"""
        try:
            df = pd.read_csv('results/所有周期背离数据_20250529_235931.csv', encoding='utf-8-sig')
            df['日期时间'] = pd.to_datetime(df['日期时间'])
            
            # 筛选目标时间周期
            target_data = df[df['时间周期'] == self.p.target_timeframe].copy()
            
            # 只保留底部背离信号（低风险策略）
            bottom_signals = target_data[target_data['背离类型'] == '底部背离'].copy()
            
            # 计算综合信号评分
            bottom_signals['综合评分'] = bottom_signals.apply(self.calculate_comprehensive_score, axis=1)
            
            # 分析市场环境
            bottom_signals['市场环境'] = bottom_signals.apply(self.analyze_market_environment, axis=1)
            
            # 设置时间索引
            bottom_signals.set_index('日期时间', inplace=True)
            
            print(f"📊 加载 {self.p.target_timeframe} 底部背离信号: {len(bottom_signals)} 条")
            print(f"📈 评分≥{self.p.min_signal_score}的信号: {len(bottom_signals[bottom_signals['综合评分'] >= self.p.min_signal_score])} 条")
            print(f"🎯 强信号(≥{self.p.strong_signal_threshold}分): {len(bottom_signals[bottom_signals['综合评分'] >= self.p.strong_signal_threshold])} 条")
            
            return bottom_signals
            
        except Exception as e:
            print(f"❌ 加载背离数据失败: {e}")
            return pd.DataFrame()

    def calculate_comprehensive_score(self, row):
        """
        计算综合信号评分（100分制）
        多重技术指标优化版
        """
        score = 0
        
        # 1. 信号强度评分 (20%) - 多指标环境下降低权重
        strength_scores = {'强': 20, '中': 15, '弱': 10}
        score += strength_scores.get(row['信号强度'], 0)
        
        # 2. J值区间评分 (25%) - 保持重要权重
        j_range_scores = {
            '极度超卖(<0)': 25,
            '超卖(0-20)': 22, 
            '偏弱(20-50)': 18,
            '中性(50-80)': 12,
            '超买(80-100)': 6,
            '极度超买(>100)': 0
        }
        score += j_range_scores.get(row['J值区间'], 0)
        
        # 3. 价格区间评分 (20%) - 保持权重
        price = row['收盘价']
        if price < 20000:      # 熊市底部
            score += 20
        elif price < 40000:    # 恢复期
            score += 18
        elif price < 70000:    # 成长期
            score += 15
        elif price < 100000:   # 牛市中期
            score += 12
        else:                  # 牛市顶部
            score += 8
        
        # 4. 时间周期评分 (15%) - 1小时加分
        timeframe_scores = {
            '1w': 15, '3d': 14, '1d': 13, '12h': 12, 
            '8h': 11, '4h': 10, '2h': 9, '1h': 15  # 1小时提高评分
        }
        score += timeframe_scores.get(row['时间周期'], 0)
        
        # 5. J值绝对位置评分 (10%) - 1小时周期更敏感
        j_value = row['J值']
        if j_value < -15:      # 极度超卖
            score += 10
        elif j_value < -5:     # 深度超卖
            score += 8
        elif j_value < 5:      # 超卖
            score += 6
        elif j_value < 20:     # 偏弱
            score += 4
        else:                  # 其他
            score += 2
        
        # 6. 多指标环境加分 (10%) - 新增
        # 根据历史数据推测可能的指标状态给予加分
        if j_value < 0:        # J值极度超卖，可能其他指标也超卖
            score += 10
        elif j_value < 15:     # J值超卖
            score += 8
        elif j_value < 30:     # J值偏弱
            score += 5
        else:
            score += 2
        
        return min(score, 100)  # 最高100分

    def analyze_market_environment(self, row):
        """分析市场环境"""
        price = row['收盘价']
        
        if price < 20000:
            return '熊市底部'
        elif price < 40000:
            return '恢复期'
        elif price < 70000:
            return '成长期'
        elif price < 100000:
            return '牛市中期'
        else:
            return '牛市顶部'

    def check_technical_indicators_confirmation(self):
        """检查所有技术指标确认信号"""
        if len(self.macd.macd) < 2 or len(self.rsi) < 2:
            return False, "指标数据不足"
        
        confirmations = []
        confirmation_score = 0
        
        # 1. MACD确认
        macd_confirmed, macd_signals = self.check_macd_signals()
        if macd_confirmed:
            confirmations.extend(macd_signals)
            confirmation_score += 25
        
        # 2. RSI确认
        rsi_confirmed, rsi_signals = self.check_rsi_signals()
        if rsi_confirmed:
            confirmations.extend(rsi_signals)
            confirmation_score += 25
        
        # 3. 布林带确认
        bb_confirmed, bb_signals = self.check_bollinger_signals()
        if bb_confirmed:
            confirmations.extend(bb_signals)
            confirmation_score += 25
        
        # 4. 趋势确认
        trend_confirmed, trend_signals = self.check_trend_signals()
        if trend_confirmed:
            confirmations.extend(trend_signals)
            confirmation_score += 25
        
        # 至少需要2个指标确认，总分≥40
        if len(confirmations) >= 2 and confirmation_score >= 40:
            return True, " | ".join(confirmations), confirmation_score
        else:
            return False, "技术指标确认不足", confirmation_score

    def check_macd_signals(self):
        """检查MACD信号"""
        current_macd = self.macd.macd[0]
        current_signal = self.macd.signal[0]
        current_histogram = self.macd_histogram[0]
        
        prev_macd = self.macd.macd[-1]
        prev_signal = self.macd.signal[-1]
        prev_histogram = self.macd_histogram[-1]
        
        signals = []
        
        # MACD看涨信号
        if current_macd < current_signal and (current_macd - current_signal) > (prev_macd - prev_signal):
            signals.append("MACD收敛")
        
        if prev_macd <= prev_signal and current_macd > current_signal:
            signals.append("MACD金叉")
        
        if current_histogram > prev_histogram and current_histogram > 0:
            signals.append("MACD柱状图转正")
        elif current_histogram > prev_histogram:
            signals.append("MACD动能增强")
        
        if current_macd < 0:
            signals.append("MACD低位")
        
        return len(signals) > 0, signals

    def check_rsi_signals(self):
        """检查RSI信号"""
        current_rsi = self.rsi[0]
        prev_rsi = self.rsi[-1]
        
        signals = []
        
        # RSI看涨信号
        if current_rsi < self.p.rsi_oversold:
            signals.append(f"RSI超卖({current_rsi:.1f})")
        elif current_rsi < 40:
            signals.append(f"RSI偏弱({current_rsi:.1f})")
        
        # RSI趋势改善
        if current_rsi > prev_rsi and current_rsi < 50:
            signals.append("RSI向上")
        
        # RSI背离（简化版，检查是否在低位但有上升趋势）
        if current_rsi < 35 and current_rsi > prev_rsi:
            signals.append("RSI底部反转")
        
        return len(signals) > 0, signals

    def check_bollinger_signals(self):
        """检查布林带信号"""
        if len(self.bb_position) < 2:
            return False, []
        
        bb_pos = self.bb_position[0]  # 当前价格在布林带中的位置
        prev_bb_pos = self.bb_position[-1]
        
        signals = []
        
        # 布林带看涨信号
        if bb_pos < 0.2:  # 价格接近下轨
            signals.append(f"布林带下轨({bb_pos:.2f})")
        elif bb_pos < 0.3:
            signals.append(f"布林带偏下({bb_pos:.2f})")
        
        # 价格从下轨反弹
        if bb_pos > prev_bb_pos and bb_pos < 0.4:
            signals.append("布林带反弹")
        
        # 布林带收窄（波动率下降）
        bb_width = (self.bollinger.lines.top[0] - self.bollinger.lines.bot[0]) / self.bollinger.lines.mid[0]
        if bb_width < 0.1:  # 布林带收窄
            signals.append("布林带收窄")
        
        return len(signals) > 0, signals

    def check_trend_signals(self):
        """检查趋势信号"""
        current_price = self.dataclose[0]
        signals = []
        
        # 均线支撑
        if current_price > self.sma_20[0] * 0.98:  # 接近20日均线
            signals.append("20日均线支撑")
        
        if current_price > self.sma_60[0] * 0.95:  # 接近60日均线
            signals.append("60日均线支撑")
        
        # 短期均线趋势
        if self.sma_20[0] > self.sma_20[-5]:  # 20日均线向上
            signals.append("短期趋势向上")
        
        return len(signals) > 0, signals

    def calculate_position_size(self, signal_score, entry_price, stop_price):
        """
        平衡型仓位计算 - 稳健风险管理
        """
        # 基础风险金额 - 保守基准
        cash = self.broker.get_cash()
        base_risk_amount = cash * self.p.max_risk_per_trade
        
        # 趋势加成 - 保守调整
        trend_multiplier = self.get_trend_multiplier()
        
        # 根据信号强度调整风险金额 - 更保守
        if signal_score >= 90:           # 超强信号
            risk_multiplier = 1.8 * min(trend_multiplier, 1.3)
        elif signal_score >= self.p.strong_signal_threshold:  # 强信号
            risk_multiplier = 1.5 * min(trend_multiplier, 1.2)
        elif signal_score >= 70:        # 中强信号
            risk_multiplier = 1.3 * min(trend_multiplier, 1.1)
        else:                           # 一般信号
            risk_multiplier = 1.0 * min(trend_multiplier, 1.05)
        
        adjusted_risk_amount = base_risk_amount * risk_multiplier
        
        # 计算每股风险
        risk_per_share = entry_price - stop_price
        if risk_per_share <= 0:
            return 0
        
        # 计算仓位大小
        position_size = adjusted_risk_amount / risk_per_share
        
        # 限制最大仓位 - 保守上限
        max_position_value = cash * self.p.max_position_ratio
        max_shares = max_position_value / entry_price
        
        # 最终仓位
        final_size = min(position_size, max_shares)
        
        return final_size if final_size > 0 else 0

    def get_trend_multiplier(self):
        """获取趋势乘数 - 顺势交易时增加仓位"""
        if len(self) < self.p.trend_period:
            return 1.0
        
        current_price = self.dataclose[0]
        trend_sma = self.sma_trend[0]
        momentum = self.momentum[0]
        ema_fast = self.ema_fast[0]
        ema_slow = self.ema_slow[0]
        
        # 趋势评分
        trend_score = 0
        
        # 价格vs趋势均线
        if current_price > trend_sma:
            trend_score += 1
        
        # 动量
        if momentum > 0:
            trend_score += 1
        
        # EMA金叉
        if ema_fast > ema_slow:
            trend_score += 1
        
        # 短期均线趋势
        if self.sma_20[0] > self.sma_20[-5]:
            trend_score += 1
        
        # 趋势乘数映射
        if trend_score >= 3:     # 强势上涨
            return 1.5
        elif trend_score >= 2:   # 温和上涨
            return 1.3
        elif trend_score >= 1:   # 弱势上涨
            return 1.1
        else:                    # 下跌趋势
            return 0.8

    def check_market_trend_filter(self, price):
        """
        市场趋势过滤器
        根据价格与均线关系判断大趋势
        """
        # 短期趋势
        short_trend = 'up' if price > self.sma_20[0] else 'down'
        
        # 中期趋势  
        medium_trend = 'up' if self.sma_20[0] > self.sma_60[0] else 'down'
        
        # 价格相对位置
        price_position = 'high' if price > self.sma_60[0] * 1.2 else 'normal'
        
        return {
            'short_trend': short_trend,
            'medium_trend': medium_trend, 
            'price_position': price_position
        }

    def should_enter_trade(self, signal_score, market_env):
        """
        平衡型交易判断 - 质量优于数量
        """
        # 基础条件检查
        if signal_score < self.p.min_signal_score:
            return False, "信号评分不足"
        
        # 连续亏损保护 - 适中限制
        if self.consecutive_losses >= 8:
            return False, "连续亏损过多，暂停交易"
        
        # 资金保护 - 适中保护
        current_value = self.broker.getvalue()
        if current_value < self.initial_cash * 0.75:  # 允许25%亏损
            return False, "账户亏损过大，暂停交易"
        
        # 每日交易限制 - 适中限制
        current_date = self.datas[0].datetime.date(0)
        if self.last_trade_date == current_date and self.daily_trades >= 6:
            return False, "今日交易次数已达上限"
        
        # 市场环境过滤 - 适中要求
        trend_info = self.check_market_trend_filter(self.dataclose[0])
        
        # 牛市顶部需要强信号
        if market_env == '牛市顶部' and signal_score < 80:
            return False, "牛市顶部需要强信号"
        
        return True, "可以交易"

    def log(self, txt, dt=None):
        """增强日志输出"""
        dt = dt or self.datas[0].datetime.date(0)
        print(f'{dt.isoformat()} {txt}')

    def notify_order(self, order):
        """订单状态通知"""
        if order.status in [order.Submitted, order.Accepted]:
            return
            
        if order.status in [order.Completed]:
            if order.isbuy():
                self.position_entry_price = order.executed.price
                self.position_entry_date = self.datas[0].datetime.date(0)
                
                # 更新交易统计
                if self.last_trade_date != self.position_entry_date:
                    self.daily_trades = 1
                    self.last_trade_date = self.position_entry_date
                else:
                    self.daily_trades += 1
                
                self.log(f'✅ 买入执行 - 价格: ${order.executed.price:.2f}, '
                        f'数量: {order.executed.size:.6f}, 手续费: ${order.executed.comm:.2f}')
                        
            elif order.issell():
                self.log(f'✅ 卖出执行 - 价格: ${order.executed.price:.2f}, '
                        f'数量: {order.executed.size:.6f}, 手续费: ${order.executed.comm:.2f}')
                
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log('❌ 订单被取消/保证金不足/被拒绝')
            
        self.order = None

    def notify_trade(self, trade):
        """交易完成通知"""
        if not trade.isclosed:
            return
            
        self.trade_count += 1
        profit_pct = (trade.pnl / abs(trade.value)) * 100 if trade.value != 0 else 0
        
        if trade.pnl > 0:
            self.win_count += 1
            self.consecutive_losses = 0
            self.log(f'🎉 盈利交易 - 利润: ${trade.pnl:.2f} ({profit_pct:.2f}%)')
        else:
            self.consecutive_losses += 1
            self.max_consecutive_losses = max(self.max_consecutive_losses, self.consecutive_losses)
            self.log(f'📉 亏损交易 - 亏损: ${trade.pnl:.2f} ({profit_pct:.2f}%)')
        
        self.total_profit += trade.pnl
        
        # 重置持仓相关变量
        self.position_entry_price = None
        self.position_entry_date = None
        self.stop_price = None
        self.profit_target = None
        self.trailing_stop = None
        
        current_value = self.broker.getvalue()
        self.log(f'📊 账户价值: ${current_value:.2f}, 总收益: ${self.total_profit:.2f}')

    def next(self):
        """策略主逻辑"""
        if self.order:
            return
        
        current_datetime = self.datas[0].datetime.datetime(0)
        current_price = self.dataclose[0]
        
        # 持仓管理
        if self.position:
            self.manage_position(current_price)
            return
        
        # 寻找买入机会
        if len(self.divergence_data) > 0 and current_datetime in self.divergence_data.index:
            signal_data = self.divergence_data.loc[current_datetime]
            self.evaluate_buy_signal(signal_data, current_price)

    def evaluate_buy_signal(self, signal_data, current_price):
        """评估买入信号 - 多重技术指标确认版"""
        signal_score = signal_data['综合评分']
        market_env = signal_data['市场环境']
        signal_strength = signal_data['信号强度']
        j_range = signal_data['J值区间']
        
        # 判断是否可以交易
        can_trade, reason = self.should_enter_trade(signal_score, market_env)
        if not can_trade:
            self.log(f'⚠️  跳过信号 - 评分: {signal_score:.0f}, 原因: {reason}')
            return
        
        # 多重技术指标确认检查
        indicators_confirmed, confirmation_details, confirmation_score = self.check_technical_indicators_confirmation()
        if not indicators_confirmed:
            self.log(f'⚠️  跳过信号 - 评分: {signal_score:.0f}, 技术指标确认不足: {confirmation_details}')
            return
        
        # 根据确认强度调整信号评分
        enhanced_score = min(signal_score + confirmation_score * 0.2, 100)  # 最多加20分
        
        # 计算止损位
        atr_value = self.atr[0]
        stop_price = current_price - atr_value * self.p.atr_stop_multiplier
        
        # 技术止损位（前低点保护）- 1小时周期缩短回看期
        lookback_period = 12
        if len(self) > lookback_period:
            recent_low = min([self.datalow[-i] for i in range(1, lookback_period + 1)])
            technical_stop = recent_low * 0.99  # 前低点下方1%
            stop_price = max(stop_price, technical_stop)  # 取较高的止损位
        
        # 计算目标价
        risk_amount = current_price - stop_price
        profit_target = current_price + risk_amount * self.p.profit_target_ratio
        
        # 计算仓位大小（使用增强评分）
        position_size = self.calculate_position_size(enhanced_score, current_price, stop_price)
        
        if position_size > 0:
            # 记录交易信息
            self.stop_price = stop_price
            self.profit_target = profit_target
            self.trailing_stop = stop_price
            
            # 执行买入
            self.order = self.buy(size=position_size)
            
            stop_pct = ((current_price - stop_price) / current_price) * 100
            target_pct = ((profit_target - current_price) / current_price) * 100
            position_value = position_size * current_price
            
            self.log(f'🚀 买入信号触发!')
            self.log(f'   📊 信号评分: {signal_score:.0f}/100 → {enhanced_score:.0f}/100 ({signal_strength})')
            self.log(f'   🌍 市场环境: {market_env}')
            self.log(f'   📈 J值区间: {j_range}')
            self.log(f'   ✅ 技术指标确认({confirmation_score}分): {confirmation_details}')
            self.log(f'   💰 仓位价值: ${position_value:.2f}')
            self.log(f'   🛡️  止损: ${stop_price:.2f} (-{stop_pct:.1f}%)')
            self.log(f'   🎯 目标: ${profit_target:.2f} (+{target_pct:.1f}%)')

    def manage_position(self, current_price):
        """持仓管理 - 平衡型风格"""
        if not self.position or not self.position_entry_price:
            return
        
        # 检查最大持仓时间
        if self.position_entry_date:
            days_held = (self.datas[0].datetime.date(0) - self.position_entry_date).days
            if days_held > self.p.max_hold_days:
                self.log(f'⏰ 超过最大持仓期限({self.p.max_hold_days}天)，强制平仓')
                self.close()
                return
        
        current_profit_pct = ((current_price - self.position_entry_price) / self.position_entry_price) * 100
        
        # 移动止损逻辑 - 平衡型
        if current_profit_pct > 5:  # 盈利5%以上才启动移动止损
            new_trailing_stop = current_price * (1 - self.p.trailing_stop_ratio)
            if new_trailing_stop > self.trailing_stop:
                self.trailing_stop = new_trailing_stop
                self.log(f'📈 移动止损更新: ${self.trailing_stop:.2f} (盈利{current_profit_pct:.1f}%)')
        elif current_profit_pct > 2:  # 盈利2%-5%时，保护利润
            protection_stop = self.position_entry_price * 1.005  # 保本+0.5%
            if protection_stop > self.trailing_stop:
                self.trailing_stop = protection_stop
                self.log(f'🛡️ 利润保护止损: ${self.trailing_stop:.2f}')
        
        # 止损检查
        if current_price <= self.trailing_stop:
            self.log(f'🛑 移动止损触发 - 当前价: ${current_price:.2f}, 收益: {current_profit_pct:.2f}%')
            self.close()
            return
        
        # 动态止盈检查 - 稍微保守
        trend_multiplier = self.get_trend_multiplier()
        dynamic_target = self.profit_target * max(trend_multiplier, 1.0)  # 最低不缩减目标
        
        # 止盈逻辑
        if current_price >= dynamic_target:
            self.log(f'🎯 目标价触达 - 当前价: ${current_price:.2f}, 收益: {current_profit_pct:.2f}%')
            self.close()
            return

    def stop(self):
        """策略结束时的统计报告"""
        win_rate = (self.win_count / self.trade_count * 100) if self.trade_count > 0 else 0
        total_return = (self.broker.getvalue() - self.initial_cash) / self.initial_cash * 100
        
        print("\n" + "="*80)
        print("📊 高级背离策略回测结果")
        print("="*80)
        print(f'💰 初始资金: ${self.initial_cash:,.2f}')
        print(f'💰 最终价值: ${self.broker.getvalue():,.2f}')
        print(f'📈 总收益率: {total_return:.2f}%')
        print(f'🎯 总交易数: {self.trade_count}')
        print(f'✅ 获胜次数: {self.win_count}')
        print(f'📊 胜率: {win_rate:.1f}%')
        print(f'📉 最大连续亏损: {self.max_consecutive_losses}')
        print(f'💵 总利润: ${self.total_profit:.2f}')
        
        if self.trade_count > 0:
            avg_profit = self.total_profit / self.trade_count
            print(f'💹 平均每笔收益: ${avg_profit:.2f}')


def create_advanced_backtest_engine():
    """创建高级回测引擎"""
    cerebro = bt.Cerebro()
    
    # 添加高级策略 - 1小时短线交易参数优化
    cerebro.addstrategy(AdvancedDivergenceStrategy,
                       target_timeframe='1h',          # 改为1小时周期，更多机会
                       min_signal_score=65,             # 降低信号门槛到65分
                       max_risk_per_trade=0.012,        # 提高单笔风险到1.2%
                       strong_signal_threshold=75,      # 降低强信号门槛到75分
                       max_position_ratio=0.15,         # 提高最大仓位到15%
                       atr_period=10,                   # 缩短ATR周期
                       atr_stop_multiplier=2.0,          # 更紧的初始止损
                       trailing_stop_ratio=0.03,         # 更紧的移动止损3%
                       profit_target_ratio=2.2,          # 提高风险回报比到2.2:1
                       max_hold_days=8)                  # 延长最大持仓到8天
    
    # 加载数据
    data_file = 'crypto_data/BTCUSDT_1h.csv'
    if not os.path.exists(data_file):
        print(f"❌ 数据文件不存在: {data_file}")
        return None
        
    df = pd.read_csv(data_file)
    df['开盘时间'] = pd.to_datetime(df['开盘时间'])
    df.set_index('开盘时间', inplace=True)
    
    # 重命名列
    df = df.rename(columns={
        '开盘价': 'open', '最高价': 'high', '最低价': 'low',
        '收盘价': 'close', '成交量': 'volume'
    })
    
    data = bt.feeds.PandasData(dataname=df)
    cerebro.adddata(data)
    
    print(f"📊 加载价格数据: {len(df)} 条记录")
    
    # 设置回测参数
    cerebro.broker.setcash(100000.0)  # 10万初始资金
    cerebro.broker.setcommission(commission=0.001)  # 0.1%手续费
    
    # 添加分析器
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
    
    return cerebro


def run_advanced_backtest():
    """运行高级背离策略回测"""
    print("🚀 启动高级背离策略回测")
    print("基于专业交易策略文档")
    print("="*80)
    
    cerebro = create_advanced_backtest_engine()
    if cerebro is None:
        return
    
    # 运行回测
    initial_cash = cerebro.broker.getvalue()
    print(f'💰 初始资金: ${initial_cash:,.2f}')
    
    results = cerebro.run()
    strategy = results[0]
    
    # 最终结果
    final_value = cerebro.broker.getvalue()
    total_return = (final_value - initial_cash) / initial_cash * 100
    
    print(f'\n📈 回测完成')
    print(f'💰 最终价值: ${final_value:,.2f}')
    print(f'📊 总收益率: {total_return:.2f}%')
    
    # 详细分析
    try:
        trades = strategy.analyzers.trades.get_analysis()
        if "total" in trades and trades.total.total > 0:
            print(f'\n📊 详细统计:')
            print(f'🎯 总交易: {trades.total.total}')
            
            if "won" in trades and "lost" in trades:
                win_rate = trades.won.total/(trades.won.total + trades.lost.total)*100
                print(f'✅ 盈利: {trades.won.total} | ❌ 亏损: {trades.lost.total}')
                print(f'📈 胜率: {win_rate:.1f}%')
                
                if trades.won.total > 0:
                    print(f'💹 平均盈利: ${trades.won.pnl.average:.2f}')
                if trades.lost.total > 0:
                    print(f'📉 平均亏损: ${trades.lost.pnl.average:.2f}')
        
        # 其他指标
        sharpe = strategy.analyzers.sharpe.get_analysis()
        if 'sharperatio' in sharpe and sharpe['sharperatio'] is not None:
            print(f'📊 夏普比率: {sharpe["sharperatio"]:.3f}')
        
        drawdown = strategy.analyzers.drawdown.get_analysis()
        if 'max' in drawdown:
            print(f'📉 最大回撤: {drawdown["max"]["drawdown"]:.2f}%')
            
    except Exception as e:
        print(f"❌ 分析结果获取失败: {e}")


if __name__ == "__main__":
    run_advanced_backtest() 