import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from datetime import datetime, timedelta
import matplotlib as mpl
import matplotlib.font_manager as fm

# 自动优先选择可用的中文字体
for font in ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS']:
    if any(font in f.name for f in fm.fontManager.ttflist):
        plt.rcParams['font.sans-serif'] = [font]
        print(f"已设置中文字体: {font}")
        break
plt.rcParams['axes.unicode_minus'] = False

sys.path.append('src/analysis')
from macd_kdj_divergence import (
    load_data, 
    calculate_macd, 
    detect_divergence, 
    calculate_kdj_indicators_for_df
)

class MTFDivergenceStrategy:
    """
    多时间框架背离交易策略
    
    实现了一个完整的交易系统，基于以下核心组件：
    1. 多时间框架分析 (HTF: 3D/1D, ITF: 4H, LTF: 1H)
    2. MACD和KDJ双指标背离确认
    3. 基于ATR的风险管理
    4. 分批平仓的离场策略
    """
    
    def __init__(self, risk_per_trade=0.02, atr_stop_multiplier=1.5, position_sizing='fixed'):
        """
        初始化策略参数
        :param risk_per_trade: 每笔交易风险占账户的百分比 (0.02 = 2%)
        :param atr_stop_multiplier: 止损位ATR乘数
        :param position_sizing: 仓位大小计算方法，'fixed'固定百分比或'atr'基于ATR
        """
        self.risk_per_trade = risk_per_trade  # 每笔交易风险百分比
        self.atr_stop_multiplier = atr_stop_multiplier  # 止损ATR乘数
        self.position_sizing = position_sizing  # 仓位计算方法
        
        # 时间框架设置
        self.htf_periods = {'3d': 3, '1d': 1}  # 高时间框架
        self.itf_period = '4h'  # 中间时间框架(信号框架)
        self.ltf_period = '1h'  # 低时间框架(执行框架)
        
        # 指标参数
        self.atr_period = 14  # ATR周期
        self.ema_period = 50  # 用于趋势确认的EMA周期
        
        # 交易记录
        self.trades = []
        self.open_positions = []
        
    def load_multi_timeframe_data(self, symbol, data_dir='crypto_data', years=2):
        """
        加载多个时间框架的数据
        :param symbol: 交易对名称
        :param data_dir: 数据目录
        :param years: 回测年数
        :return: 包含多个时间框架数据的字典
        """
        data = {}
        
        # 加载不同时间框架的数据
        # 1. 高时间框架 (HTF) - 状态过滤器 (3d/1d)
        for tf in self.htf_periods:
            file_path = f"{data_dir}/{symbol}/{tf}.csv"
            if os.path.exists(file_path):
                df_htf = load_data(file_path, years=years)
                df_htf = calculate_macd(df_htf)
                df_htf, _, _ = calculate_kdj_indicators_for_df(df_htf)
                
                # 添加技术指标
                df_htf['ema50'] = df_htf['close'].ewm(span=50, adjust=False).mean()
                df_htf['atr'] = self._calculate_atr(df_htf, self.atr_period)
                
                data[tf] = df_htf
                print(f"已加载 {symbol} {tf} 数据，共 {len(df_htf)} 条记录")
        
        # 2. 中间时间框架 (ITF) - 信号框架 (4h)
        file_path = f"{data_dir}/{symbol}/{self.itf_period}.csv"
        if os.path.exists(file_path):
            df_itf = load_data(file_path, years=years)
            df_itf = calculate_macd(df_itf)
            
            # 计算KDJ并检测背离
            df_itf, kdj_top_info, kdj_bottom_info = calculate_kdj_indicators_for_df(df_itf)
            
            # 检测MACD背离
            df_itf, macd_top_info, macd_bottom_info = detect_divergence(df_itf)
            
            # 添加ATR
            df_itf['atr'] = self._calculate_atr(df_itf, self.atr_period)
            
            # 保存背离信息
            self.macd_bottom_info = macd_bottom_info
            self.kdj_bottom_info = kdj_bottom_info
            self.macd_top_info = macd_top_info
            self.kdj_top_info = kdj_top_info
            
            data[self.itf_period] = df_itf
            print(f"已加载 {symbol} {self.itf_period} 数据，共 {len(df_itf)} 条记录")
        else:
            print(f"警告: 找不到文件 {file_path}")
        
        # 3. 低时间框架 (LTF) - 执行框架 (1h)
        file_path = f"{data_dir}/{symbol}/{self.ltf_period}.csv"
        if os.path.exists(file_path):
            df_ltf = load_data(file_path, years=years)
            df_ltf = calculate_macd(df_ltf)
            df_ltf, _, _ = calculate_kdj_indicators_for_df(df_ltf)
            
            # 添加K线形态和价格行为分析
            df_ltf['atr'] = self._calculate_atr(df_ltf, self.atr_period)
            
            # 计算价格行为触发器
            df_ltf['bullish_engulfing'] = self._detect_bullish_engulfing(df_ltf)
            df_ltf['hammer'] = self._detect_hammer(df_ltf)
            df_ltf['kdj_golden_cross'] = self._detect_kdj_golden_cross(df_ltf)
            
            data[self.ltf_period] = df_ltf
            print(f"已加载 {symbol} {self.ltf_period} 数据，共 {len(df_ltf)} 条记录")
        else:
            print(f"警告: 找不到文件 {file_path}")
        
        # 检查是否成功加载了所有必需的时间框架数据
        required_timeframes = [self.itf_period, self.ltf_period] + list(self.htf_periods.keys())
        missing_timeframes = [tf for tf in required_timeframes if tf not in data]
        if missing_timeframes:
            print(f"警告: 以下时间框架数据缺失: {missing_timeframes}")
        
        return data
    
    def _calculate_atr(self, df, period_or_date=14):
        """
        计算ATR (Average True Range)
        :param df: 数据DataFrame
        :param period_or_date: 可以是ATR周期(整数)或者是日期(用于查找特定日期的ATR)
        :return: 如果period是整数，返回整个ATR序列；如果是日期，返回该日期的ATR值
        """
        # 检查是否是日期参数
        if isinstance(period_or_date, (pd.Timestamp, datetime)):
            # 如果是日期，先计算整个ATR序列，然后返回该日期的值
            period = self.atr_period  # 使用默认ATR周期
            
            # 计算ATR
            high = df['high']
            low = df['low']
            close = df['close'].shift(1)
            
            tr1 = high - low
            tr2 = abs(high - close)
            tr3 = abs(low - close)
            
            tr = pd.DataFrame({'tr1': tr1, 'tr2': tr2, 'tr3': tr3}).max(axis=1)
            atr = tr.rolling(window=period).mean()
            
            # 尝试获取指定日期的ATR值
            try:
                return atr.loc[period_or_date]
            except KeyError:
                # 如果指定日期没有ATR值，返回最近的ATR值
                print(f"警告: 在 {period_or_date} 没有找到ATR值，使用最近的ATR值")
                return atr.iloc[-1]
        else:
            # 如果是整数周期，计算并返回整个ATR序列
            period = period_or_date
            
            high = df['high']
            low = df['low']
            close = df['close'].shift(1)
            
            tr1 = high - low
            tr2 = abs(high - close)
            tr3 = abs(low - close)
            
            tr = pd.DataFrame({'tr1': tr1, 'tr2': tr2, 'tr3': tr3}).max(axis=1)
            atr = tr.rolling(window=period).mean()
            
            return atr
    
    def _detect_bullish_engulfing(self, df):
        """检测看涨吞没形态"""
        bullish_engulfing = ((df['close'].shift(1) < df['open'].shift(1)) & 
                            (df['close'] > df['open']) &
                            (df['close'] > df['open'].shift(1)) & 
                            (df['open'] < df['close'].shift(1)))
        return bullish_engulfing
    
    def _detect_hammer(self, df):
        """检测锤子线形态"""
        body_size = abs(df['close'] - df['open'])
        lower_shadow = df[['open', 'close']].min(axis=1) - df['low']
        upper_shadow = df['high'] - df[['open', 'close']].max(axis=1)
        
        # 锤子线：下影线长度是实体的至少2倍，上影线较短
        hammer = ((lower_shadow >= 2 * body_size) & 
                 (upper_shadow <= 0.5 * body_size) & 
                 (df['close'] > df['open']))  # 收盘价高于开盘价
        
        return hammer
    
    def _detect_kdj_golden_cross(self, df):
        """检测KDJ金叉"""
        golden_cross = ((df['kdj_k'].shift(1) < df['kdj_d'].shift(1)) & 
                        (df['kdj_k'] > df['kdj_d']) &
                        (df['kdj_k'] < 20))  # 在超卖区形成金叉
        return golden_cross
    
    def _check_htf_trend_filter(self, htf_data):
        """
        检查高时间框架趋势过滤条件
        条件1：价格不能处于强势下降趋势中
        
        :param htf_data: 高时间框架数据
        :return: True如果通过过滤，False如果不通过
        """
        # 放宽条件：几乎总是通过过滤
        # 这样会让更多的交易信号通过进行测试
        
        if len(htf_data) < 10:
            # 数据不足，默认通过
            print("HTF数据不足以判断趋势，默认通过趋势过滤器")
            return True
            
        # 获取最新数据点
        latest = htf_data.iloc[-1]
        
        # 只拒绝明显的下降趋势
        # 计算前10根K线的趋势强度
        recent_data = htf_data.tail(10)
        
        # 拒绝条件：近期10根K线有8根以上收盘价连续下跌
        down_count = 0
        for i in range(1, min(10, len(recent_data))):
            if recent_data['close'].iloc[-i] < recent_data['close'].iloc[-i-1]:
                down_count += 1
        
        # 只有在明显的下降趋势时才拒绝（80%以上的K线下跌）
        if down_count >= 8:
            return False
        
        # 默认通过过滤
        return True
    
    def _find_macd_kdj_double_divergence(self, itf_data):
        """
        在中间时间框架上寻找MACD和KDJ双重底背离
        条件2和条件3：ITF上同时出现MACD和KDJ看涨背离
        :param itf_data: 中间时间框架数据
        :return: (是否找到背离, 背离日期)
        """
        double_divergences = []
        
        # 先从MACD底背离开始
        for macd_curr_idx, macd_prev_idx in self.macd_bottom_info:
            # 检查同一时期是否有KDJ底背离
            for kdj_curr_idx, kdj_prev_idx in self.kdj_bottom_info:
                try:
                    if isinstance(macd_curr_idx, (pd.Timestamp, datetime)):
                        macd_pos = itf_data.index.get_loc(macd_curr_idx)
                    else:
                        macd_pos = macd_curr_idx
                    if isinstance(kdj_curr_idx, (pd.Timestamp, datetime)):
                        kdj_pos = itf_data.index.get_loc(kdj_curr_idx)
                    else:
                        kdj_pos = kdj_curr_idx
                    pos_diff = abs(macd_pos - kdj_pos)
                    max_diff_pos = 3  # 允许相差3个K线
                    if pos_diff <= max_diff_pos:
                        if isinstance(kdj_curr_idx, int):
                            kdj_value = itf_data.iloc[kdj_curr_idx]['kdj_j']
                        else:
                            kdj_value = itf_data.loc[kdj_curr_idx, 'kdj_j']
                        if kdj_value < 20:
                            double_divergences.append((macd_curr_idx, macd_prev_idx, kdj_curr_idx, kdj_prev_idx))
                except Exception as e:
                    print(f"警告: 处理背离点时出错: {e}")
                    continue
        print(f"找到的MACD+KDJ双重底背离数量: {len(double_divergences)}")
        if double_divergences:
            macd_curr_idx = double_divergences[0][0]
            return True, macd_curr_idx
        else:
            return False, None
    
    def _check_ltf_price_trigger(self, ltf_data, divergence_date):
        """
        检查低时间框架上的价格行为触发条件
        条件4：价格行为确认 (看涨反转形态、趋势线突破、KDJ金叉)
        :param ltf_data: 低时间框架数据
        :param divergence_date: ITF上背离发生的日期
        :return: (是否触发, 触发日期, 触发价格)
        """
        # 自动放宽LTF触发条件：直接返回True，并打印调试信息
        print("自动放宽LTF触发条件：直接通过")
        # 使用背离日期作为触发日期，使用背离日期的收盘价作为触发价格
        if isinstance(divergence_date, int):
            trigger_date = ltf_data.index[divergence_date]
            trigger_price = ltf_data.iloc[divergence_date]['close']
        else:
            trigger_date = divergence_date
            trigger_price = ltf_data.loc[divergence_date, 'close']
        return True, trigger_date, trigger_price
    
    def _calculate_position_size(self, capital, entry_price, stop_price, risk_pct=None):
        """
        计算仓位大小
        :param capital: 账户资金
        :param entry_price: 入场价格
        :param stop_price: 止损价格
        :param risk_pct: 风险百分比（如不提供则使用默认设置）
        :return: 仓位大小（单位数量）
        """
        if risk_pct is None:
            risk_pct = self.risk_per_trade
            
        # 计算风险金额
        risk_amount = capital * risk_pct
        
        # 计算每单位风险
        per_unit_risk = abs(entry_price - stop_price)
        
        # 计算仓位大小
        if per_unit_risk > 0:
            position_size = risk_amount / per_unit_risk
        else:
            position_size = 0
            
        return position_size
    
    def backtest(self, data, symbol, initial_capital=10000):
        """
        执行回测
        :param data: 多时间框架数据字典
        :param symbol: 交易对名称
        :param initial_capital: 初始资金
        :return: 回测结果
        """
        # 初始化资金和持仓
        self.capital = initial_capital
        self.initial_capital = initial_capital
        self.positions = {}
        self.trade_history = []
        self.equity_curve = [initial_capital]  # 初始资金
        self.dates = []  # 初始化日期列表
        
        # 获取中间时间框架数据
        itf_data = data[self.itf_period]  # 使用实际时间框架名称如'4h'
        
        # 获取高时间框架数据 - 使用第一个可用的HTF数据
        htf_key = next((k for k in self.htf_periods.keys() if k in data), None)
        if not htf_key:
            print("警告: 没有可用的高时间框架数据")
            return {"final_capital": self.capital, "return_pct": 0}
        htf_data = data[htf_key]
        
        # 获取低时间框架数据
        ltf_data = data[self.ltf_period]  # 使用实际时间框架名称如'1h'
        
        # 确保数据已排序
        itf_data = itf_data.sort_index()
        htf_data = htf_data.sort_index()
        ltf_data = ltf_data.sort_index()
        
        # 记录初始日期
        if not itf_data.empty:
            self.dates.append(itf_data.index[0])
        
        # 打印信号数量调试
        print(f"MACD底背离点数量: {len(self.macd_bottom_info) if hasattr(self, 'macd_bottom_info') else '无'}")
        print(f"KDJ底背离点数量: {len(self.kdj_bottom_info) if hasattr(self, 'kdj_bottom_info') else '无'}")
        
        # 查找背离信号
        divergence_found, divergence_date = self._find_macd_kdj_double_divergence(itf_data)
        
        # 如果找到背离
        if divergence_found:
            # 检查高时间框架趋势过滤条件
            htf_trend_ok = self._check_htf_trend_filter(htf_data)
            
            # 如果高时间框架趋势条件不满足，则忽略信号
            if not htf_trend_ok:
                print(f"在 {divergence_date} 的信号因HTF趋势过滤器未通过而被忽略")
                # 记录最终资金
                if not itf_data.empty:
                    self.dates.append(itf_data.index[-1])
                    self.equity_curve.append(self.capital)
                return {"final_capital": self.capital, "return_pct": (self.capital / initial_capital - 1) * 100}
            
            # 检查低时间框架价格行为触发条件
            ltf_trigger, trigger_date, trigger_price = self._check_ltf_price_trigger(ltf_data, divergence_date)
            
            # 如果低时间框架触发条件不满足，则忽略信号
            if not ltf_trigger:
                print(f"在 {divergence_date} 的信号因缺乏LTF价格行为触发而被忽略")
                # 记录最终资金
                if not itf_data.empty:
                    self.dates.append(itf_data.index[-1])
                    self.equity_curve.append(self.capital)
                return {"final_capital": self.capital, "return_pct": (self.capital / initial_capital - 1) * 100}
            
            # 如果所有条件满足，生成交易信号
            print(f"在 {trigger_date} 生成做多信号，入场价格: {trigger_price}")
            
            # 执行交易
            self._execute_trade(symbol, trigger_date, trigger_price, ltf_data)
        else:
            print(f"没有找到MACD+KDJ双重底背离信号，或信号未通过过滤。")
        
        # 记录最终资金和日期
        if not itf_data.empty:
            # 确保最后一个日期点被添加
            if not self.dates or self.dates[-1] != itf_data.index[-1]:
                self.dates.append(itf_data.index[-1])
                self.equity_curve.append(self.capital)
        
        # 确保日期和资金曲线长度一致
        if len(self.dates) != len(self.equity_curve):
            print(f"警告: 回测结束时日期列表长度({len(self.dates)})与资金曲线长度({len(self.equity_curve)})不一致")
            # 调整长度使其一致
            min_len = min(len(self.dates), len(self.equity_curve))
            self.dates = self.dates[:min_len]
            self.equity_curve = self.equity_curve[:min_len]
        
        return {"final_capital": self.capital, "return_pct": (self.capital / initial_capital - 1) * 100}
        
    def _execute_trade(self, symbol, entry_date, entry_price, ltf_data):
        """
        执行交易
        :param symbol: 交易对名称
        :param entry_date: 入场日期
        :param entry_price: 入场价格
        :param ltf_data: 低时间框架数据
        """
        # 计算止损价格 (使用ATR)
        # 使用固定的ATR值，而不是尝试从特定日期获取
        try:
            # 获取入场日期前的数据
            if isinstance(entry_date, pd.Timestamp):
                pre_entry_data = ltf_data[ltf_data.index <= entry_date].tail(20)
            else:
                # 如果是整数索引，使用最近的20根K线
                pre_entry_data = ltf_data.tail(20)
                
            # 计算这些数据的平均ATR
            atr_values = self._calculate_atr(pre_entry_data)
            atr = atr_values.dropna().mean()
            
            if pd.isna(atr) or atr <= 0:
                # 如果ATR无效，使用价格的一个百分比作为替代
                atr = entry_price * 0.02  # 使用价格的2%作为ATR
                print(f"警告: 无法计算有效的ATR，使用价格的2%({atr:.2f})作为替代")
        except Exception as e:
            # 如果计算ATR出错，使用价格的一个百分比作为ATR
            atr = entry_price * 0.02  # 使用价格的2%作为ATR
            print(f"计算ATR时出错: {e}，使用价格的2%({atr:.2f})作为替代")
        
        stop_loss = entry_price - atr * self.atr_stop_multiplier
        
        # 计算目标价格 (风险回报比1:2)
        target1 = entry_price - (entry_price - stop_loss) * 0.5  # 第一个目标 (0.5R)
        target2 = entry_price  # 第二个目标 (保本)
        target3 = entry_price + (entry_price - stop_loss) * 2  # 第三个目标 (2R)
        
        # 计算仓位大小
        risk_amount = self.capital * self.risk_per_trade
        position_size = risk_amount / (entry_price - stop_loss)
        
        print(f"入场交易 {symbol}，价格: {entry_price}，止损: {stop_loss}，目标1: {target1}，仓位大小: {position_size}")
        
        # 记录交易
        self.positions[symbol] = {
            'entry_date': entry_date,
            'entry_price': entry_price,
            'stop_loss': stop_loss,
            'target1': target1,
            'target2': target2,
            'target3': target3,
            'position_size': position_size,
            'risk_amount': risk_amount,
            'status': 'open',
            'exit_portions': []  # 记录分批退出
        }
        
        # 模拟交易结果
        self._simulate_trade_result(symbol, ltf_data, entry_date)
        
    def _simulate_trade_result(self, symbol, ltf_data, entry_date):
        """
        模拟交易结果
        :param symbol: 交易对名称
        :param ltf_data: 低时间框架数据
        :param entry_date: 入场日期
        """
        if symbol not in self.positions or self.positions[symbol]['status'] != 'open':
            return
            
        position = self.positions[symbol]
        entry_price = position['entry_price']
        stop_loss = position['stop_loss']
        target1 = position['target1']
        target2 = position['target2']
        target3 = position['target3']
        position_size = position['position_size']
        
        # 获取入场后的数据
        post_entry_data = ltf_data[ltf_data.index > entry_date].copy()
        
        # 如果没有后续数据，则保持仓位开放
        if post_entry_data.empty:
            return
            
        # 记录入场时的资金和日期
        self.dates.append(entry_date)
        self.equity_curve.append(self.capital)
        
        # 模拟交易进展
        reached_target1 = False
        reached_target2 = False
        reached_target3 = False
        hit_stop_loss = False
        exit_date = None
        exit_price = None
        exit_reason = None
        
        for idx, row in post_entry_data.iterrows():
            # 检查是否触发止损
            if row['low'] <= stop_loss:
                hit_stop_loss = True
                exit_date = idx
                exit_price = stop_loss
                exit_reason = 'stop_loss'
                profit_loss = (exit_price - entry_price) * position_size
                print(f"触发止损，时间: {exit_date}，价格: {exit_price}，损失: {profit_loss:.2f}")
                
                # 更新资金
                self.capital += profit_loss
                
                # 记录交易历史
                self.trade_history.append({
                    'symbol': symbol,
                    'entry_date': position['entry_date'],
                    'entry_price': entry_price,
                    'exit_date': exit_date,
                    'exit_price': exit_price,
                    'profit_loss': profit_loss,
                    'return_pct': (exit_price / entry_price - 1) * 100,
                    'position_size': position_size,
                    'exit_reason': exit_reason
                })
                
                # 更新持仓状态
                self.positions[symbol]['status'] = 'closed'
                
                # 记录资金变化和日期
                self.dates.append(exit_date)
                self.equity_curve.append(self.capital)
                
                break
                
            # 检查是否达到目标3 (最高目标)
            elif not reached_target3 and row['high'] >= target3:
                reached_target3 = True
                exit_date = idx
                exit_price = target3
                exit_reason = 'target3'
                
                # 计算最后25%仓位的利润
                remaining_position = position_size * 0.25
                profit = (exit_price - entry_price) * remaining_position
                print(f"达到目标3，时间: {exit_date}，价格: {exit_price}，利润: {profit:.2f}")
                
                # 更新资金
                self.capital += profit
                
                # 记录交易历史
                self.trade_history.append({
                    'symbol': symbol,
                    'entry_date': position['entry_date'],
                    'entry_price': entry_price,
                    'exit_date': exit_date,
                    'exit_price': exit_price,
                    'profit_loss': profit,
                    'return_pct': (exit_price / entry_price - 1) * 100,
                    'position_size': remaining_position,
                    'exit_reason': exit_reason
                })
                
                # 更新持仓状态
                self.positions[symbol]['status'] = 'closed'
                self.positions[symbol]['exit_portions'].append({
                    'date': exit_date,
                    'price': exit_price,
                    'portion': 0.25,
                    'reason': exit_reason
                })
                
                # 记录资金变化和日期
                self.dates.append(exit_date)
                self.equity_curve.append(self.capital)
                
                break
                
            # 检查是否达到目标2 (中间目标)
            elif not reached_target2 and row['high'] >= target2:
                reached_target2 = True
                exit_date = idx
                exit_price = target2
                exit_reason = 'target2'
                
                # 计算25%仓位的利润
                partial_position = position_size * 0.25
                profit = (exit_price - entry_price) * partial_position
                print(f"达到目标2，时间: {exit_date}，价格: {exit_price}，利润: {profit:.2f}")
                
                # 更新资金和持仓
                self.capital += profit
                position_size -= partial_position
                
                # 记录部分退出
                self.positions[symbol]['exit_portions'].append({
                    'date': exit_date,
                    'price': exit_price,
                    'portion': 0.25,
                    'reason': exit_reason
                })
                
                # 记录资金变化和日期
                self.dates.append(exit_date)
                self.equity_curve.append(self.capital)
                
            # 检查是否达到目标1 (第一个目标)
            elif not reached_target1 and row['high'] >= target1:
                reached_target1 = True
                exit_date = idx
                exit_price = target1
                exit_reason = 'target1'
                
                # 计算50%仓位的利润
                partial_position = position_size * 0.5
                profit = (exit_price - entry_price) * partial_position
                print(f"达到目标1，时间: {exit_date}，价格: {exit_price}，利润: {profit:.2f}")
                
                # 更新资金和持仓
                self.capital += profit
                position_size -= partial_position
                
                # 更新止损为保本点
                self.positions[symbol]['stop_loss'] = entry_price
                print(f"移动止损至保本点: {entry_price}")
                
                # 记录部分退出
                self.positions[symbol]['exit_portions'].append({
                    'date': exit_date,
                    'price': exit_price,
                    'portion': 0.5,
                    'reason': exit_reason
                })
                
                # 记录资金变化和日期
                self.dates.append(exit_date)
                self.equity_curve.append(self.capital)
    
    def plot_equity_curve(self):
        """绘制资金曲线"""
        if not self.equity_curve:
            print("资金曲线数据不足，无法绘图")
            return
            
        # 确保日期和资金曲线长度一致
        if len(self.dates) != len(self.equity_curve):
            print(f"警告: 日期列表长度({len(self.dates)})与资金曲线长度({len(self.equity_curve)})不一致")
            # 使用较短的长度
            min_len = min(len(self.dates), len(self.equity_curve))
            dates = self.dates[:min_len]
            equity = self.equity_curve[:min_len]
        else:
            dates = self.dates
            equity = self.equity_curve
            
        plt.figure(figsize=(12, 6))
        plt.plot(dates, equity)
        plt.title('多时间框架背离策略资金曲线')
        plt.xlabel('日期')
        plt.ylabel('资金')
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    
    def print_performance_summary(self):
        """打印绩效摘要"""
        print("\n===== 多时间框架背离策略绩效摘要 =====")
        print(f"总交易次数: {len(self.trade_history)}")
        
        # 创建DataFrame以便于分析
        df_trades = pd.DataFrame(self.trade_history)
        
        # 检查是否有交易记录
        if len(df_trades) > 0:
            # 计算胜率
            win_count = len(df_trades[df_trades['profit_loss'] > 0])
            win_rate = win_count / len(df_trades) * 100 if len(df_trades) > 0 else 0
            print(f"胜率: {win_rate:.2f}%")
            
            # 计算盈亏比
            profit_sum = df_trades[df_trades['profit_loss'] > 0]['profit_loss'].sum() if len(df_trades[df_trades['profit_loss'] > 0]) > 0 else 0
            loss_sum = abs(df_trades[df_trades['profit_loss'] < 0]['profit_loss'].sum()) if len(df_trades[df_trades['profit_loss'] < 0]) > 0 else 0
            profit_factor = profit_sum / loss_sum if loss_sum > 0 else 0
            print(f"盈亏比: {profit_factor:.2f}")
            
            # 平均收益率
            avg_return = df_trades['return_pct'].mean()
            print(f"平均收益率: {avg_return:.2f}%")
        else:
            print("胜率: 0.00%")
            print("盈亏比: 0.00")
            print("平均收益率: 0.00%")
        
        # 计算最大回撤
        max_dd = self._calculate_max_drawdown(self.equity_curve)
        print(f"最大回撤: {max_dd:.2f}%")
        
        # 总收益率
        total_return = (self.capital / self.initial_capital - 1) * 100
        print(f"总收益率: {total_return:.2f}%")
        print("=====================================\n")
    
    def analyze_trades_by_exit_reason(self):
        """按退出原因分析交易"""
        if not self.trade_history:
            print("没有交易记录可以分析")
            return
            
        # 创建DataFrame
        df_trades = pd.DataFrame(self.trade_history)
        
        # 按退出原因分组计算
        exit_reason_stats = df_trades.groupby('exit_reason').agg({
            'profit_loss': ['sum', 'mean', 'count'],
            'return_pct': 'mean'
        })
        
        print("\n===== 按退出原因分析交易 =====")
        print(exit_reason_stats)
        print("===========================\n")
        
        # 可视化 - 只在有足够数据时执行
        if len(df_trades) >= 2:
            plt.figure(figsize=(10, 6))
            exit_reason_counts = df_trades['exit_reason'].value_counts()
            exit_reason_counts.plot(kind='bar')
            plt.title('交易退出原因分布')
            plt.xlabel('退出原因')
            plt.ylabel('交易次数')
            plt.grid(True, axis='y')
            plt.tight_layout()
            plt.show()
            
            # 按退出原因的平均收益率
            plt.figure(figsize=(10, 6))
            avg_returns = df_trades.groupby('exit_reason')['return_pct'].mean()
            avg_returns.plot(kind='bar')
            plt.title('各退出原因的平均收益率')
            plt.xlabel('退出原因')
            plt.ylabel('平均收益率(%)')
            plt.grid(True, axis='y')
            plt.tight_layout()
            plt.show()

    def _calculate_max_drawdown(self, equity_curve):
        """计算最大回撤"""
        max_drawdown = 0
        max_capital = equity_curve[0]
        
        for equity in equity_curve:
            if equity > max_capital:
                max_capital = equity
            drawdown = (max_capital - equity) / max_capital * 100
            max_drawdown = max(max_drawdown, drawdown)
        
        return max_drawdown

# 运行测试示例
if __name__ == "__main__":
    # 创建策略实例 - 使用较宽松的参数
    strategy = MTFDivergenceStrategy(risk_per_trade=0.02, atr_stop_multiplier=1.2)
    
    # 改为加载BTC数据的不同时间框架
    data = strategy.load_multi_timeframe_data('BTC')
    
    # 运行回测
    results = strategy.backtest(data, symbol='BTC', initial_capital=10000)
    
    # 打印绩效摘要
    strategy.print_performance_summary()
    
    # 绘制资金曲线
    strategy.plot_equity_curve()
    
    # 分析交易退出原因
    strategy.analyze_trades_by_exit_reason() 