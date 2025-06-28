"""
多时间框架背离策略

基于"MACD找结构 + KDJ找买点"的交易策略实现
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from datetime import datetime

from .base_strategy import BaseStrategy
from .config import StrategyConfig, create_strategy_config
from ..indicators import calculate_macd, calculate_kdj
from ..analysis import DivergenceAnalyzer, PatternDetector


class MultiTimeframeDivergenceStrategy(BaseStrategy):
    """多时间框架背离策略"""
    
    def __init__(self, config: Optional[StrategyConfig] = None):
        """
        初始化策略
        
        Args:
            config: 策略配置对象
        """
        super().__init__(config)
        self.divergence_analyzer = DivergenceAnalyzer()
        self.pattern_detector = PatternDetector()
        
    def analyze_market(self, market_data: Dict[str, pd.DataFrame], 
                      symbol: str = None) -> Dict[str, Any]:
        """
        分析市场数据并生成交易决策
        
        Args:
            market_data: 市场数据字典，键为时间框架，值为K线DataFrame
            symbol: 交易对名称，如果不提供则使用配置中的默认值
            
        Returns:
            包含分析结果和交易决策的字典
        """
        # 确定交易对
        if symbol is None:
            symbol = self.config.technical["trading_pairs"]["default"]
        # 验证输入数据
        required_timeframes = [
            self.config.technical["timeframes"]["macro"],    # 宏观层(日线)
            self.config.technical["timeframes"]["meso"],     # 中观层(4小时)
            self.config.technical["timeframes"]["micro"]     # 微观层(1小时)
        ]
        
        for tf in required_timeframes:
            if tf not in market_data:
                return {
                    'error': f'缺少必要的时间框架数据: {tf}',
                    'signal_type': 'neutral',
                    'signal_strength': 0.0,
                    'confidence': 0.0
                }
        
        # 分析各个时间框架
        timeframe_analysis = {}
        
        for timeframe, df in market_data.items():
            if timeframe in required_timeframes:
                analysis = self._analyze_single_timeframe(df, timeframe)
                timeframe_analysis[timeframe] = analysis
        
        # 合并多时间框架信号
        combined_signal = self._combine_timeframe_signals(timeframe_analysis)
        
        # 保存分析结果
        self.last_analysis = {
            'timestamp': datetime.now(),
            'symbol': symbol,
            'timeframe_analysis': timeframe_analysis,
            'combined_signal': combined_signal
        }
        
        return combined_signal
    
    def _analyze_single_timeframe(self, df: pd.DataFrame, timeframe: str) -> Dict[str, Any]:
        """
        分析单个时间框架
        
        Args:
            df: K线数据
            timeframe: 时间框架
            
        Returns:
            单时间框架分析结果
        """
        try:
            # 计算技术指标
            df_with_indicators = self._calculate_indicators(df, timeframe)
            
            # 分析背离
            divergence_results = self._analyze_divergences(df_with_indicators)
            
            # 分析技术模式
            pattern_results = self._analyze_patterns(df_with_indicators)
            
            # 计算ATR
            atr_value = self._calculate_atr(df_with_indicators)
            
            # 生成信号
            signal = self._generate_timeframe_signal(
                df_with_indicators, 
                divergence_results, 
                pattern_results,
                timeframe
            )
            
            return {
                'timeframe': timeframe,
                'indicators': self._extract_current_indicators(df_with_indicators),
                'divergences': divergence_results,
                'patterns': pattern_results,
                'signal': signal,
                'close_price': float(df['收盘价'].iloc[-1]),
                'atr': atr_value
            }
            
        except Exception as e:
            return {
                'timeframe': timeframe,
                'error': str(e),
                'signal': {
                    'type': 'neutral',
                    'strength': 0.0,
                    'confidence': 0.0
                }
            }
    
    def _calculate_indicators(self, df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """
        计算技术指标
        
        Args:
            df: K线数据
            timeframe: 时间框架
            
        Returns:
            添加了指标的DataFrame
        """
        # 计算MACD
        df_with_macd = calculate_macd(df)
        
        # 根据时间框架选择合适的KDJ参数
        volatility = self._determine_volatility(df)
        kdj_params = self.config.get_kdj_params(volatility)
        
        # 计算KDJ
        df_with_indicators = calculate_kdj(
            df_with_macd,
            m2=kdj_params['k'],
            m4=kdj_params['d'], 
            j_period=kdj_params['j']
        )
        
        return df_with_indicators
    
    def _calculate_atr(self, df: pd.DataFrame) -> float:
        """
        计算ATR (Average True Range)
        
        Args:
            df: K线数据DataFrame
            
        Returns:
            ATR值
        """
        try:
            if not self.config.technical["atr_config"]["enabled"]:
                # 如果未启用动态ATR，使用备用百分比
                current_price = float(df['收盘价'].iloc[-1])
                return current_price * self.config.technical["atr_config"]["fallback_percentage"]
            
            # 计算真实波幅
            high = df['最高价'].astype(float)
            low = df['最低价'].astype(float)
            close = df['收盘价'].astype(float)
            
            # 计算TR (True Range)
            tr1 = high - low
            tr2 = abs(high - close.shift(1))
            tr3 = abs(low - close.shift(1))
            
            tr = pd.DataFrame({'tr1': tr1, 'tr2': tr2, 'tr3': tr3}).max(axis=1)
            
            # 计算ATR
            atr_period = self.config.technical["atr_config"]["period"]
            atr = tr.rolling(window=atr_period, min_periods=1).mean()
            
            return float(atr.iloc[-1])
            
        except Exception as e:
            # 计算失败时使用备用方法
            current_price = float(df['收盘价'].iloc[-1])
            return current_price * self.config.technical["atr_config"]["fallback_percentage"]
    
    def _determine_volatility(self, df: pd.DataFrame) -> str:
        """
        确定市场波动性
        
        Args:
            df: K线数据
            
        Returns:
            波动性类型 ('high', 'medium', 'low')
        """
        # 计算ATR作为波动性指标
        high = df['最高价'].astype(float)
        low = df['最低价'].astype(float)
        close = df['收盘价'].astype(float)
        
        tr = pd.DataFrame({
            'hl': high - low,
            'hc': abs(high - close.shift(1)),
            'lc': abs(low - close.shift(1))
        }).max(axis=1)
        
        atr = tr.rolling(window=14).mean().iloc[-1]
        atr_pct = atr / close.iloc[-1]
        
        # 根据ATR百分比确定波动性
        if atr_pct > 0.05:  # 5%以上为高波动
            return 'high'
        elif atr_pct < 0.02:  # 2%以下为低波动
            return 'low'
        else:
            return 'medium'
    
    def _analyze_divergences(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        分析背离
        
        Args:
            df: 包含指标的DataFrame
            
        Returns:
            背离分析结果
        """
        results = {}
        
        try:
            # MACD背离分析
            if 'macd' in df.columns:
                macd_divergences = self.divergence_analyzer.analyze_macd_divergence(df)
                # 验证返回值的有效性
                if isinstance(macd_divergences, dict) and ('bullish' in macd_divergences or 'bearish' in macd_divergences):
                    results['macd'] = macd_divergences
                else:
                    results['macd'] = {'bullish': [], 'bearish': [], 'warning': 'Invalid MACD divergence result'}
            
            # KDJ背离分析
            if all(col in df.columns for col in ['kdj_j', 'kdj_j1']):
                kdj_divergences = self.divergence_analyzer.analyze_kdj_divergence(df)
                # 验证返回值的有效性
                if isinstance(kdj_divergences, dict) and ('bullish' in kdj_divergences or 'bearish' in kdj_divergences):
                    results['kdj'] = kdj_divergences
                else:
                    results['kdj'] = {'bullish': [], 'bearish': [], 'warning': 'Invalid KDJ divergence result'}
                
        except Exception as e:
            results['error'] = str(e)
        
        return results
    
    def _analyze_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        分析技术模式
        
        Args:
            df: 包含指标的DataFrame
            
        Returns:
            模式分析结果
        """
        results = {}
        
        try:
            # MACD信号
            if all(col in df.columns for col in ['macd', 'macd_signal']):
                macd_signals = self.pattern_detector.detect_macd_signals(df)
                # 验证返回值的有效性
                if isinstance(macd_signals, dict):
                    results['macd_signals'] = macd_signals
                else:
                    results['macd_signals'] = {'warning': 'Invalid MACD signals result'}
            
            # KDJ信号
            if all(col in df.columns for col in ['kdj_k', 'kdj_d', 'kdj_j']):
                kdj_signals = self.pattern_detector.detect_kdj_signals(df)
                # 验证返回值的有效性
                if isinstance(kdj_signals, dict):
                    results['kdj_signals'] = kdj_signals
                else:
                    results['kdj_signals'] = {'warning': 'Invalid KDJ signals result'}
                
        except Exception as e:
            results['error'] = str(e)
        
        return results
    
    def _extract_current_indicators(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        提取当前指标值
        
        Args:
            df: 包含指标的DataFrame
            
        Returns:
            当前指标值字典
        """
        indicators = {}
        
        # 提取最新值
        for col in df.columns:
            if col in ['macd', 'macd_signal', 'macd_histogram', 
                      'kdj_k', 'kdj_d', 'kdj_j', 'kdj_j1']:
                indicators[col] = float(df[col].iloc[-1])
        
        return indicators
    
    def _generate_timeframe_signal(self, df: pd.DataFrame, 
                                  divergences: Dict[str, Any],
                                  patterns: Dict[str, Any],
                                  timeframe: str) -> Dict[str, Any]:
        """
        生成单时间框架信号
        
        Args:
            df: 包含指标的DataFrame
            divergences: 背离分析结果
            patterns: 模式分析结果
            timeframe: 时间框架
            
        Returns:
            时间框架信号
        """
        signal_score = 0.0
        signal_type = 'neutral'
        confidence = 0.5
        
        # MACD结构分析 (用于确定大方向)
        macd_score = self._analyze_macd_structure(df, divergences, patterns)
        
        # KDJ买卖点分析 (用于确定具体入场点)
        kdj_score = self._analyze_kdj_signals(df, divergences, patterns)
        
        # 根据时间框架分配权重
        if timeframe == self.config.technical["timeframes"]["macro"]:
            # 宏观层重点看MACD结构
            signal_score = macd_score * 0.7 + kdj_score * 0.3
        elif timeframe == self.config.technical["timeframes"]["meso"]:
            # 中观层均衡考虑
            signal_score = macd_score * 0.5 + kdj_score * 0.5
        else:  # 微观层
            # 微观层重点看KDJ买卖点
            signal_score = macd_score * 0.3 + kdj_score * 0.7
        
        # 确定信号类型
        if signal_score > 0.3:
            signal_type = 'buy'
            confidence = min(0.5 + signal_score * 0.5, 1.0)
        elif signal_score < -0.3:
            signal_type = 'sell'
            confidence = min(0.5 + abs(signal_score) * 0.5, 1.0)
        
        return {
            'type': signal_type,
            'strength': abs(signal_score),
            'confidence': confidence,
            'macd_score': macd_score,
            'kdj_score': kdj_score
        }
    
    def _analyze_macd_structure(self, df: pd.DataFrame, 
                               divergences: Dict[str, Any],
                               patterns: Dict[str, Any]) -> float:
        """
        分析MACD结构
        
        Args:
            df: 包含指标的DataFrame
            divergences: 背离分析结果
            patterns: 模式分析结果
            
        Returns:
            MACD结构评分 (-1.0 到 1.0)
        """
        score = 0.0
        
        # 检查MACD背离
        if 'macd' in divergences:
            recent_bullish = self.divergence_analyzer.get_recent_divergences(
                divergences['macd']['bullish'], window=20
            )
            recent_bearish = self.divergence_analyzer.get_recent_divergences(
                divergences['macd']['bearish'], window=20
            )
            
            if recent_bullish:
                score += 0.4  # 底背离看涨
            if recent_bearish:
                score -= 0.4  # 顶背离看跌
        
        # 检查MACD金叉死叉
        if 'macd_signals' in patterns:
            golden_crosses = patterns['macd_signals'].get('golden_crosses', [])
            death_crosses = patterns['macd_signals'].get('death_crosses', [])
            
            # 检查最近的交叉
            if golden_crosses and len(df) - golden_crosses[-1]['index'] <= 3:
                score += 0.3  # 最近金叉
            if death_crosses and len(df) - death_crosses[-1]['index'] <= 3:
                score -= 0.3  # 最近死叉
        
        # 检查MACD零轴位置
        if 'macd' in df.columns:
            current_macd = df['macd'].iloc[-1]
            if current_macd > 0:
                score += 0.1  # MACD在零轴之上
            else:
                score -= 0.1  # MACD在零轴之下
        
        return np.clip(score, -1.0, 1.0)
    
    def _analyze_kdj_signals(self, df: pd.DataFrame,
                            divergences: Dict[str, Any],
                            patterns: Dict[str, Any]) -> float:
        """
        分析KDJ买卖点
        
        Args:
            df: 包含指标的DataFrame
            divergences: 背离分析结果
            patterns: 模式分析结果
            
        Returns:
            KDJ信号评分 (-1.0 到 1.0)
        """
        score = 0.0
        
        # 检查KDJ背离
        if 'kdj' in divergences:
            recent_bullish = self.divergence_analyzer.get_recent_divergences(
                divergences['kdj']['bullish'], window=10
            )
            recent_bearish = self.divergence_analyzer.get_recent_divergences(
                divergences['kdj']['bearish'], window=10
            )
            
            if recent_bullish:
                score += 0.5  # KDJ底背离强买入信号
            if recent_bearish:
                score -= 0.5  # KDJ顶背离强卖出信号
        
        # 检查KDJ超买超卖
        if 'kdj_signals' in patterns:
            overbought_oversold = patterns['kdj_signals'].get('overbought_oversold', {})
            
            # 检查最近的超买超卖
            oversold = overbought_oversold.get('oversold', [])
            overbought = overbought_oversold.get('overbought', [])
            
            if oversold and len(df) - oversold[-1]['index'] <= 5:
                score += 0.3  # 最近超卖
            if overbought and len(df) - overbought[-1]['index'] <= 5:
                score -= 0.3  # 最近超买
        
        # 检查KDJ金叉死叉
        if 'kdj_signals' in patterns:
            kd_crosses = patterns['kdj_signals'].get('kd_crosses', {})
            golden = kd_crosses.get('golden', [])
            death = kd_crosses.get('death', [])
            
            if golden and len(df) - golden[-1]['index'] <= 2:
                score += 0.2  # K线上穿D线
            if death and len(df) - death[-1]['index'] <= 2:
                score -= 0.2  # K线下穿D线
        
        return np.clip(score, -1.0, 1.0)
    
    def _combine_timeframe_signals(self, timeframe_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        合并多时间框架信号
        
        Args:
            timeframe_analysis: 各时间框架分析结果
            
        Returns:
            合并后的信号
        """
        # 时间框架权重
        weights = {
            self.config.technical["timeframes"]["macro"]: 0.4,   # 宏观层权重40%
            self.config.technical["timeframes"]["meso"]: 0.4,    # 中观层权重40%
            self.config.technical["timeframes"]["micro"]: 0.2    # 微观层权重20%
        }
        
        total_score = 0.0
        total_confidence = 0.0
        signal_count = 0
        
        for timeframe, analysis in timeframe_analysis.items():
            if 'signal' in analysis and 'error' not in analysis:
                signal = analysis['signal']
                weight = weights.get(timeframe, 0.1)
                
                # 根据信号类型计算加权分数
                if signal['type'] == 'buy':
                    total_score += signal['strength'] * weight
                elif signal['type'] == 'sell':
                    total_score -= signal['strength'] * weight
                
                total_confidence += signal['confidence'] * weight
                signal_count += 1
        
        # 确定最终信号类型
        if total_score > 0.2:
            final_signal_type = 'buy'
            final_strength = total_score
        elif total_score < -0.2:
            final_signal_type = 'sell'
            final_strength = abs(total_score)
        else:
            final_signal_type = 'neutral'
            final_strength = 0.0
        
        # 计算平均置信度
        avg_confidence = total_confidence / len(weights) if weights else 0.5
        
        return {
            'signal_type': final_signal_type,
            'signal_strength': final_strength,
            'confidence': avg_confidence,
            'analysis_details': timeframe_analysis,
            'confirmed_signals': signal_count,
            'required_signals': self.config.get_signal_threshold()
        }
    
    def generate_trading_signal(self, analysis_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        基于分析结果生成具体的交易信号
        
        Args:
            analysis_result: 市场分析结果
            
        Returns:
            交易信号字典
        """
        # 检查是否满足交易条件
        confirmed_signals = analysis_result.get('confirmed_signals', 0)
        required_signals = analysis_result.get('required_signals', 3)
        signal_strength = analysis_result.get('signal_strength', 0.0)
        
        should_trade = (confirmed_signals >= required_signals and 
                       signal_strength >= 0.6)
        
        if not should_trade:
            return {
                'action': 'hold',
                'reason': f'信号不足: {confirmed_signals}/{required_signals}, 强度: {signal_strength:.2f}'
            }
        
        # 获取当前价格（从中观层）
        meso_tf = self.config.technical["timeframes"]["meso"]
        analysis_details = analysis_result.get('analysis_details', {})
        
        if meso_tf not in analysis_details:
            return {
                'action': 'hold',
                'reason': '缺少中观层价格数据'
            }
        
        current_price = analysis_details[meso_tf].get('close_price')
        atr_value = analysis_details[meso_tf].get('atr')
        if not current_price:
            return {
                'action': 'hold',
                'reason': '无法获取当前价格'
            }
        
        # 计算止损和仓位
        signal_type = analysis_result['signal_type']
        
        if signal_type == 'buy':
            stop_loss = self._calculate_stop_loss(current_price, 'buy', atr_value)
            position_size = self.calculate_position_size(
                signal_strength, 
                self.get_account_balance() * 0.02,  # 2%风险
                current_price, 
                stop_loss
            )
            
            return {
                'action': 'buy',
                'symbol': self.last_analysis.get('symbol', self.config.technical["trading_pairs"]["default"]),
                'entry_price': current_price,
                'stop_loss': stop_loss,
                'position_size': position_size,
                'signal_strength': signal_strength,
                'confidence': analysis_result['confidence']
            }
            
        elif signal_type == 'sell':
            stop_loss = self._calculate_stop_loss(current_price, 'sell', atr_value)
            position_size = self.calculate_position_size(
                signal_strength,
                self.get_account_balance() * 0.02,  # 2%风险
                current_price,
                stop_loss
            )
            
            return {
                'action': 'sell',
                'symbol': self.last_analysis.get('symbol', self.config.technical["trading_pairs"]["default"]),
                'entry_price': current_price,
                'stop_loss': stop_loss,
                'position_size': position_size,
                'signal_strength': signal_strength,
                'confidence': analysis_result['confidence']
            }
        
        return {
            'action': 'hold',
            'reason': '信号类型为neutral'
        }
    
    def _calculate_stop_loss(self, entry_price: float, direction: str, 
                           atr_value: float = None) -> float:
        """
        计算止损价格
        
        Args:
            entry_price: 入场价格
            direction: 交易方向 ('buy' 或 'sell')
            atr_value: ATR值，如果不提供则使用备用计算
            
        Returns:
            止损价格
        """
        # 确定交易对类型
        symbol = self.last_analysis.get('symbol', self.config.technical["trading_pairs"]["default"])
        if 'BTC' in symbol:
            asset_type = 'BTC'
        elif 'ETH' in symbol:
            asset_type = 'ETH'
        else:
            asset_type = 'ALT'
        
        # 使用配置中的止损倍数
        stop_loss_multiplier = self.config.risk['stop_loss_multiplier'][asset_type]
        
        # 使用动态ATR或备用值
        if atr_value is None:
            atr_value = entry_price * self.config.technical["atr_config"]["fallback_percentage"]
        
        if direction == 'buy':
            return entry_price - (atr_value * stop_loss_multiplier)
        else:  # sell
            return entry_price + (atr_value * stop_loss_multiplier)


def create_mtf_strategy(mode: str = "standard") -> MultiTimeframeDivergenceStrategy:
    """
    创建多时间框架背离策略实例
    
    Args:
        mode: 策略模式
        
    Returns:
        策略实例
    """
    config = create_strategy_config(mode)
    return MultiTimeframeDivergenceStrategy(config)