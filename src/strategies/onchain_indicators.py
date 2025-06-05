"""
⛓️ 链上数据指标模块  
OnChain Indicators Module

包含MVRV Z-Score、Puell Multiple、巨鲸监控等核心链上指标
"""

import numpy as np
import pandas as pd
import requests
import time
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import logging
from .config import OnChainConfig, API_KEYS

class OnChainDataProvider:
    """链上数据提供者基类"""
    
    def __init__(self, config: OnChainConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
    def fetch_data(self, metric: str, asset: str = 'BTC') -> pd.DataFrame:
        """获取链上数据的抽象方法"""
        raise NotImplementedError
        
class GlassnodeProvider(OnChainDataProvider):
    """Glassnode数据提供者"""
    
    def __init__(self, config: OnChainConfig):
        super().__init__(config)
        self.api_key = API_KEYS.get('GLASSNODE_API_KEY')
        self.base_url = "https://api.glassnode.com/v1/metrics"
        
    def fetch_data(self, metric: str, asset: str = 'BTC', 
                  since: str = None, until: str = None) -> pd.DataFrame:
        """
        从Glassnode获取数据
        
        Args:
            metric: 指标名称 (如 'market/mvrv_z_score')
            asset: 资产符号
            since: 开始时间 (YYYY-MM-DD)
            until: 结束时间 (YYYY-MM-DD)
        """
        if not self.api_key:
            self.logger.warning("Glassnode API key not found, using mock data")
            return self._get_mock_data(metric)
            
        url = f"{self.base_url}/{metric}"
        params = {
            'a': asset,
            'api_key': self.api_key,
            'f': 'json'
        }
        
        if since:
            params['s'] = since
        if until:
            params['u'] = until
            
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            
            data = response.json()
            df = pd.DataFrame(data)
            df['timestamp'] = pd.to_datetime(df['t'], unit='s')
            df = df.set_index('timestamp')
            df['value'] = pd.to_numeric(df['v'], errors='coerce')
            
            return df[['value']].dropna()
            
        except Exception as e:
            self.logger.error(f"Failed to fetch {metric}: {e}")
            return self._get_mock_data(metric)
    
    def _get_mock_data(self, metric: str) -> pd.DataFrame:
        """生成模拟数据用于测试"""
        dates = pd.date_range(end=datetime.now(), periods=100, freq='D')
        
        if 'mvrv' in metric.lower():
            # MVRV Z-Score 模拟数据 (-2 到 4 之间)
            values = np.random.normal(0.5, 1.0, 100)
        elif 'puell' in metric.lower():
            # Puell Multiple 模拟数据 (0.2 到 6 之间)
            values = np.random.lognormal(0, 0.5, 100)
        else:
            # 其他指标的默认模拟数据
            values = np.random.randn(100) * 1000 + 50000
            
        return pd.DataFrame({'value': values}, index=dates)

class MVRVZScore:
    """MVRV Z-Score 指标"""
    
    def __init__(self, data_provider: OnChainDataProvider):
        self.provider = data_provider
        self.cache = {}
        
    def get_current_score(self, asset: str = 'BTC') -> float:
        """获取当前MVRV Z-Score"""
        try:
            data = self.provider.fetch_data('market/mvrv_z_score', asset)
            return float(data['value'].iloc[-1])
        except Exception as e:
            logging.error(f"Failed to get MVRV Z-Score: {e}")
            return 0.0
    
    def get_historical_data(self, asset: str = 'BTC', days: int = 365) -> pd.DataFrame:
        """获取历史MVRV Z-Score数据"""
        since = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
        return self.provider.fetch_data('market/mvrv_z_score', asset, since=since)
    
    def analyze_signal(self, current_score: float) -> Dict[str, any]:
        """
        分析MVRV Z-Score信号
        
        Returns:
            信号分析结果
        """
        signal_analysis = {
            'current_score': current_score,
            'signal_type': 'neutral',
            'strength': 'weak',
            'recommendation': 'hold'
        }
        
        if current_score < self.provider.config.mvrv_buy_threshold:
            signal_analysis.update({
                'signal_type': 'bullish',
                'strength': 'strong' if current_score < 0 else 'medium',
                'recommendation': 'buy',
                'description': f'MVRV Z-Score({current_score:.2f}) 处于历史低位，强烈买入信号'
            })
        elif current_score > self.provider.config.mvrv_sell_threshold:
            signal_analysis.update({
                'signal_type': 'bearish',
                'strength': 'strong' if current_score > 5 else 'medium',
                'recommendation': 'sell',
                'description': f'MVRV Z-Score({current_score:.2f}) 处于历史高位，强烈卖出信号'
            })
        else:
            signal_analysis['description'] = f'MVRV Z-Score({current_score:.2f}) 处于正常范围'
            
        return signal_analysis

class PuellMultiple:
    """Puell Multiple 指标"""
    
    def __init__(self, data_provider: OnChainDataProvider):
        self.provider = data_provider
        
    def get_current_value(self, asset: str = 'BTC') -> float:
        """获取当前Puell Multiple值"""
        try:
            data = self.provider.fetch_data('mining/puell_multiple', asset)
            return float(data['value'].iloc[-1])
        except Exception as e:
            logging.error(f"Failed to get Puell Multiple: {e}")
            return 1.0
    
    def get_historical_data(self, asset: str = 'BTC', days: int = 365) -> pd.DataFrame:
        """获取历史Puell Multiple数据"""
        since = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
        return self.provider.fetch_data('mining/puell_multiple', asset, since=since)
    
    def analyze_signal(self, current_value: float) -> Dict[str, any]:
        """分析Puell Multiple信号"""
        signal_analysis = {
            'current_value': current_value,
            'signal_type': 'neutral',
            'strength': 'weak',
            'recommendation': 'hold'
        }
        
        if current_value < self.provider.config.puell_bottom_threshold:
            signal_analysis.update({
                'signal_type': 'bullish',
                'strength': 'strong',
                'recommendation': 'buy',
                'description': f'Puell Multiple({current_value:.2f}) 极低，矿工抛压极小，强烈买入信号'
            })
        elif current_value > self.provider.config.puell_top_threshold:
            signal_analysis.update({
                'signal_type': 'bearish',
                'strength': 'strong',
                'recommendation': 'sell',
                'description': f'Puell Multiple({current_value:.2f}) 极高，矿工大量抛售，强烈卖出信号'
            })
        else:
            signal_analysis['description'] = f'Puell Multiple({current_value:.2f}) 处于正常范围'
            
        return signal_analysis

class WhaleActivityMonitor:
    """巨鲸活动监控"""
    
    def __init__(self, config: OnChainConfig):
        self.config = config
        self.whale_alerts = []
        
    def monitor_large_transactions(self, asset: str = 'BTC') -> List[Dict]:
        """监控大额交易"""
        # 这里应该连接到Whale Alert API或类似服务
        # 现在返回模拟数据
        mock_transactions = [
            {
                'timestamp': datetime.now(),
                'asset': asset,
                'amount': 1500.0,
                'from_exchange': True,
                'to_exchange': False,
                'signal': 'bullish',  # 从交易所流出
                'description': '1500 BTC从币安流出到未知钱包'
            },
            {
                'timestamp': datetime.now() - timedelta(hours=2),
                'asset': asset,
                'amount': 2000.0,
                'from_exchange': False,
                'to_exchange': True,
                'signal': 'bearish',  # 流入交易所
                'description': '2000 BTC从未知钱包流入Coinbase'
            }
        ]
        
        return mock_transactions
    
    def analyze_whale_behavior(self, transactions: List[Dict]) -> Dict[str, any]:
        """分析巨鲸行为"""
        if not transactions:
            return {'signal': 'neutral', 'strength': 'weak'}
            
        inflow = sum(tx['amount'] for tx in transactions if tx['to_exchange'])
        outflow = sum(tx['amount'] for tx in transactions if tx['from_exchange'])
        
        net_flow = outflow - inflow  # 正值表示净流出（看涨）
        
        if net_flow > self.config.exchange_flow_threshold:
            return {
                'signal': 'bullish',
                'strength': 'strong',
                'net_flow': net_flow,
                'description': f'巨鲸净流出 {net_flow:.0f} BTC，强烈看涨信号'
            }
        elif net_flow < -self.config.exchange_flow_threshold:
            return {
                'signal': 'bearish',
                'strength': 'strong',
                'net_flow': net_flow,
                'description': f'巨鲸净流入 {abs(net_flow):.0f} BTC，强烈看跌信号'
            }
        else:
            return {
                'signal': 'neutral',
                'strength': 'weak',
                'net_flow': net_flow,
                'description': f'巨鲸活动平衡，净流量 {net_flow:.0f} BTC'
            }

class NetworkValueIndicators:
    """网络价值指标"""
    
    def __init__(self, data_provider: OnChainDataProvider):
        self.provider = data_provider
        
    def get_nvt_ratio(self, asset: str = 'BTC') -> float:
        """获取NVT比率"""
        try:
            data = self.provider.fetch_data('indicators/nvt', asset)
            return float(data['value'].iloc[-1])
        except:
            return 50.0  # 默认值
    
    def get_active_addresses(self, asset: str = 'BTC') -> int:
        """获取活跃地址数"""
        try:
            data = self.provider.fetch_data('addresses/active_count', asset)
            return int(data['value'].iloc[-1])
        except:
            return 1000000  # 默认值
    
    def analyze_network_health(self, asset: str = 'BTC') -> Dict[str, any]:
        """分析网络健康状况"""
        nvt = self.get_nvt_ratio(asset)
        active_addresses = self.get_active_addresses(asset)
        
        # NVT分析
        if nvt < 20:
            nvt_signal = 'undervalued'
        elif nvt > 100:
            nvt_signal = 'overvalued'
        else:
            nvt_signal = 'fair_value'
        
        # 活跃地址分析（需要历史数据对比）
        address_trend = 'stable'  # 简化处理
        
        return {
            'nvt_ratio': nvt,
            'nvt_signal': nvt_signal,
            'active_addresses': active_addresses,
            'address_trend': address_trend,
            'overall_health': 'healthy' if nvt < 80 else 'concerning'
        }

class OnChainSignalAggregator:
    """链上信号聚合器"""
    
    def __init__(self, config: OnChainConfig):
        self.config = config
        self.data_provider = GlassnodeProvider(config)
        self.mvrv = MVRVZScore(self.data_provider)
        self.puell = PuellMultiple(self.data_provider)
        self.whale_monitor = WhaleActivityMonitor(config)
        self.network_indicators = NetworkValueIndicators(self.data_provider)
        
    def get_comprehensive_analysis(self, asset: str = 'BTC') -> Dict[str, any]:
        """获取综合链上分析"""
        
        # 获取各项指标
        mvrv_score = self.mvrv.get_current_score(asset)
        puell_value = self.puell.get_current_value(asset)
        whale_transactions = self.whale_monitor.monitor_large_transactions(asset)
        network_health = self.network_indicators.analyze_network_health(asset)
        
        # 分析各项指标
        mvrv_analysis = self.mvrv.analyze_signal(mvrv_score)
        puell_analysis = self.puell.analyze_signal(puell_value)
        whale_analysis = self.whale_monitor.analyze_whale_behavior(whale_transactions)
        
        # 计算综合信号强度
        signal_strength = self._calculate_combined_signal_strength(
            mvrv_analysis, puell_analysis, whale_analysis
        )
        
        return {
            'timestamp': datetime.now(),
            'asset': asset,
            'mvrv_analysis': mvrv_analysis,
            'puell_analysis': puell_analysis,
            'whale_analysis': whale_analysis,
            'network_health': network_health,
            'combined_signal': signal_strength,
            'recommendation': self._get_final_recommendation(signal_strength)
        }
    
    def _calculate_combined_signal_strength(self, mvrv_analysis: Dict, 
                                          puell_analysis: Dict, 
                                          whale_analysis: Dict) -> Dict[str, any]:
        """计算综合信号强度"""
        
        # 信号权重
        weights = {
            'mvrv': 0.4,   # MVRV权重40%
            'puell': 0.3,  # Puell权重30%
            'whale': 0.3   # 巨鲸权重30%
        }
        
        # 信号得分 (看涨+1, 中性0, 看跌-1)
        signal_scores = {
            'mvrv': self._convert_signal_to_score(mvrv_analysis['signal_type']),
            'puell': self._convert_signal_to_score(puell_analysis['signal_type']),
            'whale': self._convert_signal_to_score(whale_analysis['signal'])
        }
        
        # 计算加权综合得分
        combined_score = sum(signal_scores[key] * weights[key] for key in weights.keys())
        
        # 确定综合信号类型
        if combined_score > 0.3:
            signal_type = 'bullish'
            strength = 'strong' if combined_score > 0.6 else 'medium'
        elif combined_score < -0.3:
            signal_type = 'bearish'
            strength = 'strong' if combined_score < -0.6 else 'medium'
        else:
            signal_type = 'neutral'
            strength = 'weak'
        
        return {
            'signal_type': signal_type,
            'strength': strength,
            'score': combined_score,
            'individual_scores': signal_scores,
            'confidence': min(0.9, abs(combined_score))  # 置信度
        }
    
    def _convert_signal_to_score(self, signal_type: str) -> float:
        """将信号类型转换为数值得分"""
        signal_map = {
            'bullish': 1.0,
            'bearish': -1.0,
            'neutral': 0.0
        }
        return signal_map.get(signal_type, 0.0)
    
    def _get_final_recommendation(self, combined_signal: Dict) -> str:
        """获取最终推荐"""
        signal_type = combined_signal['signal_type']
        strength = combined_signal['strength']
        
        if signal_type == 'bullish':
            if strength == 'strong':
                return 'strong_buy'
            else:
                return 'buy'
        elif signal_type == 'bearish':
            if strength == 'strong':
                return 'strong_sell'
            else:
                return 'sell'
        else:
            return 'hold'

class HashRibbons:
    """Hash Ribbons 矿工投降指标"""
    
    def __init__(self, data_provider: OnChainDataProvider):
        self.provider = data_provider
        
    def calculate_hash_ribbons(self, asset: str = 'BTC') -> Dict[str, any]:
        """计算Hash Ribbons指标"""
        try:
            # 获取算力数据
            hashrate_data = self.provider.fetch_data('mining/hash_rate_mean', asset)
            
            if len(hashrate_data) < 60:
                return {'signal': 'insufficient_data'}
            
            # 计算30日和60日移动平均
            hashrate_data['ma_30'] = hashrate_data['value'].rolling(30).mean()
            hashrate_data['ma_60'] = hashrate_data['value'].rolling(60).mean()
            
            current_ma_30 = hashrate_data['ma_30'].iloc[-1]
            current_ma_60 = hashrate_data['ma_60'].iloc[-1]
            
            # Hash Ribbons信号
            if current_ma_30 > current_ma_60:
                # 30日均线上穿60日均线，矿工投降结束
                signal = 'bullish'
                description = '矿工投降结束，Hash Ribbons看涨信号'
            else:
                # 30日均线下穿60日均线，矿工可能投降
                signal = 'bearish'
                description = '矿工可能投降，Hash Ribbons看跌信号'
            
            return {
                'signal': signal,
                'ma_30': current_ma_30,
                'ma_60': current_ma_60,
                'description': description
            }
            
        except Exception as e:
            logging.error(f"Hash Ribbons calculation failed: {e}")
            return {'signal': 'error', 'description': 'Hash Ribbons计算失败'}

if __name__ == "__main__":
    # 测试链上指标模块
    from .config import OnChainConfig
    
    print("⛓️ 链上指标模块测试")
    
    # 初始化配置和聚合器
    config = OnChainConfig()
    aggregator = OnChainSignalAggregator(config)
    
    # 获取综合分析
    analysis = aggregator.get_comprehensive_analysis('BTC')
    
    print(f"\n📊 BTC链上分析结果:")
    print(f"MVRV Z-Score: {analysis['mvrv_analysis']['current_score']:.2f}")
    print(f"Puell Multiple: {analysis['puell_analysis']['current_value']:.2f}")
    print(f"巨鲸信号: {analysis['whale_analysis']['signal']}")
    print(f"综合信号: {analysis['combined_signal']['signal_type']} ({analysis['combined_signal']['strength']})")
    print(f"最终推荐: {analysis['recommendation']}")
    
    print("\n✅ 链上指标模块测试完成") 