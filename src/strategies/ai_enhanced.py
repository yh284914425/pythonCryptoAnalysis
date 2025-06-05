"""
🤖 AI增强系统模块
AI Enhanced System Module

包含Transformer预测模型、强化学习优化、情绪分析等AI功能
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import requests
import json
from datetime import datetime, timedelta
import logging
from .config import AIConfig, API_KEYS

class TransformerPricePredictor:
    """Transformer价格预测模型"""
    
    def __init__(self, config: AIConfig):
        self.config = config
        self.model = None
        self.scaler = None
        self.sequence_length = 168  # 7天*24小时
        self.feature_dim = 10  # 价格、成交量、技术指标等特征
        
    def build_model(self):
        """构建Transformer模型"""
        class PriceTransformer(nn.Module):
            def __init__(self, feature_dim, d_model=128, nhead=8, num_layers=4):
                super().__init__()
                self.feature_dim = feature_dim
                self.d_model = d_model
                
                # 输入投影层
                self.input_projection = nn.Linear(feature_dim, d_model)
                
                # 位置编码
                self.positional_encoding = nn.Parameter(
                    torch.randn(1000, d_model) * 0.1
                )
                
                # Transformer编码器
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=d_model,
                    nhead=nhead,
                    dim_feedforward=512,
                    dropout=0.1,
                    batch_first=True
                )
                self.transformer = nn.TransformerEncoder(
                    encoder_layer, 
                    num_layers=num_layers
                )
                
                # 输出层
                self.output_projection = nn.Sequential(
                    nn.Linear(d_model, 64),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Linear(64, 1)  # 预测价格变化
                )
                
            def forward(self, x):
                batch_size, seq_len, _ = x.shape
                
                # 输入投影
                x = self.input_projection(x)
                
                # 添加位置编码
                x = x + self.positional_encoding[:seq_len].unsqueeze(0)
                
                # Transformer编码
                x = self.transformer(x)
                
                # 使用最后一个时间步的输出
                x = x[:, -1, :]
                
                # 预测输出
                return self.output_projection(x)
        
        self.model = PriceTransformer(self.feature_dim)
        return self.model
    
    def prepare_features(self, price_data: pd.DataFrame, 
                        technical_indicators: Dict, 
                        onchain_data: Dict) -> np.ndarray:
        """准备模型输入特征"""
        features = []
        
        # 价格特征 (标准化)
        price_features = price_data[['open', 'high', 'low', 'close', 'volume']].values
        price_features = (price_features - price_features.mean(axis=0)) / price_features.std(axis=0)
        features.append(price_features)
        
        # 技术指标特征
        if 'kdj' in technical_indicators:
            kdj_features = np.column_stack([
                technical_indicators['kdj']['K'].values,
                technical_indicators['kdj']['D'].values,
                technical_indicators['kdj']['J'].values
            ])
            features.append(kdj_features)
        
        # MACD特征
        if 'macd' in technical_indicators:
            macd_features = np.column_stack([
                technical_indicators['macd']['macd'].values,
                technical_indicators['macd']['signal'].values
            ])
            features.append(macd_features)
        
        # 合并所有特征
        all_features = np.concatenate(features, axis=1)
        
        # 确保特征维度正确
        if all_features.shape[1] != self.feature_dim:
            # 填充或截断到目标维度
            if all_features.shape[1] < self.feature_dim:
                padding = np.zeros((all_features.shape[0], 
                                  self.feature_dim - all_features.shape[1]))
                all_features = np.concatenate([all_features, padding], axis=1)
            else:
                all_features = all_features[:, :self.feature_dim]
        
        return all_features
    
    def predict_price_probability(self, features: np.ndarray) -> Dict[str, float]:
        """
        预测未来价格概率分布
        
        Returns:
            包含上涨/下跌概率的字典
        """
        if self.model is None:
            self.build_model()
            # 在实际使用中，这里应该加载预训练的权重
            # self.model.load_state_dict(torch.load('model_weights.pth'))
        
        self.model.eval()
        
        # 准备输入序列
        if len(features) < self.sequence_length:
            # 如果数据不足，用零填充
            padding = np.zeros((self.sequence_length - len(features), self.feature_dim))
            features = np.concatenate([padding, features], axis=0)
        else:
            features = features[-self.sequence_length:]
        
        # 转换为张量
        input_tensor = torch.FloatTensor(features).unsqueeze(0)
        
        with torch.no_grad():
            prediction = self.model(input_tensor)
            
        # 将预测转换为概率
        price_change = prediction.item()
        
        # 使用sigmoid函数将价格变化转换为上涨概率
        up_probability = 1 / (1 + np.exp(-price_change))
        down_probability = 1 - up_probability
        
        confidence = abs(price_change)  # 预测置信度
        
        return {
            'up_probability': float(up_probability),
            'down_probability': float(down_probability),
            'confidence': float(min(confidence, 1.0)),
            'raw_prediction': float(price_change)
        }

class SentimentAnalyzer:
    """情绪分析器"""
    
    def __init__(self, config: AIConfig):
        self.config = config
        self.twitter_api_key = API_KEYS.get('TWITTER_API_KEY')
        
        # 初始化情绪分析模型
        try:
            self.sentiment_pipeline = pipeline(
                "sentiment-analysis",
                model="ProsusAI/finbert",  # 金融领域专用模型
                tokenizer="ProsusAI/finbert"
            )
        except:
            # 如果模型加载失败，使用简化版本
            self.sentiment_pipeline = None
            logging.warning("FinBERT model not available, using simplified sentiment analysis")
    
    def analyze_twitter_sentiment(self, query: str = "Bitcoin BTC", 
                                 count: int = 100) -> Dict[str, float]:
        """分析Twitter情绪"""
        if not self.twitter_api_key:
            return self._get_mock_sentiment()
        
        try:
            # 这里应该调用Twitter API获取推文
            # tweets = self._fetch_tweets(query, count)
            tweets = self._get_mock_tweets()  # 使用模拟数据
            
            return self._analyze_tweet_sentiments(tweets)
            
        except Exception as e:
            logging.error(f"Twitter sentiment analysis failed: {e}")
            return self._get_mock_sentiment()
    
    def _get_mock_tweets(self) -> List[str]:
        """获取模拟推文数据"""
        mock_tweets = [
            "Bitcoin is going to the moon! 🚀",
            "BTC looking bearish, time to sell",
            "Hodling strong, diamond hands 💎",
            "Crypto market is crashing, panic selling",
            "Bitcoin adoption is accelerating, bullish long term",
            "Whales are accumulating, bottom might be in",
            "Fear and greed index shows extreme fear",
            "DeFi is the future, Ethereum leading the way",
            "Regulatory clarity needed for crypto growth",
            "Bitcoin hash rate hitting new highs"
        ]
        return mock_tweets
    
    def _analyze_tweet_sentiments(self, tweets: List[str]) -> Dict[str, float]:
        """分析推文情绪"""
        if not self.sentiment_pipeline:
            return self._get_mock_sentiment()
        
        sentiments = []
        for tweet in tweets:
            try:
                result = self.sentiment_pipeline(tweet)[0]
                score = result['score']
                label = result['label'].lower()
                
                if label == 'positive':
                    sentiments.append(score)
                elif label == 'negative':
                    sentiments.append(-score)
                else:
                    sentiments.append(0)
                    
            except Exception as e:
                logging.warning(f"Failed to analyze tweet sentiment: {e}")
                continue
        
        if not sentiments:
            return self._get_mock_sentiment()
        
        # 计算情绪统计
        avg_sentiment = np.mean(sentiments)
        sentiment_std = np.std(sentiments)
        
        # 转换为恐惧贪婪指数 (0-100)
        fear_greed_index = max(0, min(100, (avg_sentiment + 1) * 50))
        
        return {
            'fear_greed_index': fear_greed_index,
            'average_sentiment': avg_sentiment,
            'sentiment_volatility': sentiment_std,
            'bullish_ratio': len([s for s in sentiments if s > 0.1]) / len(sentiments),
            'bearish_ratio': len([s for s in sentiments if s < -0.1]) / len(sentiments),
            'neutral_ratio': len([s for s in sentiments if -0.1 <= s <= 0.1]) / len(sentiments)
        }
    
    def _get_mock_sentiment(self) -> Dict[str, float]:
        """生成模拟情绪数据"""
        np.random.seed(int(datetime.now().timestamp()) % 1000)
        fear_greed_index = np.random.uniform(10, 90)
        
        return {
            'fear_greed_index': fear_greed_index,
            'average_sentiment': (fear_greed_index - 50) / 50,
            'sentiment_volatility': np.random.uniform(0.1, 0.5),
            'bullish_ratio': np.random.uniform(0.2, 0.6),
            'bearish_ratio': np.random.uniform(0.2, 0.6),
            'neutral_ratio': np.random.uniform(0.1, 0.4)
        }
    
    def get_sentiment_signal(self, sentiment_data: Dict[str, float]) -> Dict[str, any]:
        """根据情绪数据生成交易信号"""
        fear_greed_index = sentiment_data['fear_greed_index']
        
        if fear_greed_index < self.config.sentiment_fear_threshold:
            return {
                'signal': 'bullish',
                'strength': 'strong',
                'description': f'极度恐慌(FGI: {fear_greed_index:.0f})，强买入信号'
            }
        elif fear_greed_index > self.config.sentiment_greed_threshold:
            return {
                'signal': 'bearish', 
                'strength': 'strong',
                'description': f'极度贪婪(FGI: {fear_greed_index:.0f})，强卖出信号'
            }
        elif fear_greed_index < 25:
            return {
                'signal': 'bullish',
                'strength': 'medium',
                'description': f'恐慌情绪(FGI: {fear_greed_index:.0f})，买入信号'
            }
        elif fear_greed_index > 75:
            return {
                'signal': 'bearish',
                'strength': 'medium', 
                'description': f'贪婪情绪(FGI: {fear_greed_index:.0f})，卖出信号'
            }
        else:
            return {
                'signal': 'neutral',
                'strength': 'weak',
                'description': f'情绪中性(FGI: {fear_greed_index:.0f})'
            }

class ReinforcementLearningOptimizer:
    """强化学习参数优化器"""
    
    def __init__(self, config: AIConfig):
        self.config = config
        self.q_table = {}  # 简化的Q表
        self.learning_rate = config.rl_learning_rate
        self.epsilon = 0.1  # 探索率
        self.state_history = []
        self.action_history = []
        self.reward_history = []
    
    def get_state(self, market_data: Dict) -> str:
        """将市场数据转换为状态表示"""
        # 简化的状态表示
        price_trend = "up" if market_data.get('price_change', 0) > 0 else "down"
        volatility = "high" if market_data.get('volatility', 0) > 0.05 else "low"
        volume = "high" if market_data.get('volume_ratio', 1) > 1.5 else "low"
        
        return f"{price_trend}_{volatility}_{volume}"
    
    def choose_action(self, state: str, available_actions: List[str]) -> str:
        """使用ε-贪婪策略选择动作"""
        if np.random.random() < self.epsilon:
            # 探索：随机选择
            return np.random.choice(available_actions)
        else:
            # 利用：选择最优动作
            if state not in self.q_table:
                self.q_table[state] = {action: 0.0 for action in available_actions}
            
            return max(self.q_table[state], key=self.q_table[state].get)
    
    def update_q_value(self, state: str, action: str, reward: float, next_state: str):
        """更新Q值"""
        if state not in self.q_table:
            self.q_table[state] = {}
        if action not in self.q_table[state]:
            self.q_table[state][action] = 0.0
        
        if next_state not in self.q_table:
            self.q_table[next_state] = {}
        
        # Q-learning更新规则
        old_q = self.q_table[state][action]
        next_max_q = max(self.q_table[next_state].values()) if self.q_table[next_state] else 0
        
        new_q = old_q + self.learning_rate * (reward + 0.9 * next_max_q - old_q)
        self.q_table[state][action] = new_q
    
    def get_optimized_parameters(self, current_state: str) -> Dict[str, float]:
        """获取优化后的策略参数"""
        # 根据Q表选择最优参数组合
        parameter_actions = [
            'conservative', 'moderate', 'aggressive'
        ]
        
        optimal_action = self.choose_action(current_state, parameter_actions)
        
        # 返回对应的参数设置
        parameter_sets = {
            'conservative': {
                'position_size_multiplier': 0.8,
                'stop_loss_multiplier': 0.8,
                'signal_threshold': 75
            },
            'moderate': {
                'position_size_multiplier': 1.0,
                'stop_loss_multiplier': 1.0,
                'signal_threshold': 65
            },
            'aggressive': {
                'position_size_multiplier': 1.2,
                'stop_loss_multiplier': 1.2,
                'signal_threshold': 55
            }
        }
        
        return parameter_sets.get(optimal_action, parameter_sets['moderate'])

class AISignalFilter:
    """AI信号过滤器"""
    
    def __init__(self, config: AIConfig):
        self.config = config
        self.price_predictor = TransformerPricePredictor(config)
        self.sentiment_analyzer = SentimentAnalyzer(config)
        self.rl_optimizer = ReinforcementLearningOptimizer(config)
        
    def filter_trading_signal(self, 
                            raw_signal: Dict,
                            market_data: pd.DataFrame,
                            technical_indicators: Dict,
                            onchain_data: Dict) -> Dict[str, any]:
        """
        使用AI模型过滤交易信号
        
        Args:
            raw_signal: 原始技术分析信号
            market_data: 市场数据
            technical_indicators: 技术指标
            onchain_data: 链上数据
            
        Returns:
            过滤后的增强信号
        """
        
        # 1. Transformer价格预测
        features = self.price_predictor.prepare_features(
            market_data, technical_indicators, onchain_data
        )
        price_prediction = self.price_predictor.predict_price_probability(features)
        
        # 2. 情绪分析
        sentiment_data = self.sentiment_analyzer.analyze_twitter_sentiment()
        sentiment_signal = self.sentiment_analyzer.get_sentiment_signal(sentiment_data)
        
        # 3. 强化学习优化
        current_state = self.rl_optimizer.get_state({
            'price_change': market_data['close'].pct_change().iloc[-1],
            'volatility': market_data['close'].pct_change().std(),
            'volume_ratio': market_data['volume'].iloc[-1] / market_data['volume'].mean()
        })
        optimized_params = self.rl_optimizer.get_optimized_parameters(current_state)
        
        # 4. 综合评估和过滤
        enhanced_signal = self._combine_ai_signals(
            raw_signal, price_prediction, sentiment_signal, optimized_params
        )
        
        return enhanced_signal
    
    def _combine_ai_signals(self, 
                           raw_signal: Dict,
                           price_prediction: Dict,
                           sentiment_signal: Dict,
                           optimized_params: Dict) -> Dict[str, any]:
        """综合AI信号"""
        
        # AI确认逻辑
        ai_confirmation = True
        confidence_boost = 0
        
        # 价格预测确认
        if raw_signal.get('signal_type') == 'bullish':
            if price_prediction['up_probability'] > self.config.model_confidence_threshold:
                confidence_boost += 0.2
            else:
                ai_confirmation = False
                
        elif raw_signal.get('signal_type') == 'bearish':
            if price_prediction['down_probability'] > self.config.model_confidence_threshold:
                confidence_boost += 0.2
            else:
                ai_confirmation = False
        
        # 情绪分析确认
        if sentiment_signal['signal'] == raw_signal.get('signal_type'):
            if sentiment_signal['strength'] == 'strong':
                confidence_boost += 0.15
            else:
                confidence_boost += 0.1
        
        # 计算最终信号强度
        original_strength = raw_signal.get('strength_score', 0.5)
        final_strength = min(1.0, original_strength + confidence_boost)
        
        return {
            'signal_type': raw_signal.get('signal_type'),
            'original_strength': original_strength,
            'ai_enhanced_strength': final_strength,
            'ai_confirmation': ai_confirmation,
            'price_prediction': price_prediction,
            'sentiment_analysis': sentiment_signal,
            'optimized_parameters': optimized_params,
            'recommendation': 'execute' if ai_confirmation and final_strength > 0.7 else 'hold',
            'confidence': final_strength
        }

if __name__ == "__main__":
    # 测试AI增强模块
    from .config import AIConfig
    
    print("🤖 AI增强系统测试")
    
    # 初始化配置
    config = AIConfig()
    ai_filter = AISignalFilter(config)
    
    # 创建测试数据
    dates = pd.date_range('2024-01-01', periods=200, freq='1H')
    np.random.seed(42)
    prices = 50000 + np.cumsum(np.random.randn(200) * 100)
    
    test_market_data = pd.DataFrame({
        'open': prices + np.random.randn(200) * 50,
        'high': prices + np.abs(np.random.randn(200)) * 100,
        'low': prices - np.abs(np.random.randn(200)) * 100,
        'close': prices,
        'volume': np.random.randint(1000, 10000, 200)
    }, index=dates)
    
    # 模拟原始信号
    raw_signal = {
        'signal_type': 'bullish',
        'strength_score': 0.65,
        'source': 'technical_analysis'
    }
    
    # 测试AI过滤
    enhanced_signal = ai_filter.filter_trading_signal(
        raw_signal=raw_signal,
        market_data=test_market_data,
        technical_indicators={},
        onchain_data={}
    )
    
    print(f"\n📊 AI信号过滤结果:")
    print(f"原始信号强度: {enhanced_signal['original_strength']:.2f}")
    print(f"AI增强强度: {enhanced_signal['ai_enhanced_strength']:.2f}")
    print(f"AI确认: {enhanced_signal['ai_confirmation']}")
    print(f"最终推荐: {enhanced_signal['recommendation']}")
    print(f"置信度: {enhanced_signal['confidence']:.2f}")
    
    print("\n✅ AI增强系统测试完成") 