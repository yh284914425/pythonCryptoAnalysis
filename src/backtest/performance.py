"""
性能分析器

计算各种交易策略的性能指标，如夏普比率、最大回撤等
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
import warnings

warnings.filterwarnings('ignore')


class PerformanceAnalyzer:
    """性能分析器"""
    
    def __init__(self):
        pass
    
    def calculate_metrics(self, returns: pd.Series, 
                         risk_free_rate: float = 0.02) -> Dict[str, float]:
        """
        计算全面的性能指标
        
        Args:
            returns: 收益率序列
            risk_free_rate: 无风险利率（年化）
            
        Returns:
            性能指标字典
        """
        if len(returns) == 0:
            return self._empty_metrics()
        
        # 清理数据
        returns = returns.dropna().replace([np.inf, -np.inf], 0)
        
        if len(returns) == 0:
            return self._empty_metrics()
        
        try:
            metrics = {}
            
            # 基础统计
            metrics['total_return'] = self.calculate_total_return(returns)
            metrics['annualized_return'] = self.calculate_annualized_return(returns)
            metrics['volatility'] = self.calculate_volatility(returns)
            metrics['sharpe_ratio'] = self.calculate_sharpe_ratio(returns, risk_free_rate)
            
            # 风险指标
            metrics['max_drawdown'] = self.calculate_max_drawdown(returns)
            metrics['calmar_ratio'] = self.calculate_calmar_ratio(returns)
            metrics['sortino_ratio'] = self.calculate_sortino_ratio(returns, risk_free_rate)
            
            # 分布指标
            metrics['skewness'] = self.calculate_skewness(returns)
            metrics['kurtosis'] = self.calculate_kurtosis(returns)
            metrics['var_95'] = self.calculate_var(returns, confidence=0.95)
            metrics['cvar_95'] = self.calculate_cvar(returns, confidence=0.95)
            
            # 其他指标
            metrics['win_rate'] = self.calculate_win_rate(returns)
            metrics['profit_factor'] = self.calculate_profit_factor(returns)
            metrics['recovery_factor'] = self.calculate_recovery_factor(returns)
            
            return metrics
            
        except Exception as e:
            print(f"计算性能指标时出错: {e}")
            return self._empty_metrics()
    
    def _empty_metrics(self) -> Dict[str, float]:
        """返回空的指标字典"""
        return {
            'total_return': 0.0,
            'annualized_return': 0.0,
            'volatility': 0.0,
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0,
            'calmar_ratio': 0.0,
            'sortino_ratio': 0.0,
            'skewness': 0.0,
            'kurtosis': 0.0,
            'var_95': 0.0,
            'cvar_95': 0.0,
            'win_rate': 0.0,
            'profit_factor': 0.0,
            'recovery_factor': 0.0
        }
    
    def calculate_total_return(self, returns: pd.Series) -> float:
        """计算总收益率"""
        try:
            return (1 + returns).prod() - 1
        except:
            return 0.0
    
    def calculate_annualized_return(self, returns: pd.Series, 
                                   periods_per_year: int = 252) -> float:
        """计算年化收益率"""
        try:
            total_return = self.calculate_total_return(returns)
            n_periods = len(returns)
            if n_periods == 0:
                return 0.0
            years = n_periods / periods_per_year
            return (1 + total_return) ** (1/years) - 1
        except:
            return 0.0
    
    def calculate_volatility(self, returns: pd.Series, 
                           periods_per_year: int = 252) -> float:
        """计算年化波动率"""
        try:
            if len(returns) <= 1:
                return 0.0
            return returns.std() * np.sqrt(periods_per_year)
        except:
            return 0.0
    
    def calculate_sharpe_ratio(self, returns: pd.Series, 
                              risk_free_rate: float = 0.02,
                              periods_per_year: int = 252) -> float:
        """计算夏普比率"""
        try:
            excess_return = self.calculate_annualized_return(returns) - risk_free_rate
            volatility = self.calculate_volatility(returns, periods_per_year)
            
            if volatility == 0:
                return 0.0
            
            return excess_return / volatility
        except:
            return 0.0
    
    def calculate_max_drawdown(self, returns: pd.Series) -> float:
        """计算最大回撤"""
        try:
            cumulative = (1 + returns).cumprod()
            rolling_max = cumulative.expanding().max()
            drawdown = (cumulative - rolling_max) / rolling_max
            return drawdown.min()
        except:
            return 0.0
    
    def calculate_calmar_ratio(self, returns: pd.Series) -> float:
        """计算卡尔玛比率"""
        try:
            annualized_return = self.calculate_annualized_return(returns)
            max_drawdown = abs(self.calculate_max_drawdown(returns))
            
            if max_drawdown == 0:
                return 0.0
            
            return annualized_return / max_drawdown
        except:
            return 0.0
    
    def calculate_sortino_ratio(self, returns: pd.Series, 
                               risk_free_rate: float = 0.02,
                               periods_per_year: int = 252) -> float:
        """计算索提诺比率"""
        try:
            excess_return = self.calculate_annualized_return(returns) - risk_free_rate
            negative_returns = returns[returns < 0]
            
            if len(negative_returns) == 0:
                return float('inf') if excess_return > 0 else 0.0
            
            downside_deviation = negative_returns.std() * np.sqrt(periods_per_year)
            
            if downside_deviation == 0:
                return 0.0
            
            return excess_return / downside_deviation
        except:
            return 0.0
    
    def calculate_skewness(self, returns: pd.Series) -> float:
        """计算偏度"""
        try:
            if len(returns) < 3:
                return 0.0
            return returns.skew()
        except:
            return 0.0
    
    def calculate_kurtosis(self, returns: pd.Series) -> float:
        """计算峰度"""
        try:
            if len(returns) < 4:
                return 0.0
            return returns.kurtosis()
        except:
            return 0.0
    
    def calculate_var(self, returns: pd.Series, confidence: float = 0.95) -> float:
        """计算风险价值 (VaR)"""
        try:
            if len(returns) == 0:
                return 0.0
            return returns.quantile(1 - confidence)
        except:
            return 0.0
    
    def calculate_cvar(self, returns: pd.Series, confidence: float = 0.95) -> float:
        """计算条件风险价值 (CVaR)"""
        try:
            if len(returns) == 0:
                return 0.0
            var = self.calculate_var(returns, confidence)
            return returns[returns <= var].mean()
        except:
            return 0.0
    
    def calculate_win_rate(self, returns: pd.Series) -> float:
        """计算胜率"""
        try:
            if len(returns) == 0:
                return 0.0
            return (returns > 0).mean()
        except:
            return 0.0
    
    def calculate_profit_factor(self, returns: pd.Series) -> float:
        """计算盈利因子"""
        try:
            positive_returns = returns[returns > 0].sum()
            negative_returns = abs(returns[returns < 0].sum())
            
            if negative_returns == 0:
                return float('inf') if positive_returns > 0 else 0.0
            
            return positive_returns / negative_returns
        except:
            return 0.0
    
    def calculate_recovery_factor(self, returns: pd.Series) -> float:
        """计算恢复因子"""
        try:
            total_return = self.calculate_total_return(returns)
            max_drawdown = abs(self.calculate_max_drawdown(returns))
            
            if max_drawdown == 0:
                return 0.0
            
            return total_return / max_drawdown
        except:
            return 0.0
    
    def calculate_rolling_metrics(self, returns: pd.Series, 
                                 window: int = 252) -> pd.DataFrame:
        """
        计算滚动性能指标
        
        Args:
            returns: 收益率序列
            window: 滚动窗口大小
            
        Returns:
            滚动指标DataFrame
        """
        try:
            if len(returns) < window:
                return pd.DataFrame()
            
            rolling_metrics = pd.DataFrame(index=returns.index)
            
            # 滚动年化收益率
            rolling_metrics['rolling_return'] = returns.rolling(window).apply(
                lambda x: self.calculate_annualized_return(x)
            )
            
            # 滚动波动率
            rolling_metrics['rolling_volatility'] = returns.rolling(window).apply(
                lambda x: self.calculate_volatility(x)
            )
            
            # 滚动夏普比率
            rolling_metrics['rolling_sharpe'] = returns.rolling(window).apply(
                lambda x: self.calculate_sharpe_ratio(x)
            )
            
            # 滚动最大回撤
            rolling_metrics['rolling_max_drawdown'] = returns.rolling(window).apply(
                lambda x: self.calculate_max_drawdown(x)
            )
            
            return rolling_metrics
            
        except Exception as e:
            print(f"计算滚动指标时出错: {e}")
            return pd.DataFrame()
    
    def generate_performance_report(self, returns: pd.Series,
                                   benchmark_returns: Optional[pd.Series] = None) -> Dict[str, Any]:
        """
        生成完整的性能报告
        
        Args:
            returns: 策略收益率
            benchmark_returns: 基准收益率（可选）
            
        Returns:
            性能报告字典
        """
        try:
            report = {}
            
            # 策略指标
            report['strategy_metrics'] = self.calculate_metrics(returns)
            
            # 基准指标（如果提供）
            if benchmark_returns is not None:
                benchmark_returns = benchmark_returns.reindex(returns.index).dropna()
                if len(benchmark_returns) > 0:
                    report['benchmark_metrics'] = self.calculate_metrics(benchmark_returns)
                    
                    # 超额收益
                    excess_returns = returns - benchmark_returns
                    report['excess_metrics'] = self.calculate_metrics(excess_returns)
                    
                    # 信息比率
                    if excess_returns.std() != 0:
                        report['information_ratio'] = excess_returns.mean() / excess_returns.std() * np.sqrt(252)
                    else:
                        report['information_ratio'] = 0.0
                    
                    # Beta
                    if benchmark_returns.var() != 0:
                        report['beta'] = excess_returns.cov(benchmark_returns) / benchmark_returns.var()
                    else:
                        report['beta'] = 0.0
            
            # 时间序列统计
            report['time_series_stats'] = {
                'start_date': returns.index[0] if len(returns) > 0 else None,
                'end_date': returns.index[-1] if len(returns) > 0 else None,
                'total_periods': len(returns),
                'positive_periods': (returns > 0).sum(),
                'negative_periods': (returns < 0).sum(),
                'zero_periods': (returns == 0).sum()
            }
            
            return report
            
        except Exception as e:
            print(f"生成性能报告时出错: {e}")
            return {'error': str(e)}