import pandas as pd
import numpy as np
import sys
import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec
import datetime
from typing import Dict, List, Any, Optional
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
import warnings
import time
import logging
from contextlib import contextmanager
from functools import wraps

# 抑制常见的数学运算警告
warnings.filterwarnings('ignore', category=RuntimeWarning, message='invalid value encountered in divide')
warnings.filterwarnings('ignore', category=RuntimeWarning, message='divide by zero encountered')
warnings.filterwarnings('ignore', category=RuntimeWarning, message='invalid value encountered in true_divide')
np.seterr(divide='ignore', invalid='ignore')

# 配置日志系统 - 进一步减少输出
logging.basicConfig(
    level=logging.ERROR,  # 只显示ERROR级别，更加静默
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 也可以单独设置某些模块的日志级别
logging.getLogger('matplotlib').setLevel(logging.ERROR)
logging.getLogger('PIL').setLevel(logging.ERROR)

# 自定义异常类
class IndicatorError(Exception):
    """指标计算错误基类"""
    pass

class DataError(IndicatorError):
    """数据相关错误"""
    pass

class CalculationError(IndicatorError):
    """计算错误"""
    pass

# 数据验证装饰器
def validate_dataframe(min_rows=2):
    """验证输入数据的装饰器 - 静默版本"""
    def decorator(func):
        @wraps(func)
        def wrapper(self, df: pd.DataFrame, **kwargs):
            if not isinstance(df, pd.DataFrame):
                raise TypeError(f"需要 DataFrame 输入")
            
            if len(df) < min_rows:
                # 静默跳过，不抛出异常
                return {
                    'values': np.array([]),
                    'current': 0,
                    'signal': 'neutral',
                    'strength': 0.0,
                    'confidence': 0.5
                }
            
            required_columns = ['开盘价', '最高价', '最低价', '收盘价', '成交量']
            missing = [col for col in required_columns if col not in df.columns]
            if missing:
                return {
                    'values': np.array([]),
                    'current': 0,
                    'signal': 'neutral',
                    'strength': 0.0,
                    'confidence': 0.5
                }
            
            return func(self, df, **kwargs)
        return wrapper
    return decorator

# 智能缓存管理器
class SmartCache:
    """智能缓存管理器 - 线程安全版本，支持LRU和TTL"""
    
    def __init__(self, max_size: int = 1000, ttl: int = 3600):
        self.cache = {}
        self.access_count = {}
        self.last_access = {}
        self.max_size = max_size
        self.ttl = ttl
        # 添加线程锁确保并发安全
        import threading
        self._lock = threading.RLock()
    
    def get(self, key: str):
        """获取缓存值 - 线程安全"""
        with self._lock:
            if key in self.cache:
                # 检查过期
                if time.time() - self.last_access[key] > self.ttl:
                    self._remove_key(key)
                    return None
                
                self.access_count[key] += 1
                self.last_access[key] = time.time()
                return self.cache[key]
            return None
    
    def set(self, key: str, value):
        """设置缓存值 - 线程安全"""
        with self._lock:
            # LRU 淘汰策略
            if len(self.cache) >= self.max_size:
                lru_key = min(self.cache.keys(), 
                             key=lambda k: self.access_count.get(k, 0))
                self._remove_key(lru_key)
            
            self.cache[key] = value
            self.access_count[key] = 1
            self.last_access[key] = time.time()
    
    def _remove_key(self, key: str):
        """移除缓存键 - 内部调用，已在锁内"""
        if key in self.cache:
            del self.cache[key]
            del self.access_count[key]
            del self.last_access[key]
    
    def clear(self):
        """清空缓存 - 线程安全"""
        with self._lock:
            self.cache.clear()
            self.access_count.clear()
            self.last_access.clear()
    
    def get_stats(self):
        """获取缓存统计 - 线程安全，修复并发安全问题"""
        with self._lock:
            return {
                'size': len(self.cache),
                'max_size': self.max_size,
                'hit_keys': list(self.cache.keys()),  # list()已经创建新列表，无需.copy()
                'ttl': self.ttl
            }

# 数据预处理工具
class DataProcessor:
    """数据预处理工具 - 统一处理数据类型和格式"""
    
    @staticmethod
    def ensure_numeric(df: pd.DataFrame, required_columns: List[str] = None) -> pd.DataFrame:
        """确保数值列的类型正确，处理边界情况，避免TA-Lib输入错误"""
        if required_columns is None:
            required_columns = ['开盘价', '最高价', '最低价', '收盘价', '成交量']
        
        df_processed = df.copy()
        
        for col in required_columns:
            if col in df_processed.columns:
                # 统一转换为数值类型，错误值转为NaN
                df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce')
                
                # 检查是否整列都是NaN
                if df_processed[col].isna().all():
                    # 使用合理的默认值
                    if col == '成交量':
                        df_processed[col] = 0.0
                    else:
                        # 价格列使用其他有效列的值作为参考
                        reference_value = None
                        for ref_col in required_columns:
                            if ref_col != col and ref_col in df_processed.columns:
                                valid_values = df_processed[ref_col].dropna()
                                if len(valid_values) > 0:
                                    reference_value = valid_values.iloc[-1]
                                    break
                        
                        # 如果找到参考值，使用；否则使用默认值
                        df_processed[col] = reference_value if reference_value is not None else 100.0
                else:
                    # 正常的填充逻辑
                    df_processed[col] = df_processed[col].ffill()
                    
                    # 如果还有NaN（开头），用后向填充
                    df_processed[col] = df_processed[col].bfill()
                    
                    # 最后的保护：如果仍有NaN，用中位数填充
                    if df_processed[col].isna().any():
                        median_val = df_processed[col].median()
                        if pd.notna(median_val):
                            df_processed[col] = df_processed[col].fillna(median_val)
                        else:
                            # 极端情况，使用默认值
                            default_val = 0.0 if col == '成交量' else 100.0
                            df_processed[col] = df_processed[col].fillna(default_val)
                
                # 确保最终类型为float64
                df_processed[col] = df_processed[col].astype(np.float64)
        
        return df_processed
    
    @staticmethod
    def validate_data_quality(df: pd.DataFrame) -> Dict[str, Any]:
        """验证数据质量"""
        quality_report = {
            'valid': True,
            'issues': [],
            'stats': {}
        }
        
        required_columns = ['开盘价', '最高价', '最低价', '收盘价', '成交量']
        
        for col in required_columns:
            if col not in df.columns:
                quality_report['valid'] = False
                quality_report['issues'].append(f"缺少列: {col}")
                continue
                
            # 检查数据类型
            if not pd.api.types.is_numeric_dtype(df[col]):
                quality_report['issues'].append(f"{col} 不是数值类型")
            
            # 检查负值
            if (df[col] < 0).any():
                quality_report['issues'].append(f"{col} 包含负值")
            
            # 检查NaN比例
            nan_ratio = df[col].isnull().sum() / len(df)
            if nan_ratio > 0.1:
                quality_report['issues'].append(f"{col} NaN比例过高: {nan_ratio:.1%}")
            
            quality_report['stats'][col] = {
                'nan_count': df[col].isnull().sum(),
                'nan_ratio': nan_ratio,
                'min': df[col].min(),
                'max': df[col].max()
            }
        
        return quality_report

# 环境能力检测器
class EnvironmentChecker:
    """检测系统环境和依赖能力"""
    
    @staticmethod
    def check_numba_capability() -> Dict[str, Any]:
        """检查Numba滚动计算能力"""
        numba_info = {
            'available': False,
            'version': None,
            'pandas_support': False,
            'recommendation': ''
        }
        
        try:
            import numba
            numba_info['available'] = True
            numba_info['version'] = numba.__version__
            
            # 检查pandas版本
            pandas_version = pd.__version__
            major, minor = map(int, pandas_version.split('.')[:2])
            
            if (major > 2) or (major == 2 and minor >= 1):
                numba_info['pandas_support'] = True
                numba_info['recommendation'] = "✅ Numba滚动计算已启用"
            else:
                numba_info['recommendation'] = f"⚠️ Pandas {pandas_version} < 2.1，Numba滚动不可用"
                
        except ImportError:
            numba_info['recommendation'] = "❌ Numba未安装，使用纯Python滚动计算"
        
        return numba_info
    
    @staticmethod
    def check_signal_capability() -> Dict[str, Any]:
        """检查信号处理能力"""
        signal_info = {
            'available': False,
            'platform': None,
            'recommendation': ''
        }
        
        try:
            import signal
            import platform
            signal_info['platform'] = platform.system()
            
            if hasattr(signal, 'SIGALRM'):
                signal_info['available'] = True
                signal_info['recommendation'] = "✅ 信号超时控制可用"
            else:
                signal_info['recommendation'] = "⚠️ Windows系统，信号超时不可用"
                
        except ImportError:
            signal_info['recommendation'] = "❌ 信号模块不可用"
        
        return signal_info
    
    @staticmethod
    def print_environment_report():
        """打印环境检测报告"""
        print("\n" + "="*60)
        print("🔧 环境能力检测报告")
        print("="*60)
        
        # Numba检测
        numba_info = EnvironmentChecker.check_numba_capability()
        print(f"📊 Numba滚动计算: {numba_info['recommendation']}")
        if numba_info['available']:
            print(f"   版本: {numba_info['version']}")
        
        # 信号检测
        signal_info = EnvironmentChecker.check_signal_capability()
        print(f"⏰ 信号超时控制: {signal_info['recommendation']}")
        print(f"   平台: {signal_info['platform']}")
        
        # TA-Lib检测
        print(f"📈 TA-Lib加速: {'✅ 可用' if TALIB_AVAILABLE else '❌ 不可用'}")
        
        print("="*60)

# 配置验证器
class ConfigValidator:
    """配置验证器"""
    
    @staticmethod
    def validate_config(config) -> List[str]:
        """验证配置完整性和合理性"""
        errors = []
        
        # 检查必要的配置项
        required_sections = ['technical', 'trading', 'risk_management']
        for section in required_sections:
            if not hasattr(config, section):
                errors.append(f"缺少配置节: {section}")
        
        # 验证技术指标参数
        if hasattr(config, 'technical'):
            tech = config.technical
            
            # KDJ参数验证
            if 'kdj' in tech:
                for volatility in ['low', 'medium', 'high']:
                    if volatility in tech['kdj']:
                        params = tech['kdj'][volatility]
                        if params.get('k', 0) < 1 or params.get('k', 0) > 100:
                            errors.append(f"KDJ {volatility} K参数无效: {params.get('k')}")
            
            # ADX参数验证
            if 'adx' in tech:
                period = tech['adx'].get('period', 0)
                if period < 5 or period > 50:
                    errors.append(f"ADX周期无效: {period}")
        
        return errors
    
    @staticmethod
    def validate_data(df: pd.DataFrame, min_records: int = 30) -> List[str]:
        """验证数据质量 - 降低数据量要求"""
        errors = []
        
        # 检查数据量 - 只在数据极少时才报错
        if len(df) < min_records:
            errors.append(f"数据量极少: {len(df)} < {min_records}")
        
        # 检查必要列
        required_columns = ['开盘价', '最高价', '最低价', '收盘价', '成交量']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            errors.append(f"缺少列: {missing_columns}")
        
        # 检查数据质量
        if '收盘价' in df.columns:
            nulls = df['收盘价'].isnull().sum()
            if nulls > 0:
                errors.append(f"收盘价包含 {nulls} 个空值")
            
            # 检查价格合理性
            if df['收盘价'].min() <= 0:
                errors.append("存在无效价格 (<=0)")
        
        return errors

# 实时监控器
class RealTimeMonitor:
    """实时监控仪表板 - 防止内存泄漏"""
    
    def __init__(self, max_history: int = 10000):
        self.max_history = max_history
        self.metrics = {
            'signals': [],
            'performance': [],
            'errors': []
        }
        self.start_time = time.time()
    
    def _trim_history(self, metric_name: str):
        """限制历史记录大小，防止内存泄漏"""
        if len(self.metrics[metric_name]) > self.max_history:
            # 保留最新的记录
            self.metrics[metric_name] = self.metrics[metric_name][-self.max_history:]
    
    def log_signal(self, symbol: str, signal: Dict[str, Any]):
        """记录信号"""
        self.metrics['signals'].append({
            'timestamp': time.time(),
            'symbol': symbol,
            'direction': signal.get('direction'),
            'strength': signal.get('strength'),
            'confidence': signal.get('confidence')
        })
        self._trim_history('signals')
    
    def log_performance(self, operation: str, duration: float):
        """记录性能"""
        self.metrics['performance'].append({
            'timestamp': time.time(),
            'operation': operation,
            'duration': duration
        })
        self._trim_history('performance')
    
    def log_error(self, error: str):
        """记录错误"""
        self.metrics['errors'].append({
            'timestamp': time.time(),
            'error': error
        })
        self._trim_history('errors')
    
    def get_dashboard(self) -> Dict[str, Any]:
        """获取仪表板数据"""
        uptime = time.time() - self.start_time
        
        # 计算信号统计
        recent_signals = [s for s in self.metrics['signals'] 
                         if time.time() - s['timestamp'] < 3600]  # 最近1小时
        
        buy_signals = len([s for s in recent_signals if s['direction'] == 'buy'])
        sell_signals = len([s for s in recent_signals if s['direction'] == 'sell'])
        
        # 计算性能统计
        recent_perf = [p for p in self.metrics['performance'] 
                      if time.time() - p['timestamp'] < 300]  # 最近5分钟
        
        avg_duration = np.mean([p['duration'] for p in recent_perf]) if recent_perf else 0
        
        return {
            'uptime_hours': uptime / 3600,
            'signals_1h': {
                'total': len(recent_signals),
                'buy': buy_signals,
                'sell': sell_signals
            },
            'performance_5m': {
                'avg_duration_ms': avg_duration * 1000,
                'operations': len(recent_perf)
            },
            'errors_total': len(self.metrics['errors']),
            'health_score': self._calculate_health_score()
        }
    
    def _calculate_health_score(self) -> float:
        """计算系统健康分数"""
        # 基于错误率、性能等计算
        error_rate = len(self.metrics['errors']) / max(len(self.metrics['performance']), 1)
        health = max(0, 1 - error_rate) * 100
        return round(health, 1)

# 性能监控器
class PerformanceMonitor:
    """实时性能监控"""
    
    def __init__(self):
        self.metrics = {
            'calculation_times': [],
            'memory_usage': [],
            'cache_hit_rate': 0,
            'error_count': 0,
            'total_operations': 0
        }
    
    @contextmanager
    def monitor(self, operation_name: str):
        """监控上下文管理器"""
        start_time = time.time()
        start_memory = 0
        
        try:
            import psutil
            start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        except ImportError:
            pass  # psutil不可用时跳过内存监控
        
        error_occurred = False
        try:
            yield
        except Exception as e:
            error_occurred = True
            self.metrics['error_count'] += 1
            raise
        finally:
            end_time = time.time()
            end_memory = start_memory
            
            try:
                import psutil
                end_memory = psutil.Process().memory_info().rss / 1024 / 1024
            except ImportError:
                pass
            
            self.metrics['calculation_times'].append({
                'operation': operation_name,
                'duration': end_time - start_time,
                'memory_delta': end_memory - start_memory,
                'success': not error_occurred
            })
            self.metrics['total_operations'] += 1
    
    def get_report(self):
        """生成性能报告"""
        if not self.metrics['calculation_times']:
            return {'status': 'no_data'}
        
        times = [m['duration'] for m in self.metrics['calculation_times']]
        memory = [m['memory_delta'] for m in self.metrics['calculation_times']]
        success_rate = len([m for m in self.metrics['calculation_times'] if m['success']]) / len(self.metrics['calculation_times'])
        
        return {
            'avg_calculation_time': np.mean(times),
            'max_calculation_time': np.max(times),
            'total_memory_used': sum(memory),
            'error_rate': self.metrics['error_count'] / self.metrics['total_operations'] if self.metrics['total_operations'] > 0 else 0,
            'success_rate': success_rate,
            'total_operations': self.metrics['total_operations']
        }

# TA-Lib 检测和导入
TALIB_AVAILABLE = False
try:
    import talib
    TALIB_AVAILABLE = True
    print("✅ TA-Lib 可用，将使用高性能库计算")
except ImportError:
    print("⚠️ TA-Lib 不可用，将使用自实现算法")
    talib = None

# 添加项目根目录到路径，以便导入其他模块
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

try:
    from src.strategies.divergence_analyzer import DivergenceAnalyzer
except ImportError:
    # 如果直接运行当前文件，尝试相对导入
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from divergence_analyzer import DivergenceAnalyzer


class BaseIndicator(ABC):
    """所有指标的基类"""
    
    def __init__(self, name: str, category: str, params: Dict[str, Any] = None, use_talib: bool = True):
        self.name = name
        self.category = category
        self.params = params or {}
        self.cache = {}
        self.use_talib = use_talib and TALIB_AVAILABLE
    
    @abstractmethod
    def calculate(self, df: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """计算指标值"""
        pass
    
    @abstractmethod
    def get_signal(self, values: Dict[str, Any]) -> str:
        """获取信号类型: buy/sell/neutral"""
        pass
    
    @abstractmethod
    def get_strength(self, values: Dict[str, Any]) -> float:
        """获取信号强度: 0-1"""
        pass
    
    def get_confidence(self, values: Dict[str, Any]) -> float:
        """获取信号置信度: 0-1"""
        return 0.5  # 默认实现


class HybridIndicator(BaseIndicator):
    """混合指标基类 - 支持TA-Lib和自实现算法"""
    
    def __init__(self, name: str, category: str, params: Dict[str, Any] = None, use_talib: bool = True):
        super().__init__(name, category, params, use_talib)
        self.performance_stats = {'talib_calls': 0, 'custom_calls': 0, 'talib_time': 0.0, 'custom_time': 0.0}
    
    def _calculate_talib(self, df: pd.DataFrame, **kwargs) -> Optional[Dict[str, Any]]:
        """TA-Lib实现 - 子类需要重写"""
        return None
    
    def _calculate_custom(self, df: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """自实现算法 - 子类需要重写"""
        raise NotImplementedError("Custom implementation required")
    
    def calculate(self, df: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """智能选择算法计算指标"""
        import time
        
        start_time = time.time()
        
        # 优先尝试TA-Lib
        if self.use_talib and TALIB_AVAILABLE:
            try:
                result = self._calculate_talib(df, **kwargs)
                if result is not None:
                    self.performance_stats['talib_calls'] += 1
                    self.performance_stats['talib_time'] += time.time() - start_time
                    return result
            except Exception as e:
                warnings.warn(f"TA-Lib计算失败，回退到自实现: {str(e)}")
        
        # 回退到自实现
        try:
            result = self._calculate_custom(df, **kwargs)
            self.performance_stats['custom_calls'] += 1
            self.performance_stats['custom_time'] += time.time() - start_time
            return result
        except Exception as e:
            warnings.warn(f"指标 {self.name} 计算失败: {str(e)}")
            return {'values': np.array([]), 'current': 0, 'signal_quality': 'error'}
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """获取性能统计"""
        total_calls = self.performance_stats['talib_calls'] + self.performance_stats['custom_calls']
        if total_calls == 0:
            return {'usage': 'no_calls', 'performance_ratio': 0}
        
        talib_ratio = self.performance_stats['talib_calls'] / total_calls
        avg_talib_time = (self.performance_stats['talib_time'] / self.performance_stats['talib_calls'] 
                         if self.performance_stats['talib_calls'] > 0 else 0)
        avg_custom_time = (self.performance_stats['custom_time'] / self.performance_stats['custom_calls'] 
                          if self.performance_stats['custom_calls'] > 0 else 0)
        
        # 修复：避免除零错误
        performance_ratio = (avg_custom_time / avg_talib_time) if avg_talib_time > 0 else None
        
        return {
            'usage': f'{talib_ratio:.1%} TA-Lib, {1-talib_ratio:.1%} Custom',
            'performance_ratio': performance_ratio,
            'avg_talib_time': avg_talib_time * 1000,  # ms
            'avg_custom_time': avg_custom_time * 1000  # ms
        }


class RSIIndicator(HybridIndicator):
    """RSI指标实现 - 混合TA-Lib和自实现"""
    
    def __init__(self, period: int = 14, use_talib: bool = True):
        super().__init__("RSI", "momentum", {"period": period}, use_talib)
    
    def _calculate_talib(self, df: pd.DataFrame, **kwargs) -> Optional[Dict[str, Any]]:
        """TA-Lib实现RSI"""
        if not TALIB_AVAILABLE:
            return None
        
        closes = df['收盘价'].astype(float).values
        period = self.params['period']
        
        # 使用TA-Lib计算RSI
        rsi_values = talib.RSI(closes, timeperiod=period)
        
        # 处理NaN值
        rsi_values = np.where(np.isnan(rsi_values), 50, rsi_values)
        
        return {
            'values': rsi_values,
            'current': rsi_values[-1] if len(rsi_values) > 0 else 50,
            'previous': rsi_values[-2] if len(rsi_values) > 1 else 50,
            'algorithm': 'talib'
        }
    
    @validate_dataframe(min_rows=14)
    def _calculate_custom(self, df: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """自实现RSI算法 - 修复：使用Wilder's Smoothing (EMA)"""
        logger.debug(f"开始计算RSI自实现算法，数据量: {len(df)}")
        closes = df['收盘价'].astype(float)
        period = self.params['period']
        
        delta = closes.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        # 修复：使用Wilder's Smoothing (等同于alpha=1/period的EMA)
        # 这与TA-Lib的RSI算法一致
        alpha = 1.0 / period
        gain_ema = gain.ewm(alpha=alpha, adjust=False).mean()
        loss_ema = loss.ewm(alpha=alpha, adjust=False).mean()
        
        # 安全的除法操作
        with np.errstate(divide='ignore', invalid='ignore'):
            rs = np.where(loss_ema == 0, np.inf, gain_ema / loss_ema)
            rsi = np.where(np.isinf(rs), 100, 100 - (100 / (1 + rs)))
            rsi = np.nan_to_num(rsi, nan=50)
        
        return {
            'values': rsi,
            'current': rsi[-1] if len(rsi) > 0 else 50,
            'previous': rsi[-2] if len(rsi) > 1 else 50,
            'algorithm': 'custom_fixed'
        }
    
    def get_signal(self, values: Dict[str, Any]) -> str:
        """获取RSI信号"""
        current = values['current']
        if current > 70:
            return 'sell'
        elif current < 30:
            return 'buy'
        else:
            return 'neutral'
    
    def get_strength(self, values: Dict[str, Any]) -> float:
        """获取RSI信号强度"""
        current = values['current']
        if current > 80:
            return min((current - 80) / 20, 1.0)
        elif current < 20:
            return min((20 - current) / 20, 1.0)
        elif current > 70:
            return (current - 70) / 10 * 0.6
        elif current < 30:
            return (30 - current) / 10 * 0.6
        else:
            return 0.0


class MACDIndicator(HybridIndicator):
    """MACD指标实现 - 混合TA-Lib和自实现"""
    
    def __init__(self, fast: int = 12, slow: int = 26, signal: int = 9, use_talib: bool = True):
        super().__init__("MACD", "momentum", {"fast": fast, "slow": slow, "signal": signal}, use_talib)
    
    def _calculate_talib(self, df: pd.DataFrame, **kwargs) -> Optional[Dict[str, Any]]:
        """TA-Lib实现MACD"""
        if not TALIB_AVAILABLE:
            return None
        
        closes = df['收盘价'].astype(float).values
        fast = self.params['fast']
        slow = self.params['slow']
        signal = self.params['signal']
        
        # 使用TA-Lib计算MACD
        macd_line, signal_line, histogram = talib.MACD(closes, fastperiod=fast, slowperiod=slow, signalperiod=signal)
        
        # 智能处理NaN值 - 优化的填充逻辑
        if len(closes) < 50:  # 只有在数据极少时才提示
            # 数据极少时，使用简单填充
            macd_line = np.where(np.isnan(macd_line), 0, macd_line)
            signal_line = np.where(np.isnan(signal_line), 0, signal_line)
            histogram = np.where(np.isnan(histogram), 0, histogram)
        else:
            # 正常情况下，用智能填充处理NaN
            macd_line = pd.Series(macd_line).bfill().fillna(0).values
            signal_line = pd.Series(signal_line).bfill().fillna(0).values
            histogram = pd.Series(histogram).bfill().fillna(0).values
        
        return {
            'macd': macd_line,
            'signal': signal_line,
            'histogram': histogram,
            'current_macd': macd_line[-1] if len(macd_line) > 0 else 0,
            'current_signal': signal_line[-1] if len(signal_line) > 0 else 0,
            'current_histogram': histogram[-1] if len(histogram) > 0 else 0,
            'algorithm': 'talib'
        }
    
    def _calculate_custom(self, df: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """自实现MACD算法"""
        closes = df['收盘价'].astype(float)
        
        ema_fast = closes.ewm(span=self.params['fast']).mean()
        ema_slow = closes.ewm(span=self.params['slow']).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=self.params['signal']).mean()
        histogram = macd_line - signal_line
        
        return {
            'macd': macd_line.values,
            'signal': signal_line.values,
            'histogram': histogram.values,
            'current_macd': macd_line.iloc[-1] if len(macd_line) > 0 else 0,
            'current_signal': signal_line.iloc[-1] if len(signal_line) > 0 else 0,
            'current_histogram': histogram.iloc[-1] if len(histogram) > 0 else 0,
            'algorithm': 'custom'
        }
    
    def get_signal(self, values: Dict[str, Any]) -> str:
        """获取MACD信号"""
        macd = values['current_macd']
        signal = values['current_signal']
        histogram = values['current_histogram']
        
        if macd > signal and histogram > 0:
            return 'buy'
        elif macd < signal and histogram < 0:
            return 'sell'
        else:
            return 'neutral'
    
    def get_strength(self, values: Dict[str, Any]) -> float:
        """获取MACD信号强度"""
        histogram = abs(values['current_histogram'])
        # 简单的强度计算，实际应用中可以根据历史数据标准化
        return min(histogram * 1000, 1.0)


class EMAIndicator(HybridIndicator):
    """EMA指标实现 - 混合TA-Lib和自实现"""
    
    def __init__(self, periods: List[int] = [20, 50, 200], use_talib: bool = True):
        super().__init__("EMA", "trend", {"periods": periods}, use_talib)
    
    def _calculate_talib(self, df: pd.DataFrame, **kwargs) -> Optional[Dict[str, Any]]:
        """TA-Lib实现EMA"""
        if not TALIB_AVAILABLE:
            return None
        
        closes = df['收盘价'].astype(float).values
        result = {}
        
        for period in self.params['periods']:
            if len(closes) < 30:  # 只有数据极少时才使用简单方法
                # 数据极少时使用收盘价
                ema_values = closes
            else:
                ema_values = talib.EMA(closes, timeperiod=period)
                # 智能填充NaN值
                ema_values = pd.Series(ema_values).bfill().fillna(pd.Series(closes).iloc[-1]).values
            
            result[f'ema_{period}'] = ema_values
            result[f'current_ema_{period}'] = ema_values[-1] if len(ema_values) > 0 else closes[-1]
        
        result['algorithm'] = 'talib'
        return result
    
    def _calculate_custom(self, df: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """自实现EMA算法"""
        closes = df['收盘价'].astype(float)
        result = {}
        
        for period in self.params['periods']:
            ema = closes.ewm(span=period).mean()
            result[f'ema_{period}'] = ema.values
            result[f'current_ema_{period}'] = ema.iloc[-1] if len(ema) > 0 else closes.iloc[-1]
        
        result['algorithm'] = 'custom'
        return result
    
    def get_signal(self, values: Dict[str, Any]) -> str:
        """获取EMA信号"""
        # 使用20, 50, 200周期的EMA排列判断趋势
        ema_20 = values.get('current_ema_20', 0)
        ema_50 = values.get('current_ema_50', 0)
        ema_200 = values.get('current_ema_200', 0)
        
        if ema_20 > ema_50 > ema_200:
            return 'buy'
        elif ema_20 < ema_50 < ema_200:
            return 'sell'
        else:
            return 'neutral'
    
    def get_strength(self, values: Dict[str, Any]) -> float:
        """获取EMA信号强度"""
        ema_20 = values.get('current_ema_20', 0)
        ema_50 = values.get('current_ema_50', 0)
        
        if ema_20 == 0 or ema_50 == 0:
            return 0.0
        
        # 计算EMA间的距离作为强度指标
        strength = abs(ema_20 - ema_50) / ema_50
        return min(strength * 10, 1.0)


class VWAPIndicator(BaseIndicator):
    """VWAP指标实现 - 成交量加权平均价"""
    
    def __init__(self, period: int = 20):
        super().__init__("VWAP", "volume", {"period": period})
    
    def calculate(self, df: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """优化的向量化VWAP计算"""
        closes = df['收盘价'].astype(float)
        highs = df['最高价'].astype(float)
        lows = df['最低价'].astype(float)
        volumes = df['成交量'].astype(float)
        
        # 计算典型价格
        typical_price = (highs + lows + closes) / 3
        
        # 向量化计算VWAP - 使用滚动窗口
        period = self.params['period']
        
        # 计算价格*成交量的滚动和 - 修复：使用numba加速
        price_volume = typical_price * volumes
        try:
            cum_price_volume = price_volume.rolling(window=period, min_periods=1).sum(engine='numba')
            cum_volume = volumes.rolling(window=period, min_periods=1).sum(engine='numba')
        except Exception:
            # numba不可用时回退到默认方法
            cum_price_volume = price_volume.rolling(window=period, min_periods=1).sum()
            cum_volume = volumes.rolling(window=period, min_periods=1).sum()
        
        # 计算VWAP，避免除零
        vwap = cum_price_volume / cum_volume.replace(0, np.nan)
        
        # 处理NaN值，用典型价格填充
        vwap = vwap.fillna(typical_price)
        
        return {
            'values': vwap.values,
            'current': vwap.iloc[-1] if len(vwap) > 0 else closes.iloc[-1],
            'current_price': closes.iloc[-1],
            'algorithm': 'vectorized'
        }
    
    def get_signal(self, values: Dict[str, Any]) -> str:
        """获取VWAP信号"""
        current_price = values.get('current_price', 0)
        vwap = values.get('current', 0)
        
        if current_price > vwap * 1.002:  # 价格显著高于VWAP
            return 'sell'
        elif current_price < vwap * 0.998:  # 价格显著低于VWAP
            return 'buy'
        else:
            return 'neutral'
    
    def get_strength(self, values: Dict[str, Any]) -> float:
        """获取VWAP信号强度"""
        current_price = values.get('current_price', 0)
        vwap = values.get('current', 0)
        
        if vwap == 0:
            return 0.0
        
        deviation = abs(current_price - vwap) / vwap
        return min(deviation * 100, 1.0)  # 偏离度转换为强度


class OBVIndicator(HybridIndicator):
    """OBV指标实现 - 混合TA-Lib和向量化实现"""
    
    def __init__(self, use_talib: bool = True):
        super().__init__("OBV", "volume", {}, use_talib)
    
    def _calculate_talib(self, df: pd.DataFrame, **kwargs) -> Optional[Dict[str, Any]]:
        """TA-Lib实现OBV"""
        if not TALIB_AVAILABLE:
            return None
        
        closes = df['收盘价'].astype(float).values
        volumes = df['成交量'].astype(float).values
        
        # 使用TA-Lib OBV
        obv_values = talib.OBV(closes, volumes)
        
        # 计算MA
        obv_ma = talib.SMA(obv_values, timeperiod=10)
        obv_ma = np.where(np.isnan(obv_ma), obv_values, obv_ma)
        
        return {
            'values': obv_values,
            'ma_values': obv_ma,
            'current': obv_values[-1] if len(obv_values) > 0 else 0,
            'current_ma': obv_ma[-1] if len(obv_ma) > 0 else 0,
            'previous_ma': obv_ma[-2] if len(obv_ma) > 1 else 0,
            'algorithm': 'talib'
        }
    
    def _calculate_custom(self, df: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """修正的向量化OBV算法 - 符合传统OBV标准"""
        closes = df['收盘价'].astype(float)
        volumes = df['成交量'].astype(float)
        
        # 计算价格变化
        price_change = closes.diff()
        
        # 根据价格变化方向确定成交量符号
        # 注意：价格不变时成交量为0
        volume_direction = np.where(price_change > 0, 1, 
                                   np.where(price_change < 0, -1, 0))
        
        # 修复：确保数据类型一致性
        signed_volume = (volumes * pd.Series(volume_direction, index=volumes.index))
        
        # 设置初始值 - 第一个值使用原始成交量
        signed_volume.iloc[0] = volumes.iloc[0]
        
        # 累积计算OBV
        obv = signed_volume.cumsum()
        
        # 计算OBV的移动平均
        obv_ma = obv.rolling(window=10).mean()
        
        return {
            'values': obv.values,
            'ma_values': obv_ma.values,
            'current': obv.iloc[-1] if len(obv) > 0 else 0,
            'current_ma': obv_ma.iloc[-1] if len(obv_ma) > 0 else 0,
            'previous_ma': obv_ma.iloc[-2] if len(obv_ma) > 1 else obv.iloc[-1],
            'algorithm': 'custom'
        }
    
    def get_signal(self, values: Dict[str, Any]) -> str:
        """获取OBV信号"""
        current_ma = values.get('current_ma', 0)
        previous_ma = values.get('previous_ma', 0)
        
        if current_ma > previous_ma:
            return 'buy'
        elif current_ma < previous_ma:
            return 'sell'
        else:
            return 'neutral'
    
    def get_strength(self, values: Dict[str, Any]) -> float:
        """获取OBV信号强度"""
        current_ma = values.get('current_ma', 0)
        previous_ma = values.get('previous_ma', 0)
        
        if previous_ma == 0:
            return 0.0
        
        change_rate = abs(current_ma - previous_ma) / abs(previous_ma)
        return min(change_rate * 10, 1.0)


class MFIIndicator(HybridIndicator):
    """MFI指标实现 - 混合TA-Lib和向量化实现"""
    
    def __init__(self, period: int = 14, use_talib: bool = True):
        super().__init__("MFI", "volume", {"period": period}, use_talib)
    
    def _calculate_talib(self, df: pd.DataFrame, **kwargs) -> Optional[Dict[str, Any]]:
        """TA-Lib实现MFI"""
        if not TALIB_AVAILABLE:
            return None
        
        highs = df['最高价'].astype(float).values
        lows = df['最低价'].astype(float).values
        closes = df['收盘价'].astype(float).values
        volumes = df['成交量'].astype(float).values
        period = self.params['period']
        
        # 使用TA-Lib MFI
        mfi_values = talib.MFI(highs, lows, closes, volumes, timeperiod=period)
        mfi_values = np.where(np.isnan(mfi_values), 50, mfi_values)
        
        return {
            'values': mfi_values,
            'current': mfi_values[-1] if len(mfi_values) > 0 else 50,
            'previous': mfi_values[-2] if len(mfi_values) > 1 else 50,
            'algorithm': 'talib'
        }
    
    def _calculate_custom(self, df: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """优化的向量化MFI算法"""
        highs = df['最高价'].astype(float)
        lows = df['最低价'].astype(float)
        closes = df['收盘价'].astype(float)
        volumes = df['成交量'].astype(float)
        
        # 计算典型价格
        typical_price = (highs + lows + closes) / 3
        
        # 计算资金流量
        money_flow = typical_price * volumes
        
        # 修复：处理diff的首个NaN值
        price_diff = typical_price.diff().fillna(0)
        price_direction = np.sign(price_diff)
        
        # 分离正负资金流量
        positive_mf = money_flow.where(price_direction > 0, 0)
        negative_mf = money_flow.where(price_direction < 0, 0)
        
        # 使用滚动窗口计算
        period = self.params['period']
        pos_sum = positive_mf.rolling(window=period).sum()
        neg_sum = negative_mf.rolling(window=period).sum()
        
        # 修复：检查边界情况 - 所有资金流都是同向时
        if pos_sum.sum() == 0 and neg_sum.sum() == 0:
            # 没有任何资金流动，返回中性值
            mfi = pd.Series(50, index=df.index)
        elif neg_sum.sum() == 0:
            # 只有正向资金流，MFI接近100
            mfi = pd.Series(95, index=df.index)
        elif pos_sum.sum() == 0:
            # 只有负向资金流，MFI接近0
            mfi = pd.Series(5, index=df.index)
        else:
            # 正常计算MFI - 修复除零错误
            with np.errstate(divide='ignore', invalid='ignore'):
                # 使用更安全的除零处理，避免np.inf导致的问题
                money_ratio = np.where(
                    np.abs(neg_sum) < 1e-10,  # 接近零
                    100.0,  # 当负向流量为0时，比率设为很大的值
                    pos_sum / np.maximum(neg_sum, 1e-10)  # 确保分母不为零
                )
                mfi = 100 - (100 / (1 + money_ratio))
                mfi = np.nan_to_num(mfi, nan=50, posinf=100, neginf=0)  # 安全处理所有边界情况
        
        return {
            'values': mfi if isinstance(mfi, np.ndarray) else mfi.values,
            'current': mfi[-1] if isinstance(mfi, np.ndarray) else (mfi.iloc[-1] if len(mfi) > 0 else 50),
            'previous': mfi[-2] if isinstance(mfi, np.ndarray) else (mfi.iloc[-2] if len(mfi) > 1 else 50),
            'algorithm': 'custom'
        }
    
    def get_signal(self, values: Dict[str, Any]) -> str:
        """获取MFI信号"""
        current = values.get('current', 50)
        
        if current > 80:
            return 'sell'  # 超买
        elif current < 20:
            return 'buy'   # 超卖
        else:
            return 'neutral'
    
    def get_strength(self, values: Dict[str, Any]) -> float:
        """获取MFI信号强度"""
        current = values.get('current', 50)
        
        if current > 80:
            return min((current - 80) / 20, 1.0)
        elif current < 20:
            return min((20 - current) / 20, 1.0)
        else:
            return 0.0


class BollingerBandsIndicator(HybridIndicator):
    """布林带指标实现 - 混合TA-Lib和自实现"""
    
    def __init__(self, period: int = 20, std_dev: float = 2.0, use_talib: bool = True):
        super().__init__("Bollinger", "volatility", {"period": period, "std_dev": std_dev}, use_talib)
    
    def _calculate_talib(self, df: pd.DataFrame, **kwargs) -> Optional[Dict[str, Any]]:
        """TA-Lib实现布林带"""
        if not TALIB_AVAILABLE:
            return None
        
        closes = df['收盘价'].astype(float).values
        period = self.params['period']
        std_dev = self.params['std_dev']
        
        # 使用TA-Lib计算布林带
        upper, middle, lower = talib.BBANDS(closes, timeperiod=period, nbdevup=std_dev, nbdevdn=std_dev, matype=0)
        
        # 处理NaN值
        upper = np.where(np.isnan(upper), closes, upper)
        middle = np.where(np.isnan(middle), closes, middle)
        lower = np.where(np.isnan(lower), closes, lower)
        
        # 计算带宽和%B
        bandwidth = ((upper - lower) / middle) * 100
        percent_b = (closes - lower) / (upper - lower)
        
        # 处理除零和NaN
        bandwidth = np.where(np.isnan(bandwidth) | np.isinf(bandwidth), 0, bandwidth)
        percent_b = np.where(np.isnan(percent_b) | np.isinf(percent_b), 0.5, percent_b)
        
        return {
            'upper': upper,
            'middle': middle,
            'lower': lower,
            'bandwidth': bandwidth,
            'percent_b': percent_b,
            'current_price': closes[-1],
            'current_upper': upper[-1] if len(upper) > 0 else closes[-1],
            'current_middle': middle[-1] if len(middle) > 0 else closes[-1],
            'current_lower': lower[-1] if len(lower) > 0 else closes[-1],
            'current_percent_b': percent_b[-1] if len(percent_b) > 0 else 0.5,
            'algorithm': 'talib'
        }
    
    def _calculate_custom(self, df: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """自实现布林带算法"""
        closes = df['收盘价'].astype(float)
        
        # 计算中轨（SMA）
        middle = closes.rolling(window=self.params['period']).mean()
        
        # 计算标准差
        std = closes.rolling(window=self.params['period']).std()
        
        # 计算上下轨
        upper = middle + (std * self.params['std_dev'])
        lower = middle - (std * self.params['std_dev'])
        
        # 计算带宽
        bandwidth = ((upper - lower) / middle) * 100
        
        # 计算%B (价格在布林带中的位置)
        percent_b = (closes - lower) / (upper - lower)
        
        return {
            'upper': upper.values,
            'middle': middle.values,
            'lower': lower.values,
            'bandwidth': bandwidth.values,
            'percent_b': percent_b.values,
            'current_price': closes.iloc[-1],
            'current_upper': upper.iloc[-1] if len(upper) > 0 else closes.iloc[-1],
            'current_middle': middle.iloc[-1] if len(middle) > 0 else closes.iloc[-1],
            'current_lower': lower.iloc[-1] if len(lower) > 0 else closes.iloc[-1],
            'current_percent_b': percent_b.iloc[-1] if len(percent_b) > 0 else 0.5,
            'algorithm': 'custom'
        }
    
    def get_signal(self, values: Dict[str, Any]) -> str:
        """获取布林带信号"""
        percent_b = values.get('current_percent_b', 0.5)
        
        if percent_b > 1.0:  # 价格突破上轨
            return 'sell'
        elif percent_b < 0.0:  # 价格突破下轨
            return 'buy'
        elif percent_b > 0.8:  # 接近上轨
            return 'sell'
        elif percent_b < 0.2:  # 接近下轨
            return 'buy'
        else:
            return 'neutral'
    
    def get_strength(self, values: Dict[str, Any]) -> float:
        """获取布林带信号强度"""
        percent_b = values.get('current_percent_b', 0.5)
        
        if percent_b > 1.0:
            return min((percent_b - 1.0) * 2, 1.0)
        elif percent_b < 0.0:
            return min(abs(percent_b) * 2, 1.0)
        elif percent_b > 0.8:
            return (percent_b - 0.8) / 0.2 * 0.7
        elif percent_b < 0.2:
            return (0.2 - percent_b) / 0.2 * 0.7
        else:
            return 0.0


class ATRIndicator(HybridIndicator):
    """ATR指标实现 - 混合TA-Lib和自实现"""
    
    def __init__(self, period: int = 14, use_talib: bool = True):
        super().__init__("ATR", "volatility", {"period": period}, use_talib)
    
    def _calculate_talib(self, df: pd.DataFrame, **kwargs) -> Optional[Dict[str, Any]]:
        """TA-Lib实现ATR"""
        if not TALIB_AVAILABLE:
            return None
        
        highs = df['最高价'].astype(float).values
        lows = df['最低价'].astype(float).values
        closes = df['收盘价'].astype(float).values
        period = self.params['period']
        
        # 使用TA-Lib ATR函数
        atr_values = talib.ATR(highs, lows, closes, timeperiod=period)
        
        # 处理NaN值
        atr_values = np.where(np.isnan(atr_values), 0, atr_values)
        
        # 计算ATR百分比
        atr_percent = (atr_values / closes) * 100
        
        return {
            'values': atr_values,
            'percent_values': atr_percent,
            'current': atr_values[-1] if len(atr_values) > 0 else 0,
            'current_percent': atr_percent[-1] if len(atr_percent) > 0 else 0,
            'current_price': closes[-1],
            'algorithm': 'talib'
        }
    
    def _calculate_custom(self, df: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """自实现ATR算法 - 修复：使用EMA平滑"""
        highs = df['最高价'].astype(float)
        lows = df['最低价'].astype(float)
        closes = df['收盘价'].astype(float)
        
        # 计算True Range
        tr1 = highs - lows
        tr2 = np.abs(highs - closes.shift(1))
        tr3 = np.abs(lows - closes.shift(1))
        
        # 真实波幅是三者的最大值
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        
        # 修复：使用EMA平滑，与TA-Lib的ATR算法一致
        # Wilder's Smoothing: alpha = 1/period
        period = self.params['period']
        alpha = 1.0 / period
        atr = tr.ewm(alpha=alpha, adjust=False).mean()
        
        # 修复：计算ATR百分比，处理closes中的零值
        closes_safe = closes.replace(0, np.nan)  # 零价格替换为NaN
        atr_percent = (atr / closes_safe) * 100
        atr_percent = atr_percent.fillna(0)  # NaN填充为0
        
        return {
            'values': atr.values,
            'percent_values': atr_percent.values,
            'current': atr.iloc[-1] if len(atr) > 0 else 0,
            'current_percent': atr_percent.iloc[-1] if len(atr_percent) > 0 else 0,
            'current_price': closes.iloc[-1],
            'algorithm': 'custom_fixed'
        }
    
    def get_signal(self, values: Dict[str, Any]) -> str:
        """ATR本身不提供买卖信号，主要用于风险管理"""
        return 'neutral'
    
    def get_strength(self, values: Dict[str, Any]) -> float:
        """ATR强度表示市场波动性 - 修复：改善高波动区分度"""
        atr_percent = values.get('current_percent', 0)
        # 使用tanh函数，提供更好的高波动区分度
        return np.tanh(atr_percent / 5.0)  # tanh曲线，避免硬截断


class KeltnerChannelIndicator(BaseIndicator):
    """肯特纳通道指标实现"""
    
    def __init__(self, ema_period: int = 20, atr_period: int = 14, multiplier: float = 2.0):
        super().__init__("Keltner", "volatility", {
            "ema_period": ema_period,
            "atr_period": atr_period,
            "multiplier": multiplier
        })
    
    def calculate(self, df: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """计算肯特纳通道"""
        closes = df['收盘价'].astype(float)
        highs = df['最高价'].astype(float)
        lows = df['最低价'].astype(float)
        
        # 计算中线 (EMA)
        middle = closes.ewm(span=self.params['ema_period']).mean()
        
        # 计算ATR
        tr1 = highs - lows
        tr2 = np.abs(highs - closes.shift(1))
        tr3 = np.abs(lows - closes.shift(1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        atr = tr.rolling(window=self.params['atr_period']).mean()
        
        # 计算上下轨
        multiplier = self.params['multiplier']
        upper = middle + (atr * multiplier)
        lower = middle - (atr * multiplier)
        
        return {
            'upper': upper.values,
            'middle': middle.values,
            'lower': lower.values,
            'current_price': closes.iloc[-1],
            'current_upper': upper.iloc[-1] if len(upper) > 0 else closes.iloc[-1],
            'current_middle': middle.iloc[-1] if len(middle) > 0 else closes.iloc[-1],
            'current_lower': lower.iloc[-1] if len(lower) > 0 else closes.iloc[-1]
        }
    
    def get_signal(self, values: Dict[str, Any]) -> str:
        """获取肯特纳通道信号"""
        price = values.get('current_price', 0)
        upper = values.get('current_upper', 0)
        lower = values.get('current_lower', 0)
        
        if price > upper:
            return 'sell'  # 价格突破上轨
        elif price < lower:
            return 'buy'   # 价格突破下轨
        else:
            return 'neutral'
    
    def get_strength(self, values: Dict[str, Any]) -> float:
        """获取肯特纳通道信号强度"""
        price = values.get('current_price', 0)
        upper = values.get('current_upper', 0)
        lower = values.get('current_lower', 0)
        middle = values.get('current_middle', 0)
        
        if middle == 0:
            return 0.0
        
        if price > upper:
            return min((price - upper) / (upper - middle), 1.0)
        elif price < lower:
            return min((lower - price) / (middle - lower), 1.0)
        else:
            return 0.0


class StochasticIndicator(HybridIndicator):
    """随机指标实现 - 混合TA-Lib和自实现"""
    
    def __init__(self, k_period: int = 14, d_period: int = 3, use_talib: bool = True):
        super().__init__("Stochastic", "momentum", {"k_period": k_period, "d_period": d_period}, use_talib)
    
    def _calculate_talib(self, df: pd.DataFrame, **kwargs) -> Optional[Dict[str, Any]]:
        """TA-Lib实现随机指标"""
        if not TALIB_AVAILABLE:
            return None
        
        highs = df['最高价'].astype(float).values
        lows = df['最低价'].astype(float).values
        closes = df['收盘价'].astype(float).values
        
        k_period = self.params['k_period']
        d_period = self.params['d_period']
        
        # 检查数据充足性 - 降低阈值
        if len(closes) < 30:  # 只有数据极少时才使用简单方法
            # 数据极少时返回中性值
            k_percent = np.full(len(closes), 50.0)
            d_percent = np.full(len(closes), 50.0)
        else:
            # 使用TA-Lib STOCH函数
            k_percent, d_percent = talib.STOCH(highs, lows, closes,
                                             fastk_period=k_period,
                                             slowk_period=d_period,
                                             slowd_period=d_period)
            
            # 智能处理NaN值
            k_percent = pd.Series(k_percent).bfill().fillna(50).values
            d_percent = pd.Series(d_percent).bfill().fillna(50).values
        
        return {
            'k': k_percent,
            'd': d_percent,
            'current_k': k_percent[-1] if len(k_percent) > 0 else 50,
            'current_d': d_percent[-1] if len(d_percent) > 0 else 50,
            'previous_k': k_percent[-2] if len(k_percent) > 1 else 50,
            'previous_d': d_percent[-2] if len(d_percent) > 1 else 50,
            'algorithm': 'talib'
        }
    
    def _calculate_custom(self, df: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """自实现随机指标算法"""
        highs = df['最高价'].astype(float)
        lows = df['最低价'].astype(float)
        closes = df['收盘价'].astype(float)
        
        k_period = self.params['k_period']
        d_period = self.params['d_period']
        
        # 计算%K - 修复：避免除零
        lowest_low = lows.rolling(window=k_period).min()
        highest_high = highs.rolling(window=k_period).max()
        
        # 处理highest_high == lowest_low的情况
        range_diff = highest_high - lowest_low
        range_diff = range_diff.replace(0, np.nan)  # 零范围替换为NaN
        
        k_percent = 100 * (closes - lowest_low) / range_diff
        k_percent = k_percent.fillna(50)  # NaN填充为中性值50
        
        # 计算%D (对%K进行平滑)
        d_percent = k_percent.rolling(window=d_period).mean()
        
        # 修复：保持 Series 到最后，确保 fillna 正确应用
        k_percent_final = k_percent.fillna(50).astype(float)
        d_percent_final = d_percent.fillna(50).astype(float)
        
        return {
            'k': k_percent_final.values,
            'd': d_percent_final.values,
            'current_k': k_percent.iloc[-1] if len(k_percent) > 0 else 50,
            'current_d': d_percent.iloc[-1] if len(d_percent) > 0 else 50,
            'previous_k': k_percent.iloc[-2] if len(k_percent) > 1 else 50,
            'previous_d': d_percent.iloc[-2] if len(d_percent) > 1 else 50,
            'algorithm': 'custom'
        }
    
    def get_signal(self, values: Dict[str, Any]) -> str:
        """获取随机指标信号"""
        current_k = values.get('current_k', 50)
        current_d = values.get('current_d', 50)
        previous_k = values.get('previous_k', 50)
        previous_d = values.get('previous_d', 50)
        
        # 金叉死叉信号
        if current_k > current_d and previous_k <= previous_d:
            return 'buy'   # 金叉
        elif current_k < current_d and previous_k >= previous_d:
            return 'sell'  # 死叉
        # 超买超卖信号
        elif current_k > 80 and current_d > 80:
            return 'sell'  # 超买
        elif current_k < 20 and current_d < 20:
            return 'buy'   # 超卖
        else:
            return 'neutral'
    
    def get_strength(self, values: Dict[str, Any]) -> float:
        """获取随机指标信号强度"""
        current_k = values.get('current_k', 50)
        current_d = values.get('current_d', 50)
        
        # 计算K和D的差值作为强度参考
        kd_diff = abs(current_k - current_d)
        
        # 超买超卖区域的强度
        if current_k > 80 or current_k < 20:
            extreme_strength = min(abs(current_k - 50) - 30, 20) / 20
            return min(extreme_strength + kd_diff / 100, 1.0)
        else:
            return min(kd_diff / 50, 1.0)


class ROCIndicator(HybridIndicator):
    """变化率指标实现 - 混合TA-Lib和自实现"""
    
    def __init__(self, period: int = 12, use_talib: bool = True):
        super().__init__("ROC", "momentum", {"period": period}, use_talib)
    
    def _calculate_talib(self, df: pd.DataFrame, **kwargs) -> Optional[Dict[str, Any]]:
        """TA-Lib实现ROC"""
        if not TALIB_AVAILABLE:
            return None
        
        closes = df['收盘价'].astype(float).values
        period = self.params['period']
        
        # 使用TA-Lib ROC函数
        roc_values = talib.ROC(closes, timeperiod=period)
        
        # 处理NaN值
        roc_values = np.where(np.isnan(roc_values), 0, roc_values)
        
        # 计算ROC的移动平均
        roc_ma = talib.SMA(roc_values, timeperiod=10)
        roc_ma = np.where(np.isnan(roc_ma), 0, roc_ma)
        
        return {
            'values': roc_values,
            'ma_values': roc_ma,
            'current': roc_values[-1] if len(roc_values) > 0 else 0,
            'current_ma': roc_ma[-1] if len(roc_ma) > 0 else 0,
            'previous': roc_values[-2] if len(roc_values) > 1 else 0,
            'algorithm': 'talib'
        }
    
    def _calculate_custom(self, df: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """自实现ROC算法"""
        closes = df['收盘价'].astype(float)
        period = self.params['period']
        
        # 计算变化率
        roc = ((closes - closes.shift(period)) / closes.shift(period)) * 100
        
        # 计算ROC的移动平均
        roc_ma = roc.rolling(window=10).mean()
        
        return {
            'values': roc.values,
            'ma_values': roc_ma.values,
            'current': roc.iloc[-1] if len(roc) > 0 else 0,
            'current_ma': roc_ma.iloc[-1] if len(roc_ma) > 0 else 0,
            'previous': roc.iloc[-2] if len(roc) > 1 else 0,
            'algorithm': 'custom'
        }
    
    def get_signal(self, values: Dict[str, Any]) -> str:
        """获取ROC信号"""
        current = values.get('current', 0)
        current_ma = values.get('current_ma', 0)
        
        if current > 5 and current_ma > 0:
            return 'buy'   # 强劲上涨
        elif current < -5 and current_ma < 0:
            return 'sell'  # 强劲下跌
        elif current > current_ma and current_ma > 0:
            return 'buy'   # 上升趋势
        elif current < current_ma and current_ma < 0:
            return 'sell'  # 下降趋势
        else:
            return 'neutral'
    
    def get_strength(self, values: Dict[str, Any]) -> float:
        """获取ROC信号强度"""
        current = values.get('current', 0)
        return min(abs(current) / 10, 1.0)  # 10%变化率对应强度1.0


class SMAIndicator(HybridIndicator):
    """简单移动平均指标实现 - 混合TA-Lib和自实现"""
    
    def __init__(self, periods: List[int] = [20, 50, 200], use_talib: bool = True):
        super().__init__("SMA", "trend", {"periods": periods}, use_talib)
    
    def _calculate_talib(self, df: pd.DataFrame, **kwargs) -> Optional[Dict[str, Any]]:
        """TA-Lib实现SMA"""
        if not TALIB_AVAILABLE:
            return None
        
        closes = df['收盘价'].astype(float).values
        result = {}
        
        for period in self.params['periods']:
            sma_values = talib.SMA(closes, timeperiod=period)
            sma_values = np.where(np.isnan(sma_values), closes, sma_values)
            
            result[f'sma_{period}'] = sma_values
            result[f'current_sma_{period}'] = sma_values[-1] if len(sma_values) > 0 else closes[-1]
        
        result['current_price'] = closes[-1]
        result['algorithm'] = 'talib'
        return result
    
    def _calculate_custom(self, df: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """自实现SMA算法"""
        closes = df['收盘价'].astype(float)
        result = {}
        
        for period in self.params['periods']:
            sma = closes.rolling(window=period).mean()
            result[f'sma_{period}'] = sma.values
            result[f'current_sma_{period}'] = sma.iloc[-1] if len(sma) > 0 else closes.iloc[-1]
        
        result['current_price'] = closes.iloc[-1]
        result['algorithm'] = 'custom'
        return result
    
    def get_signal(self, values: Dict[str, Any]) -> str:
        """获取SMA信号"""
        # 使用20, 50, 200周期的SMA排列判断趋势
        sma_20 = values.get('current_sma_20', 0)
        sma_50 = values.get('current_sma_50', 0)
        sma_200 = values.get('current_sma_200', 0)
        current_price = values.get('current_price', 0)
        
        # 多头排列
        if current_price > sma_20 > sma_50 > sma_200:
            return 'buy'
        # 空头排列
        elif current_price < sma_20 < sma_50 < sma_200:
            return 'sell'
        # 价格在短期均线之上
        elif current_price > sma_20 and sma_20 > sma_50:
            return 'buy'
        # 价格在短期均线之下
        elif current_price < sma_20 and sma_20 < sma_50:
            return 'sell'
        else:
            return 'neutral'
    
    def get_strength(self, values: Dict[str, Any]) -> float:
        """获取SMA信号强度"""
        current_price = values.get('current_price', 0)
        sma_20 = values.get('current_sma_20', 0)
        
        if sma_20 == 0:
            return 0.0
        
        # 计算价格与短期均线的距离作为强度指标
        strength = abs(current_price - sma_20) / sma_20
        return min(strength * 20, 1.0)


class IchimokuIndicator(BaseIndicator):
    """一目均衡图指标实现"""
    
    def __init__(self, tenkan: int = 9, kijun: int = 26, senkou: int = 52):
        super().__init__("Ichimoku", "trend", {
            "tenkan": tenkan,
            "kijun": kijun, 
            "senkou": senkou
        })
    
    def calculate(self, df: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """计算一目均衡图"""
        highs = df['最高价'].astype(float)
        lows = df['最低价'].astype(float)
        closes = df['收盘价'].astype(float)
        
        # 转换线 (Tenkan-sen)
        tenkan_high = highs.rolling(window=self.params['tenkan']).max()
        tenkan_low = lows.rolling(window=self.params['tenkan']).min()
        tenkan = (tenkan_high + tenkan_low) / 2
        
        # 基准线 (Kijun-sen)
        kijun_high = highs.rolling(window=self.params['kijun']).max()
        kijun_low = lows.rolling(window=self.params['kijun']).min()
        kijun = (kijun_high + kijun_low) / 2
        
        # 先行带A (Senkou Span A) - 向未来偏移
        senkou_a = ((tenkan + kijun) / 2).shift(self.params['kijun'])
        
        # 先行带B (Senkou Span B) - 向未来偏移
        senkou_high = highs.rolling(window=self.params['senkou']).max()
        senkou_low = lows.rolling(window=self.params['senkou']).min()
        senkou_b = ((senkou_high + senkou_low) / 2).shift(self.params['kijun'])
        
        # 迟行线 (Chikou Span) - 向过去偏移
        chikou = closes.shift(-self.params['kijun'])
        
        return {
            'tenkan': tenkan.values,
            'kijun': kijun.values,
            'senkou_a': senkou_a.values,
            'senkou_b': senkou_b.values,
            'chikou': chikou.values,
            'current_price': closes.iloc[-1],
            'current_tenkan': tenkan.iloc[-1] if len(tenkan) > 0 else closes.iloc[-1],
            'current_kijun': kijun.iloc[-1] if len(kijun) > 0 else closes.iloc[-1],
            'current_senkou_a': senkou_a.iloc[-1] if len(senkou_a) > 0 else closes.iloc[-1],
            'current_senkou_b': senkou_b.iloc[-1] if len(senkou_b) > 0 else closes.iloc[-1]
        }
    
    def get_signal(self, values: Dict[str, Any]) -> str:
        """获取一目均衡图信号"""
        price = values.get('current_price', 0)
        tenkan = values.get('current_tenkan', 0)
        kijun = values.get('current_kijun', 0)
        senkou_a = values.get('current_senkou_a', 0)
        senkou_b = values.get('current_senkou_b', 0)
        
        # 云层分析
        cloud_top = max(senkou_a, senkou_b)
        cloud_bottom = min(senkou_a, senkou_b)
        
        # 多重条件判断
        bullish_conditions = 0
        bearish_conditions = 0
        
        # 条件1: 转换线与基准线关系
        if tenkan > kijun:
            bullish_conditions += 1
        elif tenkan < kijun:
            bearish_conditions += 1
        
        # 条件2: 价格与云层关系
        if price > cloud_top:
            bullish_conditions += 2  # 权重更高
        elif price < cloud_bottom:
            bearish_conditions += 2
        
        # 条件3: 价格与转换线关系
        if price > tenkan:
            bullish_conditions += 1
        elif price < tenkan:
            bearish_conditions += 1
        
        # 条件4: 云层颜色 (先行带A vs 先行带B)
        if senkou_a > senkou_b:
            bullish_conditions += 1
        elif senkou_a < senkou_b:
            bearish_conditions += 1
        
        # 决策逻辑
        if bullish_conditions >= 3:
            return 'buy'
        elif bearish_conditions >= 3:
            return 'sell'
        else:
            return 'neutral'
    
    def get_strength(self, values: Dict[str, Any]) -> float:
        """获取信号强度"""
        price = values.get('current_price', 0)
        tenkan = values.get('current_tenkan', 0)
        kijun = values.get('current_kijun', 0)
        senkou_a = values.get('current_senkou_a', 0)
        senkou_b = values.get('current_senkou_b', 0)
        
        if kijun == 0:
            return 0.0
        
        # 计算价格与关键线的距离
        price_kijun_dist = abs(price - kijun) / kijun
        
        # 计算云层厚度（反映波动性）
        cloud_thickness = abs(senkou_a - senkou_b) / max(senkou_a, senkou_b) if max(senkou_a, senkou_b) > 0 else 0
        
        # 计算综合强度
        distance_strength = min(price_kijun_dist * 10, 0.7)
        cloud_strength = min(cloud_thickness * 20, 0.3)
        
        return distance_strength + cloud_strength


class DonchianChannelIndicator(BaseIndicator):
    """唐奇安通道指标实现"""
    
    def __init__(self, period: int = 20):
        super().__init__("DonchianChannel", "volatility", {"period": period})
    
    def calculate(self, df: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """计算唐奇安通道"""
        highs = df['最高价'].astype(float)
        lows = df['最低价'].astype(float)
        closes = df['收盘价'].astype(float)
        
        # 计算上轨（N期最高价）
        upper = highs.rolling(window=self.params['period']).max()
        
        # 计算下轨（N期最低价）
        lower = lows.rolling(window=self.params['period']).min()
        
        # 计算中轨
        middle = (upper + lower) / 2
        
        # 计算通道宽度（波动性指标）
        channel_width = ((upper - lower) / middle) * 100
        
        return {
            'upper': upper.values,
            'middle': middle.values,
            'lower': lower.values,
            'width': channel_width.values,
            'current_price': closes.iloc[-1],
            'current_upper': upper.iloc[-1] if len(upper) > 0 else closes.iloc[-1],
            'current_middle': middle.iloc[-1] if len(middle) > 0 else closes.iloc[-1],
            'current_lower': lower.iloc[-1] if len(lower) > 0 else closes.iloc[-1],
            'current_width': channel_width.iloc[-1] if len(channel_width) > 0 else 0
        }
    
    def get_signal(self, values: Dict[str, Any]) -> str:
        """获取唐奇安通道信号"""
        price = values.get('current_price', 0)
        upper = values.get('current_upper', 0)
        lower = values.get('current_lower', 0)
        middle = values.get('current_middle', 0)
        
        # 突破策略 - 适用于趋势跟踪
        if price >= upper:
            return 'buy'  # 突破上轨，趋势向上
        elif price <= lower:
            return 'sell'  # 突破下轨，趋势向下
        # 均值回归策略 - 在通道内部
        elif price > middle * 1.02:  # 接近上轨但未突破
            return 'sell'  # 期待回归中位
        elif price < middle * 0.98:  # 接近下轨但未突破
            return 'buy'   # 期待回归中位
        else:
            return 'neutral'
    
    def get_strength(self, values: Dict[str, Any]) -> float:
        """获取信号强度"""
        price = values.get('current_price', 0)
        upper = values.get('current_upper', 0)
        lower = values.get('current_lower', 0)
        middle = values.get('current_middle', 0)
        width = values.get('current_width', 0)
        
        channel_range = upper - lower
        if channel_range == 0:
            return 0.0
        
        # 计算价格在通道中的相对位置
        if price >= upper or price <= lower:
            # 突破信号 - 强度与通道宽度相关
            breakthrough_strength = min(width / 10, 0.8)  # 宽通道突破更有意义
            distance_strength = 0.5  # 突破固定强度
            return breakthrough_strength + distance_strength
        else:
            # 通道内信号 - 强度与距离边界的远近相关
            distance_to_edge = min(abs(price - upper), abs(price - lower))
            relative_distance = distance_to_edge / (channel_range / 2) if channel_range > 0 else 0
            return max(0.0, 0.7 - relative_distance)  # 修复：确保非负数


# 时间框架层级定义
TIMEFRAME_HIERARCHY = {
    'primary': ['1h', '4h', '1d'],      # 主要分析框架
    'secondary': ['30m', '2h'],         # 次要确认框架
    'reference': ['15m', '1w']          # 参考框架
}

# 每个时间框架计算的指标 - 修复：清理无效指标名
TIMEFRAME_INDICATORS = {
    '15m': ['RSI', 'OBV'],              # 只计算快速指标
    '30m': ['RSI', 'MACD', 'OBV'], 
    '1h': ['ALL'],                      # 计算所有指标
    '2h': ['DynamicKDJ', 'RSI', 'MACD', 'ADX'],
    '4h': ['ALL'],                      # 计算所有指标
    '1d': ['ALL'],                      # 计算所有指标
    '1w': ['EMA', 'SMA', 'ATR']         # 只计算长期指标
}

# 指标分类结构
INDICATOR_CATEGORIES = {
    'momentum': ['RSI', 'MACD', 'Stochastic', 'ROC', 'KDJ'],
    'trend': ['EMA', 'SMA', 'ADX', 'Ichimoku'],
    'volatility': ['Bollinger', 'ATR', 'Keltner', 'DonchianChannel'],
    'volume': ['OBV', 'VolumeProfile', 'MFI', 'VWAP'],
    'custom': ['DynamicKDJ', 'DivergenceDetector']
}


class IndicatorManager:
    """管理所有指标的计算和缓存"""
    
    def __init__(self, use_talib: bool = True):
        self.indicators = {}
        self.smart_cache = SmartCache(max_size=1000, ttl=3600)  # 1小时TTL
        self.performance_monitor = PerformanceMonitor()
        self.use_talib = use_talib
        self.performance_stats = {}
        self._register_default_indicators()
    
    def _register_default_indicators(self):
        """注册默认指标"""
        # 动量指标 (TA-Lib优化)
        self.register_indicator(RSIIndicator(use_talib=self.use_talib))
        self.register_indicator(MACDIndicator(use_talib=self.use_talib))
        self.register_indicator(StochasticIndicator(use_talib=self.use_talib))  # 现已支持TA-Lib
        self.register_indicator(ROCIndicator(use_talib=self.use_talib))  # 现已支持TA-Lib
        
        # 趋势指标 (TA-Lib优化)
        self.register_indicator(EMAIndicator(use_talib=self.use_talib))
        self.register_indicator(SMAIndicator(use_talib=self.use_talib))
        self.register_indicator(IchimokuIndicator())  # 保持自实现
        
        # 波动性指标 (TA-Lib优化)
        self.register_indicator(BollingerBandsIndicator(use_talib=self.use_talib))
        self.register_indicator(ATRIndicator(use_talib=self.use_talib))  # 现已支持TA-Lib
        self.register_indicator(KeltnerChannelIndicator())  # 保持自实现
        self.register_indicator(DonchianChannelIndicator())  # 保持自实现
        
        # 成交量指标 (TA-Lib优化)
        self.register_indicator(VWAPIndicator())  # 保持自实现
        self.register_indicator(OBVIndicator(use_talib=self.use_talib))  # 现已支持TA-Lib
        self.register_indicator(MFIIndicator(use_talib=self.use_talib))  # 现已支持TA-Lib
    
    def register_indicator(self, indicator: BaseIndicator):
        """注册指标"""
        self.indicators[indicator.name] = indicator
    
    def calculate_indicator(self, indicator_name: str, df: pd.DataFrame, 
                           timeframe: str, symbol: str = None, config = None) -> Optional[Dict[str, Any]]:
        """计算单个指标"""
        logger.debug(f"开始计算指标: {indicator_name} for {symbol} on {timeframe}")
        
        if indicator_name not in self.indicators:
            logger.warning(f"指标 {indicator_name} 未注册")
            warnings.warn(f"指标 {indicator_name} 未注册")
            return None
        
        # 修复：使用MD5哈希改进缓存键设计，避免冲突
        import hashlib
        
        if '开盘时间' in df.columns and not df.empty:
            # 使用时间戳范围和数据特征
            first_timestamp = df['开盘时间'].iloc[0]
            last_timestamp = df['开盘时间'].iloc[-1] 
            first_price = df['收盘价'].iloc[0] if '收盘价' in df.columns else 0
            last_price = df['收盘价'].iloc[-1] if '收盘价' in df.columns else 0
            
            # 创建更稳定的数据签名
            data_signature = f"{df.shape}_{df.columns.tolist()}_{first_timestamp}_{last_timestamp}_{first_price:.2f}_{last_price:.2f}"
            df_signature = hashlib.md5(data_signature.encode()).hexdigest()[:16]
            
            # 修复缓存键冲突 - 添加参数哈希，安全处理config序列化
            params_hash = ""
            if config:
                import json
                try:
                    if hasattr(config, '__dict__'):
                        # 对象类型：提取非私有属性
                        config_dict = {k: str(v) for k, v in config.__dict__.items() if not k.startswith('_')}
                        params_str = json.dumps(config_dict, sort_keys=True)
                    else:
                        # 其他类型：转为字符串
                        params_str = str(config)
                    params_hash = hashlib.md5(params_str.encode()).hexdigest()[:8]
                except Exception:
                    # 序列化失败时使用对象ID作为备用
                    params_hash = str(id(config))[:8]
            
            cache_key = f"{indicator_name}_{timeframe}_{df_signature}_{symbol}_{params_hash}"
        else:
            # 备用方案：使用DataFrame的基本特征
            try:
                # 更安全的哈希生成
                shape_str = f"{df.shape[0]}x{df.shape[1]}"
                cols_str = "_".join(df.columns.tolist())
                if not df.empty and '收盘价' in df.columns:
                    price_summary = f"{df['收盘价'].iloc[0]:.2f}_{df['收盘价'].iloc[-1]:.2f}"
                else:
                    price_summary = "empty"
                
                data_signature = f"{shape_str}_{cols_str}_{price_summary}"
                df_signature = hashlib.md5(data_signature.encode()).hexdigest()[:16]
                
                # 修复缓存键冲突 - 添加参数哈希，安全处理config序列化
                params_hash = ""
                if config:
                    import json
                    try:
                        if hasattr(config, '__dict__'):
                            # 对象类型：提取非私有属性
                            config_dict = {k: str(v) for k, v in config.__dict__.items() if not k.startswith('_')}
                            params_str = json.dumps(config_dict, sort_keys=True)
                        else:
                            # 其他类型：转为字符串
                            params_str = str(config)
                        params_hash = hashlib.md5(params_str.encode()).hexdigest()[:8]
                    except Exception:
                        # 序列化失败时使用对象ID作为备用
                        params_hash = str(id(config))[:8]
                
                cache_key = f"{indicator_name}_{timeframe}_{df_signature}_{symbol}_{params_hash}"
            except:
                # 极简备用方案 - 安全处理config
                params_hash = ""
                if config:
                    try:
                        params_hash = hashlib.md5(str(config).encode()).hexdigest()[:8]
                    except Exception:
                        params_hash = str(id(config))[:8]
                cache_key = f"{indicator_name}_{timeframe}_{len(df)}_{hash(str(df.columns.tolist()))}_{symbol}_{params_hash}"
        
        # 检查智能缓存
        cached_result = self.smart_cache.get(cache_key)
        if cached_result is not None:
            return cached_result
        
        try:
            indicator = self.indicators[indicator_name]
            
            # 使用性能监控
            with self.performance_monitor.monitor(f"calculate_{indicator_name}"):
                # 数据验证
                if len(df) < 2:
                    raise DataError(f"数据量不足: {indicator_name} 需要至少2条数据")
                
                # 传递完整参数给指标
                kwargs = {
                    'timeframe': timeframe, 
                    'symbol': symbol or 'UNKNOWN',
                    'config': config
                }
                result = indicator.calculate(df, **kwargs)
                
                # 结果验证 - 智能验证逻辑
                if not result:
                    raise CalculationError(f"指标 {indicator_name} 返回空结果")
                
                # 检查结果是否有有效的数值 - 更智能的验证
                def has_valid_numeric_data(result_dict):
                    """检查结果是否包含有效的数值数据"""
                    # 检查关键数值字段
                    key_fields = ['current', 'values', 'current_price', 'current_ma', 'current_macd', 'current_upper', 'current_lower']
                    
                    for field in key_fields:
                        if field in result_dict:
                            value = result_dict[field]
                            # 检查是否为有效数值 - 修复：0是有效值
                            if isinstance(value, (int, float)) and not np.isnan(value) and not np.isinf(value):
                                return True
                            elif isinstance(value, np.ndarray) and len(value) > 0:
                                # 检查数组中是否有非NaN/inf值
                                valid_values = ~(np.isnan(value) | np.isinf(value))
                                if np.any(valid_values):
                                    return True
                    
                    # 如果没有找到有效数值，但有algorithm字段，说明计算已执行
                    if 'algorithm' in result_dict:
                        return True  # 允许通过，使用默认值
                    
                    return False
                
                if not has_valid_numeric_data(result):
                    # 静默处理数据不足情况，添加警告标记但不输出消息
                    result['warning'] = 'insufficient_data'
                
                # 安全地获取信号、强度和置信度 - 静默处理错误
                try:
                    result['signal'] = indicator.get_signal(result)
                except Exception:
                    result['signal'] = 'neutral'
                
                try:
                    result['strength'] = indicator.get_strength(result)
                except Exception:
                    result['strength'] = 0.0
                
                try:
                    result['confidence'] = indicator.get_confidence(result)
                except Exception:
                    result['confidence'] = 0.5
                
                # 缓存结果
                self.smart_cache.set(cache_key, result)
                return result
                
        except (DataError, CalculationError):
            # 静默处理已知错误
            return self._get_default_result(indicator_name)
        except Exception:
            # 静默处理未知错误
            return self._get_default_result(indicator_name)
    
    def _get_default_result(self, indicator_name: str) -> Dict[str, Any]:
        """获取默认结果以处理错误情况"""
        return {
            'values': np.array([]),
            'current': 0,
            'signal': 'neutral',
            'strength': 0.0,
            'confidence': 0.0,
            'error': True,
            'indicator': indicator_name
        }
    
    def calculate_all(self, df: pd.DataFrame, timeframe: str, symbol: str = None, config = None) -> Dict[str, Any]:
        """计算所有适用的指标"""
        results = {}
        
        # 确定该时间框架需要计算的指标
        required_indicators = TIMEFRAME_INDICATORS.get(timeframe, [])
        if 'ALL' in required_indicators:
            required_indicators = list(self.indicators.keys())
        
        # 根据指标复杂度设置不同的超时时间
        timeouts = {
            'DynamicKDJ': 60,  # KDJ计算复杂，需要更长时间
            'ADX': 45,
            'RSI': 30,
            'MACD': 30,
            'EMA': 30
        }
        
        # 并行计算指标 - 修复：改进超时处理机制
        from concurrent.futures import wait, TimeoutError as FutureTimeoutError
        
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {}
            for indicator_name in required_indicators:
                if indicator_name in self.indicators:
                    future = executor.submit(self.calculate_indicator, indicator_name, df, timeframe, symbol, config)
                    futures[indicator_name] = future
            
            # 逐个处理结果，每个指标最大30秒超时
            for indicator_name, future in futures.items():
                try:
                    # 使用 result(timeout) 进行显式超时控制
                    result = future.result(timeout=30)
                    if result:
                        results[indicator_name] = result
                    else:
                        # 修复：失败的指标填充占位，避免权重丢失
                        results[indicator_name] = self._get_default_result(indicator_name)
                except FutureTimeoutError:
                    # 超时时尝试取消并填充占位
                    future.cancel()
                    results[indicator_name] = self._get_default_result(indicator_name)
                except Exception:
                    # 其他错误也填充占位
                    results[indicator_name] = self._get_default_result(indicator_name)
        
        return results
    
    def get_indicator_by_category(self, category: str) -> List[str]:
        """根据类别获取指标列表"""
        return INDICATOR_CATEGORIES.get(category, [])
    
    def clear_cache(self):
        """清空缓存"""
        self.smart_cache.clear()
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """获取性能统计摘要"""
        summary = {
            'total_indicators': len(self.indicators),
            'talib_enabled_indicators': 0,
            'custom_only_indicators': 0,
            'hybrid_indicators': 0,
            'performance_details': {}
        }
        
        for name, indicator in self.indicators.items():
            if isinstance(indicator, HybridIndicator):
                summary['hybrid_indicators'] += 1
                if indicator.use_talib:
                    summary['talib_enabled_indicators'] += 1
                stats = indicator.get_performance_stats()
                summary['performance_details'][name] = stats
            else:
                summary['custom_only_indicators'] += 1
                summary['performance_details'][name] = {'usage': 'custom_only', 'performance_ratio': 'N/A'}
        
        return summary
    
    def preheat_cache(self, symbols: List[str], data_dict: Dict[str, Dict[str, pd.DataFrame]]):
        """预热缓存 - 在系统启动时计算常用指标"""
        print("🔥 预热缓存中...")
        
        critical_indicators = ['RSI', 'MACD', 'EMA', 'Bollinger', 'OBV', 'MFI']  # 关键指标
        critical_timeframes = ['1h', '4h', '1d']  # 关键时间框架
        
        with ThreadPoolExecutor(max_workers=6) as executor:
            futures = []
            
            for symbol in symbols:
                if symbol not in data_dict:
                    continue
                    
                for timeframe in critical_timeframes:
                    if timeframe not in data_dict[symbol]:
                        continue
                    
                    df = data_dict[symbol][timeframe]
                    
                    for indicator in critical_indicators:
                        if indicator in self.indicators:
                            future = executor.submit(
                                self.calculate_indicator, 
                                indicator, df, timeframe, symbol
                            )
                            futures.append((symbol, timeframe, indicator, future))
            
            # 等待所有计算完成
            completed = 0
            total = len(futures)
            
            for symbol, timeframe, indicator, future in futures:
                try:
                    future.result(timeout=10)
                    completed += 1
                except Exception as e:
                    print(f"⚠️ 预热失败: {symbol} {timeframe} {indicator} - {str(e)}")
        
        print(f"✅ 缓存预热完成: {completed}/{total} 成功")
    
    def health_check(self) -> Dict[str, Any]:
        """系统健康检查"""
        logger.info("开始系统健康检查...")
        
        health_status = {
            'indicators': {},
            'cache': {},
            'performance': {},
            'overall': 'healthy'
        }
        
        # 检查每个指标
        for name, indicator in self.indicators.items():
            try:
                # 创建测试数据
                test_df = self._create_test_dataframe()
                result = indicator.calculate(test_df)
                
                health_status['indicators'][name] = {
                    'status': 'ok' if result and 'error' not in result else 'error',
                    'type': 'hybrid' if isinstance(indicator, HybridIndicator) else 'basic'
                }
                logger.debug(f"指标 {name} 健康检查通过")
            except Exception as e:
                health_status['indicators'][name] = {
                    'status': 'error',
                    'error': str(e)
                }
                health_status['overall'] = 'degraded'
                logger.warning(f"指标 {name} 健康检查失败: {str(e)}")
        
        # 检查缓存
        cache_stats = self.smart_cache.get_stats()
        health_status['cache'] = {
            'usage': f"{cache_stats['size']}/{cache_stats['max_size']}",
            'usage_percent': cache_stats['size'] / cache_stats['max_size'] * 100
        }
        
        # 检查性能
        perf_report = self.performance_monitor.get_report()
        if perf_report.get('status') != 'no_data':
            health_status['performance'] = {
                'avg_time_ms': perf_report['avg_calculation_time'] * 1000,
                'error_rate': perf_report['error_rate'],
                'success_rate': perf_report['success_rate']
            }
        
        logger.info(f"健康检查完成，整体状态: {health_status['overall']}")
        return health_status
    
    def _create_test_dataframe(self) -> pd.DataFrame:
        """创建测试用数据框"""
        np.random.seed(42)  # 确保测试的一致性
        dates = pd.date_range(end=pd.Timestamp.now(), periods=100, freq='1h')
        base_price = 50000
        price_changes = np.random.randn(100) * 100
        prices = base_price + np.cumsum(price_changes)
        
        return pd.DataFrame({
            '开盘时间': dates,
            '开盘价': prices * 0.999,
            '最高价': prices * 1.002,
            '最低价': prices * 0.998,
            '收盘价': prices,
            '成交量': np.random.randint(1000, 10000, 100).astype(float)
        })
    
    def print_performance_report(self):
        """打印性能报告"""
        summary = self.get_performance_summary()
        
        print("\n" + "="*60)
        print("📊 指标性能统计报告")
        print("="*60)
        print(f"总指标数量: {summary['total_indicators']}")
        print(f"TA-Lib启用指标: {summary['talib_enabled_indicators']}")
        print(f"混合指标数量: {summary['hybrid_indicators']}")
        print(f"自实现指标: {summary['custom_only_indicators']}")
        
        if summary['performance_details']:
            print(f"\n📈 各指标使用情况:")
            for name, stats in summary['performance_details'].items():
                usage = stats.get('usage', 'unknown')
                ratio = stats.get('performance_ratio', None)
                talib_time = stats.get('avg_talib_time', 0)
                custom_time = stats.get('avg_custom_time', 0)
                
                # 修复：处理None比率
                if ratio is None:
                    perf_info = "(未测试)"
                elif isinstance(ratio, (int, float)) and ratio > 1:
                    perf_info = f"(自实现慢{ratio:.1f}倍)"
                elif usage == 'custom_only':
                    perf_info = "(仅自实现)"
                else:
                    perf_info = f"(加速{1/ratio:.1f}倍)" if ratio > 0 else ""
                
                print(f"  {name:>15}: {usage:>20} {perf_info}")
        
        print("="*60)


class SignalScorer:
    """统一的信号评分系统"""
    
    def __init__(self):
        # 各指标类别的基础权重
        self.category_weights = {
            'momentum': 0.3,
            'trend': 0.4,
            'volatility': 0.15,
            'volume': 0.1,
            'custom': 0.05
        }
        
        # 时间框架权重
        self.timeframe_weights = {
            '15m': 0.05,
            '30m': 0.1,
            '1h': 0.25,
            '2h': 0.15,
            '4h': 0.35,
            '1d': 0.4,
            '1w': 0.2
        }
    
    def score_single_indicator(self, indicator_result: Dict[str, Any], indicator_name: str) -> float:
        """为单个指标评分"""
        if not indicator_result:
            return 0.0
        
        signal = indicator_result.get('signal', 'neutral')
        strength = indicator_result.get('strength', 0.0)
        confidence = indicator_result.get('confidence', 0.5)
        
        # 基础分数
        if signal == 'buy':
            base_score = strength * confidence
        elif signal == 'sell':
            base_score = -strength * confidence
        elif signal in ['trending', 'sideways', 'transition']:
            # 修复：处理ADX等状态指标的特殊信号
            market_regime = signal
            if market_regime == 'trending':
                # 趋势市场给予小幅正分，有利于趋势策略
                base_score = strength * confidence * 0.3
            elif market_regime == 'sideways':
                # 震荡市场给予小幅负分，不利于趋势策略
                base_score = -strength * confidence * 0.2
            else:  # transition
                # 过渡期保持中性
                base_score = 0.0
        else:
            base_score = 0.0
        
        return base_score
    
    def combine_scores(self, indicator_scores: Dict[str, float], indicator_manager: IndicatorManager, 
                      indicator_results: Dict[str, Any] = None) -> float:
        """合并指标分数，包含协同分析 - 修复：分离方向信号和状态信号"""
        directional_score = 0.0
        directional_weight = 0.0
        regime_adjustments = {}
        
        # 分离处理方向性指标和状态指标
        for indicator_name, score in indicator_scores.items():
            # 获取指标类别
            category = None
            if hasattr(indicator_manager, 'indicators') and indicator_name in indicator_manager.indicators:
                indicator_obj = indicator_manager.indicators[indicator_name]
                category = getattr(indicator_obj, 'category', None)
            
            if not category:
                for cat, indicators in INDICATOR_CATEGORIES.items():
                    if indicator_name in indicators:
                        category = cat
                        break
            
            # 移除ADX的特殊处理，让其使用标准评分流程
            # ADX现在返回标准的buy/sell/neutral信号，无需特殊处理
            
            # 方向性指标正常计算
            if category:
                weight = self.category_weights.get(category, 0.1)
                directional_score += score * weight
                directional_weight += weight
        
        base_score = directional_score / directional_weight if directional_weight > 0 else 0.0
        
        # 应用市场状态调整 - 现在从ADX结果中获取市场状态
        regime_multiplier = 1.0
        if indicator_results and 'ADX' in indicator_results:
            adx_result = indicator_results['ADX']
            regime = adx_result.get('market_regime', 'transition')
            adx_strength = adx_result.get('strength', 0)
            
            if regime == 'trending':
                regime_multiplier *= (1.0 + adx_strength * 0.2)  # 趋势市场增强
            elif regime == 'sideways':
                regime_multiplier *= (1.0 - adx_strength * 0.3)  # 震荡市场削弱
        
        # 保留旧的regime_adjustments处理（如果有其他状态指标）
        for adj in regime_adjustments.values():
            if adj['regime'] == 'trending':
                regime_multiplier *= (1.0 + adj['strength'] * 0.2)
            elif adj['regime'] == 'sideways':
                regime_multiplier *= (1.0 - adj['strength'] * 0.3)
        
        adjusted_score = base_score * regime_multiplier
        
        # 指标协同分析增强
        if indicator_results:
            synergy_bonus = self._calculate_synergy_bonus(indicator_results)
            # 修复：避免负向信号被过度放大
            if adjusted_score >= 0:
                enhanced_score = adjusted_score * (1 + synergy_bonus)
            else:
                enhanced_score = adjusted_score * (1 + synergy_bonus * 0.5)  # 负向减少放大
            
            return max(-1.0, min(1.0, enhanced_score))
        
        return max(-1.0, min(1.0, adjusted_score))
    
    def _calculate_synergy_bonus(self, indicator_results: Dict[str, Any]) -> float:
        """计算指标协同增强分数"""
        synergy_bonus = 0.0
        
        # 获取各类指标的信号
        momentum_signals = []
        trend_signals = []
        volume_signals = []
        volatility_signals = []
        
        for indicator_name, result in indicator_results.items():
            signal = result.get('signal', 'neutral')
            strength = result.get('strength', 0)
            
            # 分类收集信号
            for category, indicators in INDICATOR_CATEGORIES.items():
                if indicator_name in indicators:
                    signal_value = 1 if signal == 'buy' else -1 if signal == 'sell' else 0
                    weighted_signal = signal_value * strength
                    
                    if category == 'momentum':
                        momentum_signals.append(weighted_signal)
                    elif category == 'trend':
                        trend_signals.append(weighted_signal)
                    elif category == 'volume':
                        volume_signals.append(weighted_signal)
                    elif category == 'volatility':
                        volatility_signals.append(weighted_signal)
        
        # 计算各类指标的一致性
        def calculate_consistency(signals):
            """计算信号一致性 - 修复边界情况"""
            if not signals or len(signals) < 2:
                return 0.0
            
            positive = sum(1 for s in signals if s > 0.1)
            negative = sum(1 for s in signals if s < -0.1)
            total = len(signals)
            
            # 防止除零错误
            if total == 0:
                return 0.0
            
            # 考虑中性信号
            neutral = total - positive - negative
            if neutral == total:  # 全部是中性信号
                return 0.0
            
            # 计算一致性：同向信号占比
            if positive > negative:
                return positive / total
            elif negative > positive:
                return negative / total
            else:
                # 正负信号数量相等，一致性较低
                return max(positive, negative) / total * 0.5
        
        momentum_consistency = calculate_consistency(momentum_signals)
        trend_consistency = calculate_consistency(trend_signals)
        volume_consistency = calculate_consistency(volume_signals)
        
        # 协同奖励计算
        if momentum_consistency > 0.7 and trend_consistency > 0.7:
            synergy_bonus += 0.2  # 动量和趋势一致性高
        
        if volume_consistency > 0.6 and (momentum_consistency > 0.6 or trend_consistency > 0.6):
            synergy_bonus += 0.15  # 成交量确认信号
        
        # 特殊组合奖励
        rsi_result = indicator_results.get('RSI', {})
        bollinger_result = indicator_results.get('Bollinger', {})
        vwap_result = indicator_results.get('VWAP', {})
        
        # RSI超买 + 布林带上轨突破 = 强卖出信号
        if (rsi_result.get('signal') == 'sell' and rsi_result.get('strength', 0) > 0.6 and
            bollinger_result.get('signal') == 'sell' and bollinger_result.get('strength', 0) > 0.6):
            synergy_bonus += 0.25
        
        # RSI超卖 + 布林带下轨突破 = 强买入信号
        if (rsi_result.get('signal') == 'buy' and rsi_result.get('strength', 0) > 0.6 and
            bollinger_result.get('signal') == 'buy' and bollinger_result.get('strength', 0) > 0.6):
            synergy_bonus += 0.25
        
        # VWAP确认信号 - 修复：momentum_consistency不会<0，改为检查信号方向
        if vwap_result.get('signal') != 'neutral' and vwap_result.get('strength', 0) > 0.4:
            # 计算动量信号的总体方向
            momentum_direction = sum(momentum_signals)
            
            if ((momentum_direction > 0 and vwap_result.get('signal') == 'buy') or
                (momentum_direction < 0 and vwap_result.get('signal') == 'sell')):
                synergy_bonus += 0.1
        
        return min(synergy_bonus, 0.5)  # 最大50%的协同奖励
    
    def apply_timeframe_weights(self, timeframe_scores: Dict[str, float]) -> float:
        """应用时间框架权重"""
        total_score = 0.0
        total_weight = 0.0
        
        for timeframe, score in timeframe_scores.items():
            weight = self.timeframe_weights.get(timeframe, 0.1)
            total_score += score * weight
            total_weight += weight
        
        return total_score / total_weight if total_weight > 0 else 0.0
    
    def get_final_decision(self, combined_score: float, threshold: float = 0.3) -> Dict[str, Any]:
        """获取最终决策"""
        if combined_score > threshold:
            return {
                'direction': 'buy',
                'strength': abs(combined_score),
                'confidence': min(abs(combined_score) / threshold, 1.0)
            }
        elif combined_score < -threshold:
            return {
                'direction': 'sell',
                'strength': abs(combined_score),
                'confidence': min(abs(combined_score) / threshold, 1.0)
            }
        else:
            return {
                'direction': 'neutral',
                'strength': 0.0,
                'confidence': 0.5
            }


class MultiTimeframeCoordinator:
    """协调不同时间框架的指标计算"""
    
    def __init__(self, timeframes: List[str] = None, use_talib: bool = True):
        self.timeframes = timeframes or TIMEFRAME_HIERARCHY['primary']
        self.indicator_manager = IndicatorManager(use_talib=use_talib)
        self.signal_scorer = SignalScorer()
        self.data_cache = {}
        self.use_talib = use_talib
    
    def load_data(self, symbol: str, data_dict: Dict[str, pd.DataFrame]) -> bool:
        """加载多时间框架数据"""
        try:
            self.data_cache[symbol] = data_dict
            return True
        except Exception as e:
            warnings.warn(f"加载数据时出错: {str(e)}")
            return False
    
    def calculate_indicators(self, symbol: str, config = None) -> Dict[str, Dict[str, Any]]:
        """计算所有时间框架的指标"""
        if symbol not in self.data_cache:
            raise ValueError(f"未找到 {symbol} 的数据")
        
        results = {}
        
        for timeframe in self.timeframes:
            if timeframe in self.data_cache[symbol]:
                df = self.data_cache[symbol][timeframe]
                indicators = self.indicator_manager.calculate_all(df, timeframe, symbol, config)
                results[timeframe] = indicators
        
        return results
    
    def align_signals(self, symbol: str, config = None) -> Dict[str, Any]:
        """对齐不同时间框架的信号"""
        indicator_results = self.calculate_indicators(symbol, config)
        
        # 计算每个时间框架的综合分数
        timeframe_scores = {}
        
        for timeframe, indicators in indicator_results.items():
            indicator_scores = {}
            for indicator_name, result in indicators.items():
                score = self.signal_scorer.score_single_indicator(result, indicator_name)
                indicator_scores[indicator_name] = score
            
            # 合并该时间框架的所有指标分数，包含协同分析
            combined_score = self.signal_scorer.combine_scores(indicator_scores, self.indicator_manager, indicators)
            timeframe_scores[timeframe] = combined_score
        
        # 应用时间框架权重，得到最终分数
        final_score = self.signal_scorer.apply_timeframe_weights(timeframe_scores)
        final_decision = self.signal_scorer.get_final_decision(final_score)
        
        return {
            'symbol': symbol,
            'timeframe_scores': timeframe_scores,
            'final_score': final_score,
            'decision': final_decision,
            'indicator_details': indicator_results
        }
    
    def analyze_market_multitimeframe_safe(self, data_dict: Dict[str, pd.DataFrame], 
                                          symbol: str, retry_count: int = 3) -> Dict[str, Any]:
        """带重试机制的多时间框架分析 - 修复：改进超时处理"""
        last_error = None
        
        for attempt in range(retry_count):
            try:
                logger.info(f"开始多时间框架分析 (尝试 {attempt + 1}/{retry_count}): {symbol}")
                
                # 修复：避免线程+信号混用，使用futures超时控制
                from concurrent.futures import ThreadPoolExecutor as TPE, TimeoutError as FutureTimeoutError
                
                # 使用单独线程执行分析，避免信号干扰
                with TPE(max_workers=1) as executor:
                    future = executor.submit(self.align_signals, symbol)
                    try:
                        result = future.result(timeout=120)  # 2分钟总超时
                        logger.info(f"多时间框架分析成功: {symbol}")
                        return result
                    except FutureTimeoutError:
                        future.cancel()
                        raise TimeoutError("Analysis timeout")
                    
            except TimeoutError:
                last_error = TimeoutError("Analysis timeout")
                logger.warning(f"分析超时 (尝试 {attempt + 1}/{retry_count})")
                # 清理缓存并重试
                self.indicator_manager.clear_cache()
                time.sleep(1)
            except Exception as e:
                last_error = e
                logger.warning(f"分析失败 (尝试 {attempt + 1}/{retry_count}): {str(e)}")
                
                # 清理可能的问题
                self.indicator_manager.clear_cache()
                time.sleep(1)  # 等待一秒再重试
        
        # 所有重试都失败，返回降级结果
        logger.error(f"所有重试失败，返回降级结果: {str(last_error)}")
        return self._get_fallback_result(symbol, data_dict)
    
    def _get_fallback_result(self, symbol: str, data_dict: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """获取降级结果"""
        main_tf = '4h' if '4h' in data_dict else list(data_dict.keys())[0] if data_dict else None
        
        fallback_result = {
            'symbol': symbol,
            'timestamp': datetime.datetime.now(),
            'close_price': 0,
            'timeframe_scores': {},
            'final_score': 0,
            'decision': {'direction': 'neutral', 'strength': 0, 'confidence': 0},
            'risk_assessment': {'level': 'high', 'recommendation': '系统异常，建议观望'},
            'error': True,
            'error_message': 'System fallback mode'
        }
        
        if main_tf and main_tf in data_dict:
            df = data_dict[main_tf]
            if not df.empty and '收盘价' in df.columns:
                fallback_result['close_price'] = float(df['收盘价'].iloc[-1])
                fallback_result['main_timeframe'] = main_tf
        
        return fallback_result


class DynamicKDJ:
    """
    动态KDJ参数系统，根据市场波动性自动调整KDJ参数
    """
    def __init__(self, lookback_period=252, cache_file="kdj_params_cache.json"):
        """
        初始化动态KDJ系统
        :param lookback_period: 历史回溯周期，默认252个交易日(约一年)
        :param cache_file: 参数缓存文件路径
        """
        self.lookback = lookback_period
        self.cache_file = cache_file
        self.atr_percentiles = {}  # 存储各币种的ATR分位数
        self.current_params = {}   # 当前使用的参数
        self.analyzer = DivergenceAnalyzer()  # 使用现有的背离分析器
        
        # 加载缓存的参数
        self._load_params_cache()
    
    def calculate_atr(self, df, period=14):
        """
        计算ATR指标 - 修复：使用标准EMA平滑
        :param df: DataFrame，包含high, low, close列
        :param period: ATR周期
        :return: ATR值列表
        """
        # 使用pandas进行计算，避免手动循环的复杂性
        highs = df['最高价'].astype(float)
        lows = df['最低价'].astype(float)
        closes = df['收盘价'].astype(float)
        
        # 计算True Range
        tr1 = highs - lows
        tr2 = np.abs(highs - closes.shift(1))
        tr3 = np.abs(lows - closes.shift(1))
        
        # 真实波幅是三者的最大值
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        
        # 修复：使用标准EMA平滑，与TA-Lib一致
        alpha = 1.0 / period
        atr = tr.ewm(alpha=alpha, adjust=False).mean()
        
        # 填充NaN值
        atr = atr.fillna(0)
        
        return atr.values
    
    def update_atr_percentiles(self, symbol, df):
        """
        更新ATR分位数
        :param symbol: 交易对符号
        :param df: DataFrame，包含价格数据
        """
        # 修复：数据太少时跳过分位数计算，避免ATR失真
        if len(df) < 60:  # ATR需要足够的历史数据才有意义
            # 使用默认分位数，避免基于不足数据计算
            self.atr_percentiles[symbol] = {
                "25%": 0.5,   # 默认低波动
                "50%": 1.0,   # 默认中等波动  
                "75%": 2.0,   # 默认高波动
                "current": 1.0  # 默认当前波动
            }
            return
        
        # 确定回溯周期 - 智能调整
        if len(df) < self.lookback:
            lookback = len(df)  # 使用可用的全部数据
        else:
            lookback = self.lookback
        
        # 计算ATR
        atr = self.calculate_atr(df.tail(lookback))
        
        # 过滤掉前14个可能为0的ATR值（因为ATR计算需要14天）
        valid_atr = atr[14:] if len(atr) > 14 else atr[atr > 0]
        
        if len(valid_atr) == 0:
            # 如果没有有效的ATR值，使用默认值
            self.atr_percentiles[symbol] = {
                "25%": 0.5, "50%": 1.0, "75%": 2.0, "current": 1.0
            }
            return
        
        # 计算分位数
        self.atr_percentiles[symbol] = {
            "25%": np.percentile(valid_atr, 25),
            "50%": np.percentile(valid_atr, 50),
            "75%": np.percentile(valid_atr, 75),
            "current": atr[-1] if len(atr) > 0 else 1.0
        }
    
    def determine_market_volatility(self, symbol):
        """
        确定市场波动状态
        :param symbol: 交易对符号
        :return: 波动状态，可能值为 "high", "medium", "low"
        """
        if symbol not in self.atr_percentiles:
            return "medium"  # 默认为中等波动
        
        percentiles = self.atr_percentiles[symbol]
        current_atr = percentiles["current"]
        
        if current_atr > percentiles["75%"]:
            return "high"
        elif current_atr < percentiles["25%"]:
            return "low"
        else:
            return "medium"
    
    def get_optimal_kdj_params(self, symbol, config):
        """
        获取最优KDJ参数
        :param symbol: 交易对符号
        :param config: 策略配置对象
        :return: KDJ参数字典
        """
        volatility = self.determine_market_volatility(symbol)
        params = config.get_kdj_params(volatility)
        
        # 更新当前参数
        self.current_params[symbol] = {
            "volatility": volatility,
            "params": params
        }
        
        # 保存参数到缓存
        self._save_params_cache()
        
        return params
    
    def _load_params_cache(self):
        """加载参数缓存"""
        try:
            import json
            with open(self.cache_file, 'r') as f:
                cached_data = json.load(f)
                self.current_params = cached_data.get('current_params', {})
                self.atr_percentiles = cached_data.get('atr_percentiles', {})
        except (FileNotFoundError, json.JSONDecodeError):
            # 缓存文件不存在或格式错误，使用空字典
            pass
    
    def _save_params_cache(self):
        """保存参数缓存 - 线程安全版本"""
        try:
            import json
            import tempfile
            import os
            
            cache_data = {
                'current_params': self.current_params,
                'atr_percentiles': self.atr_percentiles
            }
            
            # 使用临时文件避免写入冲突
            temp_fd, temp_path = tempfile.mkstemp(
                suffix='.tmp', 
                prefix='kdj_cache_',
                dir=os.path.dirname(self.cache_file) if os.path.dirname(self.cache_file) else '.'
            )
            
            try:
                with os.fdopen(temp_fd, 'w') as f:
                    # 尝试文件锁（仅Linux/Mac）
                    try:
                        import fcntl
                        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                    except (ImportError, OSError):
                        # Windows上或其他情况忽略文件锁
                        pass
                    
                    json.dump(cache_data, f, indent=2)
                
                # 原子性替换
                if os.path.exists(self.cache_file):
                    # 备份原文件
                    backup_path = f"{self.cache_file}.backup"
                    try:
                        os.rename(self.cache_file, backup_path)
                        os.rename(temp_path, self.cache_file)
                        os.remove(backup_path)  # 删除备份
                    except OSError:
                        # 如果原子替换失败，恢复备份
                        if os.path.exists(backup_path):
                            os.rename(backup_path, self.cache_file)
                        if os.path.exists(temp_path):
                            os.remove(temp_path)
                        raise
                else:
                    os.rename(temp_path, self.cache_file)
                    
            except Exception:
                # 清理临时文件
                if os.path.exists(temp_path):
                    try:
                        os.remove(temp_path)
                    except OSError:
                        pass
                raise
                
        except Exception:
            # 保存失败时静默处理
            pass
    
    def calculate_adaptive_kdj(self, df, symbol, config):
        """
        计算自适应KDJ指标和背离
        :param df: DataFrame，包含价格数据
        :param symbol: 交易对符号
        :param config: 策略配置对象
        :return: 包含KDJ和背离信息的字典
        """
        # 更新ATR分位数
        self.update_atr_percentiles(symbol, df)
        
        # 获取最优参数
        params = self.get_optimal_kdj_params(symbol, config)
        
        # 转换为列表格式，以便使用DivergenceAnalyzer
        klines_data = df.to_dict('records')
        
        # 使用背离分析器计算KDJ和背离，传入动态参数
        # 重定向标准输出来抑制外部模块的打印
        import sys
        from io import StringIO
        
        old_stdout = sys.stdout
        sys.stdout = StringIO()
        
        try:
            result = self.analyzer.calculate_kdj_indicators(klines_data, params)
        finally:
            sys.stdout = old_stdout
        
        # 添加当前使用的参数信息
        if result:
            result['current_params'] = self.current_params[symbol]
        
        return result


class ADXFilter:
    """
    ADX市场状态过滤器，用于判断市场趋势状态并调整信号强度
    """
    def __init__(self, period=14):
        """
        初始化ADX过滤器
        :param period: ADX计算周期
        """
        self.period = period
        self.trending_threshold = 25  # 趋势市场阈值
        self.sideways_threshold = 20  # 震荡市场阈值
    
    def calculate_adx(self, df):
        """
        计算ADX指标 - 修复：使用标准EMA平滑算法
        :param df: DataFrame，包含high, low, close列
        :return: ADX值列表
        """
        # 使用pandas DataFrame进行计算，提高代码可读性和正确性
        highs = df['最高价'].astype(float)
        lows = df['最低价'].astype(float)
        closes = df['收盘价'].astype(float)
        
        # 计算方向运动
        up_move = highs - highs.shift(1)
        down_move = lows.shift(1) - lows
        
        # 计算+DM和-DM
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
        
        # 计算True Range
        tr1 = highs - lows
        tr2 = np.abs(highs - closes.shift(1))
        tr3 = np.abs(lows - closes.shift(1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        
        # 修复：使用EMA进行平滑，与TA-Lib算法一致
        period = self.period
        alpha = 1.0 / period
        
        # 将numpy数组转换为pandas Series以便使用ewm
        plus_dm_series = pd.Series(plus_dm)
        minus_dm_series = pd.Series(minus_dm)
        tr_series = pd.Series(tr)
        
        # 使用EMA平滑
        plus_dm_smooth = plus_dm_series.ewm(alpha=alpha, adjust=False).mean()
        minus_dm_smooth = minus_dm_series.ewm(alpha=alpha, adjust=False).mean()
        tr_smooth = tr_series.ewm(alpha=alpha, adjust=False).mean()
        
        # 计算+DI和-DI
        plus_di = 100 * plus_dm_smooth / tr_smooth.replace(0, np.nan)
        minus_di = 100 * minus_dm_smooth / tr_smooth.replace(0, np.nan)
        
        # 填充NaN值
        plus_di = plus_di.fillna(0)
        minus_di = minus_di.fillna(0)
        
        # 计算DX
        di_sum = plus_di + minus_di
        dx = 100 * np.abs(plus_di - minus_di) / di_sum.replace(0, np.nan)
        dx = dx.fillna(0)
        
        # 计算ADX (对DX进行EMA平滑)
        adx = dx.ewm(alpha=alpha, adjust=False).mean()
        
        # 填充NaN值并转换为numpy数组
        adx = adx.fillna(0).values
        
        return adx
    
    def get_market_regime(self, adx_value):
        """
        判断市场状态
        :param adx_value: ADX值
        :return: 市场状态，可能值为 "trending", "sideways", "transition"
        """
        if adx_value > self.trending_threshold:
            return "trending"
        elif adx_value < self.sideways_threshold:
            return "sideways"
        else:
            return "transition"
    
    def adjust_signal_strength(self, base_signal, market_regime):
        """
        根据市场状态调整信号强度
        :param base_signal: 基础信号强度 (0-1)
        :param market_regime: 市场状态
        :return: 调整后的信号强度
        """
        if market_regime == "trending":
            return min(base_signal * 1.5, 1.0)  # 趋势市场增强信号，但不超过1
        elif market_regime == "sideways":
            return base_signal * 0.5  # 震荡市场减弱信号
        else:
            return base_signal  # 过渡状态保持不变
    
    def should_trade(self, signal_strength, threshold=0.4):
        """
        判断是否应该交易
        :param signal_strength: 信号强度
        :param threshold: 交易阈值
        :return: 布尔值，表示是否应该交易
        """
        return signal_strength >= threshold


class TechnicalAnalyzer:
    """
    技术分析器，整合多个技术指标并生成交易信号
    支持多时间框架分析
    """
    def __init__(self, config, timeframes: List[str] = None, use_talib: bool = True):
        """
        初始化技术分析器
        :param config: 策略配置对象
        :param timeframes: 要分析的时间框架列表
        :param use_talib: 是否使用TA-Lib库加速计算
        """
        self.config = config
        self.timeframes = timeframes or TIMEFRAME_HIERARCHY['primary']
        self.use_talib = use_talib
        
        # 初始化组件
        self.coordinator = MultiTimeframeCoordinator(self.timeframes, use_talib=use_talib)
        self.dynamic_kdj = DynamicKDJ(lookback_period=config.technical["atr"]["lookback"])
        self.adx_filter = ADXFilter(period=config.technical["adx"]["period"])
        
        # 注册自定义指标
        self._register_custom_indicators()
    
    def _register_custom_indicators(self):
        """注册自定义指标到协调器"""
        # 保存引用到局部变量，避免闭包作用域问题
        dynamic_kdj = self.dynamic_kdj
        adx_filter = self.adx_filter
        config = self.config
        
        # 注册动态KDJ为自定义指标
        class DynamicKDJIndicator(BaseIndicator):
            def __init__(self, analyzer_instance, default_config):
                super().__init__("DynamicKDJ", "custom")
                self.analyzer = analyzer_instance
                self.default_config = default_config
            
            def calculate(self, df: pd.DataFrame, **kwargs) -> Dict[str, Any]:
                symbol = kwargs.get('symbol', 'UNKNOWN')
                config = kwargs.get('config') or self.default_config
                
                try:
                    result = self.analyzer.calculate_adaptive_kdj(df, symbol, config)
                    if result:
                        # 统一数据类型为numpy array
                        return {
                            'k': np.array(result.get('k', [])),
                            'd': np.array(result.get('d', [])),
                            'j': np.array(result.get('j', [])),
                            'current_j': result['j'][-1] if result.get('j') else 50,
                            'top_divergence': result.get('top_divergence', [False])[-1] if result.get('top_divergence') else False,
                            'bottom_divergence': result.get('bottom_divergence', [False])[-1] if result.get('bottom_divergence') else False
                        }
                except Exception:
                    # 静默处理DynamicKDJ计算错误
                    pass
                    
                return {
                    'k': np.array([]),
                    'd': np.array([]),
                    'j': np.array([]),
                    'current_j': 50, 
                    'top_divergence': False, 
                    'bottom_divergence': False
                }
            
            def get_signal(self, values: Dict[str, Any]) -> str:
                if values.get('top_divergence'):
                    return 'sell'
                elif values.get('bottom_divergence'):
                    return 'buy'
                else:
                    j_current = values.get('current_j', 50)
                    if j_current > 80:
                        return 'sell'
                    elif j_current < 20:
                        return 'buy'
                    else:
                        return 'neutral'
            
            def get_strength(self, values: Dict[str, Any]) -> float:
                # 修复：更细粒度的强度计算
                if values.get('top_divergence') or values.get('bottom_divergence'):
                    j_current = values.get('current_j', 50)
                    # 背离信号基础强度0.7，根据J值极端程度调整
                    base_strength = 0.7
                    if j_current > 90 or j_current < 10:
                        return min(base_strength + 0.2, 1.0)  # 极端背离
                    elif j_current > 85 or j_current < 15:
                        return min(base_strength + 0.1, 1.0)  # 强背离
                    else:
                        return base_strength  # 一般背离
                
                j_current = values.get('current_j', 50)
                if j_current > 80:
                    # 线性缩放：80-100 -> 0.4-0.8
                    return 0.4 + (j_current - 80) / 20 * 0.4
                elif j_current < 20:
                    # 线性缩放：0-20 -> 0.8-0.4
                    return 0.4 + (20 - j_current) / 20 * 0.4
                else:
                    return 0.0
        
        # 注册ADX指标
        class ADXIndicator(BaseIndicator):
            def __init__(self, adx_filter_instance):
                super().__init__("ADX", "trend")
                self.adx_filter = adx_filter_instance
            
            def calculate(self, df: pd.DataFrame, **kwargs) -> Dict[str, Any]:
                try:
                    # 计算完整的ADX数据，包括+DI和-DI
                    high = df['最高价'].astype(float).values
                    low = df['最低价'].astype(float).values
                    close = df['收盘价'].astype(float).values
                    
                    # 计算+DI和-DI用于信号生成
                    df_calc = pd.DataFrame({
                        'high': high,
                        'low': low,
                        'close': close
                    })
                    
                    prev_high = df_calc['high'].shift(1)
                    prev_low = df_calc['low'].shift(1)
                    up_move = df_calc['high'] - prev_high
                    down_move = prev_low - df_calc['low']
                    
                    up_move.iloc[0] = 0.0
                    down_move.iloc[0] = 0.0
                    
                    # 计算方向指标
                    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
                    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
                    
                    # 计算True Range
                    prev_close = df_calc['close'].shift(1)
                    tr1 = df_calc['high'] - df_calc['low']
                    tr2 = np.abs(df_calc['high'] - prev_close)
                    tr3 = np.abs(df_calc['low'] - prev_close)
                    tr = np.maximum(tr1, np.maximum(tr2, tr3))
                    tr.iloc[0] = tr1.iloc[0]
                    
                    # 使用EMA平滑
                    period = self.adx_filter.period
                    alpha = 1.0 / period
                    
                    smoothed_plus_dm = pd.Series(plus_dm).ewm(alpha=alpha, adjust=False).mean()
                    smoothed_minus_dm = pd.Series(minus_dm).ewm(alpha=alpha, adjust=False).mean()
                    smoothed_tr = tr.ewm(alpha=alpha, adjust=False).mean()
                    
                    # 计算+DI和-DI
                    plus_di = 100 * (smoothed_plus_dm / smoothed_tr)
                    minus_di = 100 * (smoothed_minus_dm / smoothed_tr)
                    
                    # 处理异常值
                    plus_di = plus_di.fillna(0)
                    minus_di = minus_di.fillna(0)
                    
                    # 计算ADX
                    adx_values = self.adx_filter.calculate_adx(df)
                    current_adx = adx_values[-1] if len(adx_values) > 0 else 25
                    market_regime = self.adx_filter.get_market_regime(current_adx)
                    
                    # 获取当前和前一个周期的DI值
                    current_plus_di = plus_di.iloc[-1] if len(plus_di) > 0 else 0
                    current_minus_di = minus_di.iloc[-1] if len(minus_di) > 0 else 0
                    prev_plus_di = plus_di.iloc[-2] if len(plus_di) > 1 else current_plus_di
                    prev_minus_di = minus_di.iloc[-2] if len(minus_di) > 1 else current_minus_di
                    
                    return {
                        'values': np.array(adx_values),
                        'current': float(current_adx),
                        'market_regime': market_regime,
                        'plus_di': float(current_plus_di),
                        'minus_di': float(current_minus_di),
                        'prev_plus_di': float(prev_plus_di),
                        'prev_minus_di': float(prev_minus_di)
                    }
                except Exception:
                    # 静默处理ADX计算错误
                    return {
                        'values': np.array([]), 
                        'current': 25.0, 
                        'market_regime': 'transition', 
                        'plus_di': 0.0, 
                        'minus_di': 0.0, 
                        'prev_plus_di': 0.0, 
                        'prev_minus_di': 0.0
                    }
            
            def get_signal(self, values: Dict[str, Any]) -> str:
                """
                基于+DI和-DI的交叉生成标准买卖信号
                +DI上穿-DI：买入信号
                -DI上穿+DI：卖出信号
                """
                current_plus_di = values.get('plus_di', 0)
                current_minus_di = values.get('minus_di', 0)
                prev_plus_di = values.get('prev_plus_di', 0)
                prev_minus_di = values.get('prev_minus_di', 0)
                
                # 检查交叉信号
                if (current_plus_di > current_minus_di and 
                    prev_plus_di <= prev_minus_di):
                    return 'buy'  # +DI上穿-DI
                elif (current_minus_di > current_plus_di and 
                      prev_minus_di <= prev_plus_di):
                    return 'sell'  # -DI上穿+DI
                else:
                    return 'neutral'
            
            def get_strength(self, values: Dict[str, Any]) -> float:
                current_adx = values.get('current', 25)
                if current_adx > 25:
                    return min((current_adx - 25) / 50, 1.0)  # 趋势强度
                else:
                    return 0.0
        
        # 注册到协调器，传递必要的参数
        self.coordinator.indicator_manager.register_indicator(DynamicKDJIndicator(dynamic_kdj, config))
        self.coordinator.indicator_manager.register_indicator(ADXIndicator(adx_filter))
    
    def analyze_market_multitimeframe(self, data_dict: Dict[str, pd.DataFrame], symbol: str) -> Dict[str, Any]:
        """
        多时间框架市场分析
        :param data_dict: 包含不同时间框架数据的字典 {'1h': df, '4h': df, '1d': df}
        :param symbol: 交易对符号
        :return: 多时间框架分析结果
        """
        # 加载数据到协调器
        success = self.coordinator.load_data(symbol, data_dict)
        if not success:
            raise ValueError(f"加载 {symbol} 数据失败")
        
        # 执行多时间框架分析，传递配置
        analysis_result = self.coordinator.align_signals(symbol, self.config)
        
        # 添加额外的分析信息
        analysis_result['timestamp'] = datetime.datetime.now()
        
        # 获取主要时间框架的价格信息
        main_timeframe = '4h' if '4h' in data_dict else list(data_dict.keys())[0]
        if main_timeframe in data_dict:
            df = data_dict[main_timeframe]
            analysis_result['close_price'] = df['收盘价'].iloc[-1]
            analysis_result['main_timeframe'] = main_timeframe
        
        # 添加风险评估
        analysis_result['risk_assessment'] = self._assess_risk(analysis_result)
        
        return analysis_result
    
    def _assess_risk(self, analysis_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        多维度评估交易风险
        :param analysis_result: 分析结果
        :return: 风险评估结果
        """
        decision = analysis_result.get('decision', {})
        timeframe_scores = analysis_result.get('timeframe_scores', {})
        indicator_details = analysis_result.get('indicator_details', {})
        
        # 1. 计算时间框架一致性
        positive_scores = sum(1 for score in timeframe_scores.values() if score > 0.1)
        negative_scores = sum(1 for score in timeframe_scores.values() if score < -0.1)
        total_scores = len(timeframe_scores)
        
        if total_scores == 0:
            consistency = 0.0
        else:
            consistency = max(positive_scores, negative_scores) / total_scores
        
        # 2. 检查指标冲突
        conflicting_signals = self._check_signal_conflicts(indicator_details)
        
        # 3. 评估市场波动性风险
        atr_values = self._extract_atr_values(indicator_details)
        volatility_risk = self._assess_volatility_risk(atr_values)
        
        # 4. 计算时间框架分歧度
        timeframe_divergence = self._calculate_timeframe_divergence(timeframe_scores)
        
        # 5. 信号强度分布分析
        score_variance = np.var(list(timeframe_scores.values())) if timeframe_scores else 0
        
        # 综合风险评估
        risk_factors = {
            'consistency': consistency,
            'conflicts': conflicting_signals,
            'volatility': volatility_risk,
            'divergence': timeframe_divergence,
            'score_variance': score_variance
        }
        
        # 更细致的风险分级
        risk_score = self._calculate_risk_score(risk_factors)
        risk_level = self._determine_risk_level(risk_score)
        
        return {
            'level': risk_level,
            'score': risk_score,
            'factors': risk_factors,
            'recommendation': self._get_risk_recommendation(risk_level, decision)
        }
    
    def _check_signal_conflicts(self, indicator_details: Dict[str, Any]) -> float:
        """检查指标间信号冲突"""
        conflicts = 0
        total_pairs = 0
        
        for tf, indicators in indicator_details.items():
            if not isinstance(indicators, dict):
                continue
                
            signals = []
            for indicator_name, result in indicators.items():
                if isinstance(result, dict):
                    signal = result.get('signal', 'neutral')
                    if signal in ['buy', 'sell']:
                        signals.append(1 if signal == 'buy' else -1)
            
            # 计算该时间框架内的冲突
            if len(signals) > 1:
                for i in range(len(signals)):
                    for j in range(i+1, len(signals)):
                        total_pairs += 1
                        if signals[i] * signals[j] < 0:  # 相反信号
                            conflicts += 1
        
        return conflicts / total_pairs if total_pairs > 0 else 0.0
    
    def _extract_atr_values(self, indicator_details: Dict[str, Any]) -> List[float]:
        """提取ATR波动性值"""
        atr_values = []
        
        for tf, indicators in indicator_details.items():
            if isinstance(indicators, dict) and 'ATR' in indicators:
                atr_result = indicators['ATR']
                if isinstance(atr_result, dict):
                    atr_current = atr_result.get('current', 0)
                    if atr_current > 0:
                        atr_values.append(atr_current)
        
        return atr_values
    
    def _assess_volatility_risk(self, atr_values: List[float]) -> float:
        """评估波动性风险"""
        if not atr_values:
            return 0.5  # 默认中等风险
        
        avg_atr = np.mean(atr_values)
        
        # 根据ATR值判断波动性风险
        # 这里需要根据具体资产调整阈值
        if avg_atr > 1000:  # 高波动
            return 0.8
        elif avg_atr > 500:  # 中等波动
            return 0.5
        else:  # 低波动
            return 0.2
    
    def _calculate_timeframe_divergence(self, timeframe_scores: Dict[str, float]) -> float:
        """计算时间框架间的分歧度"""
        if len(timeframe_scores) < 2:
            return 0.0
        
        scores = list(timeframe_scores.values())
        
        # 计算标准差表示分歧度
        std_dev = np.std(scores)
        
        # 归一化到0-1范围
        return min(std_dev / 1.0, 1.0)
    
    def _calculate_risk_score(self, risk_factors: Dict[str, float]) -> float:
        """计算综合风险评分"""
        weights = {
            'consistency': -0.3,    # 一致性高降低风险
            'conflicts': 0.25,      # 冲突多增加风险
            'volatility': 0.2,      # 波动性高增加风险
            'divergence': 0.15,     # 分歧大增加风险
            'score_variance': 0.1   # 方差大增加风险
        }
        
        risk_score = 0.5  # 基础风险
        
        for factor, value in risk_factors.items():
            if factor in weights:
                risk_score += weights[factor] * value
        
        # 确保在0-1范围内
        return max(0.0, min(1.0, risk_score))
    
    def _determine_risk_level(self, risk_score: float) -> str:
        """确定风险等级"""
        if risk_score < 0.3:
            return 'low'
        elif risk_score < 0.7:
            return 'medium'
        else:
            return 'high'
    
    def _get_risk_recommendation(self, risk_level: str, decision: Dict[str, Any]) -> str:
        """
        获取风险建议
        :param risk_level: 风险等级
        :param decision: 交易决策
        :return: 风险建议
        """
        direction = decision.get('direction', 'neutral')
        
        if risk_level == 'low':
            if direction != 'neutral':
                return f"风险较低，可以考虑{direction}操作，建议正常仓位"
            else:
                return "风险较低，但信号不明确，建议观望"
        elif risk_level == 'medium':
            if direction != 'neutral':
                return f"风险中等，可以考虑{direction}操作，建议减少仓位"
            else:
                return "风险中等，信号不明确，建议观望"
        else:  # high risk
            return "风险较高，建议观望或使用小仓位试探"
    
    def analyze_market(self, df, symbol):
        """
        分析市场并生成交易信号
        :param df: DataFrame，包含价格数据
        :param symbol: 交易对符号
        :return: 分析结果字典
        """
        # 计算自适应KDJ和背离
        kdj_result = self.dynamic_kdj.calculate_adaptive_kdj(df, symbol, self.config)
        
        # 计算ADX
        adx = self.adx_filter.calculate_adx(df)
        current_adx = adx[-1]
        
        # 判断市场状态
        market_regime = self.adx_filter.get_market_regime(current_adx)
        
        # 提取最新的背离信号
        latest_top_divergence = kdj_result['top_divergence'][-1] if kdj_result else False
        latest_bottom_divergence = kdj_result['bottom_divergence'][-1] if kdj_result else False
        
        # 计算基础信号强度 (0-1)
        base_signal = 0
        signal_type = "neutral"
        
        if latest_top_divergence:
            base_signal = 0.8  # 顶部背离，卖出信号
            signal_type = "sell"
        elif latest_bottom_divergence:
            base_signal = 0.8  # 底部背离，买入信号
            signal_type = "buy"
        else:
            # 增加基于价格和KDJ指标的额外信号
            if kdj_result and len(kdj_result['j']) > 1:
                j_values = kdj_result['j']
                j_current = j_values[-1]
                j_prev = j_values[-2]
                
                # 超买区域的卖出信号
                if j_current > 80 and j_prev > j_current:
                    base_signal = 0.6
                    signal_type = "sell"
                # 超卖区域的买入信号
                elif j_current < 20 and j_current > j_prev:
                    base_signal = 0.6
                    signal_type = "buy"
                # J线上穿50的买入信号
                elif j_prev < 50 and j_current > 50:
                    base_signal = 0.5
                    signal_type = "buy"
                # J线下穿50的卖出信号
                elif j_prev > 50 and j_current < 50:
                    base_signal = 0.5
                    signal_type = "sell"
        
        # 根据市场状态调整信号强度
        adjusted_signal = self.adx_filter.adjust_signal_strength(base_signal, market_regime)
        
        # 判断是否应该交易
        should_trade = self.adx_filter.should_trade(adjusted_signal)
        
        # 获取时间戳
        timestamp = None
        if '开盘时间' in df.columns:
            timestamp = df['开盘时间'].iloc[-1]
        
        return {
            "symbol": symbol,
            "timestamp": timestamp,
            "close_price": df['收盘价'].iloc[-1],
            "market_regime": market_regime,
            "adx": current_adx,
            "kdj_params": self.dynamic_kdj.current_params.get(symbol, {}),
            "top_divergence": latest_top_divergence,
            "bottom_divergence": latest_bottom_divergence,
            "signal_type": signal_type,
            "signal_strength": adjusted_signal,
            "should_trade": should_trade
        }
        
    def analyze_historical_data(self, df, symbol, min_lookback=30):
        """
        分析历史数据并输出每一天的分析结果
        :param df: DataFrame，包含价格数据
        :param symbol: 交易对符号
        :param min_lookback: 最小回溯天数，确保有足够数据计算指标
        :return: 包含每日分析结果的DataFrame
        """
        results = []
        
        # 确保有足够的初始数据来计算指标
        for i in range(min_lookback, len(df)):
            # 使用截止到当前日期的数据
            current_df = df.iloc[:i+1]
            
            try:
                # 分析当前日期的市场状况
                result = self.analyze_market(current_df, symbol)
                results.append(result)
            except Exception as e:
                print(f"分析第{i}天数据时出错: {str(e)}")
        
        # 转换为DataFrame便于查看
        results_df = pd.DataFrame(results)
        return results_df

    def visualize_results(self, df, results_df, last_n_days=120, save_path=None):
        """
        可视化分析结果，将K线图与交易信号结合展示
        :param df: 原始K线数据DataFrame
        :param results_df: 分析结果DataFrame
        :param last_n_days: 展示最近的天数
        :param save_path: 保存图片的路径，如果为None则显示图片
        :return: None
        """
        # 确保数据量足够
        if len(results_df) < last_n_days:
            last_n_days = len(results_df)
            print(f"数据量不足，只展示全部 {last_n_days} 天数据")
        
        # 获取最近N天的数据
        recent_df = df.iloc[-last_n_days:].copy()
        recent_results = results_df.iloc[-last_n_days:].copy()
        
        # 将时间列转换为datetime类型
        if '开盘时间' in recent_df.columns:
            recent_df['日期'] = pd.to_datetime(recent_df['开盘时间'])
        
        if 'timestamp' in recent_results.columns:
            recent_results['日期'] = pd.to_datetime(recent_results['timestamp'])
        
        # 创建图表
        plt.style.use('seaborn-v0_8-darkgrid')
        fig = plt.figure(figsize=(18, 12))
        gs = GridSpec(4, 1, height_ratios=[3, 1, 1, 1])
        
        # 绘制K线图
        ax1 = plt.subplot(gs[0])
        ax1.set_title(f'比特币技术分析 - 最近{last_n_days}天', fontsize=16)
        
        # 绘制价格
        ax1.plot(recent_df['日期'], recent_df['收盘价'], label='收盘价', color='#1f77b4', linewidth=2)
        
        # 标记买入信号
        buy_signals = recent_results[(recent_results['signal_type'] == 'buy') & (recent_results['should_trade'] == True)]
        if not buy_signals.empty:
            ax1.scatter(buy_signals['日期'], buy_signals['close_price'], 
                       marker='^', color='green', s=150, label='买入信号')
            
            # 添加买入信号注释
            for i, signal in buy_signals.iterrows():
                ax1.annotate(f"买入\n强度:{signal['signal_strength']:.2f}", 
                           (mdates.date2num(signal['日期']), signal['close_price']),
                           xytext=(0, 30), textcoords='offset points',
                           arrowprops=dict(arrowstyle='->', color='green'),
                           ha='center', fontsize=9)
        
        # 标记卖出信号
        sell_signals = recent_results[(recent_results['signal_type'] == 'sell') & (recent_results['should_trade'] == True)]
        if not sell_signals.empty:
            ax1.scatter(sell_signals['日期'], sell_signals['close_price'], 
                       marker='v', color='red', s=150, label='卖出信号')
            
            # 添加卖出信号注释
            for i, signal in sell_signals.iterrows():
                ax1.annotate(f"卖出\n强度:{signal['signal_strength']:.2f}", 
                           (mdates.date2num(signal['日期']), signal['close_price']),
                           xytext=(0, -30), textcoords='offset points',
                           arrowprops=dict(arrowstyle='->', color='red'),
                           ha='center', fontsize=9)
        
        # 标记背离
        top_divergence = recent_results[recent_results['top_divergence'] == True]
        if not top_divergence.empty:
            ax1.scatter(top_divergence['日期'], top_divergence['close_price'], 
                       marker='X', color='purple', s=120, label='顶部背离')
        
        bottom_divergence = recent_results[recent_results['bottom_divergence'] == True]
        if not bottom_divergence.empty:
            ax1.scatter(bottom_divergence['日期'], bottom_divergence['close_price'], 
                       marker='X', color='blue', s=120, label='底部背离')
        
        # 设置x轴格式
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax1.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
        
        # 添加网格和图例
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='upper left')
        
        # 绘制ADX指标
        ax2 = plt.subplot(gs[1], sharex=ax1)
        ax2.set_title('ADX指标与市场状态', fontsize=12)
        ax2.plot(recent_results['日期'], recent_results['adx'], label='ADX', color='purple', linewidth=1.5)
        
        # 添加市场状态背景色
        for i, row in recent_results.iterrows():
            if row['market_regime'] == 'trending':
                ax2.axvspan(row['日期'], row['日期'] + pd.Timedelta(days=1), 
                           alpha=0.2, color='green', label='_trending')
            elif row['market_regime'] == 'sideways':
                ax2.axvspan(row['日期'], row['日期'] + pd.Timedelta(days=1), 
                           alpha=0.2, color='red', label='_sideways')
            else:  # transition
                ax2.axvspan(row['日期'], row['日期'] + pd.Timedelta(days=1), 
                           alpha=0.2, color='yellow', label='_transition')
        
        # 添加趋势阈值线
        ax2.axhline(y=25, color='green', linestyle='--', alpha=0.7, label='趋势阈值(25)')
        ax2.axhline(y=20, color='red', linestyle='--', alpha=0.7, label='震荡阈值(20)')
        
        # 设置y轴范围
        ax2.set_ylim(0, max(recent_results['adx']) * 1.1)
        ax2.legend(loc='upper left')
        
        # 绘制信号强度
        ax3 = plt.subplot(gs[2], sharex=ax1)
        ax3.set_title('信号强度和交易决策', fontsize=12)
        
        # 绘制信号强度柱状图
        bars = ax3.bar(recent_results['日期'], recent_results['signal_strength'], 
                      color=recent_results['signal_type'].map({'buy': 'green', 'sell': 'red', 'neutral': 'gray'}),
                      alpha=0.7, width=0.8)
        
        # 添加交易阈值线
        ax3.axhline(y=0.4, color='black', linestyle='--', alpha=0.7, label='交易阈值(0.4)')
        
        # 设置y轴范围
        ax3.set_ylim(0, 1.1)
        ax3.legend(loc='upper left')
        
        # 绘制KDJ参数变化
        ax4 = plt.subplot(gs[3], sharex=ax1)
        ax4.set_title('KDJ参数动态调整', fontsize=12)
        
        # 提取KDJ参数
        k_values = []
        d_values = []
        j_values = []
        volatility = []
        
        for i, row in recent_results.iterrows():
            if isinstance(row['kdj_params'], dict) and 'params' in row['kdj_params']:
                k_values.append(row['kdj_params']['params'].get('k', 0))
                d_values.append(row['kdj_params']['params'].get('d', 0))
                j_values.append(row['kdj_params']['params'].get('j', 0))
                volatility.append(row['kdj_params'].get('volatility', 'unknown'))
            else:
                k_values.append(0)
                d_values.append(0)
                j_values.append(0)
                volatility.append('unknown')
        
        recent_results['k_param'] = k_values
        recent_results['d_param'] = d_values
        recent_results['j_param'] = j_values
        recent_results['volatility'] = volatility
        
        # 绘制KDJ参数
        ax4.plot(recent_results['日期'], recent_results['k_param'], label='K周期', color='blue')
        ax4.plot(recent_results['日期'], recent_results['d_param'], label='D周期', color='orange')
        ax4.plot(recent_results['日期'], recent_results['j_param'], label='J周期', color='green')
        
        # 添加波动性背景色
        for i, row in recent_results.iterrows():
            if row['volatility'] == 'high':
                ax4.axvspan(row['日期'], row['日期'] + pd.Timedelta(days=1), 
                           alpha=0.2, color='red', label='_high')
            elif row['volatility'] == 'medium':
                ax4.axvspan(row['日期'], row['日期'] + pd.Timedelta(days=1), 
                           alpha=0.2, color='yellow', label='_medium')
            elif row['volatility'] == 'low':
                ax4.axvspan(row['日期'], row['日期'] + pd.Timedelta(days=1), 
                           alpha=0.2, color='green', label='_low')
        
        ax4.legend(loc='upper left')
        
        # 添加图例说明
        fig.text(0.02, 0.02, "市场状态: 绿色=趋势 黄色=过渡 红色=震荡\n"
                           "波动性: 红色=高 黄色=中 绿色=低\n"
                           "信号: 绿色=买入 红色=卖出 灰色=中性", fontsize=10)
        
        # 调整布局
        plt.tight_layout()
        
        # 保存或显示图片
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"图表已保存至: {save_path}")
        else:
            plt.show()
        
        plt.close()
        
        # 返回带有参数的结果DataFrame，方便进一步分析
        return recent_results


def performance_benchmark(data_dict: Dict[str, pd.DataFrame], config):
    """性能基准测试 - 对比TA-Lib和自实现性能"""
    print("\n" + "="*80)
    print("⚡ 性能基准测试: TA-Lib vs 自实现")
    print("="*80)
    
    import time
    
    # 测试数据
    test_symbol = "BTCUSDT" 
    test_timeframe = "4h"
    test_df = data_dict.get(test_timeframe, list(data_dict.values())[0])
    
    print(f"📊 测试数据: {len(test_df)} 条 {test_timeframe} K线数据")
    print(f"🔧 TA-Lib 可用性: {'✅' if TALIB_AVAILABLE else '❌'}")
    
    if not TALIB_AVAILABLE:
        print("⚠️ TA-Lib 不可用，跳过性能对比测试")
        return
    
    # 创建两个分析器对比
    print(f"\n🏃‍♂️ 开始性能测试...")
    
    # 测试TA-Lib版本
    start_time = time.time()
    analyzer_talib = TechnicalAnalyzer(config, use_talib=True)
    result_talib = analyzer_talib.analyze_market_multitimeframe(data_dict, test_symbol)
    talib_time = time.time() - start_time
    
    # 测试自实现版本
    start_time = time.time()
    analyzer_custom = TechnicalAnalyzer(config, use_talib=False)
    result_custom = analyzer_custom.analyze_market_multitimeframe(data_dict, test_symbol)
    custom_time = time.time() - start_time
    
    # 性能统计
    speedup = custom_time / talib_time if talib_time > 0 else 0
    
    print(f"\n📈 性能测试结果:")
    print(f"  TA-Lib版本耗时: {talib_time:.3f}秒")
    print(f"  自实现版本耗时: {custom_time:.3f}秒")
    print(f"  性能提升倍数: {speedup:.2f}x")
    print(f"  时间节省: {((custom_time - talib_time) / custom_time * 100):.1f}%")
    
    # 结果一致性检查
    print(f"\n🔍 结果一致性检查:")
    talib_score = result_talib.get('final_score', 0)
    custom_score = result_custom.get('final_score', 0)
    score_diff = abs(talib_score - custom_score)
    
    print(f"  TA-Lib最终得分: {talib_score:.4f}")
    print(f"  自实现最终得分: {custom_score:.4f}")
    print(f"  得分差异: {score_diff:.4f}")
    
    if score_diff < 0.001:
        print("  ✅ 结果高度一致")
    elif score_diff < 0.01:
        print("  ⚠️ 结果基本一致，存在微小差异")
    else:
        print("  ❌ 结果存在显著差异，需要检查算法")
    
    # 详细性能报告
    analyzer_talib.coordinator.indicator_manager.print_performance_report()
    
    print("="*80)


def generate_detailed_performance_report(analyzer: 'TechnicalAnalyzer') -> str:
    """生成详细的性能报告"""
    manager = analyzer.coordinator.indicator_manager
    monitor = manager.performance_monitor
    cache = manager.smart_cache
    
    report = []
    report.append("\n" + "="*80)
    report.append("📊 详细性能分析报告")
    report.append("="*80)
    
    # 1. 缓存统计
    cache_stats = cache.get_stats()
    report.append(f"\n📦 缓存统计:")
    report.append(f"  • 当前大小: {cache_stats['size']}/{cache_stats['max_size']}")
    report.append(f"  • 使用率: {cache_stats['size']/cache_stats['max_size']*100:.1f}%")
    
    # 2. 性能统计
    perf_report = monitor.get_report()
    if perf_report.get('status') != 'no_data':
        report.append(f"\n⚡ 性能指标:")
        report.append(f"  • 平均计算时间: {perf_report['avg_calculation_time']*1000:.2f}ms")
        report.append(f"  • 最大计算时间: {perf_report['max_calculation_time']*1000:.2f}ms")
        report.append(f"  • 成功率: {perf_report['success_rate']*100:.1f}%")
        report.append(f"  • 总操作数: {perf_report['total_operations']}")
    
    # 3. 指标使用统计
    summary = manager.get_performance_summary()
    report.append(f"\n📈 指标统计:")
    report.append(f"  • TA-Lib加速指标: {summary['talib_enabled_indicators']}/{summary['total_indicators']}")
    
    # 4. 性能排行
    report.append(f"\n🏆 指标性能排行 (最快到最慢):")
    
    perf_details = []
    for name, stats in summary['performance_details'].items():
        if isinstance(stats.get('avg_talib_time', 0), (int, float)):
            time = stats.get('avg_talib_time', 0) or stats.get('avg_custom_time', 0)
            if time > 0:
                perf_details.append((name, time))
    
    perf_details.sort(key=lambda x: x[1])
    
    for i, (name, time) in enumerate(perf_details[:10]):
        report.append(f"  {i+1:2d}. {name:>15}: {time:6.2f}ms")
    
    return "\n".join(report)


def load_real_timeframe_data(unlimited: bool = False, symbol: str = 'BTCUSDT'):
    """
    从crypto_data文件夹加载真实的多时间框架数据
    :param unlimited: 是否加载全部数据（不限制条数）
    :param symbol: 交易对符号，例如 'BTCUSDT'
    :return: 包含不同时间框架数据的字典
    """
    data_dict = {}
    timeframes = ['1h', '4h', '1d']  # 主要分析时间框架
    
    # 提取币种名称
    coin_name = symbol.replace('USDT', '')
    
    # 构建币种特定的数据目录
    coin_data_dir = os.path.join('crypto_data', coin_name)
    
    if not os.path.exists(coin_data_dir):
        print(f"数据目录不存在: {coin_data_dir}")
        return None
    
    for tf in timeframes:
        try:
            file_path = os.path.join(coin_data_dir, f"{tf}.csv")
            print(f"正在加载 {tf} 数据from {file_path}...")
            
            if not os.path.exists(file_path):
                print(f"文件不存在: {file_path}")
                continue
            
            df = pd.read_csv(file_path)
            
            # 转换时间列
            if '开盘时间' in df.columns:
                df['开盘时间'] = pd.to_datetime(df['开盘时间'])
            
            # 修复：使用统一的数据预处理
            df = DataProcessor.ensure_numeric(df)
            
            # 数据质量验证
            quality_report = DataProcessor.validate_data_quality(df)
            if not quality_report['valid']:
                print(f"⚠️ {tf} 数据质量问题: {quality_report['issues']}")
            
            # 按时间排序
            df = df.sort_values('开盘时间')
            
            if not unlimited:
                # 可配置的数据量，默认加载更多历史数据
                max_records = {
                    '1h': 2000,   # 1小时取最近2000条（约83天）
                    '4h': 1000,   # 4小时取最近1000条（约166天）
                    '1d': 500     # 日线取最近500条（约1.4年）
                }
                
                # 如果数据总量小于限制，则使用全部数据
                max_limit = max_records.get(tf, 1000)
                if len(df) <= max_limit:
                    print(f"  使用全部 {len(df)} 条数据")
                else:
                    df = df.tail(max_limit)
                    print(f"  数据裁剪到最近 {max_limit} 条记录")
            else:
                print(f"  无限制模式: 使用全部 {len(df)} 条数据")
            
            data_dict[tf] = df
            print(f"✓ {tf} 数据加载成功: {len(df)} 条记录，时间范围: {df['开盘时间'].iloc[0]} 到 {df['开盘时间'].iloc[-1]}")
            
        except Exception as e:
            print(f"❌ 加载 {tf} 数据失败: {str(e)}")
    
    if not data_dict:
        print(f"❌ 未能加载任何数据")
        return None
    
    return data_dict


def analyze_market_decisions(data_dict: Dict[str, pd.DataFrame], symbol: str = "BTCUSDT", 
                           lookback_days: int = 200, frequency: str = "daily", use_full_history: bool = False) -> pd.DataFrame:
    """
    分析市场决策 - 生成清晰的决策表格
    :param data_dict: 多时间框架数据
    :param symbol: 交易对符号
    :param lookback_days: 回溯天数
    :param frequency: 分析频率 ("daily"=每天, "twice_daily"=每12小时, "4hourly"=每4小时)
    :return: 决策分析DataFrame
    """
    try:
        from src.strategies.config import create_strategy_config
    except ImportError:
        from config import create_strategy_config
    
    config = create_strategy_config("standard")
    analyzer = TechnicalAnalyzer(config, use_talib=True)
    
    # 取数据进行分析 - 支持完整历史数据
    main_tf = '4h' if '4h' in data_dict else '1d' if '1d' in data_dict else list(data_dict.keys())[0]
    
    if use_full_history:
        df = data_dict[main_tf]  # 使用全部历史数据
        print(f"🔍 使用完整历史数据: {len(df)} 条记录")
        print(f"📅 数据时间范围: {df['开盘时间'].iloc[0]} 到 {df['开盘时间'].iloc[-1]}")
    else:
        df = data_dict[main_tf].tail(lookback_days * (24//4 if main_tf == '4h' else 1))
        print(f"🔍 使用最近 {lookback_days} 天数据: {len(df)} 条记录")
    
    decisions = []
    
    if use_full_history:
        print(f"📈 正在分析 {symbol} 完整历史数据的市场决策...")
    else:
        print(f"📈 正在分析 {symbol} 最近 {lookback_days} 天的市场决策...")
    print(f"📊 使用 {main_tf} 数据，共 {len(df)} 条记录")
    print()
    
    # 根据频率设置分析间隔
    min_lookback = 30  # 减少最小数据要求
    
    if frequency == "daily":
        step = 6  # 每天分析一次(4小时数据，6个周期=24小时)
        freq_desc = "每天一次"
    elif frequency == "twice_daily":
        step = 3  # 每12小时分析一次(4小时数据，3个周期=12小时)
        freq_desc = "每12小时一次"  
    elif frequency == "4hourly":
        step = 1  # 每4小时分析一次
        freq_desc = "每4小时一次"
    else:
        step = 6  # 默认每天一次
        freq_desc = "每天一次(默认)"
    
    total_points = (len(df) - min_lookback) // step
    print(f"🔍 将分析 {total_points} 个时间点 ({freq_desc}决策)")
    print(f"📅 分析周期: 每{step*4}小时一次决策")
    
    for i in range(min_lookback, len(df), step):
        current_df = df.iloc[:i+1]
        
        if len(current_df) < min_lookback:
            continue
            
        try:
            # 创建简化的单时间框架分析
            result = analyzer.analyze_market(current_df, symbol)
            
            # 获取当前时间点信息
            current_time = current_df['开盘时间'].iloc[-1]
            current_price = current_df['收盘价'].iloc[-1]
            
            # 从单时间框架结果中提取决策信息
            signal_type = result.get('signal_type', 'neutral')
            signal_strength = result.get('signal_strength', 0)
            should_trade = result.get('should_trade', False)
            market_regime = result.get('market_regime', 'unknown')
            
            # 简化的强度判断
            strength_level = "强" if signal_strength > 0.7 else "中" if signal_strength > 0.3 else "弱"
            
            # 方向判断
            direction_cn = {"buy": "买入", "sell": "卖出", "neutral": "观望"}[signal_type]
            
            # 修复：更合理的风险等级计算
            # 结合市场状态和波动性评估风险
            if market_regime == "trending":
                # 趋势市场：看波动性
                risk_level = "中" if signal_strength > 0.6 else "低"
            elif market_regime == "sideways":
                # 震荡市场：一般为中等风险，除非信号很弱
                risk_level = "高" if signal_strength < 0.3 else "中"
            else:  # transition
                # 过渡期：通常较高风险
                risk_level = "高"
            
            # 建议
            recommendation = f"{'可交易' if should_trade else '观望'} ({market_regime})"
            
            decisions.append({
                '时间': current_time.strftime('%Y-%m-%d %H:%M'),
                '价格': f"${current_price:,.2f}",
                '决策': direction_cn,
                '强度': strength_level,
                '置信度': f"{signal_strength:.2f}",
                '综合得分': f"{signal_strength:.3f}",
                '风险等级': risk_level,
                '建议': recommendation
            })
            
        except Exception as e:
            # 临时打开调试，看看什么地方出错
            print(f"分析第{i}个时间点时出错: {str(e)}")
            continue
    
    return pd.DataFrame(decisions)


def print_decision_table(decisions_df: pd.DataFrame):
    """打印美观的决策表格"""
    if decisions_df.empty:
        print("❌ 没有生成决策数据")
        return
    
    print("📋 市场决策分析表格")
    print("=" * 120)
    
    # 设置pandas显示选项
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 120)
    pd.set_option('display.max_colwidth', 20)
    
    # 按决策类型分组统计
    buy_decisions = len(decisions_df[decisions_df['决策'] == '买入'])
    sell_decisions = len(decisions_df[decisions_df['决策'] == '卖出'])
    neutral_decisions = len(decisions_df[decisions_df['决策'] == '观望'])
    
    print(f"📊 决策统计: 买入 {buy_decisions} 次 | 卖出 {sell_decisions} 次 | 观望 {neutral_decisions} 次")
    print()
    
    # 显示表格
    print(decisions_df.to_string(index=False))
    
    print("\n" + "=" * 120)
    
    # 显示关键决策点
    strong_decisions = decisions_df[decisions_df['强度'] == '强']
    if not strong_decisions.empty:
        print("🎯 强信号决策点:")
        for _, row in strong_decisions.iterrows():
            emoji = "🟢" if row['决策'] == '买入' else "🔴" if row['决策'] == '卖出' else "🔵"
            print(f"  {emoji} {row['时间']} | {row['价格']} | {row['决策']} (置信度:{row['置信度']})")
    
    # 显示风险提醒
    high_risk = decisions_df[decisions_df['风险等级'] == '高']
    if not high_risk.empty:
        print(f"\n⚠️  高风险时段 ({len(high_risk)} 次):")
        for _, row in high_risk.tail(3).iterrows():  # 只显示最近3次
            print(f"  🔴 {row['时间']} | {row['价格']} | {row['建议']}")


if __name__ == "__main__":
    try:
        from src.strategies.config import create_strategy_config
        from src.strategies.divergence_analyzer import load_bitcoin_data
    except ImportError:
        from config import create_strategy_config
        from divergence_analyzer import load_bitcoin_data
    import pandas as pd
    import os
    
    print("=" * 80)
    print("📊 比特币市场决策分析")
    print("=" * 80)
    
    # 环境检测
    EnvironmentChecker.print_environment_report()
    
    # 加载数据 - 支持无限制模式
    print("🔧 数据加载选项:")
    print("  1. 标准模式 (最近1-2千条数据)")
    print("  2. 无限制模式 (全部历史数据)")
    
    # 这里可以设置为True来加载全部数据
    unlimited_mode = False  # 改为True来分析全部数据
    coin = 'ETHUSDT'
    
    data_dict = load_real_timeframe_data(unlimited=unlimited_mode, symbol=coin)
    
    if data_dict:
        print("✅ 数据加载成功")
        print()
        
        # 生成决策分析表格 - 现在分析更多时间点
        if unlimited_mode:
            # 无限制模式：分析完整历史数据
            decisions_df = analyze_market_decisions(data_dict, coin, use_full_history=True, frequency="4hourly")
        else:
            # 标准模式：只分析最近数据
            lookback_days = 200
            decisions_df = analyze_market_decisions(data_dict, coin, lookback_days=lookback_days, frequency="4hourly")
        
        # 打印美观的决策表格
        print_decision_table(decisions_df)
        
        # 创建decisions文件夹（如果不存在）
        decisions_dir = "decisions"
        os.makedirs(decisions_dir, exist_ok=True)
        
        # 提取币种名称
        coin_name = coin.replace('USDT', '')
        
        # 保存决策表格到CSV
        try:
            # 构建保存路径
            file_path = os.path.join(decisions_dir, f"{coin_name}_decisions.csv")
            decisions_df.to_csv(file_path, index=False, encoding='utf-8-sig')
            print(f"\n💾 决策分析表已保存至: {file_path}")
        except Exception as e:
            print(f"❌ 保存决策表时出错: {str(e)}")
    else:
        print("❌ 无法加载数据")
