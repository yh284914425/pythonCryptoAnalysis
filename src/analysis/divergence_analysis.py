import pandas as pd
import numpy as np
from datetime import datetime
import pytz
import os

class DivergenceAnalyzer:
    def __init__(self):
        """初始化背离分析器"""
        pass
    
    def MA(self, data, period):
        """简单移动平均"""
        return pd.Series(data).rolling(window=period, min_periods=1).mean().fillna(0).tolist()
    
    def EMA(self, data, period):
        """指数移动平均"""
        return pd.Series(data).ewm(span=period, adjust=False).mean().tolist()
    
    def SMA(self, data, n, m):
        """平滑移动平均 (SMA)"""
        series = pd.Series(data)
        # 初始化第一个值
        result = [series.iloc[0]]
        # 应用SMA公式: (m * current + (n - m) * previous) / n
        for i in range(1, len(series)):
            sma = (m * series.iloc[i] + (n - m) * result[i-1]) / n
            result.append(sma)
        return result
    
    def HHV(self, data, period):
        """最高值"""
        return pd.Series(data).rolling(window=period, min_periods=1).max().tolist()
    
    def LLV(self, data, period):
        """最低值"""
        return pd.Series(data).rolling(window=period, min_periods=1).min().tolist()
    
    def CROSS(self, a1, b1, a2, b2):
        """交叉判断：前一根a1<=b1，当前a2>b2"""
        return a1 <= b1 and a2 > b2
    
    def calculate_kdj_indicators(self, klines_data, params=None):
        """
        计算KDJ指标和顶底背离
        :param klines_data: K线数据，格式为list of dict，包含high, low, close等字段
        :param params: KDJ参数字典，包含k, d, j等参数。如果为None，则使用默认参数
        :return: 包含j, j1, 顶部背离, 底部背离的字典
        """
        if len(klines_data) < 34:
            print("数据量不足，需要至少34根K线")
            return None
        
        # 提取价格数据并转换为DataFrame
        df = pd.DataFrame({
            'high': [float(k['最高价']) for k in klines_data],
            'low': [float(k['最低价']) for k in klines_data],
            'close': [float(k['收盘价']) for k in klines_data]
        })
        
        # 使用默认参数或传入的参数
        n = 34  # RSV周期
        m1 = 3  # RSV平滑
        m2 = 8  # K值周期
        m3 = 1  # K值权重
        m4 = 6  # D值周期
        m5 = 1  # D值权重
        j_period = 3  # J1周期
        
        # 如果传入了参数，则使用传入的参数
        if params:
            if "k" in params:
                m2 = params["k"]
            if "d" in params:
                m4 = params["d"]
            if "j" in params:
                j_period = params["j"]
        
        # 计算LLV和HHV
        df['llv'] = df['low'].rolling(window=n, min_periods=1).min()
        df['hhv'] = df['high'].rolling(window=n, min_periods=1).max()
        df['lowv'] = df['llv'].ewm(span=m1, adjust=False).mean()
        df['highv'] = df['hhv'].ewm(span=m1, adjust=False).mean()
        
        # 计算RSV
        df['rsv'] = np.where(
            df['highv'] == df['lowv'],
            50,
            100 * (df['close'] - df['lowv']) / (df['highv'] - df['lowv'])
        )
        
        df['rsv_ema'] = df['rsv'].ewm(span=m1, adjust=False).mean()
        
        # 计算K、D、J值
        # 由于SMA有特殊计算，仍需使用原方法
        k = self.SMA(df['rsv_ema'].tolist(), m2, m3)
        d = self.SMA(k, m4, m5)
        
        # 计算J值和J1
        df['k'] = k
        df['d'] = d
        df['j'] = 3 * df['k'] - 2 * df['d']
        df['j1'] = df['j'].rolling(window=j_period, min_periods=1).mean()
        
        # 转换为列表方便后续处理
        j = df['j'].tolist()
        j1 = df['j1'].tolist()
        
        # 初始化背离数组
        top_divergence = [False] * len(klines_data)
        bottom_divergence = [False] * len(klines_data)
        
        # 检测背离
        for i in range(n, len(klines_data)):
            # J上穿J1
            j_cross_up_j1 = self.CROSS(j[i-1], j1[i-1], j[i], j1[i])
            # J1上穿J
            j1_cross_up_j = self.CROSS(j1[i-1], j[i-1], j1[i], j[i])
            
            # 底部背离检测
            if j_cross_up_j1:
                # 寻找上一个J上穿J1的位置
                last_cross_index = -1
                for k_idx in range(i - 1, n - 1, -1):
                    if self.CROSS(j[k_idx-1], j1[k_idx-1], j[k_idx], j1[k_idx]):
                        last_cross_index = k_idx
                        break
                
                if last_cross_index != -1:
                    # 判断底部背离条件
                    if (df['close'].iloc[last_cross_index] > df['close'].iloc[i] and 
                        j[i] > j[last_cross_index] and 
                        j[i] < 20):
                        bottom_divergence[i] = True
            
            # 顶部背离检测
            if j1_cross_up_j:
                # 寻找上一个J1上穿J的位置
                last_cross_index = -1
                for k_idx in range(i - 1, n - 1, -1):
                    if self.CROSS(j1[k_idx-1], j[k_idx-1], j1[k_idx], j[k_idx]):
                        last_cross_index = k_idx
                        break
                
                if last_cross_index != -1:
                    # 判断顶部背离条件
                    if (df['close'].iloc[last_cross_index] < df['close'].iloc[i] and 
                        j1[last_cross_index] > j1[i] and 
                        j[i] > 90):
                        top_divergence[i] = True
        
        return {
            'j': j,
            'j1': j1,
            'top_divergence': top_divergence,
            'bottom_divergence': bottom_divergence
        }

def load_bitcoin_data(data_dir='crypto_data', symbol='BTC', interval='1d'):
    """加载比特币数据"""
    filename = f"{interval}.csv"
    filepath = os.path.join(data_dir, symbol, filename)
    
    if not os.path.exists(filepath):
        print(f"数据文件不存在: {filepath}")
        print("请先运行 downData.py 下载数据")
        return None
    
    try:
        df = pd.read_csv(filepath, encoding='utf-8-sig')
        print(f"成功加载数据: {len(df)} 条记录")
        print(f"数据时间范围: {df['开盘时间'].iloc[0]} 到 {df['开盘时间'].iloc[-1]}")
        return df.to_dict('records')
    except Exception as e:
        print(f"加载数据失败: {e}")
        return None

def analyze_divergence(start_date=None, end_date=None):
    """
    分析比特币日线数据的顶底背离
    :param start_date: 开始日期，格式 'YYYY-MM-DD'
    :param end_date: 结束日期，格式 'YYYY-MM-DD'
    """
    print("🔍 开始分析比特币日线数据的顶底背离...")
    
    # 加载数据
    klines_data = load_bitcoin_data()
    if not klines_data:
        return
    
    # 过滤日期范围
    if start_date or end_date:
        filtered_data = []
        for k in klines_data:
            kline_date = pd.to_datetime(k['开盘时间']).strftime('%Y-%m-%d')
            if start_date and kline_date < start_date:
                continue
            if end_date and kline_date > end_date:
                break
            filtered_data.append(k)
        klines_data = filtered_data
        print(f"过滤后数据范围: {len(klines_data)} 条记录")
    
    # 创建分析器并计算指标
    analyzer = DivergenceAnalyzer()
    result = analyzer.calculate_kdj_indicators(klines_data)
    
    if not result:
        return
    
    # 提取背离结果
    top_divergence = result['top_divergence']
    bottom_divergence = result['bottom_divergence']
    j = result['j']
    j1 = result['j1']
    
    # 收集背离日期
    top_divergence_dates = []
    bottom_divergence_dates = []
    
    for i, k in enumerate(klines_data):
        if top_divergence[i]:
            top_divergence_dates.append({
                'date': k['开盘时间'],
                'price': float(k['收盘价']),
                'j': j[i],
                'j1': j1[i]
            })
        
        if bottom_divergence[i]:
            bottom_divergence_dates.append({
                'date': k['开盘时间'],
                'price': float(k['收盘价']),
                'j': j[i],
                'j1': j1[i]
            })
    
    # 输出结果
    print(f"\n📈 顶部背离信号 (共{len(top_divergence_dates)}个):")
    print("=" * 80)
    if top_divergence_dates:
        for signal in top_divergence_dates:
            date_str = pd.to_datetime(signal['date']).strftime('%Y-%m-%d')
            print(f"日期: {date_str}, 价格: ${signal['price']:,.2f}, J: {signal['j']:.2f}, J1: {signal['j1']:.2f}")
    else:
        print("未检测到顶部背离信号")
    
    print(f"\n📉 底部背离信号 (共{len(bottom_divergence_dates)}个):")
    print("=" * 80)
    if bottom_divergence_dates:
        for signal in bottom_divergence_dates:
            date_str = pd.to_datetime(signal['date']).strftime('%Y-%m-%d')
            print(f"日期: {date_str}, 价格: ${signal['price']:,.2f}, J: {signal['j']:.2f}, J1: {signal['j1']:.2f}")
    else:
        print("未检测到底部背离信号")
    
    # 统计信息
    total_signals = len(top_divergence_dates) + len(bottom_divergence_dates)
    print(f"\n📊 统计信息:")
    print("=" * 80)
    print(f"总数据量: {len(klines_data)} 天")
    print(f"顶部背离: {len(top_divergence_dates)} 次")
    print(f"底部背离: {len(bottom_divergence_dates)} 次")
    print(f"背离频率: {total_signals/len(klines_data)*100:.2f}%")
    
    return {
        'top_divergence': top_divergence_dates,
        'bottom_divergence': bottom_divergence_dates,
        'total_signals': total_signals
    }

def analyze_recent_divergence(days=90):
    """分析最近N天的背离信号"""
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - pd.Timedelta(days=days)).strftime('%Y-%m-%d')
    
    print(f"🔍 分析最近{days}天的背离信号 ({start_date} 到 {end_date})")
    return analyze_divergence(start_date, end_date)

if __name__ == "__main__":
    print("🚀 比特币背离分析工具")
    print("=" * 50)
    
    # 分析所有历史数据
    print("\n1️⃣ 分析所有历史数据:")
    all_result = analyze_divergence()
    