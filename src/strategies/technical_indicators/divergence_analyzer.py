import pandas as pd
import numpy as np
import os

class DivergenceAnalyzer:
    def __init__(self):
        """初始化背离分析器"""
        pass
    
    def MA(self, data, period):
        """简单移动平均"""
        return pd.Series(data).rolling(window=period).mean().fillna(0).tolist()
    
    def EMA(self, data, period):
        """指数移动平均"""
        return pd.Series(data).ewm(span=period, adjust=False).mean().tolist()
    
    def SMA(self, data, n, m):
        """平滑移动平均 (SMA)"""
        result = []
        if len(data) == 0:
            return result
        
        sma = data[0]
        result.append(sma)
        
        for i in range(1, len(data)):
            sma = (m * data[i] + (n - m) * result[i-1]) / n
            result.append(sma)
        
        return result
    
    def HHV(self, data, period):
        """最高值"""
        result = []
        for i in range(len(data)):
            start_idx = max(0, i - period + 1)
            max_val = max(data[start_idx:i+1])
            result.append(max_val)
        return result
    
    def LLV(self, data, period):
        """最低值"""
        result = []
        for i in range(len(data)):
            start_idx = max(0, i - period + 1)
            min_val = min(data[start_idx:i+1])
            result.append(min_val)
        return result
    
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
        
        # 提取价格数据
        high = [float(k['最高价']) for k in klines_data]
        low = [float(k['最低价']) for k in klines_data]
        close = [float(k['收盘价']) for k in klines_data]
        
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
        llv = self.LLV(low, n)
        hhv = self.HHV(high, n)
        lowv = self.EMA(llv, m1)
        highv = self.EMA(hhv, m1)
        
        # 计算RSV
        rsv = []
        for i in range(len(klines_data)):
            if highv[i] == lowv[i]:
                rsv.append(50)
            else:
                rsv_val = ((close[i] - lowv[i]) / (highv[i] - lowv[i])) * 100
                rsv.append(rsv_val)
        
        rsv_ema = self.EMA(rsv, m1)
        
        # 计算K、D、J值
        k = self.SMA(rsv_ema, m2, m3)
        d = self.SMA(k, m4, m5)
        j = [3 * k[i] - 2 * d[i] for i in range(len(k))]
        j1 = self.MA(j, j_period)
        
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
                    if (close[last_cross_index] > close[i] and 
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
                    if (close[last_cross_index] < close[i] and 
                        j1[last_cross_index] > j1[i] and 
                        j[i] > 90):
                        top_divergence[i] = True
        
        return {
            'j': j,
            'j1': j1,
            'top_divergence': top_divergence,
            'bottom_divergence': bottom_divergence
        }

def load_bitcoin_data(data_dir='crypto_data', symbol='BTCUSDT', interval='1d'):
    """加载比特币数据"""
    # 提取币种名称
    coin_name = symbol.replace('USDT', '')
    
    # 构建币种特定的数据目录和文件路径
    coin_data_dir = os.path.join(data_dir, coin_name)
    filename = f"{interval}.csv"
    filepath = os.path.join(coin_data_dir, filename)
    
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

def analyze_divergence():
    """分析比特币KDJ背离信号"""
    print("开始加载比特币数据...")
    data = load_bitcoin_data()
    if not data:
        print("数据加载失败")
        return
    
    print(f"成功加载数据: {len(data)} 条记录")
    
    # 创建分析器
    analyzer = DivergenceAnalyzer()
    
    # 计算KDJ指标和背离信号
    print("计算KDJ指标和背离信号...")
    result = analyzer.calculate_kdj_indicators(data)
    
    # 统计背离信号
    top_divergence_count = sum(result["top_divergence"])
    bottom_divergence_count = sum(result["bottom_divergence"])
    
    print(f"顶背离信号数: {top_divergence_count}")
    print(f"底背离信号数: {bottom_divergence_count}")
    
    # 收集所有顶背离信号
    top_divergences = []
    for i in range(len(data)):
        if result['top_divergence'][i]:
            top_divergences.append({
                "日期": data[i]["开盘时间"],
                "价格": float(data[i]["收盘价"]),
                "J值": result['j'][i],
                "J1值": result['j1'][i],
                "索引": i
            })
    
    # 收集所有底背离信号
    bottom_divergences = []
    for i in range(len(data)):
        if result['bottom_divergence'][i]:
            bottom_divergences.append({
                "日期": data[i]["开盘时间"],
                "价格": float(data[i]["收盘价"]),
                "J值": result['j'][i],
                "J1值": result['j1'][i],
                "索引": i
            })
    
    # 打印顶背离信号详情
    print("\n顶背离信号详情:")
    for div in top_divergences:
        print(f"日期: {div['日期']}, 价格: {div['价格']}, J值: {div['J值']:.2f}, J1值: {div['J1值']:.2f}")
    
    # 打印底背离信号详情
    print("\n底背离信号详情:")
    for div in bottom_divergences:
        print(f"日期: {div['日期']}, 价格: {div['价格']}, J值: {div['J值']:.2f}, J1值: {div['J1值']:.2f}")
    
    # 分析顶背离后的价格变化
    print("\n顶背离后的价格变化分析:")
    price_changes_after_top = []
    for div in top_divergences:
        idx = div["索引"]
        if idx + 10 < len(data):  # 确保有足够的后续数据
            future_price = float(data[idx + 10]["收盘价"])
            price_change = (future_price - div["价格"]) / div["价格"] * 100
            price_changes_after_top.append(price_change)
            print(f"日期: {div['日期']}, 10天后价格变化: {price_change:.2f}%")
    
    if price_changes_after_top:
        avg_change = sum(price_changes_after_top) / len(price_changes_after_top)
        print(f"顶背离后10天平均价格变化: {avg_change:.2f}%")
        negative_count = sum(1 for change in price_changes_after_top if change < 0)
        print(f"顶背离后10天价格下跌概率: {negative_count / len(price_changes_after_top) * 100:.2f}%")
    
    # 分析底背离后的价格变化
    print("\n底背离后的价格变化分析:")
    price_changes_after_bottom = []
    for div in bottom_divergences:
        idx = div["索引"]
        if idx + 10 < len(data):  # 确保有足够的后续数据
            future_price = float(data[idx + 10]["收盘价"])
            price_change = (future_price - div["价格"]) / div["价格"] * 100
            price_changes_after_bottom.append(price_change)
            print(f"日期: {div['日期']}, 10天后价格变化: {price_change:.2f}%")
    
    if price_changes_after_bottom:
        avg_change = sum(price_changes_after_bottom) / len(price_changes_after_bottom)
        print(f"底背离后10天平均价格变化: {avg_change:.2f}%")
        positive_count = sum(1 for change in price_changes_after_bottom if change > 0)
        print(f"底背离后10天价格上涨概率: {positive_count / len(price_changes_after_bottom) * 100:.2f}%")
    
    # 分析最近90天的数据
    print("\n最近90天的背离分析:")
    recent_data = data[-90:]
    recent_result = analyzer.calculate_kdj_indicators(recent_data)
    recent_top = sum(recent_result["top_divergence"])
    recent_bottom = sum(recent_result["bottom_divergence"])
    print(f"最近90天顶背离信号数: {recent_top}")
    print(f"最近90天底背离信号数: {recent_bottom}")
    
    # 分析2024年的数据
    print("\n2024年的背离分析:")
    data_2024 = [d for d in data if "2024" in d["开盘时间"]]
    if data_2024:
        result_2024 = analyzer.calculate_kdj_indicators(data_2024)
        top_2024 = sum(result_2024["top_divergence"])
        bottom_2024 = sum(result_2024["bottom_divergence"])
        print(f"2024年顶背离信号数: {top_2024}")
        print(f"2024年底背离信号数: {bottom_2024}")
        
        # 打印2024年的顶背离信号
        print("\n2024年顶背离信号:")
        for i in range(len(data_2024)):
            if result_2024['top_divergence'][i]:
                date_str = data_2024[i]["开盘时间"]
                price = data_2024[i]["收盘价"]
                j_val = result_2024['j'][i]
                j1_val = result_2024['j1'][i]
                print(f"日期: {date_str}, 价格: {price}, J值: {j_val:.2f}, J1值: {j1_val:.2f}")
        
        # 打印2024年的底背离信号
        print("\n2024年底背离信号:")
        for i in range(len(data_2024)):
            if result_2024['bottom_divergence'][i]:
                date_str = data_2024[i]["开盘时间"]
                price = data_2024[i]["收盘价"]
                j_val = result_2024['j'][i]
                j1_val = result_2024['j1'][i]
                print(f"日期: {date_str}, 价格: {price}, J值: {j_val:.2f}, J1值: {j1_val:.2f}")
    else:
        print("没有2024年的数据")
    
    print("\n分析完成!")
    
    print("\nKDJ背离交易提示:")
    print("1. 顶背离通常是看跌信号，表明价格可能即将下跌")
    print("2. 底背离通常是看涨信号，表明价格可能即将上涨")
    print("3. 背离信号最好与其他技术指标结合使用，增加交易确认度")
    print("4. 在强势趋势中，背离信号可能会连续出现，需要谨慎对待")

# 如果直接运行此文件，则执行分析
if __name__ == "__main__":
    analyze_divergence() 