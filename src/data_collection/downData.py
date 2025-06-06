import requests
import pandas as pd
from datetime import datetime
import time
import os
import pytz

def get_binance_klines(symbol='BTCUSDT', interval='1h', limit=1000, end_time=None, start_time=None):
    """
    获取币安K线数据
    :param symbol: 交易对，例如 'BTCUSDT'
    :param interval: K线间隔，例如 '1h'（1小时）
    :param limit: 获取的K线数量（最大1000）
    :param end_time: 结束时间戳（毫秒）
    :param start_time: 开始时间戳（毫秒）
    :return: DataFrame格式的K线数据
    """
    url = 'https://api.binance.com/api/v3/klines'
    params = {
        'symbol': symbol,
        'interval': interval,
        'limit': limit
    }
    
    if end_time:
        params['endTime'] = end_time
    if start_time:
        params['startTime'] = start_time
    
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        
        # 将数据转换为DataFrame
        df = pd.DataFrame(response.json(), columns=[
            '开盘时间', '开盘价', '最高价', '最低价', '收盘价', '成交量',
            '收盘时间', '成交额', '成交笔数', '主动买入成交量', '主动买入成交额', '忽略'
        ])
        
        # 转换数据类型
        beijing_tz = pytz.timezone('Asia/Shanghai')
        df['开盘时间'] = pd.to_datetime(df['开盘时间'], unit='ms', utc=True).dt.tz_convert(beijing_tz)
        df['收盘时间'] = pd.to_datetime(df['收盘时间'], unit='ms', utc=True).dt.tz_convert(beijing_tz)
        for col in ['开盘价', '最高价', '最低价', '收盘价', '成交量', '成交额']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
        return df
        
    except requests.exceptions.RequestException as e:
        print(f"请求发生错误: {str(e)}")
        return None
    except requests.exceptions.HTTPError as e:
        print(f"HTTP错误: {str(e)}")
        print(f"响应内容: {response.text}")
        return None

def load_existing_data(filepath):
    """
    加载现有的CSV文件数据
    :param filepath: 文件路径
    :return: DataFrame或None
    """
    if os.path.exists(filepath):
        try:
            df = pd.read_csv(filepath, encoding='utf-8-sig')
            beijing_tz = pytz.timezone('Asia/Shanghai')
            # 如果时间列没有时区信息，假设是北京时间
            df['开盘时间'] = pd.to_datetime(df['开盘时间'])
            df['收盘时间'] = pd.to_datetime(df['收盘时间'])
            
            # 如果时间是naive（没有时区），添加北京时区
            if df['开盘时间'].dt.tz is None:
                df['开盘时间'] = df['开盘时间'].dt.tz_localize(beijing_tz)
            if df['收盘时间'].dt.tz is None:
                df['收盘时间'] = df['收盘时间'].dt.tz_localize(beijing_tz)
                
            print(f"加载现有数据文件: {filepath}")
            print(f"现有数据条数: {len(df)}")
            if len(df) > 0:
                print(f"数据时间范围: {df['开盘时间'].min().strftime('%Y-%m-%d %H:%M %Z')} 到 {df['开盘时间'].max().strftime('%Y-%m-%d %H:%M %Z')}")
            return df
        except Exception as e:
            print(f"加载现有文件失败: {e}")
            return None
    return None

def get_complete_historical_data(symbol='BTCUSDT', interval='1h', data_dir='crypto_data'):
    """
    获取完整的历史数据，支持增量更新
    :param symbol: 交易对
    :param interval: 时间间隔
    :param data_dir: 数据存储目录
    :return: DataFrame格式的完整K线数据
    """
    # 创建数据目录
    os.makedirs(data_dir, exist_ok=True)
    
    # 文件路径
    filename = f"{symbol}_{interval}.csv"
    filepath = os.path.join(data_dir, filename)
    
    # 加载现有数据
    existing_df = load_existing_data(filepath)
    
    all_data = []
    total_count = 0
    
    if existing_df is not None and len(existing_df) > 0:
        # 有现有数据，从最新数据开始更新
        all_data.append(existing_df)
        total_count = len(existing_df)
        latest_time = existing_df['开盘时间'].max()
        # 转换为UTC时间戳用于API调用
        start_time = int(latest_time.tz_convert(pytz.UTC).timestamp() * 1000) + 1  # 从最新数据的下一毫秒开始
        print(f"从 {latest_time.strftime('%Y-%m-%d %H:%M %Z')} 开始增量更新...")
        
        # 获取新数据（向前获取）
        while True:
            df = get_binance_klines(symbol=symbol, interval=interval, start_time=start_time, limit=1000)
            if df is None or len(df) == 0:
                print("没有新的数据可获取")
                break
            
            # 过滤掉可能重复的数据
            df = df[df['开盘时间'] > latest_time]
            if len(df) == 0:
                print("没有新的数据可获取")
                break
            
            all_data.append(df)
            total_count += len(df)
            print(f"新增 {len(df)} 条数据，总计 {total_count} 条")
            
            # 更新start_time为当前批次最新的开盘时间的下一毫秒
            max_time = df['开盘时间'].max()
            start_time = int(max_time.tz_convert(pytz.UTC).timestamp() * 1000) + 1
            
            # 如果获取的数据少于1000条，说明已经到达最新数据
            if len(df) < 1000:
                print("已获取到最新数据")
                break
            
            time.sleep(0.5)  # 避免请求过于频繁
    
    # 获取历史数据（向后获取到最早的数据）
    print("开始获取历史数据...")
    if existing_df is not None and len(existing_df) > 0:
        earliest_time = existing_df['开盘时间'].min()
        end_time = int(earliest_time.tz_convert(pytz.UTC).timestamp() * 1000) - 1
    else:
        end_time = None  # 从最新开始向后获取
    
    historical_count = 0
    while True:
        df = get_binance_klines(symbol=symbol, interval=interval, end_time=end_time, limit=1000)
        if df is None or len(df) == 0:
            print("已到达最早的历史数据")
            break
        
        # 如果有现有数据，过滤掉重复的数据
        if existing_df is not None and len(existing_df) > 0:
            earliest_existing = existing_df['开盘时间'].min()
            df = df[df['开盘时间'] < earliest_existing]
            if len(df) == 0:
                print("历史数据已完整，无需继续获取")
                break
        
        all_data.insert(-1 if existing_df is not None else 0, df)  # 插入到现有数据之前
        historical_count += len(df)
        total_count += len(df)
        
        # 更新end_time为当前批次最早的开盘时间减1毫秒
        min_time = df['开盘时间'].min()
        end_time = int(min_time.tz_convert(pytz.UTC).timestamp() * 1000) - 1
        
        print(f"获取历史数据 {len(df)} 条，历史数据总计 {historical_count} 条，全部数据 {total_count} 条")
        
        # 如果获取的数据少于1000条，说明已经到达最早的数据
        if len(df) < 1000:
            print("已到达最早的历史数据")
            break
        
        time.sleep(0.5)  # 避免请求过于频繁

    if all_data:
        # 合并所有数据
        final_df = pd.concat(all_data, ignore_index=True)
        final_df = final_df.drop_duplicates(subset=['开盘时间'])
        final_df = final_df.sort_values('开盘时间').reset_index(drop=True)
        
        # 保存到CSV文件时，先转换时间格式为字符串以避免时区问题
        save_df = final_df.copy()
        save_df['开盘时间'] = final_df['开盘时间'].dt.strftime('%Y-%m-%d %H:%M:%S')
        save_df['收盘时间'] = final_df['收盘时间'].dt.strftime('%Y-%m-%d %H:%M:%S')
        save_df.to_csv(filepath, index=False, encoding='utf-8-sig')
        
        print(f"\n数据已保存到 {filepath}")
        print(f"总共 {len(final_df)} 条不重复的数据")
        
        # 显示数据概览
        if len(final_df) > 0:
            print(f"数据时间范围：从 {final_df['开盘时间'].min().strftime('%Y-%m-%d %H:%M %Z')} 到 {final_df['开盘时间'].max().strftime('%Y-%m-%d %H:%M %Z')}")
            print("\n最近5条数据示例：")
            print(final_df[['开盘时间', '开盘价', '最高价', '最低价', '收盘价', '成交量']].tail().to_string())
        
        return final_df
    
    return None

if __name__ == "__main__":
    # 定义所有需要获取的时间周期
    intervals = ['15m', '30m', '1h', '2h', '4h', '6h', '8h', '12h', '1d', '3d', '1w']  # 获取所有周期
    symbol = 'BTCUSDT'
    data_dir = 'crypto_data'
    
    print(f"开始获取 {symbol} 的K线数据...")
    print(f"数据将保存在 {data_dir} 目录中")
    
    for interval in intervals:
        print(f"\n{'='*50}")
        print(f"开始处理 {interval} 周期数据...")
        print(f"{'='*50}")
        
        btc_data = get_complete_historical_data(symbol=symbol, interval=interval, data_dir=data_dir)
        
        if btc_data is not None:
            print(f"{interval} 数据处理完成")
        else:
            print(f"{interval} 数据处理失败")
        
        # 在处理下一个时间周期之前暂停一下
        time.sleep(2)
    
    print(f"\n{'='*50}")
    print("所有数据处理完成！")
    print(f"数据文件保存在 {data_dir} 目录中")
    print(f"{'='*50}")
