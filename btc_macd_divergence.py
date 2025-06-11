import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
from mplfinance.original_flavor import candlestick_ohlc
import ta
import datetime

# 读取CSV数据
def load_data(file_path, years=1):
    # 读取CSV文件
    df = pd.read_csv(file_path)
    
    # 重命名列名为英文，方便处理
    df.columns = ['open_time', 'open', 'high', 'low', 'close', 'volume', 'close_time', 
                  'quote_volume', 'count', 'taker_buy_volume', 'taker_buy_quote_volume', 'ignore']
    
    # 将时间列转换为datetime格式
    df['open_time'] = pd.to_datetime(df['open_time'])
    df['close_time'] = pd.to_datetime(df['close_time'])
    
    # 只保留最近N年的数据
    if years > 0:
        cutoff_date = datetime.datetime.now() - datetime.timedelta(days=365 * years)
        df = df[df['open_time'] >= cutoff_date]
    
    # 设置索引
    df.set_index('open_time', inplace=True)
    
    return df

# 计算MACD指标
def calculate_macd(df):
    # 使用ta.trend计算MACD
    macd_indicator = ta.trend.MACD(close=df['close'], window_slow=26, window_fast=12, window_sign=9)
    
    # 将结果添加到DataFrame
    df['macd'] = macd_indicator.macd()
    df['signal'] = macd_indicator.macd_signal()
    df['hist'] = macd_indicator.macd_diff()
    
    return df

# 检测MACD顶底背离
def detect_divergence(df, window_size=20):
    # 添加新列以标记顶部和底部背离
    df['top_divergence'] = False
    df['bottom_divergence'] = False
    
    # 用于存储背离点的索引和数据
    top_divergence_info = []  # 存储元组 (索引, 前一个背离点索引)
    bottom_divergence_info = []  # 存储元组 (索引, 前一个背离点索引)
    
    # 检测价格和MACD的局部最大/最小值
    for i in range(window_size, len(df) - window_size):
        # 提取窗口数据
        window = df.iloc[i-window_size:i+window_size+1]
        middle_idx = window_size
        
        # 检测价格局部最高点
        if window['high'].iloc[middle_idx] == window['high'].max():
            # 向前查找MACD局部最高点
            prev_window = df.iloc[max(0, i-window_size*2):i+1]
            if len(prev_window) > 10:  # 确保有足够的数据点
                # 如果当前MACD低于前一个高点，则可能是顶背离
                if prev_window['macd'].iloc[-1] < prev_window['macd'].max() and \
                   prev_window['macd'].idxmax() != prev_window.index[-1]:
                    df.loc[window.index[middle_idx], 'top_divergence'] = True
                    
                    # 查找前一个顶背离点
                    prev_idx = prev_window['macd'].idxmax()
                    prev_point = df.index.get_loc(prev_idx)
                    
                    top_divergence_info.append((i, prev_point))
        
        # 检测价格局部最低点
        if window['low'].iloc[middle_idx] == window['low'].min():
            # 向前查找MACD局部最低点
            prev_window = df.iloc[max(0, i-window_size*2):i+1]
            if len(prev_window) > 10:  # 确保有足够的数据点
                # 如果当前MACD高于前一个低点，则可能是底背离
                if prev_window['macd'].iloc[-1] > prev_window['macd'].min() and \
                   prev_window['macd'].idxmin() != prev_window.index[-1]:
                    df.loc[window.index[middle_idx], 'bottom_divergence'] = True
                    
                    # 查找前一个底背离点
                    prev_idx = prev_window['macd'].idxmin()
                    prev_point = df.index.get_loc(prev_idx)
                    
                    bottom_divergence_info.append((i, prev_point))
    
    return df, top_divergence_info, bottom_divergence_info

# 绘制K线图和MACD指标，标注顶底背离
def plot_chart_with_divergence(df, top_divergence_info, bottom_divergence_info, title="BTC/USDT K线图与MACD背离"):
    # 准备K线图数据
    ohlc = df.reset_index()
    ohlc['date_num'] = mdates.date2num(ohlc['open_time'])
    ohlc_data = ohlc[['date_num', 'open', 'high', 'low', 'close']].values
    
    # 创建图形和子图
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12), gridspec_kw={'height_ratios': [3, 1]})
    
    # 绘制K线图
    candlestick_ohlc(ax1, ohlc_data, width=0.6, colorup='red', colordown='green', alpha=0.8)
    
    # 设置日期格式
    date_format = mdates.DateFormatter('%Y-%m-%d')
    ax1.xaxis.set_major_formatter(date_format)
    
    # 获取价格的y轴范围，用于计算文本位置
    y_min, y_max = ax1.get_ylim()
    y_text_offset = (y_max - y_min) * 0.02  # 文本偏移量为价格范围的2%
    
    # 标注顶部背离点并连线
    for idx, prev_idx in top_divergence_info:
        # 当前顶背离点
        point_date = ohlc['date_num'].iloc[idx]
        point_price = ohlc['high'].iloc[idx]
        ax1.plot(point_date, point_price, 'rv', markersize=10)  # 绘制红色三角形
        
        # 添加日期标注
        date_str = ohlc['open_time'].iloc[idx].strftime('%Y-%m-%d')
        ax1.annotate(date_str, 
                    (point_date, point_price + y_text_offset), 
                    xytext=(0, 5), 
                    textcoords='offset points',
                    ha='center', 
                    va='bottom',
                    fontsize=8,
                    color='red',
                    rotation=45)
        
        # 前一个MACD高点对应的价格点
        prev_date = ohlc['date_num'].iloc[prev_idx]
        prev_price = ohlc['high'].iloc[prev_idx]
        
        # 绘制连线
        ax1.plot([prev_date, point_date], [prev_price, point_price], 'r--', linewidth=1.5)
    
    # 标注底部背离点并连线
    for idx, prev_idx in bottom_divergence_info:
        # 当前底背离点
        point_date = ohlc['date_num'].iloc[idx]
        point_price = ohlc['low'].iloc[idx]
        ax1.plot(point_date, point_price, 'g^', markersize=10)  # 绘制绿色三角形
        
        # 添加日期标注
        date_str = ohlc['open_time'].iloc[idx].strftime('%Y-%m-%d')
        ax1.annotate(date_str, 
                    (point_date, point_price - y_text_offset), 
                    xytext=(0, -5), 
                    textcoords='offset points',
                    ha='center', 
                    va='top',
                    fontsize=8,
                    color='green',
                    rotation=45)
        
        # 前一个MACD低点对应的价格点
        prev_date = ohlc['date_num'].iloc[prev_idx]
        prev_price = ohlc['low'].iloc[prev_idx]
        
        # 绘制连线
        ax1.plot([prev_date, point_date], [prev_price, point_price], 'g--', linewidth=1.5)
    
    # 设置K线图标题和标签
    ax1.set_title(title, fontsize=15)
    ax1.set_ylabel('价格', fontsize=12)
    ax1.grid(True)
    
    # 绘制MACD指标
    ax2.plot(ohlc['open_time'], df['macd'], label='MACD', color='blue', linewidth=1.5)
    ax2.plot(ohlc['open_time'], df['signal'], label='Signal', color='red', linewidth=1.5)
    
    # 绘制MACD柱状图
    pos_hist = ohlc.copy()
    pos_hist['hist'] = df['hist'].values
    pos_hist = pos_hist[pos_hist['hist'] > 0]
    
    neg_hist = ohlc.copy()
    neg_hist['hist'] = df['hist'].values
    neg_hist = neg_hist[neg_hist['hist'] <= 0]
    
    ax2.bar(pos_hist['open_time'], pos_hist['hist'], color='red', alpha=0.5, width=1)
    ax2.bar(neg_hist['open_time'], neg_hist['hist'], color='green', alpha=0.5, width=1)
    
    # 获取MACD的y轴范围，用于计算文本位置
    y_min_macd, y_max_macd = ax2.get_ylim()
    y_text_offset_macd = (y_max_macd - y_min_macd) * 0.05  # 文本偏移量为MACD范围的5%
    
    # 在MACD图上标注背离点
    for idx, _ in top_divergence_info:
        point_date = ohlc['open_time'].iloc[idx]
        macd_value = df['macd'].iloc[idx]
        ax2.plot(point_date, macd_value, 'rv', markersize=8)
        
        # 添加日期标注
        date_str = ohlc['open_time'].iloc[idx].strftime('%Y-%m-%d')
        ax2.annotate(date_str, 
                    (point_date, macd_value + y_text_offset_macd), 
                    xytext=(0, 5), 
                    textcoords='offset points',
                    ha='center', 
                    va='bottom',
                    fontsize=8,
                    color='red',
                    rotation=45)
    
    for idx, _ in bottom_divergence_info:
        point_date = ohlc['open_time'].iloc[idx]
        macd_value = df['macd'].iloc[idx]
        ax2.plot(point_date, macd_value, 'g^', markersize=8)
        
        # 添加日期标注
        date_str = ohlc['open_time'].iloc[idx].strftime('%Y-%m-%d')
        ax2.annotate(date_str, 
                    (point_date, macd_value - y_text_offset_macd), 
                    xytext=(0, -5), 
                    textcoords='offset points',
                    ha='center', 
                    va='top',
                    fontsize=8,
                    color='green',
                    rotation=45)
    
    # 设置MACD图标签
    ax2.set_xlabel('日期', fontsize=12)
    ax2.set_ylabel('MACD', fontsize=12)
    ax2.grid(True)
    ax2.legend(loc='upper left')
    
    # 调整布局和显示图形
    plt.tight_layout()
    plt.savefig('btc_macd_divergence.png', dpi=300)
    plt.show()

if __name__ == "__main__":
    # 设置全局变量
    window_size = 20  # 用于检测局部最大/最小值的窗口大小
    
    # 加载数据 - 只使用最近一年的数据
    file_path = 'crypto_data/BTC/1d.csv'
    df = load_data(file_path, years=1)
    
    # 计算MACD
    df = calculate_macd(df)
    
    # 检测背离
    df, top_divergence_info, bottom_divergence_info = detect_divergence(df, window_size)
    
    # 绘制图表
    plot_chart_with_divergence(df, top_divergence_info, bottom_divergence_info)
    
    # 输出检测到的背离点数量
    print(f"检测到 {len(top_divergence_info)} 个顶背离点")
    print(f"检测到 {len(bottom_divergence_info)} 个底背离点") 