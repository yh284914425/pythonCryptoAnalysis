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

# 检测MACD顶底背离（以金叉/死叉为主线）
def detect_divergence(df):
    # 标记金叉和死叉
    df['golden_cross'] = (df['macd'].shift(1) < df['signal'].shift(1)) & (df['macd'] > df['signal'])
    df['death_cross'] = (df['macd'].shift(1) > df['signal'].shift(1)) & (df['macd'] < df['signal'])

    df['top_divergence'] = False
    df['bottom_divergence'] = False
    top_divergence_info = []
    bottom_divergence_info = []

    # 找出所有金叉和死叉的索引
    golden_cross_idx = df.index[df['golden_cross']].tolist()
    death_cross_idx = df.index[df['death_cross']].tolist()

    # 检查底背离（金叉）
    for i in range(1, len(golden_cross_idx)):
        prev_idx = golden_cross_idx[i-1]
        curr_idx = golden_cross_idx[i]
        prev_pos = df.index.get_loc(prev_idx)
        curr_pos = df.index.get_loc(curr_idx)
        # 当前价格创新低，MACD未创新低
        if df['low'].iloc[curr_pos] < df['low'].iloc[prev_pos] and df['macd'].iloc[curr_pos] > df['macd'].iloc[prev_pos]:
            df.loc[curr_idx, 'bottom_divergence'] = True
            bottom_divergence_info.append((curr_pos, prev_pos))

    # 检查顶背离（死叉）
    for i in range(1, len(death_cross_idx)):
        prev_idx = death_cross_idx[i-1]
        curr_idx = death_cross_idx[i]
        prev_pos = df.index.get_loc(prev_idx)
        curr_pos = df.index.get_loc(curr_idx)
        # 当前价格创新高，MACD未创新高
        if df['high'].iloc[curr_pos] > df['high'].iloc[prev_pos] and df['macd'].iloc[curr_pos] < df['macd'].iloc[prev_pos]:
            df.loc[curr_idx, 'top_divergence'] = True
            top_divergence_info.append((curr_pos, prev_pos))

    return df, top_divergence_info, bottom_divergence_info

# 新增KDJ计算函数
def calculate_kdj(df, n=9, k_period=3, d_period=3):
    low_list = df['low'].rolling(window=n, min_periods=1).min()
    high_list = df['high'].rolling(window=n, min_periods=1).max()
    rsv = (df['close'] - low_list) / (high_list - low_list) * 100
    k = rsv.ewm(com=(k_period-1), adjust=False).mean()
    d = k.ewm(com=(d_period-1), adjust=False).mean()
    j = 3 * k - 2 * d
    df['kdj_k'] = k
    df['kdj_d'] = d
    df['kdj_j'] = j
    return df

# KDJ顶底背离检测（以J线上穿/下穿D为主线）
def detect_kdj_divergence(df):
    df['kdj_top_divergence'] = False
    df['kdj_bottom_divergence'] = False
    top_divergence_info = []
    bottom_divergence_info = []
    # 找出所有J上穿D（金叉）和J下穿D（死叉）点
    golden_cross = (df['kdj_j'].shift(1) < df['kdj_d'].shift(1)) & (df['kdj_j'] > df['kdj_d'])
    death_cross = (df['kdj_j'].shift(1) > df['kdj_d'].shift(1)) & (df['kdj_j'] < df['kdj_d'])
    golden_cross_idx = df.index[golden_cross].tolist()
    death_cross_idx = df.index[death_cross].tolist()
    # 检查底背离（金叉）
    for i in range(1, len(golden_cross_idx)):
        prev_idx = golden_cross_idx[i-1]
        curr_idx = golden_cross_idx[i]
        prev_pos = df.index.get_loc(prev_idx)
        curr_pos = df.index.get_loc(curr_idx)
        if df['low'].iloc[curr_pos] < df['low'].iloc[prev_pos] and df['kdj_j'].iloc[curr_pos] > df['kdj_j'].iloc[prev_pos]:
            df.loc[curr_idx, 'kdj_bottom_divergence'] = True
            bottom_divergence_info.append((curr_pos, prev_pos))
    # 检查顶背离（死叉）
    for i in range(1, len(death_cross_idx)):
        prev_idx = death_cross_idx[i-1]
        curr_idx = death_cross_idx[i]
        prev_pos = df.index.get_loc(prev_idx)
        curr_pos = df.index.get_loc(curr_idx)
        if df['high'].iloc[curr_pos] > df['high'].iloc[prev_pos] and df['kdj_j'].iloc[curr_pos] < df['kdj_j'].iloc[prev_pos]:
            df.loc[curr_idx, 'kdj_top_divergence'] = True
            top_divergence_info.append((curr_pos, prev_pos))
    return df, top_divergence_info, bottom_divergence_info

# 修改绘图函数，增加KDJ区域
def plot_chart_with_divergence(df, top_divergence_info, bottom_divergence_info, kdj_top_divergence_info=None, kdj_bottom_divergence_info=None, title="BTC/USDT K线图与MACD/KDJ背离"):
    # 准备K线图数据
    ohlc = df.reset_index()
    ohlc['date_num'] = mdates.date2num(ohlc['open_time'])
    ohlc_data = ohlc[['date_num', 'open', 'high', 'low', 'close']].values
    # 创建图形和子图（3行）
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(16, 16), gridspec_kw={'height_ratios': [3, 1, 1]})
    # K线图
    candlestick_ohlc(ax1, ohlc_data, width=0.6, colorup='red', colordown='green', alpha=0.8)
    date_format = mdates.DateFormatter('%Y-%m-%d')
    ax1.xaxis.set_major_formatter(date_format)
    y_min, y_max = ax1.get_ylim()
    y_text_offset = (y_max - y_min) * 0.02
    for idx, prev_idx in top_divergence_info:
        point_date = ohlc['date_num'].iloc[idx]
        point_price = ohlc['high'].iloc[idx]
        ax1.plot(point_date, point_price, 'rv', markersize=10)
        date_str = ohlc['open_time'].iloc[idx].strftime('%Y-%m-%d')
        ax1.annotate(date_str, (point_date, point_price + y_text_offset), xytext=(0, 5), textcoords='offset points', ha='center', va='bottom', fontsize=8, color='red', rotation=45)
        prev_date = ohlc['date_num'].iloc[prev_idx]
        prev_price = ohlc['high'].iloc[prev_idx]
        ax1.plot([prev_date, point_date], [prev_price, point_price], 'r--', linewidth=1.5)
    for idx, prev_idx in bottom_divergence_info:
        point_date = ohlc['date_num'].iloc[idx]
        point_price = ohlc['low'].iloc[idx]
        ax1.plot(point_date, point_price, 'g^', markersize=10)
        date_str = ohlc['open_time'].iloc[idx].strftime('%Y-%m-%d')
        ax1.annotate(date_str, (point_date, point_price - y_text_offset), xytext=(0, -5), textcoords='offset points', ha='center', va='top', fontsize=8, color='green', rotation=45)
        prev_date = ohlc['date_num'].iloc[prev_idx]
        prev_price = ohlc['low'].iloc[prev_idx]
        ax1.plot([prev_date, point_date], [prev_price, point_price], 'g--', linewidth=1.5)
    ax1.set_title(title, fontsize=15)
    ax1.set_ylabel('价格', fontsize=12)
    ax1.grid(True)
    # MACD
    ax2.plot(ohlc['open_time'], df['macd'], label='MACD', color='blue', linewidth=1.5)
    ax2.plot(ohlc['open_time'], df['signal'], label='Signal', color='red', linewidth=1.5)
    pos_hist = ohlc.copy()
    pos_hist['hist'] = df['hist'].values
    pos_hist = pos_hist[pos_hist['hist'] > 0]
    neg_hist = ohlc.copy()
    neg_hist['hist'] = df['hist'].values
    neg_hist = neg_hist[neg_hist['hist'] <= 0]
    ax2.bar(pos_hist['open_time'], pos_hist['hist'], color='red', alpha=0.5, width=1)
    ax2.bar(neg_hist['open_time'], neg_hist['hist'], color='green', alpha=0.5, width=1)
    y_min_macd, y_max_macd = ax2.get_ylim()
    y_text_offset_macd = (y_max_macd - y_min_macd) * 0.05
    for idx, _ in top_divergence_info:
        point_date = ohlc['open_time'].iloc[idx]
        macd_value = df['macd'].iloc[idx]
        ax2.plot(point_date, macd_value, 'rv', markersize=8)
        date_str = ohlc['open_time'].iloc[idx].strftime('%Y-%m-%d')
        ax2.annotate(date_str, (point_date, macd_value + y_text_offset_macd), xytext=(0, 5), textcoords='offset points', ha='center', va='bottom', fontsize=8, color='red', rotation=45)
    for idx, _ in bottom_divergence_info:
        point_date = ohlc['open_time'].iloc[idx]
        macd_value = df['macd'].iloc[idx]
        ax2.plot(point_date, macd_value, 'g^', markersize=8)
        date_str = ohlc['open_time'].iloc[idx].strftime('%Y-%m-%d')
        ax2.annotate(date_str, (point_date, macd_value - y_text_offset_macd), xytext=(0, -5), textcoords='offset points', ha='center', va='top', fontsize=8, color='green', rotation=45)
    ax2.set_xlabel('日期', fontsize=12)
    ax2.set_ylabel('MACD', fontsize=12)
    ax2.grid(True)
    ax2.legend(loc='upper left')
    # KDJ
    ax3.plot(ohlc['open_time'], df['kdj_k'], label='K', color='blue', linewidth=1)
    ax3.plot(ohlc['open_time'], df['kdj_d'], label='D', color='orange', linewidth=1)
    ax3.plot(ohlc['open_time'], df['kdj_j'], label='J', color='purple', linewidth=1)
    # 标注KDJ顶底背离
    if kdj_top_divergence_info is not None:
        for idx, prev_idx in kdj_top_divergence_info:
            point_date = ohlc['open_time'].iloc[idx]
            j_value = df['kdj_j'].iloc[idx]
            ax3.plot(point_date, j_value, 'rv', markersize=8)  # 红色三角形
            prev_date = ohlc['open_time'].iloc[prev_idx]
            prev_j = df['kdj_j'].iloc[prev_idx]
            ax3.plot([prev_date, point_date], [prev_j, j_value], 'r--', linewidth=1)
    if kdj_bottom_divergence_info is not None:
        for idx, prev_idx in kdj_bottom_divergence_info:
            point_date = ohlc['open_time'].iloc[idx]
            j_value = df['kdj_j'].iloc[idx]
            ax3.plot(point_date, j_value, 'g^', markersize=8)  # 绿色三角形
            prev_date = ohlc['open_time'].iloc[prev_idx]
            prev_j = df['kdj_j'].iloc[prev_idx]
            ax3.plot([prev_date, point_date], [prev_j, j_value], 'g--', linewidth=1)
    ax3.set_ylabel('KDJ', fontsize=12)
    ax3.set_xlabel('日期', fontsize=12)
    ax3.legend(loc='upper left')
    ax3.grid(True)
    plt.tight_layout()
    plt.savefig('btc_macd_kdj_divergence.png', dpi=300)
    plt.show()

if __name__ == "__main__":
    # 设置全局变量
    window_size = 20  # 用于检测局部最大/最小值的窗口大小
    
    # 加载数据 - 只使用最近一年的数据
    file_path = 'crypto_data/BTC/1d.csv'
    df = load_data(file_path, years=1)
    
    # 计算MACD
    df = calculate_macd(df)
    
    # 新增KDJ计算
    df = calculate_kdj(df)
    
    # 检测MACD背离
    df, top_divergence_info, bottom_divergence_info = detect_divergence(df)
    
    # 检测KDJ背离
    df, kdj_top_divergence_info, kdj_bottom_divergence_info = detect_kdj_divergence(df)
    
    # 绘制图表，传入KDJ背离信息
    plot_chart_with_divergence(df, top_divergence_info, bottom_divergence_info, kdj_top_divergence_info, kdj_bottom_divergence_info)
    
    # 输出检测到的背离点数量
    print(f"检测到 {len(top_divergence_info)} 个MACD顶背离点")
    print(f"检测到 {len(bottom_divergence_info)} 个MACD底背离点")
    print(f"检测到 {len(kdj_top_divergence_info)} 个KDJ顶背离点")
    print(f"检测到 {len(kdj_bottom_divergence_info)} 个KDJ底背离点") 