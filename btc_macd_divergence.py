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

# 只保留经典KDJ顶底背离检测（与divergence_analyzer.py一致）
def calculate_kdj_indicators_for_df(df, n=34, m1=3, m2=8, m3=1, m4=6, m5=1, j_period=3):
    high = df['high'].astype(float).tolist()
    low = df['low'].astype(float).tolist()
    close = df['close'].astype(float).tolist()
    llv = pd.Series(low).rolling(window=n, min_periods=1).min().tolist()
    hhv = pd.Series(high).rolling(window=n, min_periods=1).max().tolist()
    lowv = pd.Series(llv).ewm(span=m1, adjust=False).mean().tolist()
    highv = pd.Series(hhv).ewm(span=m1, adjust=False).mean().tolist()
    rsv = []
    for i in range(len(close)):
        if highv[i] == lowv[i]:
            rsv.append(50)
        else:
            rsv_val = ((close[i] - lowv[i]) / (highv[i] - lowv[i])) * 100
            rsv.append(rsv_val)
    rsv_ema = pd.Series(rsv).ewm(span=m1, adjust=False).mean().tolist()
    def SMA(data, n, m):
        result = []
        if len(data) == 0:
            return result
        sma = data[0]
        result.append(sma)
        for i in range(1, len(data)):
            sma = (m * data[i] + (n - m) * result[i-1]) / n
            result.append(sma)
        return result
    k = SMA(rsv_ema, m2, m3)
    d = SMA(k, m4, m5)
    j = [3 * k[i] - 2 * d[i] for i in range(len(k))]
    j1 = pd.Series(j).rolling(window=j_period, min_periods=1).mean().tolist()
    # 记录所有J上穿J1和J1上穿J的点
    j_cross_up_j1_indices = []
    j1_cross_up_j_indices = []
    for i in range(n, len(close)):
        # J上穿J1
        if j[i-1] < j1[i-1] and j[i] >= j1[i]:
            j_cross_up_j1_indices.append(i)
        # J1上穿J
        if j1[i-1] < j[i-1] and j1[i] >= j[i]:
            j1_cross_up_j_indices.append(i)
    # 顶底背离检测
    top_divergence = [False] * len(close)
    bottom_divergence = [False] * len(close)
    kdj_top_divergence_info = []
    kdj_bottom_divergence_info = []
    # 底背离
    for idx in range(1, len(j_cross_up_j1_indices)):
        prev = j_cross_up_j1_indices[idx-1]
        curr = j_cross_up_j1_indices[idx]
        if close[prev] > close[curr] and j[curr] > j[prev] and j[curr] < 20:
            bottom_divergence[curr] = True
            kdj_bottom_divergence_info.append((curr, prev))
    # 顶背离
    for idx in range(1, len(j1_cross_up_j_indices)):
        prev = j1_cross_up_j_indices[idx-1]
        curr = j1_cross_up_j_indices[idx]
        if close[prev] < close[curr] and j1[prev] > j1[curr] and j[curr] > 90:
            top_divergence[curr] = True
            kdj_top_divergence_info.append((curr, prev))
    df['kdj_k'] = k
    df['kdj_d'] = d
    df['kdj_j'] = j
    df['kdj_j1'] = j1
    df['kdj_top_divergence'] = top_divergence
    df['kdj_bottom_divergence'] = bottom_divergence
    return df, kdj_top_divergence_info, kdj_bottom_divergence_info

# KDJ背离专用标注函数，支持时间戳和数值索引两种格式
def plot_kdj_divergence_fixed(ax3, ohlc, df, kdj_top_divergence_info, kdj_bottom_divergence_info):
    # KDJ线
    ax3.plot(ohlc['open_time'], df['kdj_k'], label='K', color='blue', linewidth=1)
    ax3.plot(ohlc['open_time'], df['kdj_d'], label='D', color='orange', linewidth=1)
    ax3.plot(ohlc['open_time'], df['kdj_j'], label='J', color='purple', linewidth=1)
    # 标注KDJ顶背离
    if kdj_top_divergence_info is not None:
        for curr_idx, prev_idx in kdj_top_divergence_info:
            if isinstance(curr_idx, (np.datetime64, pd.Timestamp)):
                curr_idx = ohlc[ohlc['open_time'] == curr_idx].index[0]
            if isinstance(prev_idx, (np.datetime64, pd.Timestamp)):
                prev_idx = ohlc[ohlc['open_time'] == prev_idx].index[0]
            point_date = ohlc['open_time'].iloc[curr_idx]
            j_value = df['kdj_j'].iloc[curr_idx]
            ax3.plot(point_date, j_value, 'rv', markersize=8)
            prev_date = ohlc['open_time'].iloc[prev_idx]
            prev_j = df['kdj_j'].iloc[prev_idx]
            ax3.plot([prev_date, point_date], [prev_j, j_value], 'r--', linewidth=1)
            # 日期标注（上方）
            date_str = point_date.strftime('%Y-%m-%d')
            ax3.annotate(date_str, (point_date, j_value), xytext=(0, 8), textcoords='offset points', ha='center', va='bottom', fontsize=8, color='red', rotation=45)
    # 标注KDJ底背离
    if kdj_bottom_divergence_info is not None:
        for curr_idx, prev_idx in kdj_bottom_divergence_info:
            if isinstance(curr_idx, (np.datetime64, pd.Timestamp)):
                curr_idx = ohlc[ohlc['open_time'] == curr_idx].index[0]
            if isinstance(prev_idx, (np.datetime64, pd.Timestamp)):
                prev_idx = ohlc[ohlc['open_time'] == prev_idx].index[0]
            point_date = ohlc['open_time'].iloc[curr_idx]
            j_value = df['kdj_j'].iloc[curr_idx]
            ax3.plot(point_date, j_value, 'g^', markersize=8)
            prev_date = ohlc['open_time'].iloc[prev_idx]
            prev_j = df['kdj_j'].iloc[prev_idx]
            ax3.plot([prev_date, point_date], [prev_j, j_value], 'g--', linewidth=1)
            # 日期标注（下方）
            date_str = point_date.strftime('%Y-%m-%d')
            ax3.annotate(date_str, (point_date, j_value), xytext=(0, -10), textcoords='offset points', ha='center', va='top', fontsize=8, color='green', rotation=45)
    ax3.set_ylabel('KDJ', fontsize=12)
    ax3.set_xlabel('日期', fontsize=12)
    ax3.legend(loc='upper left')
    ax3.grid(True)

# 修改主绘图函数，KDJ部分调用plot_kdj_divergence_fixed

def plot_chart_with_divergence(df, top_divergence_info, bottom_divergence_info, kdj_top_divergence_info=None, kdj_bottom_divergence_info=None, title="BTC/USDT K线图与MACD/KDJ背离"):
    # 准备K线图数据
    ohlc = df.reset_index()
    ohlc['date_num'] = mdates.date2num(ohlc['open_time'])
    ohlc['kdj_j'] = df['kdj_j'].values  # 保证KDJ极值索引一致
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
    plot_kdj_divergence_fixed(ax3, ohlc, df, kdj_top_divergence_info, kdj_bottom_divergence_info)
    plt.tight_layout()
    plt.savefig('btc_macd_kdj_divergence.png', dpi=300)
    plt.show()

if __name__ == "__main__":
    window_size = 20
    file_path = 'crypto_data/BTC/1d.csv'
    df = load_data(file_path, years=1)
    df = calculate_macd(df)
    # 用KDJ交叉极值法检测背离
    df, kdj_top_divergence_info, kdj_bottom_divergence_info = calculate_kdj_indicators_for_df(df)
    df, top_divergence_info, bottom_divergence_info = detect_divergence(df)
    plot_chart_with_divergence(df, top_divergence_info, bottom_divergence_info, kdj_top_divergence_info, kdj_bottom_divergence_info)
    print(f"检测到 {len(top_divergence_info)} 个MACD顶背离点")
    print(f"检测到 {len(bottom_divergence_info)} 个MACD底背离点")
    print(f"检测到 {len(kdj_top_divergence_info)} 个KDJ顶背离点")
    print(f"检测到 {len(kdj_bottom_divergence_info)} 个KDJ底背离点") 