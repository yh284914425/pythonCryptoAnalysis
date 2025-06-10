from divergence_analysis import DivergenceAnalyzer, load_bitcoin_data
import pandas as pd
import os
from datetime import datetime

def analyze_all_timeframes():
    """分析所有时间周期的背离信号"""
    
    # 获取所有可用的数据文件
    data_dir = 'crypto_data'
    symbol = 'BTC'
    available_files = []
    
    # 检查BTC目录
    btc_dir = os.path.join(data_dir, symbol)
    if os.path.exists(btc_dir):
        for file in os.listdir(btc_dir):
            if file.endswith('.csv'):
                interval = file.replace('.csv', '')
                available_files.append(interval)
    
    # 按时间周期排序（从小到大）
    interval_order = ['15m', '30m', '1h', '2h', '4h', '6h', '8h', '12h', '1d', '3d', '1w']
    available_files = [interval for interval in interval_order if interval in available_files]
    
    print("🚀 多时间周期背离分析")
    print("=" * 80)
    print(f"发现 {len(available_files)} 个时间周期的数据文件")
    print(f"可用周期: {', '.join(available_files)}")
    print("=" * 80)
    
    all_results = {}
    
    for interval in available_files:
        print(f"\n📊 分析 {interval} 周期数据...")
        print("-" * 50)
        
        # 加载特定周期的数据
        klines_data = load_bitcoin_data(data_dir='crypto_data', symbol='BTC', interval=interval)
        
        if not klines_data:
            print(f"❌ 无法加载 {interval} 数据")
            continue
        
        if len(klines_data) < 34:
            print(f"❌ {interval} 数据量不足 (少于34条)")
            continue
        
        # 创建分析器并计算指标
        analyzer = DivergenceAnalyzer()
        result = analyzer.calculate_kdj_indicators(klines_data)
        
        if not result:
            print(f"❌ {interval} 计算失败")
            continue
        
        # 统计背离信号
        top_count = sum(result['top_divergence'])
        bottom_count = sum(result['bottom_divergence'])
        total_signals = top_count + bottom_count
        frequency = total_signals / len(klines_data) * 100
        
        # 找出最近的背离信号
        recent_top = None
        recent_bottom = None
        
        for i in reversed(range(len(klines_data))):
            if result['top_divergence'][i] and not recent_top:
                recent_top = {
                    'date': klines_data[i]['开盘时间'],
                    'price': float(klines_data[i]['收盘价']),
                    'j': result['j'][i]
                }
            if result['bottom_divergence'][i] and not recent_bottom:
                recent_bottom = {
                    'date': klines_data[i]['开盘时间'],
                    'price': float(klines_data[i]['收盘价']),
                    'j': result['j'][i]
                }
        
        # 保存结果
        all_results[interval] = {
            'total_data': len(klines_data),
            'top_count': top_count,
            'bottom_count': bottom_count,
            'frequency': frequency,
            'recent_top': recent_top,
            'recent_bottom': recent_bottom,
            'data_range': f"{klines_data[0]['开盘时间']} 到 {klines_data[-1]['开盘时间']}"
        }
        
        # 输出该周期的摘要
        print(f"✅ 数据量: {len(klines_data)} 条")
        print(f"📈 顶部背离: {top_count} 次")
        print(f"📉 底部背离: {bottom_count} 次")
        print(f"📊 背离频率: {frequency:.2f}%")
        
        if recent_top:
            date_str = pd.to_datetime(recent_top['date']).strftime('%Y-%m-%d %H:%M')
            print(f"🔴 最近顶背离: {date_str} (${recent_top['price']:,.0f})")
        
        if recent_bottom:
            date_str = pd.to_datetime(recent_bottom['date']).strftime('%Y-%m-%d %H:%M')
            print(f"🟢 最近底背离: {date_str} (${recent_bottom['price']:,.0f})")
    
    # 生成汇总报告
    print("\n" + "=" * 80)
    print("📋 汇总报告")
    print("=" * 80)
    
    # 创建汇总表格
    summary_data = []
    for interval, data in all_results.items():
        summary_data.append({
            '周期': interval,
            '数据量': data['total_data'],
            '顶背离': data['top_count'],
            '底背离': data['bottom_count'],
            '总信号': data['top_count'] + data['bottom_count'],
            '频率%': f"{data['frequency']:.2f}%"
        })
    
    # 按频率排序
    summary_data.sort(key=lambda x: float(x['频率%'].replace('%', '')), reverse=True)
    
    print(f"{'周期':<8} {'数据量':<8} {'顶背离':<8} {'底背离':<8} {'总信号':<8} {'频率':<8}")
    print("-" * 55)
    for data in summary_data:
        print(f"{data['周期']:<8} {data['数据量']:<8} {data['顶背离']:<8} {data['底背离']:<8} {data['总信号']:<8} {data['频率%']:<8}")
    
    return all_results

def analyze_specific_timeframe(interval):
    """分析特定时间周期的详细背离信号"""
    print(f"🔍 详细分析 {interval} 周期背离信号")
    print("=" * 60)
    
    # 加载数据
    klines_data = load_bitcoin_data(data_dir='crypto_data', symbol='BTC', interval=interval)
    
    if not klines_data:
        print(f"❌ 无法加载 {interval} 数据")
        return
    
    # 创建分析器并计算指标
    analyzer = DivergenceAnalyzer()
    result = analyzer.calculate_kdj_indicators(klines_data)
    
    if not result:
        print(f"❌ {interval} 计算失败")
        return
    
    # 收集背离信号
    top_signals = []
    bottom_signals = []
    
    for i, k in enumerate(klines_data):
        if result['top_divergence'][i]:
            top_signals.append({
                'date': k['开盘时间'],
                'price': float(k['收盘价']),
                'j': result['j'][i],
                'j1': result['j1'][i]
            })
        
        if result['bottom_divergence'][i]:
            bottom_signals.append({
                'date': k['开盘时间'],
                'price': float(k['收盘价']),
                'j': result['j'][i],
                'j1': result['j1'][i]
            })
    
    # 输出详细结果
    print(f"📈 {interval} 顶部背离信号 (共{len(top_signals)}个):")
    print("-" * 60)
    if top_signals:
        for signal in reversed(top_signals[-10:]):  # 显示最近10个
            date_str = pd.to_datetime(signal['date']).strftime('%Y-%m-%d %H:%M')
            print(f"🔴 {date_str} - ${signal['price']:,.0f} (J:{signal['j']:.1f})")
    else:
        print("未检测到顶部背离信号")
    
    print(f"\n📉 {interval} 底部背离信号 (共{len(bottom_signals)}个):")
    print("-" * 60)
    if bottom_signals:
        for signal in reversed(bottom_signals[-10:]):  # 显示最近10个
            date_str = pd.to_datetime(signal['date']).strftime('%Y-%m-%d %H:%M')
            print(f"🟢 {date_str} - ${signal['price']:,.0f} (J:{signal['j']:.1f})")
    else:
        print("未检测到底部背离信号")

def list_all_divergences_by_time():
    """列出所有周期的背离信号并按时间倒序排列"""
    print("🔍 所有周期背离信号（按时间倒序排列）")
    print("=" * 80)
    
    # 获取所有可用的数据文件
    data_dir = 'crypto_data'
    symbol = 'BTC'
    available_files = []
    
    # 检查BTC目录
    btc_dir = os.path.join(data_dir, symbol)
    if os.path.exists(btc_dir):
        for file in os.listdir(btc_dir):
            if file.endswith('.csv'):
                interval = file.replace('.csv', '')
                available_files.append(interval)
    
    # 按时间周期排序（从小到大）
    interval_order = ['15m', '30m', '1h', '2h', '4h', '6h', '8h', '12h', '1d', '3d', '1w']
    available_files = [interval for interval in interval_order if interval in available_files]
    
    all_divergences = []
    
    for interval in available_files:
        print(f"📊 正在处理 {interval} 周期数据...")
        
        # 加载特定周期的数据
        klines_data = load_bitcoin_data(data_dir='crypto_data', symbol='BTC', interval=interval)
        
        if not klines_data or len(klines_data) < 34:
            print(f"❌ 跳过 {interval} 数据（不可用或数据量不足）")
            continue
        
        # 创建分析器并计算指标
        analyzer = DivergenceAnalyzer()
        result = analyzer.calculate_kdj_indicators(klines_data)
        
        if not result:
            print(f"❌ {interval} 计算失败")
            continue
        
        # 收集背离信号
        for i, k in enumerate(klines_data):
            if result['top_divergence'][i]:
                all_divergences.append({
                    'type': '顶背离',
                    'interval': interval,
                    'date': pd.to_datetime(k['开盘时间']),
                    'price': float(k['收盘价']),
                    'j': result['j'][i]
                })
            
            if result['bottom_divergence'][i]:
                all_divergences.append({
                    'type': '底背离',
                    'interval': interval,
                    'date': pd.to_datetime(k['开盘时间']),
                    'price': float(k['收盘价']),
                    'j': result['j'][i]
                })
    
    # 按时间倒序排列
    all_divergences.sort(key=lambda x: x['date'], reverse=True)
    
    # 输出结果
    print("\n📋 所有周期背离信号（最近100个）:")
    print("=" * 80)
    print(f"{'日期':<20} {'周期':<6} {'类型':<8} {'价格':<12} {'J值':<8}")
    print("-" * 80)
    
    for signal in all_divergences[:100]:  # 显示最近100个
        date_str = signal['date'].strftime('%Y-%m-%d %H:%M')
        price_str = f"${signal['price']:,.0f}"
        signal_type = signal['type']
        emoji = "🔴" if signal_type == "顶背离" else "🟢"
        print(f"{date_str:<20} {signal['interval']:<6} {emoji} {signal_type:<6} {price_str:<12} {signal['j']:.1f}")
    
    return all_divergences

if __name__ == "__main__":
    # 分析所有时间周期
    all_results = analyze_all_timeframes()
    
    # 列出所有背离信号（按时间倒序）
    print("\n")
    all_divergences = list_all_divergences_by_time()
    
    # 用户可以选择详细分析特定周期
    print(f"\n💡 提示: 运行以下命令可详细分析特定周期:")
    print("python3 -c \"from multi_timeframe_analysis import analyze_specific_timeframe; analyze_specific_timeframe('4h')\"")
    print("python3 -c \"from multi_timeframe_analysis import analyze_specific_timeframe; analyze_specific_timeframe('1h')\"") 