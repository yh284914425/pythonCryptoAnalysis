import pandas as pd
import numpy as np
from datetime import datetime
import pytz
import os
from divergence_analysis import DivergenceAnalyzer, load_bitcoin_data

def analyze_divergence_effectiveness(days_ahead=30):
    """
    分析背离信号的有效性
    :param days_ahead: 分析信号后多少天的价格变化
    """
    print(f"📊 分析背离信号的有效性 (观察{days_ahead}天后的价格变化)")
    print("=" * 80)
    
    # 加载数据
    klines_data = load_bitcoin_data()
    if not klines_data:
        return
    
    # 创建分析器并计算指标
    analyzer = DivergenceAnalyzer()
    result = analyzer.calculate_kdj_indicators(klines_data)
    
    if not result:
        return
    
    top_divergence = result['top_divergence']
    bottom_divergence = result['bottom_divergence']
    
    # 分析顶部背离的有效性
    top_success = 0
    top_total = 0
    top_details = []
    
    for i, k in enumerate(klines_data):
        if top_divergence[i] and i + days_ahead < len(klines_data):
            top_total += 1
            current_price = float(k['收盘价'])
            future_price = float(klines_data[i + days_ahead]['收盘价'])
            change_pct = (future_price - current_price) / current_price * 100
            
            # 顶部背离应该预示价格下跌
            if change_pct < 0:
                top_success += 1
                success = "✅"
            else:
                success = "❌"
            
            top_details.append({
                'date': k['开盘时间'],
                'price': current_price,
                'future_price': future_price,
                'change_pct': change_pct,
                'success': success
            })
    
    # 分析底部背离的有效性
    bottom_success = 0
    bottom_total = 0
    bottom_details = []
    
    for i, k in enumerate(klines_data):
        if bottom_divergence[i] and i + days_ahead < len(klines_data):
            bottom_total += 1
            current_price = float(k['收盘价'])
            future_price = float(klines_data[i + days_ahead]['收盘价'])
            change_pct = (future_price - current_price) / current_price * 100
            
            # 底部背离应该预示价格上涨
            if change_pct > 0:
                bottom_success += 1
                success = "✅"
            else:
                success = "❌"
            
            bottom_details.append({
                'date': k['开盘时间'],
                'price': current_price,
                'future_price': future_price,
                'change_pct': change_pct,
                'success': success
            })
    
    # 输出结果
    print(f"\n📈 顶部背离有效性分析:")
    print("-" * 60)
    if top_total > 0:
        success_rate = top_success / top_total * 100
        print(f"总信号数: {top_total}")
        print(f"成功预测: {top_success}")
        print(f"成功率: {success_rate:.1f}%")
        print(f"\n详细结果:")
        for detail in top_details:
            date_str = pd.to_datetime(detail['date']).strftime('%Y-%m-%d')
            print(f"{detail['success']} {date_str}: ${detail['price']:,.0f} → ${detail['future_price']:,.0f} ({detail['change_pct']:+.1f}%)")
    else:
        print("没有足够的顶部背离信号进行分析")
    
    print(f"\n📉 底部背离有效性分析:")
    print("-" * 60)
    if bottom_total > 0:
        success_rate = bottom_success / bottom_total * 100
        print(f"总信号数: {bottom_total}")
        print(f"成功预测: {bottom_success}")
        print(f"成功率: {success_rate:.1f}%")
        print(f"\n详细结果:")
        for detail in bottom_details:
            date_str = pd.to_datetime(detail['date']).strftime('%Y-%m-%d')
            print(f"{detail['success']} {date_str}: ${detail['price']:,.0f} → ${detail['future_price']:,.0f} ({detail['change_pct']:+.1f}%)")
    else:
        print("没有足够的底部背离信号进行分析")

def analyze_by_year():
    """按年份分析背离信号"""
    print("📅 按年份分析背离信号")
    print("=" * 80)
    
    # 加载数据
    klines_data = load_bitcoin_data()
    if not klines_data:
        return
    
    # 创建分析器并计算指标
    analyzer = DivergenceAnalyzer()
    result = analyzer.calculate_kdj_indicators(klines_data)
    
    if not result:
        return
    
    top_divergence = result['top_divergence']
    bottom_divergence = result['bottom_divergence']
    
    # 按年份统计
    yearly_stats = {}
    
    for i, k in enumerate(klines_data):
        year = pd.to_datetime(k['开盘时间']).year
        if year not in yearly_stats:
            yearly_stats[year] = {
                'total_days': 0,
                'top_divergence': 0,
                'bottom_divergence': 0,
                'top_signals': [],
                'bottom_signals': []
            }
        
        yearly_stats[year]['total_days'] += 1
        
        if top_divergence[i]:
            yearly_stats[year]['top_divergence'] += 1
            yearly_stats[year]['top_signals'].append({
                'date': k['开盘时间'],
                'price': float(k['收盘价'])
            })
        
        if bottom_divergence[i]:
            yearly_stats[year]['bottom_divergence'] += 1
            yearly_stats[year]['bottom_signals'].append({
                'date': k['开盘时间'],
                'price': float(k['收盘价'])
            })
    
    # 输出年度统计
    for year in sorted(yearly_stats.keys()):
        stats = yearly_stats[year]
        total_signals = stats['top_divergence'] + stats['bottom_divergence']
        frequency = total_signals / stats['total_days'] * 100 if stats['total_days'] > 0 else 0
        
        print(f"\n📊 {year}年:")
        print(f"  交易天数: {stats['total_days']}")
        print(f"  顶部背离: {stats['top_divergence']} 次")
        print(f"  底部背离: {stats['bottom_divergence']} 次")
        print(f"  背离频率: {frequency:.2f}%")
        
        if stats['top_signals']:
            avg_price = np.mean([s['price'] for s in stats['top_signals']])
            print(f"  顶背平均价格: ${avg_price:,.0f}")
        
        if stats['bottom_signals']:
            avg_price = np.mean([s['price'] for s in stats['bottom_signals']])
            print(f"  底背平均价格: ${avg_price:,.0f}")

def find_recent_strong_signals(days=180):
    """寻找最近的强信号"""
    print(f"🎯 寻找最近{days}天的强背离信号")
    print("=" * 80)
    
    # 加载数据
    klines_data = load_bitcoin_data()
    if not klines_data:
        return
    
    # 只分析最近的数据
    recent_data = klines_data[-days:] if len(klines_data) > days else klines_data
    
    # 创建分析器并计算指标
    analyzer = DivergenceAnalyzer()
    result = analyzer.calculate_kdj_indicators(klines_data)  # 使用全部数据计算以保证准确性
    
    if not result:
        return
    
    # 只查看最近的信号
    top_divergence = result['top_divergence'][-days:]
    bottom_divergence = result['bottom_divergence'][-days:]
    j = result['j'][-days:]
    j1 = result['j1'][-days:]
    
    recent_top_signals = []
    recent_bottom_signals = []
    
    for i, k in enumerate(recent_data):
        actual_index = len(klines_data) - days + i
        
        if top_divergence[i]:
            # 判断信号强度
            strength = "强" if j[i] > 95 else "中" if j[i] > 90 else "弱"
            recent_top_signals.append({
                'date': k['开盘时间'],
                'price': float(k['收盘价']),
                'j': j[i],
                'j1': j1[i],
                'strength': strength
            })
        
        if bottom_divergence[i]:
            # 判断信号强度
            strength = "强" if j[i] < 5 else "中" if j[i] < 15 else "弱"
            recent_bottom_signals.append({
                'date': k['开盘时间'],
                'price': float(k['收盘价']),
                'j': j[i],
                'j1': j1[i],
                'strength': strength
            })
    
    print(f"\n📈 最近顶部背离信号 ({len(recent_top_signals)}个):")
    if recent_top_signals:
        for signal in recent_top_signals:
            date_str = pd.to_datetime(signal['date']).strftime('%Y-%m-%d')
            print(f"  🔴 {date_str}: ${signal['price']:,.0f} [J:{signal['j']:.1f}, J1:{signal['j1']:.1f}] - {signal['strength']}信号")
    else:
        print("  无顶部背离信号")
    
    print(f"\n📉 最近底部背离信号 ({len(recent_bottom_signals)}个):")
    if recent_bottom_signals:
        for signal in recent_bottom_signals:
            date_str = pd.to_datetime(signal['date']).strftime('%Y-%m-%d')
            print(f"  🟢 {date_str}: ${signal['price']:,.0f} [J:{signal['j']:.1f}, J1:{signal['j1']:.1f}] - {signal['strength']}信号")
    else:
        print("  无底部背离信号")

def export_signals_to_csv():
    """导出背离信号到CSV文件"""
    print("💾 导出背离信号到CSV文件")
    print("=" * 50)
    
    # 加载数据
    klines_data = load_bitcoin_data()
    if not klines_data:
        return
    
    # 创建分析器并计算指标
    analyzer = DivergenceAnalyzer()
    result = analyzer.calculate_kdj_indicators(klines_data)
    
    if not result:
        return
    
    # 准备导出数据
    export_data = []
    
    for i, k in enumerate(klines_data):
        row = {
            '日期': k['开盘时间'],
            '开盘价': k['开盘价'],
            '最高价': k['最高价'],
            '最低价': k['最低价'],
            '收盘价': k['收盘价'],
            '成交量': k['成交量'],
            'J值': result['j'][i],
            'J1值': result['j1'][i],
            '顶部背离': 1 if result['top_divergence'][i] else 0,
            '底部背离': 1 if result['bottom_divergence'][i] else 0
        }
        export_data.append(row)
    
    # 创建DataFrame并保存
    df = pd.DataFrame(export_data)
    filename = f"BTC_divergence_analysis_{datetime.now().strftime('%Y%m%d')}.csv"
    df.to_csv(filename, index=False, encoding='utf-8-sig')
    
    # 统计信息
    top_count = df['顶部背离'].sum()
    bottom_count = df['底部背离'].sum()
    
    print(f"✅ 数据已导出到: {filename}")
    print(f"📊 导出统计:")
    print(f"  总记录数: {len(df)}")
    print(f"  顶部背离: {top_count} 次")
    print(f"  底部背离: {bottom_count} 次")

if __name__ == "__main__":
    print("🚀 高级比特币背离分析工具")
    print("=" * 60)
    
    # 1. 分析背离信号的有效性
    print("\n1️⃣ 背离信号有效性分析 (30天后)")
    analyze_divergence_effectiveness(30)
    
    # 2. 按年份分析
    print("\n" + "="*80)
    print("2️⃣ 按年份分析背离信号")
    analyze_by_year()
    
    # 3. 寻找最近的强信号
    print("\n" + "="*80)
    print("3️⃣ 最近强背离信号")
    find_recent_strong_signals(180)
    
    # 4. 导出数据
    print("\n" + "="*80)
    print("4️⃣ 导出分析数据")
    export_signals_to_csv()
    
    print("\n" + "="*80)
    print("✅ 高级分析完成！")
    print("💡 建议: 结合其他技术指标和基本面分析来确认背离信号")
    print("⚠️  风险提示: 背离信号并非100%准确，请做好风险管理") 