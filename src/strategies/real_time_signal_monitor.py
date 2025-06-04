import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

class RealTimeSignalMonitor:
    """实时信号监控器"""
    
    def __init__(self, crypto_data_dir="crypto_data", results_file="results/所有周期背离数据_20250529_235931.csv"):
        self.crypto_data_dir = crypto_data_dir
        self.results_file = results_file
        self.divergence_data = None
        
    def load_divergence_data(self):
        """加载背离数据"""
        if os.path.exists(self.results_file):
            self.divergence_data = pd.read_csv(self.results_file, encoding='utf-8-sig')
            self.divergence_data['日期时间'] = pd.to_datetime(self.divergence_data['日期时间'])
            return True
        return False
    
    def calculate_signal_score(self, signal):
        """计算信号评分"""
        score = 0
        
        # 1. 信号强度评分 (30%)
        strength_scores = {'强': 30, '中': 20, '弱': 10}
        score += strength_scores.get(signal['信号强度'], 0)
        
        # 2. J值区间评分 (25%)
        j_scores = {
            '极度超卖(<0)': 25,
            '超卖(0-20)': 20,
            '偏弱(20-50)': 15,
            '中性(50-80)': 10,
            '超买(80-100)': 5,
            '极度超买(>100)': 0
        }
        score += j_scores.get(signal['J值区间'], 0)
        
        # 3. 价格区间评分 (20%)
        price = signal['收盘价']
        if price < 30000:
            score += 20
        elif price < 60000:
            score += 18
        elif price < 90000:
            score += 15
        elif price < 120000:
            score += 12
        else:
            score += 8
        
        # 4. 时间周期评分 (15%)
        timeframe_scores = {
            '1w': 15, '3d': 14, '1d': 13, '12h': 12,
            '8h': 11, '4h': 10, '2h': 8, '1h': 6
        }
        score += timeframe_scores.get(signal['时间周期'], 0)
        
        # 5. J值绝对位置评分 (10%)
        j_value = signal['J值']
        if j_value < 0:
            score += 10
        elif j_value < 10:
            score += 8
        elif j_value < 20:
            score += 6
        elif j_value < 30:
            score += 4
        else:
            score += 2
        
        return score
    
    def get_recent_signals(self, days=30):
        """获取最近的信号"""
        if not self.load_divergence_data():
            print("❌ 无法加载背离数据")
            return None
        
        # 获取最近30天的信号
        recent_date = datetime.now() - timedelta(days=days)
        recent_signals = self.divergence_data[
            self.divergence_data['日期时间'] >= recent_date
        ].copy()
        
        print(f"📅 最近{days}天内的信号数量: {len(recent_signals)}")
        return recent_signals
    
    def find_current_opportunities(self, min_score=60):
        """寻找当前交易机会"""
        print("🔍 正在扫描当前交易机会...")
        print("="*80)
        
        if not self.load_divergence_data():
            return None
        
        # 获取最近7天的底部背离信号
        recent_date = datetime.now() - timedelta(days=7)
        recent_bottom_signals = self.divergence_data[
            (self.divergence_data['日期时间'] >= recent_date) &
            (self.divergence_data['背离类型'] == '底部背离')
        ].copy()
        
        if len(recent_bottom_signals) == 0:
            print("📭 最近7天内没有发现底部背离信号")
            return None
        
        # 计算信号评分
        recent_bottom_signals['信号评分'] = recent_bottom_signals.apply(
            lambda row: self.calculate_signal_score(row), axis=1
        )
        
        # 筛选高分信号
        high_score_signals = recent_bottom_signals[
            recent_bottom_signals['信号评分'] >= min_score
        ].sort_values('信号评分', ascending=False)
        
        print(f"🎯 发现 {len(high_score_signals)} 个优质买入机会 (评分≥{min_score}分):")
        
        if len(high_score_signals) == 0:
            print(f"📉 最近没有发现评分≥{min_score}分的优质信号")
            # 显示最高分的信号
            if len(recent_bottom_signals) > 0:
                best_signal = recent_bottom_signals.nlargest(1, '信号评分').iloc[0]
                print(f"\n💡 最高评分信号 ({best_signal['信号评分']:.0f}分):")
                self.display_signal_details(best_signal)
        else:
            for i, (_, signal) in enumerate(high_score_signals.head(5).iterrows(), 1):
                print(f"\n🔥 机会 #{i} (评分: {signal['信号评分']:.0f}分)")
                self.display_signal_details(signal)
                self.generate_trading_advice(signal)
        
        return high_score_signals
    
    def display_signal_details(self, signal):
        """显示信号详情"""
        print(f"   📅 时间: {signal['日期时间'].strftime('%Y-%m-%d %H:%M')} ({signal['时间周期']})")
        print(f"   💰 价格: ${signal['收盘价']:,.2f}")
        print(f"   📊 J值: {signal['J值']:.2f} ({signal['J值区间']})")
        print(f"   💪 信号强度: {signal['信号强度']}")
        print(f"   🎯 价格区间: {signal.get('价格区间', '未知')}")
    
    def generate_trading_advice(self, signal):
        """生成交易建议"""
        score = signal['信号评分']
        price = signal['收盘价']
        
        # 根据评分给出建议
        if score >= 80:
            position_size = "15-20%"
            confidence = "高"
        elif score >= 70:
            position_size = "10-15%"
            confidence = "中高"
        elif score >= 60:
            position_size = "8-12%"
            confidence = "中等"
        else:
            position_size = "5-8%"
            confidence = "谨慎"
        
        # 计算建议的止损止盈
        stop_loss = price * 0.94  # 6%止损
        take_profit = price * 1.25  # 25%止盈
        
        print(f"   📋 交易建议:")
        print(f"      💡 置信度: {confidence}")
        print(f"      📦 建议仓位: {position_size}")
        print(f"      🛡️  止损价: ${stop_loss:,.2f} (-6%)")
        print(f"      🎯 止盈价: ${take_profit:,.2f} (+25%)")
        print(f"      ⏰ 最大持仓: 45天")
    
    def analyze_market_context(self):
        """分析当前市场环境"""
        print("\n📊 当前市场环境分析")
        print("="*50)
        
        if not self.load_divergence_data():
            return
        
        # 获取最近的价格信息
        latest_signals = self.divergence_data.nlargest(10, '日期时间')
        
        if len(latest_signals) > 0:
            latest_price = latest_signals.iloc[0]['收盘价']
            
            # 市场阶段判断
            if latest_price < 20000:
                stage = "熊市底部"
                advice = "极佳买入时机，底部背离信号可靠性很高"
            elif latest_price < 40000:
                stage = "恢复期"
                advice = "良好买入时机，注意风险控制"
            elif latest_price < 70000:
                stage = "成长期"
                advice = "谨慎买入，优选强信号"
            elif latest_price < 100000:
                stage = "牛市中期"
                advice = "高度谨慎，严格筛选信号"
            else:
                stage = "牛市顶部"
                advice = "极度谨慎，建议观望"
            
            print(f"🏛️  市场阶段: {stage}")
            print(f"💰 当前价格: ${latest_price:,.2f}")
            print(f"💡 策略建议: {advice}")
        
        # 统计最近信号分布
        recent_signals = self.get_recent_signals(30)
        if recent_signals is not None and len(recent_signals) > 0:
            bottom_count = len(recent_signals[recent_signals['背离类型'] == '底部背离'])
            top_count = len(recent_signals[recent_signals['背离类型'] == '顶部背离'])
            
            print(f"\n📈 最近30天信号分布:")
            print(f"   🟢 底部背离: {bottom_count} 个")
            print(f"   🔴 顶部背离: {top_count} 个")
            
            if bottom_count > top_count:
                print(f"   📊 信号倾向: 偏向买入机会")
            elif top_count > bottom_count:
                print(f"   📊 信号倾向: 偏向卖出信号")
            else:
                print(f"   📊 信号倾向: 相对平衡")
    
    def show_historical_performance(self):
        """显示历史信号表现"""
        print("\n📊 历史信号表现统计")
        print("="*50)
        
        if not self.load_divergence_data():
            return
        
        # 按信号强度统计
        strength_stats = self.divergence_data.groupby(['背离类型', '信号强度']).size().unstack(fill_value=0)
        print("📈 信号强度分布:")
        print(strength_stats)
        
        # 按时间周期统计
        timeframe_stats = self.divergence_data.groupby(['背离类型', '时间周期']).size().unstack(fill_value=0)
        print("\n⏰ 时间周期分布:")
        print(timeframe_stats)
    
    def run_monitor(self):
        """运行监控"""
        print("🚀 启动实时信号监控系统")
        print("="*80)
        
        # 1. 分析市场环境
        self.analyze_market_context()
        
        # 2. 寻找当前机会
        opportunities = self.find_current_opportunities(min_score=60)
        
        # 3. 显示历史表现
        self.show_historical_performance()
        
        print(f"\n🎯 监控总结:")
        if opportunities is not None and len(opportunities) > 0:
            print(f"✅ 发现 {len(opportunities)} 个潜在交易机会")
            print(f"💡 建议重点关注评分最高的前3个信号")
        else:
            print(f"📭 当前暂无优质交易机会")
            print(f"💡 建议继续观察，等待更好的信号")
        
        print(f"\n⚠️  风险提醒:")
        print(f"   • 所有信号仅供参考，请结合自己的判断")
        print(f"   • 严格控制仓位，不要重仓")
        print(f"   • 设置好止损，保护本金安全")
        print(f"   • 保持冷静，不要因为错过机会而冲动")

def main():
    """主函数"""
    monitor = RealTimeSignalMonitor()
    monitor.run_monitor()

if __name__ == "__main__":
    main() 