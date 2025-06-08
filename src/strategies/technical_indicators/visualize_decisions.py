#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
比特币决策分析可视化工具
读取 btc_decisions.csv 文件并绘制价格图表，标注买卖信号
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
import numpy as np
from datetime import datetime, timedelta
import seaborn as sns
import os

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 设置样式
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class DecisionVisualizer:
    """决策可视化器"""
    
    def __init__(self, csv_file: str = "btc_decisions.csv"):
        """
        初始化可视化器
        :param csv_file: CSV文件路径
        """
        self.csv_file = csv_file
        self.df = None
        self.fig = None
        self.ax = None
        
    def load_data(self):
        """加载决策数据"""
        if not os.path.exists(self.csv_file):
            raise FileNotFoundError(f"找不到文件: {self.csv_file}")
        
        # 读取CSV文件
        self.df = pd.read_csv(self.csv_file)
        
        # 转换时间列为datetime
        self.df['时间'] = pd.to_datetime(self.df['时间'])
        
        # 确保价格列为数值类型
        self.df['价格'] = pd.to_numeric(self.df['价格'].str.replace('$', '').str.replace(',', ''))
        
        print(f"✅ 成功加载 {len(self.df)} 条决策记录")
        print(f"📅 时间范围: {self.df['时间'].min()} 到 {self.df['时间'].max()}")
        
        return self.df
    
    def create_price_chart(self, time_range: str = "all", save_path: str = None, 
                          figsize: tuple = (16, 10), show_confidence: bool = True):
        """
        创建价格图表并标注买卖信号
        :param time_range: 时间范围 ("all", "1y", "6m", "3m", "1m")
        :param save_path: 保存路径，None则不保存
        :param figsize: 图像大小
        :param show_confidence: 是否显示置信度
        """
        if self.df is None:
            self.load_data()
        
        # 筛选时间范围
        df_filtered = self._filter_time_range(time_range)
        
        # 创建图像
        self.fig, self.ax = plt.subplots(figsize=figsize)
        
        # 绘制价格线
        self.ax.plot(df_filtered['时间'], df_filtered['价格'], 
                    color='#2E86C1', linewidth=1.5, alpha=0.8, label='BTC Price')
        
        # 标注买卖信号
        self._plot_signals(df_filtered, show_confidence)
        
        # 设置图表样式
        self._style_chart(df_filtered, time_range)
        
        # 添加图例和标题
        self._add_legend_and_title(time_range)
        
        # 保存或显示
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
            print(f"📊 图表已保存至: {save_path}")
        
        plt.tight_layout()
        plt.show()
        
        return self.fig, self.ax
    
    def _filter_time_range(self, time_range: str) -> pd.DataFrame:
        """根据时间范围筛选数据"""
        if time_range == "all":
            return self.df
        
        now = self.df['时间'].max()
        
        if time_range == "1y":
            start_date = now - timedelta(days=365)
        elif time_range == "6m":
            start_date = now - timedelta(days=180)
        elif time_range == "3m":
            start_date = now - timedelta(days=90)
        elif time_range == "1m":
            start_date = now - timedelta(days=30)
        else:
            return self.df
        
        return self.df[self.df['时间'] >= start_date]
    
    def _plot_signals(self, df: pd.DataFrame, show_confidence: bool):
        """绘制买卖信号"""
        buy_signals = df[df['决策'] == '买入']
        sell_signals = df[df['决策'] == '卖出']
        
        # 买入信号 - 绿色向上箭头
        if len(buy_signals) > 0:
            for _, signal in buy_signals.iterrows():
                confidence = signal['置信度']
                alpha = 0.6 + (confidence * 0.4)  # 置信度影响透明度
                size = 50 + (confidence * 100)    # 置信度影响大小
                
                self.ax.scatter(signal['时间'], signal['价格'], 
                              marker='^', s=size, color='#27AE60', 
                              alpha=alpha, edgecolors='darkgreen', linewidth=1,
                              zorder=5, label='买入信号' if signal.name == buy_signals.index[0] else "")
                
                # 添加置信度标注
                if show_confidence and confidence >= 0.9:
                    self.ax.annotate(f'{confidence:.2f}', 
                                   xy=(signal['时间'], signal['价格']),
                                   xytext=(5, 15), textcoords='offset points',
                                   fontsize=8, color='darkgreen', weight='bold',
                                   bbox=dict(boxstyle='round,pad=0.2', facecolor='lightgreen', alpha=0.7))
        
        # 卖出信号 - 红色向下箭头
        if len(sell_signals) > 0:
            for _, signal in sell_signals.iterrows():
                confidence = signal['置信度']
                alpha = 0.6 + (confidence * 0.4)
                size = 50 + (confidence * 100)
                
                self.ax.scatter(signal['时间'], signal['价格'], 
                              marker='v', s=size, color='#E74C3C', 
                              alpha=alpha, edgecolors='darkred', linewidth=1,
                              zorder=5, label='卖出信号' if signal.name == sell_signals.index[0] else "")
                
                # 添加置信度标注
                if show_confidence and confidence >= 0.9:
                    self.ax.annotate(f'{confidence:.2f}', 
                                   xy=(signal['时间'], signal['价格']),
                                   xytext=(5, -15), textcoords='offset points',
                                   fontsize=8, color='darkred', weight='bold',
                                   bbox=dict(boxstyle='round,pad=0.2', facecolor='lightcoral', alpha=0.7))
    
    def _style_chart(self, df: pd.DataFrame, time_range: str):
        """设置图表样式"""
        # Y轴格式化（价格）
        self.ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
        
        # X轴格式化（时间）
        if time_range in ["1m", "3m"]:
            self.ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
            self.ax.xaxis.set_major_locator(mdates.DayLocator(interval=7))
        elif time_range == "6m":
            self.ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            self.ax.xaxis.set_major_locator(mdates.MonthLocator())
        else:
            self.ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            self.ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        
        # 旋转x轴标签
        plt.setp(self.ax.xaxis.get_majorticklabels(), rotation=45)
        
        # 设置网格
        self.ax.grid(True, alpha=0.3, linestyle='--')
        
        # 设置背景色
        self.ax.set_facecolor('#FAFAFA')
        
        # 设置轴标签
        self.ax.set_xlabel('时间', fontsize=12, weight='bold')
        self.ax.set_ylabel('比特币价格 (USD)', fontsize=12, weight='bold')
        
        # 添加价格区间阴影
        self._add_price_zones(df)
    
    def _add_price_zones(self, df: pd.DataFrame):
        """添加价格区间背景色"""
        min_price = df['价格'].min()
        max_price = df['价格'].max()
        price_range = max_price - min_price
        
        # 高价区域 (顶部20%)
        high_zone_start = max_price - (price_range * 0.2)
        self.ax.axhspan(high_zone_start, max_price, alpha=0.1, color='red', label='高价区域')
        
        # 低价区域 (底部20%)
        low_zone_end = min_price + (price_range * 0.2)
        self.ax.axhspan(min_price, low_zone_end, alpha=0.1, color='green', label='低价区域')
    
    def _add_legend_and_title(self, time_range: str):
        """添加图例和标题"""
        # 时间范围描述
        time_desc = {
            "all": "完整历史",
            "1y": "最近1年", 
            "6m": "最近6个月",
            "3m": "最近3个月",
            "1m": "最近1个月"
        }
        
        # 设置标题
        title = f"比特币技术分析决策图表 - {time_desc.get(time_range, time_range)}"
        self.ax.set_title(title, fontsize=16, weight='bold', pad=20)
        
        # 添加副标题
        buy_count = len(self.df[self.df['决策'] == '买入'])
        sell_count = len(self.df[self.df['决策'] == '卖出'])
        subtitle = f"🟢 买入信号: {buy_count} 次  🔴 卖出信号: {sell_count} 次  📊 总决策: {len(self.df)} 次"
        self.ax.text(0.5, 0.98, subtitle, transform=self.ax.transAxes, 
                    ha='center', va='top', fontsize=11, 
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8))
        
        # 设置图例
        handles, labels = self.ax.get_legend_handles_labels()
        # 去重图例
        by_label = dict(zip(labels, handles))
        self.ax.legend(by_label.values(), by_label.keys(), 
                      loc='upper left', frameon=True, fancybox=True, shadow=True)
    
    def create_statistics_chart(self, save_path: str = None, figsize: tuple = (14, 8)):
        """创建统计分析图表"""
        if self.df is None:
            self.load_data()
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)
        
        # 1. 月度买卖信号分布
        monthly_stats = self._get_monthly_stats()
        if not monthly_stats.empty:
            x_labels = [str(idx) for idx in monthly_stats.index]
            ax1.bar(x_labels, monthly_stats.get('买入', 0), alpha=0.7, color='green', label='买入')
            ax1.bar(x_labels, monthly_stats.get('卖出', 0), alpha=0.7, color='red', label='卖出')
            plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
        ax1.set_title('月度买卖信号分布', fontsize=12, weight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. 置信度分布
        confidence_bins = np.arange(0.7, 1.05, 0.05)
        ax2.hist(self.df['置信度'], bins=confidence_bins, alpha=0.7, color='skyblue', edgecolor='black')
        ax2.set_title('决策置信度分布', fontsize=12, weight='bold')
        ax2.set_xlabel('置信度')
        ax2.set_ylabel('频次')
        ax2.grid(True, alpha=0.3)
        
        # 3. 价格区间决策分布
        price_stats = self._get_price_zone_stats()
        ax3.pie(price_stats.values, labels=price_stats.index, autopct='%1.1f%%', startangle=90)
        ax3.set_title('价格区间决策分布', fontsize=12, weight='bold')
        
        # 4. 风险等级分布
        risk_stats = self.df['风险等级'].value_counts()
        colors = {'low': 'green', 'medium': 'orange', 'high': 'red'}
        risk_colors = [colors.get(x, 'gray') for x in risk_stats.index]
        ax4.bar(risk_stats.index, risk_stats.values, color=risk_colors, alpha=0.7)
        ax4.set_title('风险等级分布', fontsize=12, weight='bold')
        ax4.set_ylabel('决策次数')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 统计图表已保存至: {save_path}")
        
        plt.show()
        return fig
    
    def _get_monthly_stats(self) -> pd.DataFrame:
        """获取月度统计"""
        self.df['年月'] = self.df['时间'].dt.to_period('M')
        monthly = self.df.groupby(['年月', '决策']).size().unstack(fill_value=0)
        return monthly
    
    def _get_price_zone_stats(self) -> pd.Series:
        """获取价格区间统计"""
        min_price = self.df['价格'].min()
        max_price = self.df['价格'].max()
        price_range = max_price - min_price
        
        def get_price_zone(price):
            if price <= min_price + price_range * 0.3:
                return '低价区 (0-30%)'
            elif price <= min_price + price_range * 0.7:
                return '中价区 (30-70%)'
            else:
                return '高价区 (70-100%)'
        
        self.df['价格区间'] = self.df['价格'].apply(get_price_zone)
        return self.df['价格区间'].value_counts()
    
    def create_performance_analysis(self, save_path: str = None, figsize: tuple = (16, 10)):
        """创建性能分析图表"""
        if self.df is None:
            self.load_data()
        
        # 计算模拟交易性能
        performance = self._calculate_performance()
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)
        
        # 1. 累计收益率
        ax1.plot(performance['时间'], performance['累计收益率'], 
                linewidth=2, color='blue', label='策略收益')
        ax1.axhline(y=0, color='red', linestyle='--', alpha=0.5)
        ax1.set_title('累计收益率', fontsize=12, weight='bold')
        ax1.set_ylabel('收益率 (%)')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # 2. 回撤分析
        ax2.fill_between(performance['时间'], performance['回撤'], 
                        alpha=0.7, color='red', label='回撤')
        ax2.set_title('最大回撤分析', fontsize=12, weight='bold')
        ax2.set_ylabel('回撤 (%)')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # 3. 胜率分析
        win_rate_data = self._calculate_win_rate()
        ax3.bar(['胜', '负'], [win_rate_data['胜率'], 100-win_rate_data['胜率']], 
               color=['green', 'red'], alpha=0.7)
        ax3.set_title(f"胜率分析 ({win_rate_data['胜率']:.1f}%)", fontsize=12, weight='bold')
        ax3.set_ylabel('百分比 (%)')
        ax3.grid(True, alpha=0.3)
        
        # 4. 年化收益
        yearly_returns = self._calculate_yearly_returns()
        ax4.bar(yearly_returns.index.astype(str), yearly_returns.values, 
               alpha=0.7, color='skyblue')
        ax4.set_title('年度收益率', fontsize=12, weight='bold')
        ax4.set_ylabel('收益率 (%)')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 性能分析图表已保存至: {save_path}")
        
        plt.show()
        return fig
    
    def _calculate_performance(self) -> pd.DataFrame:
        """计算策略性能"""
        df = self.df[self.df['决策'].isin(['买入', '卖出'])].copy()
        df = df.sort_values('时间')
        
        position = 0  # 0: 空仓, 1: 持仓
        entry_price = 0
        returns = []
        cumulative_return = 0
        max_return = 0
        drawdowns = []
        
        for _, row in df.iterrows():
            if row['决策'] == '买入' and position == 0:
                position = 1
                entry_price = row['价格']
                returns.append(0)
                drawdowns.append(0)
            elif row['决策'] == '卖出' and position == 1:
                position = 0
                trade_return = (row['价格'] - entry_price) / entry_price * 100
                returns.append(trade_return)
                cumulative_return += trade_return
                max_return = max(max_return, cumulative_return)
                drawdown = (max_return - cumulative_return)
                drawdowns.append(drawdown)
            else:
                returns.append(0)
                drawdowns.append(drawdowns[-1] if drawdowns else 0)
        
        # 计算累计收益率
        cumulative_returns = np.cumsum(returns)
        
        return pd.DataFrame({
            '时间': df['时间'],
            '累计收益率': cumulative_returns,
            '回撤': drawdowns
        })
    
    def _calculate_win_rate(self) -> dict:
        """计算胜率"""
        df = self.df[self.df['决策'].isin(['买入', '卖出'])].copy()
        df = df.sort_values('时间')
        
        trades = []
        position = 0
        entry_price = 0
        
        for _, row in df.iterrows():
            if row['决策'] == '买入' and position == 0:
                position = 1
                entry_price = row['价格']
            elif row['决策'] == '卖出' and position == 1:
                position = 0
                trade_return = (row['价格'] - entry_price) / entry_price
                trades.append(trade_return)
        
        if trades:
            winning_trades = sum(1 for t in trades if t > 0)
            win_rate = winning_trades / len(trades) * 100
            return {'胜率': win_rate, '总交易': len(trades), '盈利交易': winning_trades}
        
        return {'胜率': 0, '总交易': 0, '盈利交易': 0}
    
    def _calculate_yearly_returns(self) -> pd.Series:
        """计算年度收益率"""
        performance = self._calculate_performance()
        performance['年份'] = performance['时间'].dt.year
        
        yearly_final_returns = performance.groupby('年份')['累计收益率'].last()
        yearly_initial_returns = performance.groupby('年份')['累计收益率'].first()
        
        return yearly_final_returns - yearly_initial_returns


def main():
    """主函数 - 生成各种图表"""
    print("📊 比特币决策分析可视化工具")
    print("=" * 50)
    
    # 创建可视化器
    visualizer = DecisionVisualizer()
    
    try:
        # 加载数据
        visualizer.load_data()
        
        # 1. 创建完整历史价格图表
        print("\n📈 生成完整历史价格图表...")
        visualizer.create_price_chart(
            time_range="all", 
            save_path="btc_full_history_chart.png",
            show_confidence=True
        )
        
        # 2. 创建最近1年的详细图表
        print("\n📈 生成最近1年详细图表...")
        visualizer.create_price_chart(
            time_range="1y", 
            save_path="btc_1year_chart.png",
            show_confidence=True
        )
        
        # 3. 创建统计分析图表
        print("\n📊 生成统计分析图表...")
        visualizer.create_statistics_chart(save_path="btc_statistics_chart.png")
        
        # 4. 创建性能分析图表
        print("\n📈 生成性能分析图表...")
        visualizer.create_performance_analysis(save_path="btc_performance_chart.png")
        
        print("\n✅ 所有图表生成完成！")
        print("📁 生成的文件:")
        print("   - btc_full_history_chart.png (完整历史)")
        print("   - btc_1year_chart.png (最近1年)")
        print("   - btc_statistics_chart.png (统计分析)")
        print("   - btc_performance_chart.png (性能分析)")
        
    except FileNotFoundError as e:
        print(f"❌ 错误: {e}")
        print("💡 请确保 btc_decisions.csv 文件存在")
    except Exception as e:
        print(f"❌ 发生错误: {e}")


if __name__ == "__main__":
    main() 