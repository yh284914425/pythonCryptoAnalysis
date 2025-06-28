#!/usr/bin/env python3
"""
回测和模拟交易结果分析工具

提供简单的数据分析和可视化功能
"""

import os
import sys
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
import argparse

# 设置中文字体和样式
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']  # 中文字体
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号
sns.set_style("whitegrid")
plt.style.use('seaborn-v0_8')

class ResultAnalyzer:
    """结果分析器"""
    
    def __init__(self, results_dir: str = "backtest_results", 
                 paper_trading_dir: str = "paper_trading_logs"):
        """
        初始化分析器
        
        Args:
            results_dir: 回测结果目录
            paper_trading_dir: 模拟交易日志目录
        """
        self.results_dir = Path(results_dir)
        self.paper_trading_dir = Path(paper_trading_dir)
        
        # 创建输出目录
        self.output_dir = Path("analysis_output")
        self.output_dir.mkdir(exist_ok=True)
        
    def load_backtest_results(self) -> List[Dict[str, Any]]:
        """加载回测结果"""
        results = []
        
        if not self.results_dir.exists():
            print(f"警告: 回测结果目录不存在: {self.results_dir}")
            return results
        
        for file_path in self.results_dir.glob("*.json"):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    data['file_name'] = file_path.name
                    data['file_path'] = str(file_path)
                    results.append(data)
                    print(f"加载回测结果: {file_path.name}")
            except Exception as e:
                print(f"加载文件失败 {file_path}: {e}")
        
        return results
    
    def load_paper_trading_results(self) -> List[Dict[str, Any]]:
        """加载模拟交易结果"""
        results = []
        
        if not self.paper_trading_dir.exists():
            print(f"警告: 模拟交易目录不存在: {self.paper_trading_dir}")
            return results
        
        for file_path in self.paper_trading_dir.glob("final_report_*.json"):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    data['file_name'] = file_path.name
                    data['file_path'] = str(file_path)
                    results.append(data)
                    print(f"加载模拟交易结果: {file_path.name}")
            except Exception as e:
                print(f"加载文件失败 {file_path}: {e}")
        
        return results
    
    def analyze_backtest_performance(self, results: List[Dict]) -> pd.DataFrame:
        """分析回测性能"""
        if not results:
            return pd.DataFrame()
        
        performance_data = []
        
        for result in results:
            if 'error' in result:
                continue
            
            config = result.get('config', {})
            performance = result.get('performance_metrics', {})
            trade_stats = result.get('trade_statistics', {})
            
            row = {
                'file_name': result['file_name'],
                'mode': config.get('mode', 'unknown'),
                'symbol': config.get('symbol', 'unknown'),
                'start_date': config.get('start_date', ''),
                'end_date': config.get('end_date', ''),
                'initial_capital': config.get('initial_capital', 0),
                'final_value': result.get('final_value', 0),
                'total_return': result.get('total_return', 0),
                'sharpe_ratio': performance.get('sharpe_ratio', 0),
                'max_drawdown': performance.get('max_drawdown', 0),
                'volatility': performance.get('volatility', 0),
                'calmar_ratio': performance.get('calmar_ratio', 0),
                'win_rate': trade_stats.get('win_rate', 0),
                'total_trades': trade_stats.get('total_trades', 0),
                'profit_factor': trade_stats.get('profit_factor', 0),
                'max_profit': trade_stats.get('max_profit', 0),
                'max_loss': trade_stats.get('max_loss', 0)
            }
            
            performance_data.append(row)
        
        return pd.DataFrame(performance_data)
    
    def create_performance_comparison_chart(self, df: pd.DataFrame):
        """创建性能对比图表"""
        if df.empty:
            print("无数据可用于性能对比图表")
            return
        
        # 创建子图
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('回测性能对比分析', fontsize=16, fontweight='bold')
        
        # 1. 总收益率对比
        axes[0, 0].bar(range(len(df)), df['total_return'] * 100)
        axes[0, 0].set_title('总收益率 (%)')
        axes[0, 0].set_xlabel('测试编号')
        axes[0, 0].set_ylabel('收益率 (%)')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # 2. 夏普比率对比
        axes[0, 1].bar(range(len(df)), df['sharpe_ratio'])
        axes[0, 1].set_title('夏普比率')
        axes[0, 1].set_xlabel('测试编号')
        axes[0, 1].set_ylabel('夏普比率')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # 3. 最大回撤对比
        axes[0, 2].bar(range(len(df)), df['max_drawdown'] * 100)
        axes[0, 2].set_title('最大回撤 (%)')
        axes[0, 2].set_xlabel('测试编号')
        axes[0, 2].set_ylabel('回撤 (%)')
        axes[0, 2].tick_params(axis='x', rotation=45)
        
        # 4. 胜率对比
        axes[1, 0].bar(range(len(df)), df['win_rate'])
        axes[1, 0].set_title('胜率 (%)')
        axes[1, 0].set_xlabel('测试编号')
        axes[1, 0].set_ylabel('胜率 (%)')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # 5. 交易次数对比
        axes[1, 1].bar(range(len(df)), df['total_trades'])
        axes[1, 1].set_title('交易次数')
        axes[1, 1].set_xlabel('测试编号')
        axes[1, 1].set_ylabel('交易次数')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        # 6. 盈利因子对比
        axes[1, 2].bar(range(len(df)), df['profit_factor'])
        axes[1, 2].set_title('盈利因子')
        axes[1, 2].set_xlabel('测试编号')
        axes[1, 2].set_ylabel('盈利因子')
        axes[1, 2].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        # 保存图表
        chart_file = self.output_dir / 'performance_comparison.png'
        plt.savefig(chart_file, dpi=300, bbox_inches='tight')
        print(f"性能对比图表已保存: {chart_file}")
        
        plt.show()
    
    def create_risk_return_scatter(self, df: pd.DataFrame):
        """创建风险-收益散点图"""
        if df.empty:
            print("无数据可用于风险收益散点图")
            return
        
        plt.figure(figsize=(10, 8))
        
        # 创建散点图
        scatter = plt.scatter(df['volatility'] * 100, df['total_return'] * 100, 
                            c=df['sharpe_ratio'], cmap='viridis', 
                            s=100, alpha=0.7, edgecolors='black')
        
        # 添加颜色条
        cbar = plt.colorbar(scatter)
        cbar.set_label('夏普比率', rotation=270, labelpad=15)
        
        # 添加标签
        for i, row in df.iterrows():
            plt.annotate(f"{row['mode']}", 
                        (row['volatility'] * 100, row['total_return'] * 100),
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        plt.xlabel('年化波动率 (%)')
        plt.ylabel('总收益率 (%)')
        plt.title('风险-收益分析\n(颜色表示夏普比率)', fontweight='bold')
        plt.grid(True, alpha=0.3)
        
        # 保存图表
        chart_file = self.output_dir / 'risk_return_scatter.png'
        plt.savefig(chart_file, dpi=300, bbox_inches='tight')
        print(f"风险收益散点图已保存: {chart_file}")
        
        plt.show()
    
    def create_mode_comparison_table(self, df: pd.DataFrame):
        """创建模式对比表"""
        if df.empty:
            print("无数据可用于模式对比")
            return
        
        # 按模式分组计算平均值
        mode_stats = df.groupby('mode').agg({
            'total_return': 'mean',
            'sharpe_ratio': 'mean',
            'max_drawdown': 'mean',
            'volatility': 'mean',
            'win_rate': 'mean',
            'total_trades': 'mean',
            'profit_factor': 'mean'
        }).round(4)
        
        print("\\n" + "="*80)
        print("策略模式对比分析")
        print("="*80)
        print(mode_stats.to_string())
        
        # 保存到CSV
        csv_file = self.output_dir / 'mode_comparison.csv'
        mode_stats.to_csv(csv_file)
        print(f"\\n模式对比表已保存: {csv_file}")
        
        return mode_stats
    
    def create_equity_curve_chart(self, results: List[Dict]):
        """创建权益曲线图"""
        if not results:
            print("无数据可用于权益曲线图")
            return
        
        plt.figure(figsize=(15, 8))
        
        for i, result in enumerate(results):
            if 'portfolio_history' in result and result['portfolio_history']:
                # 转换为DataFrame
                portfolio_df = pd.DataFrame(result['portfolio_history'])
                
                if 'timestamp' in portfolio_df.columns:
                    # 如果timestamp是字符串，转换为datetime
                    if isinstance(portfolio_df['timestamp'].iloc[0], str):
                        portfolio_df['timestamp'] = pd.to_datetime(portfolio_df['timestamp'])
                    
                    # 绘制权益曲线
                    config = result.get('config', {})
                    label = f"{config.get('mode', 'unknown')} - {config.get('symbol', 'unknown')}"
                    
                    if 'equity' in portfolio_df.columns:
                        plt.plot(portfolio_df['timestamp'], portfolio_df['equity'], 
                               label=label, linewidth=2, alpha=0.8)
                    elif 'total_value' in portfolio_df.columns:
                        plt.plot(portfolio_df['timestamp'], portfolio_df['total_value'], 
                               label=label, linewidth=2, alpha=0.8)
        
        plt.xlabel('时间')
        plt.ylabel('投资组合价值 ($)')
        plt.title('投资组合权益曲线对比', fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        
        # 保存图表
        chart_file = self.output_dir / 'equity_curves.png'
        plt.savefig(chart_file, dpi=300, bbox_inches='tight')
        print(f"权益曲线图已保存: {chart_file}")
        
        plt.show()
    
    def generate_comprehensive_report(self):
        """生成综合分析报告"""
        print("="*80)
        print("策略回测与模拟交易综合分析报告")
        print(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*80)
        
        # 加载数据
        print("\\n1. 加载数据...")
        backtest_results = self.load_backtest_results()
        paper_trading_results = self.load_paper_trading_results()
        
        print(f"找到 {len(backtest_results)} 个回测结果")
        print(f"找到 {len(paper_trading_results)} 个模拟交易结果")
        
        if not backtest_results and not paper_trading_results:
            print("\\n⚠️  未找到任何结果文件，请先运行回测或模拟交易")
            return
        
        # 分析回测结果
        if backtest_results:
            print("\\n2. 分析回测结果...")
            performance_df = self.analyze_backtest_performance(backtest_results)
            
            if not performance_df.empty:
                print(f"成功分析 {len(performance_df)} 个回测结果")
                
                # 生成图表
                print("\\n3. 生成图表...")
                self.create_performance_comparison_chart(performance_df)
                self.create_risk_return_scatter(performance_df)
                self.create_equity_curve_chart(backtest_results)
                
                # 生成对比表
                print("\\n4. 生成对比分析...")
                self.create_mode_comparison_table(performance_df)
                
                # 保存详细结果
                detailed_file = self.output_dir / 'detailed_backtest_results.csv'
                performance_df.to_csv(detailed_file, index=False)
                print(f"详细回测结果已保存: {detailed_file}")
        
        # 分析模拟交易结果
        if paper_trading_results:
            print("\\n5. 分析模拟交易结果...")
            for result in paper_trading_results:
                summary = result.get('summary', {})
                print(f"\\n模拟交易报告: {result['file_name']}")
                print(f"  运行时长: {summary.get('duration_hours', 0):.1f} 小时")
                print(f"  总收益率: {summary.get('total_return', 0):.2%}")
                print(f"  信号总数: {result.get('trading_statistics', {}).get('total_signals', 0)}")
                print(f"  交易总数: {result.get('trading_statistics', {}).get('total_trades', 0)}")
        
        print("\\n" + "="*80)
        print("🎉 综合分析报告生成完成！")
        print(f"所有输出文件保存在: {self.output_dir}")
        print("="*80)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='回测和模拟交易结果分析工具')
    parser.add_argument('--backtest-dir', default='backtest_results', help='回测结果目录')
    parser.add_argument('--paper-dir', default='paper_trading_logs', help='模拟交易日志目录')
    parser.add_argument('--output-dir', default='analysis_output', help='输出目录')
    
    args = parser.parse_args()
    
    # 创建分析器
    analyzer = ResultAnalyzer(
        results_dir=args.backtest_dir,
        paper_trading_dir=args.paper_dir
    )
    
    # 设置输出目录
    analyzer.output_dir = Path(args.output_dir)
    analyzer.output_dir.mkdir(exist_ok=True)
    
    try:
        # 生成综合报告
        analyzer.generate_comprehensive_report()
        
    except Exception as e:
        print(f"\\n❌ 分析过程中发生错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()