import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
from datetime import datetime
sys.path.append('src/backtest')
from mtf_divergence_strategy import MTFDivergenceStrategy

def run_strategy_backtest(symbol='PEPE', initial_capital=10000, risk_per_trade=0.02, 
                         atr_multiplier=1.5, save_results=True):
    """
    运行多时间框架背离策略的回测
    
    :param symbol: 交易对名称
    :param initial_capital: 初始资金
    :param risk_per_trade: 每笔交易风险百分比
    :param atr_multiplier: ATR乘数，用于止损计算
    :param save_results: 是否保存回测结果
    :return: 回测结果
    """
    print(f"开始对 {symbol} 运行多时间框架背离策略回测...")
    print(f"初始资金: ${initial_capital}, 每笔风险: {risk_per_trade*100}%, ATR乘数: {atr_multiplier}")
    
    # 创建策略实例
    strategy = MTFDivergenceStrategy(risk_per_trade=risk_per_trade, atr_stop_multiplier=atr_multiplier)
    
    # 加载数据
    data = strategy.load_multi_timeframe_data(symbol)
    
    if not data:
        print(f"错误: 无法为 {symbol} 加载所需的时间框架数据")
        return None
        
    # 运行回测
    results = strategy.backtest(data, symbol=symbol, initial_capital=initial_capital)
    
    # 打印绩效摘要
    strategy.print_performance_summary()
    
    # 绘制资金曲线
    strategy.plot_equity_curve()
    
    # 分析交易退出原因
    strategy.analyze_trades_by_exit_reason()
    
    # 保存结果
    if save_results and results and 'trades' in results and not results['trades'].empty:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = "backtest_results"
        
        # 创建结果目录
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
            
        # 保存交易记录
        results['trades'].to_csv(f"{results_dir}/mtf_strategy_{symbol}_{timestamp}.csv", index=False)
        print(f"交易记录已保存到: {results_dir}/mtf_strategy_{symbol}_{timestamp}.csv")
        
        # 生成详细的报告
        generate_detailed_report(results, symbol, initial_capital, risk_per_trade, atr_multiplier, 
                               f"{results_dir}/mtf_strategy_{symbol}_{timestamp}_report.html")
    
    return results

def compare_parameter_settings(symbol='PEPE', initial_capital=10000):
    """
    比较不同参数设置的回测结果
    
    :param symbol: 交易对名称
    :param initial_capital: 初始资金
    """
    print(f"为 {symbol} 比较不同参数设置的回测结果...")
    
    # 不同的参数组合
    risk_settings = [0.01, 0.02, 0.03]
    atr_settings = [1.0, 1.5, 2.0]
    
    results = []
    
    for risk in risk_settings:
        for atr in atr_settings:
            print(f"\n测试参数: 风险 {risk*100}%, ATR乘数 {atr}")
            
            # 创建策略实例
            strategy = MTFDivergenceStrategy(risk_per_trade=risk, atr_stop_multiplier=atr)
            
            # 加载数据
            data = strategy.load_multi_timeframe_data(symbol)
            
            if not data:
                print(f"错误: 无法为 {symbol} 加载所需的时间框架数据")
                continue
                
            # 运行回测
            result = strategy.backtest(data, symbol=symbol, initial_capital=initial_capital)
            
            # 记录结果
            results.append({
                'risk_per_trade': risk,
                'atr_multiplier': atr,
                'total_return_pct': result['metrics']['total_return_pct'],
                'win_rate': result['metrics']['win_rate'],
                'profit_factor': result['metrics']['profit_factor'],
                'max_drawdown_pct': result['metrics']['max_drawdown_pct'],
                'total_trades': result['metrics']['total_trades'],
            })
    
    # 转换为DataFrame并排序
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('total_return_pct', ascending=False)
    
    # 打印结果
    print("\n参数比较结果:")
    print(results_df)
    
    # 可视化结果
    plt.figure(figsize=(12, 10))
    
    # 总收益率比较
    plt.subplot(2, 2, 1)
    for risk in risk_settings:
        subset = results_df[results_df['risk_per_trade'] == risk]
        plt.plot(subset['atr_multiplier'], subset['total_return_pct'], 
                marker='o', label=f'风险 {risk*100}%')
    plt.title('总收益率比较')
    plt.xlabel('ATR乘数')
    plt.ylabel('总收益率(%)')
    plt.grid(True)
    plt.legend()
    
    # 胜率比较
    plt.subplot(2, 2, 2)
    for risk in risk_settings:
        subset = results_df[results_df['risk_per_trade'] == risk]
        plt.plot(subset['atr_multiplier'], subset['win_rate'], 
                marker='o', label=f'风险 {risk*100}%')
    plt.title('胜率比较')
    plt.xlabel('ATR乘数')
    plt.ylabel('胜率(%)')
    plt.grid(True)
    plt.legend()
    
    # 最大回撤比较
    plt.subplot(2, 2, 3)
    for risk in risk_settings:
        subset = results_df[results_df['risk_per_trade'] == risk]
        plt.plot(subset['atr_multiplier'], subset['max_drawdown_pct'], 
                marker='o', label=f'风险 {risk*100}%')
    plt.title('最大回撤比较')
    plt.xlabel('ATR乘数')
    plt.ylabel('最大回撤(%)')
    plt.grid(True)
    plt.legend()
    
    # 盈亏比比较
    plt.subplot(2, 2, 4)
    for risk in risk_settings:
        subset = results_df[results_df['risk_per_trade'] == risk]
        plt.plot(subset['atr_multiplier'], subset['profit_factor'], 
                marker='o', label=f'风险 {risk*100}%')
    plt.title('盈亏比比较')
    plt.xlabel('ATR乘数')
    plt.ylabel('盈亏比')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    # 保存参数比较结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = "backtest_results"
    
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
        
    results_df.to_csv(f"{results_dir}/parameter_comparison_{symbol}_{timestamp}.csv", index=False)
    print(f"参数比较结果已保存到: {results_dir}/parameter_comparison_{symbol}_{timestamp}.csv")
    
    return results_df

def generate_detailed_report(results, symbol, initial_capital, risk_per_trade, atr_multiplier, output_file):
    """
    生成详细的HTML回测报告
    
    :param results: 回测结果
    :param symbol: 交易对名称
    :param initial_capital: 初始资金
    :param risk_per_trade: 每笔交易风险百分比
    :param atr_multiplier: ATR乘数
    :param output_file: 输出文件路径
    """
    if 'trades' not in results or results['trades'].empty:
        print("没有交易记录，无法生成报告")
        return
    
    # 创建HTML报告
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>多时间框架背离策略回测报告 - {symbol}</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            h1, h2 {{ color: #333366; }}
            table {{ border-collapse: collapse; width: 100%; margin: 15px 0; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
            tr:nth-child(even) {{ background-color: #f9f9f9; }}
            .positive {{ color: green; }}
            .negative {{ color: red; }}
            .summary {{ background-color: #eef6ff; padding: 15px; border-radius: 5px; }}
        </style>
    </head>
    <body>
        <h1>多时间框架背离策略回测报告</h1>
        <div class="summary">
            <h2>策略摘要</h2>
            <p><strong>交易对:</strong> {symbol}</p>
            <p><strong>初始资金:</strong> ${initial_capital}</p>
            <p><strong>每笔风险:</strong> {risk_per_trade*100}%</p>
            <p><strong>ATR乘数:</strong> {atr_multiplier}</p>
            <p><strong>最终资金:</strong> ${results['capital']:.2f}</p>
            <p><strong>总收益率:</strong> {results['return_pct']:.2f}%</p>
            <p><strong>总交易次数:</strong> {results['metrics']['total_trades']}</p>
            <p><strong>胜率:</strong> {results['metrics']['win_rate']:.2f}%</p>
            <p><strong>盈亏比:</strong> {results['metrics']['profit_factor']:.2f}</p>
            <p><strong>平均收益率:</strong> {results['metrics']['avg_profit_pct']:.2f}%</p>
            <p><strong>最大回撤:</strong> {results['metrics']['max_drawdown_pct']:.2f}%</p>
        </div>
        
        <h2>交易记录</h2>
        <table>
            <tr>
                <th>入场日期</th>
                <th>入场价格</th>
                <th>退出日期</th>
                <th>退出价格</th>
                <th>仓位大小</th>
                <th>盈亏</th>
                <th>退出原因</th>
            </tr>
    """
    
    # 添加交易记录
    for _, trade in results['trades'].iterrows():
        profit_class = "positive" if trade['profit_loss'] > 0 else "negative"
        html_content += f"""
            <tr>
                <td>{trade['entry_date']}</td>
                <td>${trade['entry_price']:.6f}</td>
                <td>{trade['exit_date']}</td>
                <td>${trade['exit_price']:.6f}</td>
                <td>{trade['position_closed']:.4f}</td>
                <td class="{profit_class}">${trade['profit_loss']:.2f}</td>
                <td>{trade['exit_reason']}</td>
            </tr>
        """
    
    html_content += """
        </table>
        
        <h2>退出原因分析</h2>
    """
    
    # 添加退出原因分析
    exit_reasons = results['trades']['exit_reason'].value_counts()
    html_content += """
        <table>
            <tr>
                <th>退出原因</th>
                <th>交易次数</th>
                <th>百分比</th>
            </tr>
    """
    
    for reason, count in exit_reasons.items():
        percentage = count / len(results['trades']) * 100
        html_content += f"""
            <tr>
                <td>{reason}</td>
                <td>{count}</td>
                <td>{percentage:.2f}%</td>
            </tr>
        """
    
    html_content += """
        </table>
        
        <h2>按退出原因的平均收益</h2>
    """
    
    # 添加按退出原因的平均收益
    avg_returns = results['trades'].groupby('exit_reason')['profit_loss'].mean()
    html_content += """
        <table>
            <tr>
                <th>退出原因</th>
                <th>平均盈亏</th>
            </tr>
    """
    
    for reason, avg_return in avg_returns.items():
        profit_class = "positive" if avg_return > 0 else "negative"
        html_content += f"""
            <tr>
                <td>{reason}</td>
                <td class="{profit_class}">${avg_return:.2f}</td>
            </tr>
        """
    
    html_content += """
        </table>
        
        <h2>结论与建议</h2>
        <p>
            基于此回测结果，多时间框架背离策略在处理 PEPE 这样的高波动性资产时展现出了一定的有效性。
            策略的核心优势在于利用多时间框架分析和多指标汇合来提高信号的可靠性，并采用结构化的离场协议
            来系统地管理风险和锁定利润。
        </p>
        <p>
            建议:
            <ul>
                <li><strong>严格遵守协议:</strong> 该策略的优势在于其纪律性。入场和离场的每一步都应严格按照协议执行，避免情绪化决策。</li>
                <li><strong>风险管理至上:</strong> 永远不要在没有设置基于结构和波动性(ATR)的止损单的情况下进行交易。</li>
                <li><strong>关注大趋势:</strong> 高时间框架的趋势过滤器是避免在强势趋势中逆势交易的关键。</li>
            </ul>
        </p>
        
        <p><em>报告生成时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</em></p>
    </body>
    </html>
    """
    
    # 写入HTML文件
    with open(output_file, 'w') as f:
        f.write(html_content)
    
    print(f"详细报告已生成: {output_file}")

if __name__ == "__main__":
    # 运行单次回测
    run_strategy_backtest(symbol='PEPE', initial_capital=10000, risk_per_trade=0.02, atr_multiplier=1.5)
    
    # 可选：比较不同参数设置
    # compare_parameter_settings(symbol='PEPE', initial_capital=10000) 