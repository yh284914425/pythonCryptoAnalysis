#!/usr/bin/env python3
"""
比特币KDJ背离分析系统 - 项目管理脚本
"""

import argparse
import subprocess
import sys
import os
from pathlib import Path

def run_command(command, description):
    """运行命令并显示结果"""
    print(f"\n🚀 {description}")
    print("=" * 60)
    
    try:
        result = subprocess.run(command, shell=True, check=True, 
                              capture_output=False, text=True)
        print(f"✅ {description} 完成!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} 失败: {e}")
        return False

def setup_environment():
    """设置环境"""
    print("🔧 检查环境和依赖...")
    
    # 检查Python版本
    python_version = sys.version_info
    if python_version.major < 3 or python_version.minor < 7:
        print("❌ 需要Python 3.7或更高版本")
        return False
    
    print(f"✅ Python版本: {python_version.major}.{python_version.minor}")
    
    # 检查依赖文件
    if not Path("config/requirements.txt").exists():
        print("❌ 找不到 config/requirements.txt")
        return False
    
    # 安装依赖
    return run_command("pip install -r config/requirements.txt", "安装依赖包")

def download_data():
    """下载数据"""
    return run_command("python src/data_collection/downData.py", "下载币安数据")

def run_multi_timeframe():
    """运行多周期分析"""
    return run_command("python src/analysis/multi_timeframe_analysis.py", "多周期背离分析")

def run_backtest():
    """运行Backtrader回测"""
    return run_command("python src/strategies/optimized_backtrader_strategy.py", "运行优化回测策略")

def run_monitor():
    """运行实时监控"""
    return run_command("python src/strategies/real_time_signal_monitor.py", "实时信号监控")

def show_project_status():
    """显示项目状态"""
    print("📋 项目状态检查")
    print("=" * 60)
    
    # 检查数据文件
    data_dir = Path("crypto_data")
    if data_dir.exists():
        data_files = list(data_dir.glob("*.csv"))
        print(f"📁 数据文件: {len(data_files)} 个")
        for file in data_files:
            size_mb = file.stat().st_size / (1024 * 1024)
            print(f"   - {file.name}: {size_mb:.1f}MB")
    else:
        print("📁 数据文件: 无")
    
    # 检查结果文件
    results_dir = Path("results")
    if results_dir.exists():
        result_files = list(results_dir.glob("*.csv"))
        print(f"📊 结果文件: {len(result_files)} 个")
        for file in result_files:
            print(f"   - {file.name}")
    else:
        print("📊 结果文件: 无")
    
    # 检查依赖
    try:
        import pandas, requests, pytz, backtrader
        print("✅ 依赖包: 已安装")
    except ImportError as e:
        print(f"❌ 依赖包: 缺失 {e}")

def clean_results():
    """清理结果文件"""
    import shutil
    
    if Path("results").exists():
        shutil.rmtree("results")
        os.makedirs("results")
        print("🧹 结果文件已清理")
    else:
        print("📁 results目录不存在，已创建")
        os.makedirs("results")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='比特币KDJ背离分析系统',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  python run.py --setup       # 安装依赖
  python run.py --download    # 下载数据
  python run.py --analyze     # 多周期分析
  python run.py --backtest    # 运行回测
  python run.py --monitor     # 实时监控
  python run.py --status      # 项目状态
  python run.py --clean       # 清理结果
        """
    )
    
    parser.add_argument('--setup', action='store_true', help='安装依赖')
    parser.add_argument('--download', action='store_true', help='下载数据')
    parser.add_argument('--analyze', action='store_true', help='多周期背离分析')
    parser.add_argument('--backtest', action='store_true', help='运行优化回测')
    parser.add_argument('--monitor', action='store_true', help='实时信号监控')
    parser.add_argument('--status', action='store_true', help='项目状态')
    parser.add_argument('--clean', action='store_true', help='清理结果')
    
    args = parser.parse_args()
    
    print("🚀 比特币KDJ背离分析系统")
    print("=" * 60)
    
    if args.setup:
        setup_environment()
    elif args.download:
        download_data()
    elif args.analyze:
        run_multi_timeframe()
    elif args.backtest:
        run_backtest()
    elif args.monitor:
        run_monitor()
    elif args.status:
        show_project_status()
    elif args.clean:
        clean_results()
    else:
        print("请指定一个操作选项，使用 --help 查看帮助")
        print("\n🎯 常用命令:")
        print("  python run.py --status      # 查看项目状态")
        print("  python run.py --analyze     # 运行背离分析")
        print("  python run.py --backtest    # 运行回测策略")
        print("  python run.py --monitor     # 实时监控信号")

if __name__ == "__main__":
    main() 