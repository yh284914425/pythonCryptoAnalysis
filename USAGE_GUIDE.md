# 加密货币策略回测与模拟交易系统使用指南

## 项目概述

这是一个完整的加密货币多时间框架背离策略回测与模拟交易系统，基于 **"MACD找结构 + KDJ找买点"** 的核心交易逻辑。系统经过完整的重构，具有清晰的模块化架构，并通过严格的算法验证确保与原始交易逻辑100%一致。

## 系统架构

```
src/
├── indicators/           # 指标计算层
│   ├── macd.py          # MACD指标计算
│   └── kdj.py           # KDJ指标计算
├── analysis/            # 分析引擎层
│   ├── divergence.py    # 背离检测算法
│   ├── pattern_detector.py  # 技术模式识别
│   └── peak_trough_finder.py  # 高低点检测
├── strategies/          # 策略决策层
│   ├── mtf_divergence_strategy.py  # 主策略实现
│   ├── config.py        # 策略配置管理
│   └── base_strategy.py  # 策略基类
└── backtest/           # 回测与执行层
    ├── engine.py       # 回测引擎
    ├── portfolio.py    # 投资组合管理
    └── performance.py  # 性能分析
```

## 主要功能模块

### 1. 数据收集模块 (`src/data_collection/downData.py`)
- ✅ 自动下载币安历史K线数据
- ✅ 支持多个时间框架和交易对
- ✅ 增量更新机制
- ✅ 高精度小值币种支持

### 2. 策略核心算法
- ✅ **KDJ背离检测**: 完全保留原有J上穿J1/J1上穿J精确逻辑
- ✅ **MACD背离检测**: 基于金叉死叉的背离识别
- ✅ **多时间框架分析**: 宏观层(日线)、中观层(4小时)、微观层(1小时)
- ✅ **动态ATR计算**: 智能止损价格设定
- ✅ **可配置参数**: 支持保守、标准、激进三种模式

### 3. 回测系统 (`run_backtest.py`)
- ✅ **基准测试**: 标准配置完整历史数据测试
- ✅ **参数敏感性分析**: 止损倍率和信号阈值测试
- ✅ **鲁棒性检验**: 跨资产和不同市场周期测试
- ✅ **完整性能指标**: 夏普比率、最大回撤、胜率等

### 4. 模拟交易系统 (`paper_trading.py`)
- ✅ **实时数据获取**: 自动获取币安最新K线数据
- ✅ **定时任务调度**: 可配置的分析周期
- ✅ **完整交易日志**: 信号、交易、投资组合状态记录
- ✅ **实时监控**: 状态报告和性能追踪

### 5. 结果分析工具 (`analyze_results.py`)
- ✅ **性能对比图表**: 多维度可视化分析
- ✅ **风险收益分析**: 散点图和相关性分析
- ✅ **模式对比**: 不同策略模式效果比较
- ✅ **综合报告**: 自动生成分析文档

## 快速开始

### 环境准备

1. 确保Python 3.8+环境
2. 安装依赖：
```bash
uv sync  # 或 pip install -r requirements.txt
```

### 第一步：数据准备

```bash
# 下载历史数据（首次运行需要一些时间）
python src/data_collection/downData.py

# 数据将保存在 crypto_data/ 目录下，按币种分类：
# crypto_data/BTC/1h.csv, 4h.csv, 1d.csv
# crypto_data/ETH/1h.csv, 4h.csv, 1d.csv  
# crypto_data/PEPE/1h.csv, 4h.csv, 1d.csv
```

### 第二步：运行回测

```bash
# 基准测试（推荐首次运行）
python run_backtest.py --test-type baseline --symbol BTCUSDT --capital 100000

# 完整测试套件
python run_backtest.py --test-type all --symbol BTCUSDT --capital 100000

# 参数敏感性分析
python run_backtest.py --test-type sensitivity --symbol BTCUSDT

# 鲁棒性检验
python run_backtest.py --test-type robustness
```

### 第三步：模拟交易

```bash
# 启动模拟交易（标准模式，每小时检查一次）
python paper_trading.py --mode standard --capital 10000 --interval 60

# 使用不同策略模式
python paper_trading.py --mode aggressive --capital 10000 --interval 30

# 运行指定时长
python paper_trading.py --mode conservative --duration 24  # 运行24小时
```

### 第四步：结果分析

```bash
# 生成综合分析报告
python analyze_results.py

# 指定特定目录
python analyze_results.py --backtest-dir backtest_results --paper-dir paper_trading_logs
```

## 策略配置说明

系统支持三种预设策略模式：

### 保守模式 (Conservative)
- 最大单仓：20%
- 最大总仓：60% 
- 信号确认：需要4个指标
- AI置信度：80%
- 适合风险厌恶型投资者

### 标准模式 (Standard) 
- 最大单仓：30%
- 最大总仓：80%
- 信号确认：需要3个指标
- AI置信度：70%
- 平衡风险和收益

### 激进模式 (Aggressive)
- 最大单仓：40%
- 最大总仓：90%
- 信号确认：需要2个指标
- AI置信度：60%
- 追求高收益，承担更高风险

## 核心算法说明

### KDJ背离算法
保留原有精确逻辑：
- **底部背离**: J上穿J1时，价格创新低但J值未创新低，且J<20
- **顶部背离**: J1上穿J时，价格创新高但J1值未创新高，且J>90

### MACD背离算法  
基于金叉死叉：
- **底部背离**: 金叉时，价格创新低但MACD未创新低
- **顶部背离**: 死叉时，价格创新高但MACD未创新高

### 止损机制
动态ATR计算：
- **BTC**: ATR × 2.0倍
- **ETH**: ATR × 1.8倍  
- **山寨币**: ATR × 4.0倍

## 输出文件说明

### 回测结果 (`backtest_results/`)
- `baseline_test_*.json`: 基准测试结果
- `sensitivity_analysis_*.json`: 参数敏感性分析
- `robustness_check_*.json`: 鲁棒性检验结果

### 模拟交易日志 (`paper_trading_logs/`)
- `paper_trading_*.log`: 详细运行日志
- `final_report_*.json`: 最终交易报告
- `signals_history_*.json`: 信号历史记录
- `trades_history_*.json`: 交易历史记录

### 分析输出 (`analysis_output/`)
- `performance_comparison.png`: 性能对比图表
- `risk_return_scatter.png`: 风险收益散点图
- `equity_curves.png`: 权益曲线图
- `mode_comparison.csv`: 模式对比表
- `detailed_backtest_results.csv`: 详细回测数据

## 常见问题解答

### Q1: 数据下载失败怎么办？
A: 检查网络连接，币安API有速率限制。可以多次运行数据下载脚本，系统会自动增量更新。

### Q2: 回测结果没有交易信号？
A: 这是正常的，策略有严格的信号确认机制。可以尝试：
- 使用激进模式降低信号阈值
- 选择不同的时间范围
- 检查是否有足够的历史数据

### Q3: 模拟交易如何停止？
A: 使用 Ctrl+C 停止，系统会自动生成最终报告。

### Q4: 如何修改策略参数？
A: 编辑 `src/strategies/config.py` 文件，或者创建新的配置类。

### Q5: 支持其他交易所数据吗？
A: 目前仅支持币安。如需其他交易所，需要修改 `src/data_collection/downData.py` 中的API调用。

## 系统测试

提供了完整的测试脚本：

```bash
# 测试回测系统
python test_backtest.py

# 测试模拟交易系统  
python test_paper_trading.py
```

## 注意事项

1. **策略仅供学习研究**: 本系统设计用于教育和研究目的，实盘交易前请充分测试
2. **数据准确性**: 确保网络连接稳定，避免数据不完整影响分析结果
3. **资金管理**: 建议先用小资金测试，确认策略效果后再考虑增加投入
4. **风险控制**: 严格遵循止损设置，避免过度杠杆和重仓操作

## 版本历史

- **v1.0**: 原始策略实现
- **v2.0**: 完整重构，模块化架构  
- **v2.1**: 添加动态ATR和配置管理
- **v2.2**: 完成回测和模拟交易系统
- **v2.3**: 添加结果分析和可视化工具

---

🎯 **策略核心理念**: MACD看大方向，KDJ找精确入场点，多时间框架确认，严格风险控制

📈 **期望效果**: 在趋势行情中捕捉主要机会，在震荡行情中控制回撤

⚠️  **风险提示**: 任何交易策略都存在风险，过往表现不代表未来收益，请理性投资