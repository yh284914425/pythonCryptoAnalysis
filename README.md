# 🚀 比特币KDJ背离分析系统

一个专业的比特币技术分析工具，专注于KDJ指标的背离信号检测和交易策略分析。

## 📁 项目结构

```
crypto/
├── src/                          # 源代码目录
│   ├── data_collection/          # 数据收集模块
│   │   ├── __init__.py
│   │   └── downData.py          # 币安数据下载工具
│   ├── analysis/                 # 分析模块
│   │   ├── __init__.py
│   │   ├── divergence_analysis.py        # 基础背离分析
│   │   ├── advanced_divergence_analysis.py # 高级分析和回测
│   │   └── multi_timeframe_analysis.py   # 多周期分析
│   └── strategies/               # 交易策略模块
│       ├── __init__.py
│       ├── strategy_analysis.py         # 策略分析
│       └── exit_strategy_analysis.py    # 卖出策略分析
├── crypto_data/                  # 原始数据文件
├── results/                      # 分析结果
├── config/                       # 配置文件
│   ├── requirements.txt
│   └── .gitignore
├── docs/                         # 文档
└── venv/                         # 虚拟环境
```

## 🛠️ 安装和使用

### 1. 环境准备
```bash
# 激活虚拟环境
source venv/bin/activate  # macOS/Linux
# 或 venv\Scripts\activate  # Windows

# 安装依赖
pip install -r config/requirements.txt
```

### 2. 数据收集
```bash
# 下载币安数据
python src/data_collection/downData.py
```

### 3. 背离分析
```bash
# 基础背离分析
python src/analysis/divergence_analysis.py

# 高级分析和回测
python src/analysis/advanced_divergence_analysis.py

# 多周期分析
python src/analysis/multi_timeframe_analysis.py
```

### 4. 策略分析
```bash
# 交易策略分析
python src/strategies/strategy_analysis.py

# 卖出策略分析
python src/strategies/exit_strategy_analysis.py
```

## 📊 核心功能

### 🔍 背离检测
- **KDJ指标计算**：K、D、J值的精确计算
- **背离识别**：自动识别顶部和底部背离信号
- **信号强度分级**：强、中、弱三个等级
- **多周期支持**：1h、2h、4h、8h、12h、1d、3d、1w

### 📈 策略分析
- **底部背离策略**：70%成功率的买入策略
- **多周期共振**：提高信号可靠性
- **价格区间策略**：根据价格位置调整策略
- **风险管理**：完整的止损和止盈机制

### 💰 卖出策略
- **涨幅目标法**：根据价格区间设定目标
- **KDJ指标法**：基于技术指标的卖出时机
- **顶部背离确认**：反向信号确认卖出
- **综合智能策略**：多重确认机制

## 📝 使用示例

```python
# 导入模块
from src.analysis.divergence_analysis import DivergenceAnalyzer
from src.strategies.strategy_analysis import analyze_trading_strategies

# 创建分析器
analyzer = DivergenceAnalyzer()

# 分析背离信号
signals = analyzer.analyze_data('crypto_data/BTCUSDT_1h.csv')

# 策略分析
analyze_trading_strategies('results/所有周期背离数据_20250529_235931.csv')
```

## 📋 主要特性

- ✅ **高精度算法**：基于JavaScript原版算法精确转换
- ✅ **多周期分析**：支持8个不同时间周期
- ✅ **历史回测**：验证策略有效性
- ✅ **风险控制**：完整的资金管理体系
- ✅ **数据导出**：CSV格式结果导出
- ✅ **可视化支持**：详细的分析报告

## ⚠️ 免责声明

本工具仅供教育和研究目的使用。任何投资决策都应该基于您自己的研究和风险承受能力。加密货币投资存在高风险，可能导致全部资金损失。

## 📧 联系方式

如有问题或建议，请通过Issues页面联系我们。

---

**Happy Trading! 📈🚀** 