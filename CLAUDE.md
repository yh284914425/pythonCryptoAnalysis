# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a sophisticated cryptocurrency trading strategy system that implements multi-timeframe divergence analysis for automated trading. The system combines MACD and KDJ technical indicators with advanced risk management and AI-enhanced decision making.

## Core Architecture

### Main Components

1. **Trading Strategy Engine** (`src/strategies/`)
   - `main_strategy.py`: Multi-timeframe divergence strategy controller
   - `config.py`: Strategy configuration management (conservative/standard/aggressive modes)
   - `technical_indicators/`: Technical analysis modules including KDJ, MACD, ATR calculations

2. **Backtesting Framework** (`src/backtest/`)
   - `mtf_divergence_strategy.py`: Complete backtesting implementation
   - `run_mtf_strategy.py`: Backtest execution and parameter optimization
   - `backtest.py`: General backtesting utilities

3. **Data Management** (`src/data_collection/`)
   - `downData.py`: Binance API integration for historical data fetching
   - `crypto_data/`: Historical price data organized by symbol and timeframe

4. **Analysis Tools** (`src/analysis/`)
   - `macd_kdj_divergence.py`: Core divergence detection algorithms
   - `multi_timeframe_analysis.py`: Cross-timeframe signal confirmation
   - `divergence_analysis.py`: Advanced divergence pattern recognition

### Strategy Logic

The system operates on a **multi-timeframe approach**:
- **HTF (High Time Frame)**: 3d/1d - Trend filtering and market state assessment
- **ITF (Intermediate Time Frame)**: 4h - Primary signal generation
- **LTF (Low Time Frame)**: 1h - Entry timing and execution

**Signal Requirements**:
- MACD bottom divergence on ITF
- KDJ bottom divergence on ITF (J < 20)
- HTF trend filter confirmation
- LTF price action trigger (engulfing patterns, hammer candles, KDJ golden cross)

## Common Development Commands

### Environment Setup
```bash
# Install dependencies using UV package manager
uv pip install -r requirements.txt

# Or using standard pip
pip install -r requirements.txt
```

### Running Backtests
```bash
# Run multi-timeframe strategy backtest
python src/backtest/run_mtf_strategy.py

# Run main strategy test
python src/strategies/main_strategy.py

# Run divergence analysis
python src/analysis/macd_kdj_divergence.py
```

### Data Collection
```bash
# Download historical data for specific symbols
python src/data_collection/downData.py
```

### Testing
```bash
# Run basic functionality test
python test.py

# No formal test framework - each module has __main__ section for testing
```

## Key Configuration

### Strategy Modes
- **Conservative**: 20% max single position, 80% AI confidence, 4 signal confirmations
- **Standard**: 30% max single position, 70% AI confidence, 3 signal confirmations  
- **Aggressive**: 40% max single position, 60% AI confidence, 2 signal confirmations

### Risk Management Parameters
- **Stop Loss**: ATR-based (BTC: 2.0x, ETH: 1.8x, ALT: 4.0x)
- **Position Sizing**: Kelly criterion with volatility adjustment
- **Risk per Trade**: Configurable (default 2% of capital)

### Technical Indicators
- **KDJ Parameters**: Adaptive based on volatility (high: 18,5,5 | medium: 14,7,7 | low: 21,10,10)
- **MACD**: Standard 12-26-9 configuration
- **ATR Period**: 14 (for volatility and stop-loss calculation)

## Data Structure

### Price Data Format
Expected CSV columns (Chinese headers):
- `开盘时间` (Open Time)
- `开盘价` (Open Price)  
- `最高价` (High Price)
- `最低价` (Low Price)
- `收盘价` (Close Price)
- `成交量` (Volume)
- `成交额` (Quote Volume)

### File Organization
```
crypto_data/
├── BTC/
│   ├── 1h.csv
│   ├── 4h.csv
│   ├── 1d.csv
│   └── 3d.csv
├── ETH/
└── PEPE/
```

## Development Guidelines

### Adding New Indicators
1. Implement in `src/strategies/technical_indicators/`
2. Add to `TechnicalAnalyzer` class
3. Include in signal generation logic
4. Update configuration parameters

### Creating New Strategies
1. Inherit from base strategy pattern in `main_strategy.py`
2. Implement required methods: `analyze_market()`, `execute_trade()`
3. Add to configuration options
4. Create corresponding backtest module

### Risk Management Integration
- All trades must go through `RiskManager` validation
- Position sizing calculated using `calculate_position_size()`
- Stop-loss automatically set based on ATR
- Maximum exposure limits enforced

## Important Notes

### Dependencies
- Core: pandas, numpy, matplotlib, requests
- Technical Analysis: ta, mplfinance
- Backtesting: backtrader
- AI Components: torch, transformers (if using AI features)

### Limitations
- Chinese language comments and variable names throughout codebase
- No formal unit testing framework
- Hardcoded timeframes in some modules
- Manual data download process

### Performance Considerations
- Large CSV files for historical data (multiple timeframes per symbol)
- Memory intensive operations in backtesting
- API rate limits for data collection from Binance

### File Naming Conventions
- Strategy modules use snake_case
- Data files organized by symbol/timeframe.csv
- Results saved with timestamp suffixes
- Log files in dedicated `logs/` directory

This system is designed for educational and research purposes in cryptocurrency trading strategy development. All trading involves significant risk and past performance does not guarantee future results.