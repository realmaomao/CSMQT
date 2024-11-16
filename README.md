# Chinese Stock Market Quant Trading

## Overview
This is a quantitative trading program designed for the Chinese stock market, utilizing multiple technical indicators and risk management strategies. The system uses Akshare for data retrieval and implements a comprehensive trading strategy based on technical analysis.

## Features
- Automated stock data retrieval using Akshare
- Multiple technical indicators:
  - RSI (Relative Strength Index, 14-period Wilder's RSI)
  - MACD (12,26,9 standard parameters)
  - Bollinger Bands (20-period, 2 standard deviations)
  - Moving Averages (Fast: 5, Mid: 10, Slow: 20, Trend: 30)
- Advanced risk management:
  - Position sizing controls
  - Stop-loss mechanisms (5%)
  - Trailing stops (8%)
  - Maximum position limits
- Comprehensive backtesting framework

## Technical Details
### Indicator Calculations
- **RSI**: Uses Wilder's original calculation method with 14-period lookback
  - Oversold threshold: 30
  - Overbought threshold: 70
- **MACD**: Standard parameters
  - Fast EMA: 12 periods
  - Slow EMA: 26 periods
  - Signal line: 9 periods
- **Bollinger Bands**: Standard implementation
  - Period: 20 days
  - Width: 2 standard deviations

### Risk Management
- Maximum simultaneous positions: 5
- Position size: 10% of portfolio
- Stop-loss: 5% from entry
- Trailing stop: 8% from highest point

## Setup
1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Configure trading parameters in `config.py`
   - Adjust technical indicator parameters
   - Set risk management preferences
   - Configure data retrieval settings

## Project Structure
- `main.py`: Entry point and main execution logic
- `trading_strategy.py`: Core trading strategy implementation
- `data_retrieval.py`: Data fetching and technical analysis
- `config.py`: Configuration and parameter settings

## Usage
1. Run the main trading script:
```bash
python main.py
```

2. Monitor the logs for:
   - Trade signals
   - Position updates
   - Risk management actions
   - Portfolio performance

## Performance Monitoring
The system provides detailed logging of:
- Trade entry/exit points
- Position sizes and risk levels
- Technical indicator signals
- Portfolio value updates

## Disclaimer
This is a research project intended for educational purposes. Always perform thorough testing and risk assessment before applying any trading strategy with real money. Past performance does not guarantee future results.
