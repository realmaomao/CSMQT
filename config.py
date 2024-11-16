import os

# Trading Configuration
CONFIG = {
    'INITIAL_CAPITAL': 100000,  # Initial trading capital in CNY
    'RISK_PER_TRADE': 0.02,     # Maximum risk per trade (2%)
    'TRADING_PERIOD': 'daily',  # Trading timeframe
    'STOCK_UNIVERSE': [         # List of ETFs to trade
    #    '510300.SH',  # 沪深300ETF
    #    '510500.SH',  # 中证500ETF
        '159915.SZ',  # 创业板ETF
        '588000.SH',  # 科创50ETF
        '512170.SH',  # 医疗ETF
        '512690.SH',  # 酒ETF
        '515050.SH',  # 5GETF
        '512480.SH',  # 半导体芯片ETF
        '159870.SZ',  # 鸿蒙概念ETF
    ],
    
    # ETF中文名称映射
    'ETF_NAMES': {
    #    '510300.SH': '沪深300ETF',
    #    '510500.SH': '中证500ETF',
        '159915.SZ': '创业板ETF',
        '588000.SH': '科创50ETF',
        '512170.SH': '医疗ETF',
        '512690.SH': '白酒ETF',
        '515050.SH': '5GETF',
        '512480.SH': '半导体ETF',
        '159870.SZ': '鸿蒙ETF'
    },
    
    'START_DATE': '2020-01-01',  # 使用更近的开始日期
    'END_DATE': '2024-11-10',    # 使用更近的结束日期
    
    # Technical Strategy Parameters
    'MA_FAST': 5,    # Fast Moving Average
    'MA_MID': 10,    # Mid-term Moving Average
    'MA_SLOW': 20,   # Slow Moving Average
    'MA_TREND': 30,  # Trend Moving Average
    
    # RSI Parameters
    'RSI_PERIOD': 14,
    'RSI_OVERSOLD': 30,    # 调整RSI超卖阈值
    'RSI_OVERBOUGHT': 70,  # 调整RSI超买阈值
    
    # MACD Parameters (仅用于参考，实际计算使用标准参数)
    'MACD_FAST': 12,    # 标准MACD快线
    'MACD_SLOW': 26,    # 标准MACD慢线
    'MACD_SIGNAL': 9,   # 标准MACD信号线
    
    # Bollinger Bands Parameters
    'BB_PERIOD': 20,    # 标准布林带周期
    'BB_STD': 2.0,      # 标准布林带宽度
    
    # Risk Management
    'MAX_POSITIONS': 5,          # Maximum number of simultaneous positions
    'STOP_LOSS_PCT': 0.05,       # 5% stop loss
    'TRAILING_STOP_PCT': 0.08,   # 8% trailing stop
    'MAX_DRAWDOWN_PCT': 0.15,    # 15% maximum drawdown
    'POSITION_SIZE_PCT': 0.1,    # Maximum 10% per position
    'MIN_PROFIT_TARGET': 0.03,   # 3% minimum profit target
    
    # Trading Rules
    'MIN_VOLUME': 1000000,       # Minimum daily volume
    'MIN_PRICE': 1.0,            # Minimum price
    'MAX_SPREAD_PCT': 0.02,      # Maximum bid-ask spread
    'MIN_SIGNALS': 2,            # Minimum number of confirming signals
    
    # Performance Metrics
    'BENCHMARK': '510300.SH',    # Benchmark index
    'RISK_FREE_RATE': 0.03,      # Annual risk-free rate
}

# Ensure logs directory exists
os.makedirs('logs', exist_ok=True)
