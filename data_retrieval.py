import akshare as ak
import pandas as pd
import numpy as np
from tqdm import tqdm
import logging

# Configure logging
logging.basicConfig(filename='logs/data_retrieval.dat', level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s: %(message)s')

def fetch_stock_data(stock_code, start_date, end_date):
    """
    Retrieve historical stock data using Akshare
    """
    try:
        print(f"\nTrying to fetch data for {stock_code}")
        print(f"Period: {start_date} to {end_date}")
        
        # 从股票代码中提取数字部分
        symbol = stock_code.split('.')[0]
        
        try:
            # 尝试使用ETF接口获取数据
            df = ak.fund_etf_hist_em(
                symbol=symbol,
                period="daily",
                start_date=start_date.replace('-', ''),
                end_date=end_date.replace('-', ''),
                adjust="hfq"
            )
            print(f"Successfully fetched ETF data for {stock_code}")
        except Exception as e:
            print(f"Failed to fetch ETF data, trying stock interface: {e}")
            # 如果ETF接口失败，尝试使用股票接口
            df = ak.stock_zh_a_hist(
                symbol=symbol,
                period="daily",
                start_date=start_date.replace('-', ''),
                end_date=end_date.replace('-', ''),
                adjust="hfq"
            )
            print(f"Successfully fetched stock data for {stock_code}")
        
        if df.empty:
            print(f"No data returned for {stock_code}")
            return pd.DataFrame()
            
        # 统一列名
        column_mapping = {
            '日期': 'date',
            '开盘': 'open',
            '收盘': 'close',
            '最高': 'high',
            '最低': 'low',
            '成交量': 'volume',
            '成交额': 'amount'
        }
        df.rename(columns=column_mapping, inplace=True)
        
        # 设置日期索引
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        
        # 确保数据按日期排序
        df.sort_index(inplace=True)
        
        print(f"Processed data shape: {df.shape}")
        return df
        
    except Exception as e:
        print(f"Error fetching data for {stock_code}: {str(e)}")
        return pd.DataFrame()

def retrieve_stock_data(stock_code, start_date, end_date):
    """
    获取单个股票数据并计算技术指标
    """
    df = fetch_stock_data(stock_code, start_date, end_date)
    if not df.empty:
        df = calculate_technical_indicators(df)
    return df

def retrieve_multiple_stocks(stock_codes, start_date, end_date):
    """
    获取多个股票的数据
    """
    stock_data = {}
    for code in tqdm(stock_codes, desc="Fetching stock data"):
        try:
            df = retrieve_stock_data(code, start_date, end_date)
            if not df.empty:
                stock_data[code] = df
                print(f"Successfully retrieved data for {code}")
            else:
                print(f"No data retrieved for {code}")
        except Exception as e:
            print(f"Error processing {code}: {str(e)}")
            logging.error(f"Error processing {code}: {str(e)}")
    
    return stock_data

def calculate_rsi(prices, periods=None):
    """
    Calculate Relative Strength Index (RSI)
    """
    from config import CONFIG
    
    # 使用配置文件中的参数，如果没有提供参数的话
    periods = periods or CONFIG['RSI_PERIOD']
    
    # 计算价格变化
    delta = prices.diff()
    
    # 分离上涨和下跌
    gain = (delta.where(delta > 0, 0)).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)
    
    # 首先计算简单移动平均
    avg_gain = gain.rolling(window=periods, min_periods=1).mean()
    avg_loss = loss.rolling(window=periods, min_periods=1).mean()
    
    # 然后计算指数移动平均
    avg_gain = avg_gain.ewm(alpha=1/periods, min_periods=periods, adjust=False).mean()
    avg_loss = avg_loss.ewm(alpha=1/periods, min_periods=periods, adjust=False).mean()
    
    # 计算相对强度，避免除以零
    rs = avg_gain / avg_loss.replace(0, float('inf'))
    
    # 计算RSI
    rsi = 100 - (100 / (1 + rs))
    
    return rsi

def calculate_macd(prices):
    """
    Calculate MACD (Moving Average Convergence Divergence)
    使用标准的MACD参数：快线=12，慢线=26，信号线=9
    """
    # 使用标准的MACD参数，而不是配置文件中的参数
    exp1 = prices.ewm(span=12, adjust=False).mean()
    exp2 = prices.ewm(span=26, adjust=False).mean()
    macd = exp1 - exp2
    signal = macd.ewm(span=9, adjust=False).mean()
    
    return macd, signal

def calculate_bollinger_bands(prices, window=None, num_std=None):
    """
    Calculate Bollinger Bands with minimum periods requirement
    """
    from config import CONFIG
    
    # 使用配置文件中的参数，如果没有提供参数的话
    window = window or CONFIG['BB_PERIOD']
    num_std = num_std or CONFIG['BB_STD']
    
    # 添加最小周期要求
    sma = prices.rolling(window=window, min_periods=1).mean()
    std = prices.rolling(window=window, min_periods=1).std()
    
    upper_band = sma + (std * num_std)
    lower_band = sma - (std * num_std)
    
    return upper_band, lower_band

def calculate_atr(df, window=None):
    """
    Calculate Average True Range (ATR)
    """
    from config import CONFIG
    
    # 使用配置文件中的参数，如果没有提供参数的话
    window = window or CONFIG['BB_PERIOD']  # 使用布林带周期作为默认ATR周期
    
    high = df['high']
    low = df['low']
    close = df['close']
    
    # 计算真实范围
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # 计算ATR
    atr = tr.rolling(window=window).mean()
    
    return atr

def calculate_fear_greed_index(df):
    """
    计算贪婪恐慌指数
    基于市场动量、波动率、RSI和MACD等指标综合计算
    """
    from config import CONFIG
    
    # 计算动量分数 (基于快速MA周期)
    momentum = df['Momentum'].rolling(window=CONFIG['MA_FAST']).mean()
    momentum_score = ((momentum - momentum.rolling(window=CONFIG['MA_SLOW']).min()) / 
                     (momentum.rolling(window=CONFIG['MA_SLOW']).max() - 
                      momentum.rolling(window=CONFIG['MA_SLOW']).min()) * 100)
    
    # 计算波动率分数
    volatility = df['Volatility']
    volatility_score = 100 - ((volatility - volatility.rolling(window=CONFIG['BB_PERIOD']).min()) / 
                             (volatility.rolling(window=CONFIG['BB_PERIOD']).max() - 
                              volatility.rolling(window=CONFIG['BB_PERIOD']).min()) * 100)
    
    # 计算RSI分数
    rsi = df['RSI']
    rsi_score = rsi
    
    # 计算MACD分数
    macd_hist = df['MACD_Hist']
    macd_score = ((macd_hist - macd_hist.rolling(window=CONFIG['MACD_SLOW']).min()) / 
                  (macd_hist.rolling(window=CONFIG['MACD_SLOW']).max() - 
                   macd_hist.rolling(window=CONFIG['MACD_SLOW']).min()) * 100)
    
    # 计算布林带位置分数
    bb_score = ((df['close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower']) * 100)
    
    # 综合计算贪婪恐慌指数
    fear_greed = (
        momentum_score * 0.2 +     # 动量权重20%
        volatility_score * 0.2 +   # 波动率权重20%
        rsi_score * 0.2 +          # RSI权重20%
        macd_score * 0.2 +         # MACD权重20%
        bb_score * 0.2             # 布林带权重20%
    ).fillna(50)
    
    # 将指数限制在0-100之间
    fear_greed = fear_greed.clip(0, 100)
    
    return fear_greed

def calculate_technical_indicators(df):
    """
    Calculate all technical indicators
    """
    from config import CONFIG
    
    # Moving Averages
    df[f'MA{CONFIG["MA_FAST"]}'] = df['close'].rolling(window=CONFIG['MA_FAST']).mean()
    df[f'MA{CONFIG["MA_MID"]}'] = df['close'].rolling(window=CONFIG['MA_MID']).mean()
    df[f'MA{CONFIG["MA_SLOW"]}'] = df['close'].rolling(window=CONFIG['MA_SLOW']).mean()
    df[f'MA{CONFIG["MA_TREND"]}'] = df['close'].rolling(window=CONFIG['MA_TREND']).mean()
    
    # RSI
    df['RSI'] = calculate_rsi(df['close'], periods=CONFIG['RSI_PERIOD'])
    
    # MACD
    df['MACD'], df['MACD_Signal'] = calculate_macd(df['close'])
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
    
    # Bollinger Bands
    df['BB_Upper'], df['BB_Lower'] = calculate_bollinger_bands(df['close'], 
                                                             window=CONFIG['BB_PERIOD'], 
                                                             num_std=CONFIG['BB_STD'])
    df['BB_Middle'] = df['close'].rolling(window=CONFIG['BB_PERIOD']).mean()
    
    # ATR
    df['ATR'] = calculate_atr(df)
    
    # Momentum
    df['Momentum'] = df['close'] - df['close'].shift(CONFIG['MA_FAST'])
    
    # Volatility
    df['Volatility'] = df['close'].rolling(window=CONFIG['BB_PERIOD']).std()
    
    # Fear & Greed Index
    df['Fear_Greed'] = calculate_fear_greed_index(df)
    
    return df

def main():
    from config import CONFIG
    
    # Retrieve stock data
    stock_data = retrieve_multiple_stocks(
        CONFIG['STOCK_UNIVERSE'], 
        CONFIG['START_DATE'], 
        CONFIG['END_DATE']
    )
    
    # Save retrieved data
    for stock, df in stock_data.items():
        df.to_csv(f'data/{stock}_historical_data.csv', index=True)
    
    logging.info("Data retrieval completed successfully")

if __name__ == "__main__":
    # Create data directory if it doesn't exist
    import os
    os.makedirs('data', exist_ok=True)
    
    main()
