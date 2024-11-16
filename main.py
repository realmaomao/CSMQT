import logging
import os
from pathlib import Path
from data_retrieval import retrieve_multiple_stocks
from trading_strategy import backtest_strategy
from config import CONFIG

# Create necessary directories
Path('logs').mkdir(exist_ok=True)
Path('plots').mkdir(exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler('logs/main.log'),
        logging.StreamHandler()
    ]
)

def generate_plots(stock_data):
    """
    Generate and save plots with bilingual labels (Chinese and English)
    """
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    import datetime
    
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 设置基本样式
    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.alpha'] = 0.3
    
    plots_dir = Path('plots')
    plots_dir.mkdir(exist_ok=True)
    
    for symbol in stock_data.keys():
        df = stock_data[symbol]
        etf_name = CONFIG['ETF_NAMES'].get(symbol, symbol)
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Price Plot
        plt.figure(figsize=(15, 7))
        plt.plot(df.index, df['close'], label='Close Price / 收盘价', linewidth=2, color='#1f77b4')
        
        title = f"{symbol} - {etf_name}\nPrice Chart / 价格走势图"
        plt.title(title, fontsize=12, pad=15)
        plt.xlabel('Date / 日期', fontsize=10)
        plt.ylabel('Price / 价格 (CNY)', fontsize=10)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        
        # 格式化x轴日期
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.gcf().autofmt_xdate()  # 自动旋转日期标签
        
        filename = f"{symbol}_{etf_name}_price_{timestamp}.png"
        plt.savefig(plots_dir / filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Volume Plot
        plt.figure(figsize=(15, 5))
        plt.bar(df.index, df['volume'], label='Volume / 成交量', alpha=0.6, color='#2ecc71')
        
        title = f"{symbol} - {etf_name}\nVolume Chart / 成交量图"
        plt.title(title, fontsize=12, pad=15)
        plt.xlabel('Date / 日期', fontsize=10)
        plt.ylabel('Volume / 成交量', fontsize=10)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.gcf().autofmt_xdate()
        
        filename = f"{symbol}_{etf_name}_volume_{timestamp}.png"
        plt.savefig(plots_dir / filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        logging.info(f"Generated plots for {symbol} ({etf_name})")

def main():
    """
    Main entry point for the quantitative trading program
    """
    try:
        print(f"\nWorking directory: {os.getcwd()}")
        print(f"Plots directory: {os.path.abspath('plots')}")
        
        # Retrieve stock data
        logging.info("Starting data retrieval...")
        stock_data = retrieve_multiple_stocks(
            CONFIG['STOCK_UNIVERSE'], 
            CONFIG['START_DATE'], 
            CONFIG['END_DATE']
        )
        
        # Generate basic plots
        logging.info("Generating basic plots...")
        generate_plots(stock_data)
        
        # Run backtesting
        logging.info("Starting backtesting...")
        print("\nStarting backtesting strategy...")
        results = backtest_strategy()  # 获取回测结果
        
        if results is not None:
            logging.info("Backtesting completed successfully.")
            print("\nAll plots have been saved to:", os.path.abspath('plots'))
        else:
            logging.warning("Backtesting completed but no results were returned.")
        
    except Exception as e:
        logging.error(f"An error occurred during program execution: {e}")
        import traceback
        print(f"Error traceback:\n{traceback.format_exc()}")

if __name__ == "__main__":
    main()
