import pandas as pd
import numpy as np
from typing import List, Dict, Any
import logging
from pathlib import Path
import os
from strategies.base_strategy import BaseStrategy
from strategies.ma_crossover_strategy import MACrossoverStrategy
from strategies.rsi_strategy import RSIStrategy
from strategies.macd_strategy import MACDStrategy
from strategies.bollinger_strategy import BollingerStrategy
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
from data_retrieval import retrieve_stock_data
from config import CONFIG  # 导入CONFIG

# 创建必要的目录
os.makedirs('logs', exist_ok=True)
os.makedirs('plots', exist_ok=True)

# 配置日志
logging.basicConfig(
    filename='logs/trading_strategy.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s: %(message)s'
)

class TradingStrategy:
    """Main trading strategy class that combines multiple sub-strategies"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.strategies: List[BaseStrategy] = [
            MACrossoverStrategy(config),
            RSIStrategy(config),
            MACDStrategy(config),
            BollingerStrategy(config)
        ]
        self.current_position = 0
        self.entry_price = 0
        self.high_since_entry = 0
        
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate signals from all strategies"""
        for strategy in self.strategies:
            data = strategy.generate_signals(data)
            
        # Combine signals (require majority agreement)
        signal_columns = [col for col in data.columns if col.endswith('_Signal')]
        data['Combined_Signal'] = data[signal_columns].sum(axis=1)
        data['Trade_Signal'] = 0
        
        # Generate trade signals when majority of strategies agree
        min_agreement = len(self.strategies) // 2 + 1
        data.loc[data['Combined_Signal'] >= min_agreement, 'Trade_Signal'] = 1
        data.loc[data['Combined_Signal'] <= -min_agreement, 'Trade_Signal'] = -1
        
        return data
        
    def should_enter_trade(self, data: pd.DataFrame, index: int) -> bool:
        """Check if we should enter a trade based on all strategies"""
        if self.current_position != 0:
            return False
            
        # Count strategies suggesting entry
        entry_votes = 0
        for strategy in self.strategies:
            should_enter, _ = strategy.should_enter_trade(data, index)
            if should_enter:
                entry_votes += 1
                
        # Enter if majority of strategies agree
        return entry_votes >= len(self.strategies) // 2 + 1
        
    def should_exit_trade(self, data: pd.DataFrame, index: int) -> bool:
        """Check if we should exit based on any strategy"""
        if self.current_position == 0:
            return False
            
        # Exit if any strategy suggests exit
        for strategy in self.strategies:
            if strategy.should_exit_trade(data, index, self.entry_price):
                return True
                
        return False
        
    def calculate_position_size(self, portfolio_value: float, price: float) -> float:
        """Calculate position size based on portfolio value"""
        position_pct = self.config.get('POSITION_SIZE_PCT', 0.1)
        return (portfolio_value * position_pct) / price
        
    def execute_trade(self, data: pd.DataFrame, index: int, portfolio: Dict[str, Any]) -> Dict[str, Any]:
        """Execute trading decision"""
        current_price = data.iloc[index]['close']
        
        # Update high since entry if in position
        if self.current_position > 0:
            self.high_since_entry = max(self.high_since_entry, current_price)
            
        # Check exit conditions
        if self.current_position != 0 and self.should_exit_trade(data, index):
            # Calculate profit/loss
            pnl = (current_price - self.entry_price) * self.current_position
            portfolio['cash'] += current_price * self.current_position + pnl
            
            logging.info(f"Exiting position at {current_price:.2f}, PnL: {pnl:.2f}")
            
            self.current_position = 0
            self.entry_price = 0
            self.high_since_entry = 0
            
        # Check entry conditions
        elif self.current_position == 0 and self.should_enter_trade(data, index):
            # Calculate position size
            position_size = self.calculate_position_size(
                portfolio['cash'],
                current_price
            )
            
            # Enter position
            cost = position_size * current_price
            if cost <= portfolio['cash']:
                self.current_position = position_size
                self.entry_price = current_price
                self.high_since_entry = current_price
                portfolio['cash'] -= cost
                
                logging.info(f"Entering position at {current_price:.2f}, Size: {position_size:.0f}")
                
        # Update portfolio value
        portfolio['value'] = portfolio['cash'] + (self.current_position * current_price)
        
        return portfolio
        
    def backtest_strategy(self, data: pd.DataFrame) -> pd.DataFrame:
        """Run backtest on historical data"""
        # Initialize portfolio
        portfolio = {
            'cash': self.config['INITIAL_CAPITAL'],
            'value': self.config['INITIAL_CAPITAL']
        }
        
        # Generate signals
        data = self.generate_signals(data)
        
        # Initialize results storage
        results = []
        
        # Run through each day
        for i in range(len(data)):
            # Execute trading logic
            portfolio = self.execute_trade(data, i, portfolio)
            
            # Store results
            results.append({
                'date': data.index[i],
                'close': data.iloc[i]['close'],
                'position': self.current_position,
                'portfolio_value': portfolio['value'],
                'cash': portfolio['cash']
            })
            
        # Convert results to DataFrame
        results_df = pd.DataFrame(results)
        results_df.set_index('date', inplace=True)
        
        return results_df

def print_performance_summary(portfolio_values, total_trades, winning_trades, trades):
    """打印策略表现总结"""
    total_profit = portfolio_values[-1] - CONFIG['INITIAL_CAPITAL']
    
    # 计算最大回撤
    peak = np.maximum.accumulate(portfolio_values)
    drawdown = (peak - portfolio_values) / peak * 100
    max_drawdown = np.max(drawdown)
    
    # 打印总体表现
    header = "策略回测结果 | Strategy Backtest Results"
    print(f"\n{'='*70}")
    print(f"{header:^70}")
    print(f"{'='*70}")
    
    metrics = [
        ('初始资金 | Initial Capital', f"{CONFIG['INITIAL_CAPITAL']:,.2f} CNY"),
        ('最终资金 | Final Capital', f"{portfolio_values[-1]:,.2f} CNY"),
        ('总收益 | Total Profit', f"{total_profit:,.2f} CNY"),
        ('收益率 | Return Rate', f"{(portfolio_values[-1]/CONFIG['INITIAL_CAPITAL'] - 1)*100:.2f}%"),
        ('总交易次数 | Total Trades', f"{total_trades}"),
        ('胜率 | Win Rate', f"{(winning_trades/total_trades*100 if total_trades > 0 else 0):.2f}%"),
        ('最大回撤 | Max Drawdown', f"{max_drawdown:.2f}%")
    ]
    
    for metric, value in metrics:
        print(f"{metric:<35} | {value:>32}")
    
    # 按ETF整理交易统计
    etf_stats = {}
    for trade in trades:
        stock_code = trade['stock']
        if stock_code not in etf_stats:
            etf_stats[stock_code] = {
                'trades': [],
                'total_trades': 0,
                'winning_trades': 0,
                'total_return': 0,
                'buy_trades': 0,
                'sell_trades': 0
            }
        
        etf_stats[stock_code]['trades'].append(trade)
        etf_stats[stock_code]['total_trades'] += 1
        
        if trade['type'] == 'buy':
            etf_stats[stock_code]['buy_trades'] += 1
        elif trade['type'] == 'sell':
            etf_stats[stock_code]['sell_trades'] += 1
            if trade['profit'] > 0:
                etf_stats[stock_code]['winning_trades'] += 1
            etf_stats[stock_code]['total_return'] += trade['profit_pct']
    
    # 打印每个ETF的交易统计
    if etf_stats:
        header = "ETF交易明细 | ETF Trading Details"
        print(f"\n{'='*70}")
        print(f"{header:^70}")
        print(f"{'='*70}")
        
        headers = ['ETF代码 | Code', '买入/卖出 | B/S', '平均收益率 | Avg Return', '胜率 | Win Rate']
        print(f"{headers[0]:<15} | {headers[1]:<15} | {headers[2]:<20} | {headers[3]:<15}")
        print(f"{'-'*70}")
        
        for stock_code, stats in etf_stats.items():
            etf_name = f"{stock_code} ({CONFIG['ETF_NAMES'][stock_code]})"
            sell_trades = stats['sell_trades']
            if sell_trades > 0:
                avg_return = stats['total_return'] / sell_trades
                win_rate = (stats['winning_trades'] / sell_trades * 100)
                trades_info = f"{stats['buy_trades']}/{sell_trades}"
                print(f"{etf_name:<15} | {trades_info:^15} | {avg_return:>18.2f}% | {win_rate:>13.2f}%")

def backtest_strategy():
    """
    执行回测策略
    """
    try:
        # 获取股票数据
        stock_data = {}
        for symbol in CONFIG['STOCK_UNIVERSE']:
            try:
                data = retrieve_stock_data(symbol, CONFIG['START_DATE'], CONFIG['END_DATE'])  # 调用retrieve_stock_data函数
                if not data.empty:
                    stock_data[symbol] = data
                    # 生成单个ETF的技术分析图表
                    plot_single_etf_metrics(symbol, data, None, 'plots')
            except Exception as e:
                logging.error(f"获取{symbol}数据时出错: {e}")
        
        if not stock_data:
            raise ValueError("没有成功获取任何股票数据")
        
        # 初始化结果数据结构
        results = pd.DataFrame(index=list(stock_data.values())[0].index)
        results['portfolio_value'] = CONFIG['INITIAL_CAPITAL']
        results['portfolio_value'] = results['portfolio_value'].astype(float)
        results['drawdown'] = 0.0
        
        # 为每个ETF创建position列，确保是float类型
        for symbol in CONFIG['STOCK_UNIVERSE']:
            results[f'position_{symbol}'] = 0.0
        
        # 执行回测逻辑...
        current_capital = CONFIG['INITIAL_CAPITAL']
        positions = {}  # 当前持仓
        portfolio_values = [current_capital]  # 记录每日的组合价值
        dates = [results.index[0]]  # 记录日期
        
        # 创建交易策略实例
        strategy = TradingStrategy(CONFIG)
        
        for date in results.index[1:]:
            # 对每个股票执行交易策略
            for symbol in CONFIG['STOCK_UNIVERSE']:
                if symbol in stock_data:
                    # 获取到当前日期的历史数据
                    hist_data = stock_data[symbol].loc[:date]
                    if not hist_data.empty:
                        # 执行交易策略
                        portfolio = {
                            'cash': current_capital,
                            'value': current_capital
                        }
                        portfolio = strategy.execute_trade(hist_data, -1, portfolio)
            
            # 更新持仓价值
            portfolio_value = portfolio['value']  # 现金部分
            portfolio_values.append(portfolio_value)
            dates.append(date)
            
            # 更新结果数据
            results.loc[date, 'portfolio_value'] = portfolio_value
            
            # 计算持仓比例
            total_value = portfolio_value
            for symbol in CONFIG['STOCK_UNIVERSE']:
                if symbol in positions:
                    if symbol in stock_data and date in stock_data[symbol].index:
                        position_value = positions[symbol]['shares'] * stock_data[symbol].loc[date, 'close']
                        results.loc[date, f'position_{symbol}'] = position_value / total_value
                else:
                    results.loc[date, f'position_{symbol}'] = 0.0
            
            # 计算回撤
            peak = results['portfolio_value'][:date].max()
            drawdown = (portfolio_value - peak) / peak
            results.loc[date, 'drawdown'] = drawdown
        
        # 生成投资组合相关的图表
        plot_portfolio_metrics(results, 'plots')
        
        # 打印最终结果
        final_value = portfolio_values[-1]
        total_return = (final_value - CONFIG['INITIAL_CAPITAL']) / CONFIG['INITIAL_CAPITAL'] * 100
        max_drawdown = results['drawdown'].min() * 100
        
        print(f"\n=== 回测结果 ===")
        print(f"初始资金: {CONFIG['INITIAL_CAPITAL']:,.2f}")
        print(f"最终资金: {final_value:,.2f}")
        print(f"总收益率: {total_return:.2f}%")
        print(f"最大回撤: {max_drawdown:.2f}%")
        
        return results
        
    except Exception as e:
        logging.error(f"回测过程中出错: {e}")
        raise

def plot_portfolio_metrics(results, save_dir='plots'):
    """Plot portfolio metrics."""
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # 设置中文字体和基本样式
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.alpha'] = 0.3
    
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Portfolio Value Plot / 投资组合价值图
    plt.figure(figsize=(15, 6))
    plt.plot(results['portfolio_value'], label='Portfolio Value / 投资组合价值', color='#1f77b4', linewidth=2)
    plt.title('Portfolio Value / 投资组合价值', fontsize=12, pad=15)
    plt.xlabel('Date / 日期', fontsize=10)
    plt.ylabel('Value (CNY) / 价值 (人民币)', fontsize=10)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gcf().autofmt_xdate()
    filename = f"portfolio_value_投资组合价值_{timestamp}.png"
    plt.savefig(os.path.join(save_dir, filename), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Positions Plot / 持仓比例图
    plt.figure(figsize=(16, 8))  # 加宽图表以适应更长的标签
    for symbol in CONFIG['STOCK_UNIVERSE']:
        etf_name = CONFIG['ETF_NAMES'][symbol]
        label = f"{symbol} - {etf_name}"
        plt.plot(results[f'position_{symbol}'], label=label)
    
    plt.title('ETF Positions / ETF持仓比例', fontsize=12, pad=15)
    plt.xlabel('Date / 日期', fontsize=10)
    plt.ylabel('Position Ratio / 持仓比例', fontsize=10)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10,
              borderaxespad=0., frameon=True, fancybox=True, framealpha=0.8,
              title='ETF Names / ETF名称', title_fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gcf().autofmt_xdate()
    plt.tight_layout()
    filename = f"positions_持仓比例_{timestamp}.png"
    plt.savefig(os.path.join(save_dir, filename), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Drawdown Plot / 回撤图
    plt.figure(figsize=(15, 6))
    plt.fill_between(results.index, results['drawdown'], 0, color='#e74c3c', alpha=0.3, 
                    label='Drawdown / 回撤')
    plt.title('Portfolio Drawdown / 投资组合回撤', fontsize=12, pad=15)
    plt.xlabel('Date / 日期', fontsize=10)
    plt.ylabel('Drawdown Ratio / 回撤比例', fontsize=10)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gcf().autofmt_xdate()
    filename = f"drawdown_回撤_{timestamp}.png"
    plt.savefig(os.path.join(save_dir, filename), dpi=300, bbox_inches='tight')
    plt.close()

def plot_single_etf_metrics(etf_code, prices, signals, save_dir='plots'):
    """Plot metrics for a single ETF."""
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
                
    etf_name = CONFIG['ETF_NAMES'][etf_code]
        
    # 设置中文字体和基本样式
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.alpha'] = 0.3
        
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 12))
        
    # Price and Moving Averages / 价格和移动平均线
    ax1.plot(prices.index, prices['close'], label='Price / 价格', color='#1f77b4', linewidth=2)
    ax1.plot(prices.index, prices['MA5'], label='MA5', color='#e74c3c', alpha=0.6)
    ax1.plot(prices.index, prices['MA10'], label='MA10', color='#2ecc71', alpha=0.6)
    ax1.plot(prices.index, prices['MA20'], label='MA20', color='#f39c12', alpha=0.6)
    ax1.plot(prices.index, prices['MA30'], label='MA30', color='#9b59b6', alpha=0.6)
        
    title = f"{etf_code} - {etf_name}\nPrice and Moving Averages / 价格和移动平均线"
    ax1.set_title(title, fontsize=12, pad=15)
    ax1.set_xlabel('Date / 日期', fontsize=10)
    ax1.set_ylabel('Price / 价格', fontsize=10)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        
    # RSI Plot / RSI指标
    ax2.plot(prices.index, prices['RSI'], label='RSI', color='#3498db', linewidth=2)
    ax2.axhline(y=70, color='#e74c3c', linestyle='--', alpha=0.5)
    ax2.axhline(y=30, color='#2ecc71', linestyle='--', alpha=0.5)
    ax2.set_title('RSI Indicator / RSI指标', fontsize=12, pad=15)
    ax2.set_xlabel('Date / 日期', fontsize=10)
    ax2.set_ylabel('RSI Value / RSI值', fontsize=10)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        
    # MACD Plot / MACD指标
    ax3.plot(prices.index, prices['MACD'], label='MACD', color='#3498db', linewidth=2)
    ax3.plot(prices.index, prices['Signal'], label='Signal / 信号线', color='#e74c3c', linewidth=1)
    ax3.bar(prices.index, prices['Histogram'], label='Histogram / 柱状图', color='#7f8c8d', alpha=0.3)
    ax3.set_title('MACD Indicator / MACD指标', fontsize=12, pad=15)
    ax3.set_xlabel('Date / 日期', fontsize=10)
    ax3.set_ylabel('MACD Value / MACD值', fontsize=10)
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        
    plt.tight_layout()
    filename = f"{etf_code}_{etf_name}_metrics_{timestamp}.png"
    plt.savefig(os.path.join(save_dir, filename), dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    backtest_strategy()
