import pandas as pd
import numpy as np
import logging
import matplotlib
matplotlib.use('TkAgg')  # 显式设置后端
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.dates import DateFormatter
from config import CONFIG
import datetime
import os
from pathlib import Path
import matplotlib.dates as mdates
from data_retrieval import retrieve_stock_data

# 创建必要的目录
os.makedirs('logs', exist_ok=True)
os.makedirs('plots', exist_ok=True)

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# Configure logging
logging.basicConfig(filename='logs/trading_strategy.log', level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s: %(message)s')

class TradingStrategy:
    def __init__(self, initial_capital=CONFIG['INITIAL_CAPITAL']):
        """
        Initialize trading strategy
        
        Args:
            initial_capital (float): Starting trading capital
        """
        self.capital = initial_capital
        self.positions = {}
        self.trade_history = []
    
    def calculate_macd(self, data):
        """计算MACD指标"""
        exp1 = data['close'].ewm(span=CONFIG['MACD_FAST'], adjust=False).mean()
        exp2 = data['close'].ewm(span=CONFIG['MACD_SLOW'], adjust=False).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=CONFIG['MACD_SIGNAL'], adjust=False).mean()
        return macd, signal
    
    def calculate_bollinger_bands(self, data, window=None, num_std=None):
        """计算布林带"""
        window = window or CONFIG['BB_PERIOD']
        num_std = num_std or CONFIG['BB_STD']
        sma = data['close'].rolling(window=window).mean()
        std = data['close'].rolling(window=window).std()
        upper_band = sma + (std * num_std)
        lower_band = sma - (std * num_std)
        return upper_band, lower_band
    
    def moving_average_crossover(self, data):
        """
        Enhanced Moving Average Crossover Strategy
        """
        if f'MA{CONFIG["MA_FAST"]}' not in data.columns or f'MA{CONFIG["MA_SLOW"]}' not in data.columns:
            return 'hold'
        
        # 获取最近的MA值
        ma_fast = data[f'MA{CONFIG["MA_FAST"]}'].iloc[-3:]
        ma_slow = data[f'MA{CONFIG["MA_SLOW"]}'].iloc[-3:]
        close_price = data['close'].iloc[-1]
        
        # 计算MA的斜率
        ma_fast_slope = (ma_fast.iloc[-1] - ma_fast.iloc[0]) / 2
        ma_slow_slope = (ma_slow.iloc[-1] - ma_slow.iloc[0]) / 2
        
        # 判断趋势强度
        trend_strength = abs(ma_fast_slope) > abs(ma_slow_slope) * 1.2  # 降低趋势强度要求
        
        # 交叉信号
        if ma_fast.iloc[-1] > ma_slow.iloc[-1] and \
           ma_fast.iloc[-2] <= ma_slow.iloc[-2] and \
           (trend_strength or ma_fast_slope > 0):  # 放宽条件
            return 'buy'
        elif ma_fast.iloc[-1] < ma_slow.iloc[-1] and \
             ma_fast.iloc[-2] >= ma_slow.iloc[-2] and \
             (trend_strength or ma_fast_slope < 0):  # 放宽条件
            return 'sell'
        
        return 'hold'
    
    def rsi_strategy(self, data):
        """
        Enhanced RSI Strategy with trend confirmation
        """
        if 'RSI' not in data.columns:
            return 'hold'
        
        # 获取最近的RSI值和收盘价
        rsi_values = data['RSI'].iloc[-5:]
        close_prices = data['close'].iloc[-5:]
        
        # 计算RSI和价格趋势
        rsi_trend = (rsi_values.iloc[-1] - rsi_values.iloc[0]) / 4
        price_trend = (close_prices.iloc[-1] - close_prices.iloc[0]) / close_prices.iloc[0]
        
        current_rsi = rsi_values.iloc[-1]
        
        # RSI超卖且趋势向上
        if current_rsi < CONFIG['RSI_OVERSOLD'] and (rsi_trend > 0 or price_trend > -0.01):  # 放宽价格趋势要求
            return 'buy'
        # RSI超买且趋势向下
        elif current_rsi > CONFIG['RSI_OVERBOUGHT'] and (rsi_trend < 0 or price_trend < 0.01):  # 放宽价格趋势要求
            return 'sell'
        
        return 'hold'
    
    def macd_strategy(self, data):
        """
        MACD Strategy
        """
        macd, signal = self.calculate_macd(data)
        
        # 获取最近的MACD值
        macd_values = macd.iloc[-3:]
        signal_values = signal.iloc[-3:]
        
        # MACD金叉
        if macd_values.iloc[-1] > signal_values.iloc[-1] and \
           macd_values.iloc[-2] <= signal_values.iloc[-2] and \
           macd_values.iloc[-1] > macd_values.iloc[-2]:
            return 'buy'
        # MACD死叉
        elif macd_values.iloc[-1] < signal_values.iloc[-1] and \
             macd_values.iloc[-2] >= signal_values.iloc[-2] and \
             macd_values.iloc[-1] < macd_values.iloc[-2]:
            return 'sell'
        
        return 'hold'
    
    def bollinger_strategy(self, data):
        """
        布林带策略
        """
        upper_band, lower_band = self.calculate_bollinger_bands(data)
        
        if len(data) < CONFIG['BB_PERIOD']:
            return 'hold'
        
        close_price = data['close'].iloc[-1]
        
        # 价格突破下轨且RSI超卖
        if close_price < lower_band.iloc[-1] and \
           data['RSI'].iloc[-1] < CONFIG['RSI_OVERSOLD']:
            return 'buy'
        # 价格突破上轨且RSI超买
        elif close_price > upper_band.iloc[-1] and \
             data['RSI'].iloc[-1] > CONFIG['RSI_OVERBOUGHT']:
            return 'sell'
        
        return 'hold'
    
    def fear_greed_strategy(self, data):
        """
        基于贪婪恐慌指数的交易策略
        """
        if 'Fear_Greed' not in data.columns:
            return 'hold'
        
        fear_greed = data['Fear_Greed'].iloc[-1]
        fear_greed_ma = data['Fear_Greed'].rolling(window=5).mean().iloc[-1]
        
        # 获取趋势
        fear_greed_trend = data['Fear_Greed'].iloc[-5:].diff().mean()
        
        # 极度恐慌时买入（贪婪指数 < 20 且呈上升趋势）
        if fear_greed < 20 and fear_greed > fear_greed_ma and fear_greed_trend > 0:
            return 'buy'
        
        # 极度贪婪时卖出（贪婪指数 > 80 且呈下降趋势）
        elif fear_greed > 80 and fear_greed < fear_greed_ma and fear_greed_trend < 0:
            return 'sell'
        
        return 'hold'
    
    def combined_strategy(self, data):
        """
        Enhanced Combined Strategy with multiple indicators
        """
        if len(data) < 30:
            return 'hold'
        
        # 获取各个策略的信号
        ma_signal = self.moving_average_crossover(data)
        rsi_signal = self.rsi_strategy(data)
        macd_signal = self.macd_strategy(data)
        bb_signal = self.bollinger_strategy(data)
        fg_signal = self.fear_greed_strategy(data)
        
        # 记录各个指标的信号
        logging.info(f"\n当前信号状态：")
        logging.info(f"MA信号: {ma_signal}")
        logging.info(f"RSI信号: {rsi_signal}")
        logging.info(f"MACD信号: {macd_signal}")
        logging.info(f"布林带信号: {bb_signal}")
        logging.info(f"恐慌指数信号: {fg_signal}")
        
        # 计算市场趋势
        ma_fast = data[f'MA{CONFIG["MA_FAST"]}'].iloc[-1]
        ma_slow = data[f'MA{CONFIG["MA_SLOW"]}'].iloc[-1]
        trend = 'up' if ma_fast > ma_slow else 'down'
        logging.info(f"市场趋势: {trend}")
        
        # 统计买入和卖出信号
        signals = [ma_signal, rsi_signal, macd_signal, bb_signal, fg_signal]
        buy_signals = sum(1 for signal in signals if signal == 'buy')
        sell_signals = sum(1 for signal in signals if signal == 'sell')
        
        # 计算贪婪恐慌指数的权重
        fear_greed = data['Fear_Greed'].iloc[-1]
        logging.info(f"恐慌指数: {fear_greed:.2f}")
        
        if fear_greed < 20:  # 极度恐慌时增加买入倾向
            buy_signals += 1
            logging.info("恐慌指数低于20，增加买入信号")
        elif fear_greed > 80:  # 极度贪婪时增加卖出倾向
            sell_signals += 1
            logging.info("恐慌指数高于80，增加卖出信号")
        
        logging.info(f"买入信号数: {buy_signals}, 卖出信号数: {sell_signals}")
        
        # 交易信号确认条件
        # 放宽条件：只需要1个以上的信号，且没有相反信号即可
        if buy_signals >= 1 and sell_signals == 0:
            logging.info("生成买入信号")
            return 'buy'
        elif sell_signals >= 1 and buy_signals == 0:
            logging.info("生成卖出信号")
            return 'sell'
        
        logging.info("保持观望")
        return 'hold'
    
    def calculate_position_size(self, stock_price):
        """
        Enhanced position size calculation with risk management
        """
        # 设置每笔交易的最大风险比例（占总资本的百分比）
        max_risk_per_trade = 0.02  # 2%
        
        # 计算可用于该笔交易的最大资金
        max_trade_capital = self.capital * 0.1  # 最多使用10%资金
        
        # 计算基于风险的头寸规模
        risk_based_position = (self.capital * max_risk_per_trade) / (stock_price * 0.1)  # 假设止损设在10%
        
        # 计算基于资金的头寸规模
        capital_based_position = max_trade_capital / stock_price
        
        # 取两者的较小值
        position_size = min(int(risk_based_position), int(capital_based_position))
        
        return max(0, position_size)  # 确保不返回负数
    
    def execute_trade(self, stock_code, data):
        """
        Execute trading logic for a single stock
        
        Args:
            stock_code (str): Stock identifier
            data (pd.DataFrame): Stock price data
        """
        try:
            # 检查是否有足够的历史数据来计算指标
            if len(data) < 30:  # 至少需要30天数据
                return
            
            # 获取当前持仓情况
            current_position = self.positions.get(stock_code, None)
            current_price = data['close'].iloc[-1]
            
            # 计算止损价格（如果有持仓）
            if current_position:
                stop_loss_price = current_position['entry_price'] * (1 - CONFIG['STOP_LOSS_PCT'])
                # 检查是否触发止损
                if current_price <= stop_loss_price:
                    # 执行止损
                    sell_value = current_position['shares'] * current_price
                    self.capital += sell_value
                    del self.positions[stock_code]
                    logging.info(f"止损: 卖出 {stock_code}, 价格 {current_price}, 止损价 {stop_loss_price}")
                    return
            
            # 获取交易信号
            signal = self.combined_strategy(data)
            
            # 计算持仓总价值
            total_value = self.capital
            for pos in self.positions.values():
                total_value += pos['shares'] * current_price
            
            if signal == 'buy' and len(self.positions) < CONFIG['MAX_POSITIONS']:
                # 确保没有过度集中
                if current_position is None:  # 只在没有持仓时买入
                    # 计算可用资金，考虑风险管理
                    available_capital = min(
                        self.capital,
                        total_value * CONFIG['POSITION_SIZE_PCT']
                    )
                    
                    if available_capital > 1000:  # 确保有足够资金进行有意义的交易
                        shares = int(available_capital / current_price)
                        trade_cost = shares * current_price
                        
                        if trade_cost <= self.capital:
                            self.positions[stock_code] = {
                                'shares': shares,
                                'entry_price': current_price
                            }
                            self.capital -= trade_cost
                            logging.info(f"买入 {shares} 股 {stock_code}，价格 {current_price}")
            
            elif signal == 'sell' and stock_code in self.positions:
                position = self.positions[stock_code]
                sell_value = position['shares'] * current_price
                profit = sell_value - (position['shares'] * position['entry_price'])
                profit_pct = (profit / (position['shares'] * position['entry_price'])) * 100
                
                self.capital += sell_value
                del self.positions[stock_code]
                
                logging.info(f"卖出 {position['shares']} 股 {stock_code}，价格 {current_price}，"
                           f"收益率 {profit_pct:.2f}%")
        
        except Exception as e:
            logging.error(f"交易执行错误 - 股票 {stock_code}：{e}")

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
        # 创建保存图表的目录
        plots_dir = Path('plots')
        plots_dir.mkdir(exist_ok=True)
        
        # 获取股票数据
        stock_data = {}
        for symbol in CONFIG['STOCK_UNIVERSE']:
            try:
                data = retrieve_stock_data(symbol, CONFIG['START_DATE'], CONFIG['END_DATE'])
                if not data.empty:
                    stock_data[symbol] = data
                    # 生成单个ETF的技术分析图表
                    plot_single_etf_metrics(symbol, data, None, plots_dir)
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
        strategy = TradingStrategy(initial_capital=current_capital)
        
        for date in results.index[1:]:
            # 对每个股票执行交易策略
            for symbol in CONFIG['STOCK_UNIVERSE']:
                if symbol in stock_data:
                    # 获取到当前日期的历史数据
                    hist_data = stock_data[symbol].loc[:date]
                    if not hist_data.empty:
                        # 执行交易策略
                        strategy.execute_trade(symbol, hist_data)
            
            # 更新持仓价值
            portfolio_value = strategy.capital  # 现金部分
            for symbol, position in strategy.positions.items():
                if symbol in stock_data and date in stock_data[symbol].index:
                    current_price = stock_data[symbol].loc[date, 'close']
                    portfolio_value += position['shares'] * current_price
            
            portfolio_values.append(portfolio_value)
            dates.append(date)
            
            # 更新结果数据
            results.loc[date, 'portfolio_value'] = portfolio_value
            
            # 计算持仓比例
            total_value = portfolio_value
            for symbol in CONFIG['STOCK_UNIVERSE']:
                if symbol in strategy.positions:
                    if symbol in stock_data and date in stock_data[symbol].index:
                        position_value = strategy.positions[symbol]['shares'] * stock_data[symbol].loc[date, 'close']
                        results.loc[date, f'position_{symbol}'] = position_value / total_value
                else:
                    results.loc[date, f'position_{symbol}'] = 0.0
            
            # 计算回撤
            peak = results['portfolio_value'][:date].max()
            drawdown = (portfolio_value - peak) / peak
            results.loc[date, 'drawdown'] = drawdown
        
        # 生成投资组合相关的图表
        plot_portfolio_metrics(results, str(plots_dir))
        
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
