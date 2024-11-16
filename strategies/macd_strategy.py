import pandas as pd
from typing import Dict, Any, Tuple
from .base_strategy import BaseStrategy

class MACDStrategy(BaseStrategy):
    """MACD Trend Following Strategy"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        # Using standard MACD parameters
        self.fast_period = 12
        self.slow_period = 26
        self.signal_period = 9
        
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate trading signals based on MACD"""
        # Calculate MACD
        exp1 = data['close'].ewm(span=self.fast_period, adjust=False).mean()
        exp2 = data['close'].ewm(span=self.slow_period, adjust=False).mean()
        data['MACD'] = exp1 - exp2
        data['Signal_Line'] = data['MACD'].ewm(span=self.signal_period, adjust=False).mean()
        data['MACD_Hist'] = data['MACD'] - data['Signal_Line']
        
        # Generate signals
        data['Signal'] = 0
        # Buy signal when MACD crosses above signal line
        data.loc[data['MACD'] > data['Signal_Line'], 'Signal'] = 1
        # Sell signal when MACD crosses below signal line
        data.loc[data['MACD'] < data['Signal_Line'], 'Signal'] = -1
        
        return data
        
    def should_enter_trade(self, data: pd.DataFrame, index: int) -> Tuple[bool, float]:
        """Check if we should enter a trade based on MACD"""
        if index < self.slow_period:  # Need enough data for MACD
            return False, 0.0
            
        current_signal = data.iloc[index]['Signal']
        prev_signal = data.iloc[index-1]['Signal']
        current_hist = data.iloc[index]['MACD_Hist']
        
        # Enter long when MACD crosses above signal line with positive momentum
        if current_signal == 1 and prev_signal <= 0 and current_hist > 0:
            position_size = self.calculate_position_size(
                portfolio_value=self.config['INITIAL_CAPITAL'],
                price=data.iloc[index]['close']
            )
            return True, position_size
            
        return False, 0.0
        
    def should_exit_trade(self, data: pd.DataFrame, index: int, entry_price: float) -> bool:
        """Check if we should exit an existing trade"""
        current_price = data.iloc[index]['close']
        high_since_entry = data.iloc[index-1:index+1]['high'].max()
        
        # Check risk management rules
        if self.apply_risk_management(entry_price, current_price, high_since_entry):
            return True
            
        # Exit when MACD crosses below signal line
        current_signal = data.iloc[index]['Signal']
        prev_signal = data.iloc[index-1]['Signal']
        if current_signal == -1 and prev_signal >= 0:
            return True
            
        return False
