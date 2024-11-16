import pandas as pd
from typing import Dict, Any, Tuple
from .base_strategy import BaseStrategy

class BollingerStrategy(BaseStrategy):
    """Bollinger Bands Mean Reversion Strategy"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.bb_period = config.get('BB_PERIOD', 20)
        self.bb_std = config.get('BB_STD', 2.0)
        
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate trading signals based on Bollinger Bands"""
        # Calculate Bollinger Bands
        sma = data['close'].rolling(window=self.bb_period, min_periods=1).mean()
        std = data['close'].rolling(window=self.bb_period, min_periods=1).std()
        
        data['BB_Upper'] = sma + (std * self.bb_std)
        data['BB_Lower'] = sma - (std * self.bb_std)
        data['BB_Middle'] = sma
        
        # Generate signals
        data['Signal'] = 0
        # Buy signal when price crosses below lower band
        data.loc[data['close'] < data['BB_Lower'], 'Signal'] = 1
        # Sell signal when price crosses above upper band
        data.loc[data['close'] > data['BB_Upper'], 'Signal'] = -1
        
        # Calculate %B indicator
        data['BB_PCT'] = (data['close'] - data['BB_Lower']) / (data['BB_Upper'] - data['BB_Lower'])
        
        return data
        
    def should_enter_trade(self, data: pd.DataFrame, index: int) -> Tuple[bool, float]:
        """Check if we should enter a trade based on Bollinger Bands"""
        if index < self.bb_period:  # Need enough data for Bollinger Bands
            return False, 0.0
            
        current_signal = data.iloc[index]['Signal']
        current_bb_pct = data.iloc[index]['BB_PCT']
        
        # Enter long when price is below lower band and %B is low
        if current_signal == 1 and current_bb_pct < 0.05:
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
            
        # Exit when price moves above middle band
        if current_price > data.iloc[index]['BB_Middle']:
            return True
            
        # Exit when %B becomes high
        if data.iloc[index]['BB_PCT'] > 0.95:
            return True
            
        return False
