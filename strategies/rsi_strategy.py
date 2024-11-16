import pandas as pd
from typing import Dict, Any, Tuple
from .base_strategy import BaseStrategy

class RSIStrategy(BaseStrategy):
    """RSI Mean Reversion Strategy"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.rsi_period = config.get('RSI_PERIOD', 14)
        self.oversold = config.get('RSI_OVERSOLD', 30)
        self.overbought = config.get('RSI_OVERBOUGHT', 70)
        
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate trading signals based on RSI"""
        # Calculate RSI
        delta = data['close'].diff()
        gain = (delta.where(delta > 0, 0)).fillna(0)
        loss = (-delta.where(delta < 0, 0)).fillna(0)
        
        # First calculate SMA
        avg_gain = gain.rolling(window=self.rsi_period, min_periods=1).mean()
        avg_loss = loss.rolling(window=self.rsi_period, min_periods=1).mean()
        
        # Then calculate EMA
        avg_gain = avg_gain.ewm(alpha=1/self.rsi_period, min_periods=self.rsi_period, adjust=False).mean()
        avg_loss = avg_loss.ewm(alpha=1/self.rsi_period, min_periods=self.rsi_period, adjust=False).mean()
        
        # Calculate RS and RSI
        rs = avg_gain / avg_loss.replace(0, float('inf'))
        data['RSI'] = 100 - (100 / (1 + rs))
        
        # Generate signals
        data['Signal'] = 0
        data.loc[data['RSI'] < self.oversold, 'Signal'] = 1  # Oversold, potential buy
        data.loc[data['RSI'] > self.overbought, 'Signal'] = -1  # Overbought, potential sell
        
        return data
        
    def should_enter_trade(self, data: pd.DataFrame, index: int) -> Tuple[bool, float]:
        """Check if we should enter a trade based on RSI"""
        if index < self.rsi_period:  # Need enough data for RSI
            return False, 0.0
            
        current_signal = data.iloc[index]['Signal']
        current_rsi = data.iloc[index]['RSI']
        
        # Enter long when RSI is oversold
        if current_signal == 1 and current_rsi < self.oversold:
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
            
        # Exit when RSI becomes overbought
        current_rsi = data.iloc[index]['RSI']
        if current_rsi > self.overbought:
            return True
            
        return False
