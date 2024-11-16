import pandas as pd
from typing import Dict, Any, Tuple
from .base_strategy import BaseStrategy

class MACrossoverStrategy(BaseStrategy):
    """Moving Average Crossover Strategy"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.fast_period = config.get('MA_FAST', 5)
        self.slow_period = config.get('MA_SLOW', 20)
        
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate trading signals based on MA crossover"""
        # Calculate moving averages
        data['MA_Fast'] = data['close'].rolling(window=self.fast_period).mean()
        data['MA_Slow'] = data['close'].rolling(window=self.slow_period).mean()
        
        # Generate crossover signals
        data['Signal'] = 0
        data.loc[data['MA_Fast'] > data['MA_Slow'], 'Signal'] = 1
        data.loc[data['MA_Fast'] < data['MA_Slow'], 'Signal'] = -1
        
        return data
        
    def should_enter_trade(self, data: pd.DataFrame, index: int) -> Tuple[bool, float]:
        """Check if we should enter a trade based on MA crossover"""
        if index < 1:  # Need at least 2 periods
            return False, 0.0
            
        current_signal = data.iloc[index]['Signal']
        prev_signal = data.iloc[index-1]['Signal']
        
        # Enter long when fast MA crosses above slow MA
        if current_signal == 1 and prev_signal <= 0:
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
            
        # Exit when fast MA crosses below slow MA
        current_signal = data.iloc[index]['Signal']
        prev_signal = data.iloc[index-1]['Signal']
        if current_signal == -1 and prev_signal >= 0:
            return True
            
        return False
