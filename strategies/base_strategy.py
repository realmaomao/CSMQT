from abc import ABC, abstractmethod
import pandas as pd
from typing import Dict, Any, Tuple, Optional

class BaseStrategy(ABC):
    """Base class for all trading strategies"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals based on strategy rules
        
        Args:
            data: DataFrame with OHLCV data and technical indicators
            
        Returns:
            DataFrame with added signal columns
        """
        pass
    
    @abstractmethod
    def should_enter_trade(self, data: pd.DataFrame, index: int) -> Tuple[bool, float]:
        """
        Determine if we should enter a trade at the given index
        
        Args:
            data: DataFrame with signals
            index: Current index to check
            
        Returns:
            Tuple of (should_enter: bool, suggested_position_size: float)
        """
        pass
    
    @abstractmethod
    def should_exit_trade(self, data: pd.DataFrame, index: int, entry_price: float) -> bool:
        """
        Determine if we should exit an existing trade
        
        Args:
            data: DataFrame with signals
            index: Current index to check
            entry_price: Price at which we entered the trade
            
        Returns:
            True if we should exit, False otherwise
        """
        pass
    
    def calculate_position_size(self, portfolio_value: float, price: float) -> float:
        """
        Calculate the position size based on portfolio value and current price
        
        Args:
            portfolio_value: Current portfolio value
            price: Current price of the asset
            
        Returns:
            Number of shares to trade
        """
        position_pct = self.config.get('POSITION_SIZE_PCT', 0.1)  # Default 10%
        position_value = portfolio_value * position_pct
        return position_value / price
    
    def apply_risk_management(self, entry_price: float, current_price: float,
                            high_since_entry: float) -> bool:
        """
        Apply risk management rules to determine if we should exit
        
        Args:
            entry_price: Price at which we entered the trade
            current_price: Current price
            high_since_entry: Highest price since entry
            
        Returns:
            True if we should exit based on risk management rules
        """
        # Stop loss check
        stop_loss_pct = self.config.get('STOP_LOSS_PCT', 0.05)
        if (entry_price - current_price) / entry_price > stop_loss_pct:
            return True
            
        # Trailing stop check
        trailing_stop_pct = self.config.get('TRAILING_STOP_PCT', 0.08)
        if (high_since_entry - current_price) / high_since_entry > trailing_stop_pct:
            return True
            
        return False
