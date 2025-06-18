import pandas as pd
import numpy as np
from typing import Dict, Any
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.backtest import calculate_sma, calculate_rsi

class MeanReversionStrategy:
    """
    Mean Reversion Strategy using RSI and Moving Averages
    
    Strategy Logic:
    - Buy when price is below SMA and RSI is oversold (< 30)
    - Sell when price is above SMA and RSI is overbought (> 70)
    """
    
    def __init__(self, parameters: Dict[str, Any] = None):
        # Default parameters - keeping old names for compatibility
        self.period = 20
        self.threshold = 0.02
        self.position_size = 0.1
        self.version = '1.0'
        
        # Internal parameters for RSI strategy
        self.params = {
            'sma_period': 20,
            'rsi_period': 14,
            'rsi_oversold': 30,
            'rsi_overbought': 70,
            'lookback_period': 50
        }
        
        # Update with custom parameters
        if parameters:
            if 'period' in parameters:
                if parameters['period'] <= 0:
                    raise ValueError("Period must be positive")
                self.period = parameters['period']
                self.params['sma_period'] = parameters['period']
            
            if 'threshold' in parameters:
                if parameters['threshold'] < 0:
                    raise ValueError("Threshold must be non-negative")
                self.threshold = parameters['threshold']
            
            if 'position_size' in parameters:
                if parameters['position_size'] <= 0 or parameters['position_size'] > 1:
                    raise ValueError("Position size must be between 0 and 1")
                self.position_size = parameters['position_size']
            
            # Update internal params
            for key in ['sma_period', 'rsi_period', 'rsi_oversold', 'rsi_overbought']:
                if key in parameters:
                    self.params[key] = parameters[key]
        
        self.name = "Mean Reversion Strategy"
    
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        Generate trading signals based on the strategy logic
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            Series with signals: 1 for buy, -1 for sell, 0 for hold
        """
        if len(data) < max(self.params['sma_period'], self.params['rsi_period']):
            return pd.Series(0, index=data.index)
        
        # Calculate indicators
        closes = data['close'].values
        
        # Simple Moving Average
        sma = calculate_sma(closes, self.params['sma_period'])
        
        # RSI
        rsi = calculate_rsi(closes, self.params['rsi_period'])
        
        # Create signals
        signals = pd.Series(0, index=data.index)
        
        for i in range(len(data)):
            if np.isnan(sma[i]) or np.isnan(rsi[i]):
                continue
            
            current_price = closes[i]
            
            # Buy signal: price below SMA and RSI oversold
            if (current_price < sma[i] and 
                rsi[i] < self.params['rsi_oversold']):
                signals.iloc[i] = 1
            
            # Sell signal: price above SMA and RSI overbought
            elif (current_price > sma[i] and 
                  rsi[i] > self.params['rsi_overbought']):
                signals.iloc[i] = -1
        
        return signals
    
    def get_parameters(self) -> Dict[str, Any]:
        """Get strategy parameters"""
        return self.params.copy()
    
    def set_parameters(self, parameters: Dict[str, Any]):
        """Set strategy parameters"""
        self.params.update(parameters)
    
    def get_required_data_length(self) -> int:
        """Get minimum required data length"""
        return max(self.params['sma_period'], self.params['rsi_period']) + 10
    
    def describe(self) -> str:
        """Get strategy description"""
        return f"""
{self.name} v{self.version}

Strategy Logic:
- Buy Signal: Price < SMA({self.params['sma_period']}) AND RSI({self.params['rsi_period']}) < {self.params['rsi_oversold']}
- Sell Signal: Price > SMA({self.params['sma_period']}) AND RSI({self.params['rsi_period']}) > {self.params['rsi_overbought']}

Parameters:
- SMA Period: {self.params['sma_period']}
- RSI Period: {self.params['rsi_period']}
- RSI Oversold Level: {self.params['rsi_oversold']}
- RSI Overbought Level: {self.params['rsi_overbought']}

This is a mean reversion strategy that assumes prices will revert to their moving average.
It uses RSI to identify overbought/oversold conditions for entry/exit signals.
"""

class BollingerBandsMeanReversion:
    """
    Alternative Mean Reversion Strategy using Bollinger Bands
    """
    
    def __init__(self, parameters: Dict[str, Any] = None):
        self.params = {
            'bb_period': 20,
            'bb_std': 2.0,
            'rsi_period': 14,
            'rsi_oversold': 30,
            'rsi_overbought': 70
        }
        
        if parameters:
            self.params.update(parameters)
        
        self.name = "Bollinger Bands Mean Reversion"
        self.version = "1.0"
    
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Generate signals using Bollinger Bands"""
        if len(data) < self.params['bb_period']:
            return pd.Series(0, index=data.index)
        
        # Calculate Bollinger Bands
        sma = data['close'].rolling(window=self.params['bb_period']).mean()
        std = data['close'].rolling(window=self.params['bb_period']).std()
        
        upper_band = sma + (std * self.params['bb_std'])
        lower_band = sma - (std * self.params['bb_std'])
        
        # Calculate RSI
        rsi = self._calculate_rsi(data['close'], self.params['rsi_period'])
        
        # Generate signals
        signals = pd.Series(0, index=data.index)
        
        # Buy when price touches lower band and RSI is oversold
        buy_condition = (
            (data['close'] <= lower_band) & 
            (rsi < self.params['rsi_oversold'])
        )
        
        # Sell when price touches upper band and RSI is overbought
        sell_condition = (
            (data['close'] >= upper_band) & 
            (rsi > self.params['rsi_overbought'])
        )
        
        signals[buy_condition] = 1
        signals[sell_condition] = -1
        
        return signals
    
    def _calculate_rsi(self, prices: pd.Series, period: int) -> pd.Series:
        """Calculate RSI using pandas"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def get_parameters(self) -> Dict[str, Any]:
        return self.params.copy()
    
    def set_parameters(self, parameters: Dict[str, Any]):
        self.params.update(parameters)
    
    def get_required_data_length(self) -> int:
        return max(self.params['bb_period'], self.params['rsi_period']) + 10
    
    def describe(self) -> str:
        return f"""
{self.name} v{self.version}

Strategy Logic:
- Buy Signal: Price <= Lower Bollinger Band AND RSI < {self.params['rsi_oversold']}
- Sell Signal: Price >= Upper Bollinger Band AND RSI > {self.params['rsi_overbought']}

Parameters:
- Bollinger Bands Period: {self.params['bb_period']}
- Bollinger Bands Standard Deviation: {self.params['bb_std']}
- RSI Period: {self.params['rsi_period']}
- RSI Oversold Level: {self.params['rsi_oversold']}
- RSI Overbought Level: {self.params['rsi_overbought']}

This strategy uses Bollinger Bands to identify when prices deviate significantly
from their moving average, combined with RSI for confirmation.
"""

# Factory function to create strategy instances
def create_strategy(strategy_name: str, parameters: Dict[str, Any] = None):
    """
    Factory function to create strategy instances
    
    Args:
        strategy_name: Name of the strategy ('meanrev' or 'bollinger')
        parameters: Strategy parameters
        
    Returns:
        Strategy instance
    """
    strategies = {
        'meanrev': MeanReversionStrategy,
        'bollinger': BollingerBandsMeanReversion
    }
    
    if strategy_name not in strategies:
        raise ValueError(f"Unknown strategy: {strategy_name}. Available: {list(strategies.keys())}")
    
    return strategies[strategy_name](parameters)