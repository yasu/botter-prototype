import pytest
import pandas as pd
import numpy as np
from datetime import datetime
import sys
import os

# Add strategies directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'strategies'))

from strategies.sample_meanrev import MeanReversionStrategy

class TestMeanReversionStrategy:
    def test_strategy_initialization(self):
        strategy = MeanReversionStrategy()
        assert strategy.period == 20
        assert strategy.threshold == 0.02
        assert strategy.position_size == 0.1
        assert strategy.version == '1.0'
    
    def test_get_parameters(self):
        strategy = MeanReversionStrategy()
        params = strategy.get_parameters()
        
        expected_params = {
            'period': 20,
            'threshold': 0.02,
            'position_size': 0.1
        }
        
        assert params == expected_params
    
    def test_set_parameters(self):
        strategy = MeanReversionStrategy()
        new_params = {
            'period': 30,
            'threshold': 0.03,
            'position_size': 0.2
        }
        
        strategy.set_parameters(new_params)
        
        assert strategy.period == 30
        assert strategy.threshold == 0.03
        assert strategy.position_size == 0.2
    
    def test_generate_signals_insufficient_data(self):
        strategy = MeanReversionStrategy()
        
        # Create data with less than required period
        short_data = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=10, freq='1min'),
            'open': [50000] * 10,
            'high': [50100] * 10,
            'low': [49900] * 10,
            'close': [50000] * 10,
            'volume': [1000] * 10
        })
        
        signals = strategy.generate_signals(short_data)
        
        # Should return all zeros for insufficient data
        assert all(signals == 0)
    
    def test_generate_signals_with_trend(self):
        strategy = MeanReversionStrategy()
        
        # Create trending data
        periods = 50
        dates = pd.date_range('2024-01-01', periods=periods, freq='1min')
        
        # Create uptrend data
        base_price = 50000
        close_prices = []
        for i in range(periods):
            # Add trend + some noise
            price = base_price + (i * 10) + np.random.normal(0, 50)
            close_prices.append(price)
        
        data = pd.DataFrame({
            'timestamp': dates,
            'open': close_prices,
            'high': [p + 50 for p in close_prices],
            'low': [p - 50 for p in close_prices],
            'close': close_prices,
            'volume': [1000] * periods
        })
        
        signals = strategy.generate_signals(data)
        
        # Should generate some signals (not all zeros)
        assert not all(signals == 0)
        assert len(signals) == periods
    
    def test_generate_signals_buy_condition(self):
        strategy = MeanReversionStrategy()
        strategy.period = 5  # Short period for testing
        strategy.threshold = 0.05  # 5% threshold
        
        # Create data where price drops significantly below moving average
        close_prices = [100, 100, 100, 100, 100, 80]  # 20% drop
        
        data = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=6, freq='1min'),
            'open': close_prices,
            'high': [p + 5 for p in close_prices],
            'low': [p - 5 for p in close_prices],
            'close': close_prices,
            'volume': [1000] * 6
        })
        
        signals = strategy.generate_signals(data)
        
        # Last signal should be buy (1) because price dropped significantly
        assert signals.iloc[-1] == 1
    
    def test_generate_signals_sell_condition(self):
        strategy = MeanReversionStrategy()
        strategy.period = 5  # Short period for testing
        strategy.threshold = 0.05  # 5% threshold
        
        # Create data where price rises significantly above moving average
        close_prices = [100, 100, 100, 100, 100, 120]  # 20% rise
        
        data = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=6, freq='1min'),
            'open': close_prices,
            'high': [p + 5 for p in close_prices],
            'low': [p - 5 for p in close_prices],
            'close': close_prices,
            'volume': [1000] * 6
        })
        
        signals = strategy.generate_signals(data)
        
        # Last signal should be sell (-1) because price rose significantly
        assert signals.iloc[-1] == -1
    
    def test_generate_signals_no_signal_condition(self):
        strategy = MeanReversionStrategy()
        strategy.period = 5
        strategy.threshold = 0.05
        
        # Create stable data around moving average
        close_prices = [100, 101, 99, 100, 102, 101]  # Small variations
        
        data = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=6, freq='1min'),
            'open': close_prices,
            'high': [p + 2 for p in close_prices],
            'low': [p - 2 for p in close_prices],
            'close': close_prices,
            'volume': [1000] * 6
        })
        
        signals = strategy.generate_signals(data)
        
        # Last signal should be hold (0) because price is near moving average
        assert signals.iloc[-1] == 0
    
    def test_strategy_edge_cases(self):
        strategy = MeanReversionStrategy()
        
        # Test with empty DataFrame
        empty_df = pd.DataFrame()
        signals = strategy.generate_signals(empty_df)
        assert len(signals) == 0
        
        # Test with single row
        single_row = pd.DataFrame({
            'timestamp': [datetime(2024, 1, 1)],
            'open': [50000],
            'high': [50100],
            'low': [49900],
            'close': [50000],
            'volume': [1000]
        })
        
        signals = strategy.generate_signals(single_row)
        assert len(signals) == 1
        assert signals.iloc[0] == 0  # Should be no signal
    
    def test_parameter_validation(self):
        strategy = MeanReversionStrategy()
        
        # Test invalid parameters
        with pytest.raises(ValueError):
            strategy.set_parameters({'period': 0})  # Invalid period
        
        with pytest.raises(ValueError):
            strategy.set_parameters({'threshold': -0.1})  # Negative threshold
        
        with pytest.raises(ValueError):
            strategy.set_parameters({'position_size': 1.5})  # Position size > 1