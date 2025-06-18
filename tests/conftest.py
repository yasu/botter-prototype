import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, MagicMock
import os

@pytest.fixture
def sample_ohlcv_data():
    """Generate sample OHLCV data for testing"""
    dates = pd.date_range(start='2024-01-01', periods=100, freq='1min')
    data = []
    
    base_price = 50000
    for i, date in enumerate(dates):
        # Simple price movement simulation
        price = base_price + np.sin(i * 0.1) * 1000 + np.random.normal(0, 100)
        high = price + np.random.uniform(10, 100)
        low = price - np.random.uniform(10, 100)
        open_price = price + np.random.uniform(-50, 50)
        close_price = price
        volume = np.random.uniform(1000, 10000)
        
        data.append({
            'timestamp': date,
            'open': open_price,
            'high': high,
            'low': low,
            'close': close_price,
            'volume': volume
        })
    
    return pd.DataFrame(data)

@pytest.fixture
def mock_exchange():
    """Mock ccxt exchange for testing"""
    exchange = Mock()
    exchange.fetch_ohlcv = Mock(return_value=[
        [1704067200000, 50000, 50500, 49500, 50200, 1000],  # timestamp, o, h, l, c, v
        [1704067260000, 50200, 50700, 49800, 50400, 1100],
        [1704067320000, 50400, 50900, 50000, 50600, 1200],
    ])
    exchange.fetch_balance = Mock(return_value={
        'total': {'USDT': 10000.0},
        'free': {'USDT': 10000.0},
        'used': {'USDT': 0.0}
    })
    return exchange

@pytest.fixture
def test_env_vars():
    """Set test environment variables"""
    os.environ['DATABASE_URL'] = 'sqlite:///:memory:'
    os.environ['BYBIT_API_KEY'] = 'test_key'
    os.environ['BYBIT_API_SECRET'] = 'test_secret'
    yield
    # Cleanup
    if 'DATABASE_URL' in os.environ:
        del os.environ['DATABASE_URL']

@pytest.fixture
def sample_strategy():
    """Create a simple test strategy"""
    class TestStrategy:
        def __init__(self):
            self.parameters = {'period': 20, 'threshold': 0.02}
            self.version = '1.0'
        
        def get_parameters(self):
            return self.parameters
        
        def set_parameters(self, params):
            self.parameters.update(params)
        
        def generate_signals(self, df):
            signals = pd.Series(0, index=df.index)
            # Simple moving average crossover
            if len(df) > self.parameters['period']:
                ma = df['close'].rolling(window=self.parameters['period']).mean()
                signals[df['close'] > ma * (1 + self.parameters['threshold'])] = 1  # buy
                signals[df['close'] < ma * (1 - self.parameters['threshold'])] = -1  # sell
            return signals
    
    return TestStrategy()