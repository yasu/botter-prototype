import pytest
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime
from core.data_loader import DataLoader

class TestDataLoader:
    def test_initialization(self, test_env_vars):
        loader = DataLoader()
        assert loader.api_key == 'test_key'
        assert loader.api_secret == 'test_secret'
        assert loader.exchange is not None
    
    def test_timeframe_to_minutes(self):
        loader = DataLoader()
        assert loader._timeframe_to_minutes('1m') == 1
        assert loader._timeframe_to_minutes('5m') == 5
        assert loader._timeframe_to_minutes('1h') == 60
        assert loader._timeframe_to_minutes('1d') == 1440
        assert loader._timeframe_to_minutes('invalid') == 60  # default
    
    def test_parse_timestamp(self):
        loader = DataLoader()
        
        # Test with milliseconds timestamp
        timestamp_ms = 1704067200000
        result = loader._parse_timestamp(timestamp_ms)
        assert isinstance(result, datetime)
        
        # Test with datetime object
        dt = datetime(2024, 1, 1)
        result = loader._parse_timestamp(dt)
        assert result == dt
        
        # Test with string
        result = loader._parse_timestamp('2024-01-01 00:00:00')
        assert isinstance(result, datetime)
    
    @patch('core.data_loader.DataLoader.exchange')
    def test_get_account_balance(self, mock_exchange, test_env_vars):
        mock_exchange.fetch_balance.return_value = {
            'total': {'USDT': 10000.0, 'BTC': 0.5},
            'free': {'USDT': 9000.0, 'BTC': 0.5},
            'used': {'USDT': 1000.0, 'BTC': 0.0}
        }
        
        loader = DataLoader()
        loader.exchange = mock_exchange
        balance = loader.get_account_balance()
        
        assert balance['total']['USDT'] == 10000.0
        assert balance['free']['USDT'] == 9000.0
        assert balance['used']['USDT'] == 1000.0
    
    @patch('core.data_loader.DataLoader.exchange')
    def test_fetch_live_data(self, mock_exchange, test_env_vars):
        mock_exchange.fetch_ohlcv.return_value = [
            [1704067200000, 50000, 50500, 49500, 50200, 1000],
            [1704067260000, 50200, 50700, 49800, 50400, 1100],
        ]
        
        loader = DataLoader()
        loader.exchange = mock_exchange
        data = loader.fetch_live_data('BTC/USDT', '1m', limit=2)
        
        assert isinstance(data, pd.DataFrame)
        assert len(data) == 2
        assert 'timestamp' in data.columns
        assert 'open' in data.columns
        assert 'high' in data.columns
        assert 'low' in data.columns
        assert 'close' in data.columns
        assert 'volume' in data.columns
    
    @patch('core.data_loader.Database')
    def test_get_historical_data_from_db(self, mock_db_class, test_env_vars):
        # Mock database response
        mock_db = MagicMock()
        mock_db_class.return_value = mock_db
        
        mock_data = [
            (datetime(2024, 1, 1), 50000, 50500, 49500, 50200, 1000),
            (datetime(2024, 1, 1, 0, 1), 50200, 50700, 49800, 50400, 1100),
        ]
        mock_db.get_session.return_value.query.return_value.filter.return_value.order_by.return_value.all.return_value = mock_data
        
        loader = DataLoader()
        df = loader.get_historical_data_from_db('BTCUSDT', '1m', '2024-01-01', '2024-01-02')
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
    
    def test_format_symbol_for_api(self):
        loader = DataLoader()
        
        # Test standard format
        symbol = loader._format_symbol_for_api('BTCUSDT')
        assert symbol == 'BTC/USDT'
        
        # Test already formatted
        symbol = loader._format_symbol_for_api('BTC/USDT')
        assert symbol == 'BTC/USDT'
        
        # Test edge cases
        symbol = loader._format_symbol_for_api('ETHUSDT')
        assert symbol == 'ETH/USDT'
    
    def test_format_symbol_for_db(self):
        loader = DataLoader()
        
        # Test API format
        symbol = loader._format_symbol_for_db('BTC/USDT')
        assert symbol == 'BTCUSDT'
        
        # Test already formatted
        symbol = loader._format_symbol_for_db('BTCUSDT')
        assert symbol == 'BTCUSDT'
    
    @patch('core.data_loader.DataLoader.exchange')
    @patch('core.data_loader.Database')
    async def test_download_historical_data(self, mock_db_class, mock_exchange, test_env_vars):
        # Mock exchange response
        mock_exchange.fetch_ohlcv.return_value = [
            [1704067200000, 50000, 50500, 49500, 50200, 1000],
            [1704067260000, 50200, 50700, 49800, 50400, 1100],
        ]
        
        # Mock database
        mock_db = MagicMock()
        mock_db_class.return_value = mock_db
        
        loader = DataLoader()
        loader.exchange = mock_exchange
        
        df = await loader.download_historical_data(
            symbol='BTC/USDT',
            start_date='2024-01-01',
            end_date='2024-01-02',
            timeframe='1m',
            save_to_db=True
        )
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert 'timestamp' in df.columns
    
    def test_data_validation(self):
        loader = DataLoader()
        
        # Test valid data
        valid_data = pd.DataFrame({
            'timestamp': [datetime(2024, 1, 1), datetime(2024, 1, 1, 0, 1)],
            'open': [50000, 50200],
            'high': [50500, 50700],
            'low': [49500, 49800],
            'close': [50200, 50400],
            'volume': [1000, 1100]
        })
        
        is_valid = loader._validate_ohlcv_data(valid_data)
        assert is_valid
        
        # Test invalid data (missing columns)
        invalid_data = pd.DataFrame({
            'timestamp': [datetime(2024, 1, 1)],
            'open': [50000],
            'high': [50500]
            # Missing low, close, volume
        })
        
        is_valid = loader._validate_ohlcv_data(invalid_data)
        assert not is_valid
        
        # Test invalid data (negative prices)
        invalid_prices = pd.DataFrame({
            'timestamp': [datetime(2024, 1, 1)],
            'open': [-50000],  # Negative price
            'high': [50500],
            'low': [49500],
            'close': [50200],
            'volume': [1000]
        })
        
        is_valid = loader._validate_ohlcv_data(invalid_prices)
        assert not is_valid