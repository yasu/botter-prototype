import pytest
import os
from datetime import datetime
from unittest.mock import patch, MagicMock
from core.db import Database

class TestDatabase:
    @patch.dict(os.environ, {'DATABASE_URL': 'sqlite:///:memory:'})
    def test_database_initialization(self):
        db = Database()
        assert db.database_url == 'sqlite:///:memory:'
        assert db.engine is not None
    
    @patch.dict(os.environ, {'DATABASE_URL': 'sqlite:///:memory:'})
    def test_create_tables(self):
        db = Database()
        # Should not raise an exception
        db.create_tables()
        
        # Test table creation by checking if we can get a session
        session = db.get_session()
        assert session is not None
        session.close()
    
    @patch.dict(os.environ, {'DATABASE_URL': 'sqlite:///:memory:'})
    def test_get_session(self):
        db = Database()
        db.create_tables()
        
        session = db.get_session()
        assert session is not None
        
        # Test that we can execute a simple query
        result = session.execute("SELECT 1").scalar()
        assert result == 1
        
        session.close()
    
    @patch.dict(os.environ, {'DATABASE_URL': 'sqlite:///:memory:'})
    def test_save_strategy(self):
        db = Database()
        db.create_tables()
        
        strategy_id = db.save_strategy(
            name='TestStrategy',
            version='1.0',
            parameters={'param1': 'value1'},
            file='test_strategy.py'
        )
        
        assert isinstance(strategy_id, int)
        assert strategy_id > 0
    
    @patch.dict(os.environ, {'DATABASE_URL': 'sqlite:///:memory:'})
    def test_save_backtest_result(self):
        db = Database()
        db.create_tables()
        
        # First save a strategy
        strategy_id = db.save_strategy(
            name='TestStrategy',
            version='1.0',
            parameters={},
            file='test.py'
        )
        
        # Then save backtest result
        backtest_id = db.save_backtest_result(
            strategy_id=strategy_id,
            symbol='BTCUSDT',
            timeframe='1m',
            period_start=datetime(2024, 1, 1),
            period_end=datetime(2024, 1, 2),
            initial_balance=10000.0,
            metrics={
                'total_return': 5.5,
                'sharpe_ratio': 1.2,
                'max_drawdown': -2.1,
                'win_rate': 0.65,
                'total_trades': 10,
                'profit_factor': 1.8
            }
        )
        
        assert isinstance(backtest_id, int)
        assert backtest_id > 0
    
    @patch.dict(os.environ, {'DATABASE_URL': 'sqlite:///:memory:'})
    def test_save_ohlcv_data(self):
        db = Database()
        db.create_tables()
        
        # Test saving single row
        db.save_ohlcv_data(
            symbol='BTCUSDT',
            timeframe='1m',
            timestamp=datetime(2024, 1, 1),
            open_price=50000.0,
            high_price=50500.0,
            low_price=49500.0,
            close_price=50200.0,
            volume=1000.0
        )
        
        # Should not raise an exception
        session = db.get_session()
        from core.db import OHLCVData
        count = session.query(OHLCVData).count()
        assert count == 1
        session.close()
    
    @patch.dict(os.environ, {'DATABASE_URL': 'sqlite:///:memory:'})
    def test_get_ohlcv_data(self):
        db = Database()
        db.create_tables()
        
        # Save test data
        test_data = [
            {
                'symbol': 'BTCUSDT',
                'timeframe': '1m',
                'timestamp': datetime(2024, 1, 1, 0, 0),
                'open_price': 50000.0,
                'high_price': 50500.0,
                'low_price': 49500.0,
                'close_price': 50200.0,
                'volume': 1000.0
            },
            {
                'symbol': 'BTCUSDT',
                'timeframe': '1m',
                'timestamp': datetime(2024, 1, 1, 0, 1),
                'open_price': 50200.0,
                'high_price': 50700.0,
                'low_price': 49800.0,
                'close_price': 50400.0,
                'volume': 1100.0
            }
        ]
        
        for data in test_data:
            db.save_ohlcv_data(**data)
        
        # Retrieve data
        retrieved_data = db.get_ohlcv_data(
            symbol='BTCUSDT',
            timeframe='1m',
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 1, 2)
        )
        
        assert len(retrieved_data) == 2
        assert retrieved_data[0][1] == 50000.0  # open price of first row
    
    @patch.dict(os.environ, {'DATABASE_URL': 'sqlite:///:memory:'})
    def test_database_error_handling(self):
        db = Database()
        db.create_tables()
        
        # Test saving strategy with invalid data
        with pytest.raises(Exception):
            db.save_strategy(
                name=None,  # Invalid name
                version='1.0',
                parameters={},
                file='test.py'
            )
    
    @patch.dict(os.environ, {'DATABASE_URL': 'sqlite:///:memory:'})
    def test_duplicate_ohlcv_data(self):
        db = Database()
        db.create_tables()
        
        # Save same data twice
        data = {
            'symbol': 'BTCUSDT',
            'timeframe': '1m',
            'timestamp': datetime(2024, 1, 1),
            'open_price': 50000.0,
            'high_price': 50500.0,
            'low_price': 49500.0,
            'close_price': 50200.0,
            'volume': 1000.0
        }
        
        db.save_ohlcv_data(**data)
        
        # Second save should handle duplicates gracefully
        db.save_ohlcv_data(**data)
        
        # Should still have only one record
        session = db.get_session()
        from core.db import OHLCVData
        count = session.query(OHLCVData).count()
        assert count == 1
        session.close()