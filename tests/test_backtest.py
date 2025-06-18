import pytest
import pandas as pd
import numpy as np
from datetime import datetime
from core.backtest import Backtester, Trade, BacktestResult

class TestTrade:
    def test_trade_creation(self):
        trade = Trade(
            timestamp=datetime(2024, 1, 1),
            side='buy',
            price=50000.0,
            quantity=0.1,
            value=5000.0,
            commission=5.0
        )
        assert trade.timestamp == datetime(2024, 1, 1)
        assert trade.side == 'buy'
        assert trade.price == 50000.0
        assert trade.quantity == 0.1
        assert trade.value == 5000.0
        assert trade.commission == 5.0
        assert trade.pnl == 0.0

class TestBacktester:
    def test_backtester_initialization(self):
        backtester = Backtester(initial_balance=10000, commission_rate=0.001)
        assert backtester.initial_balance == 10000
        assert backtester.commission_rate == 0.001
        assert backtester.balance == 10000
        assert backtester.position == 0.0
        assert len(backtester.trades) == 0
    
    def test_reset(self):
        backtester = Backtester(initial_balance=10000)
        backtester.balance = 8000
        backtester.position = 0.5
        backtester.trades.append(Trade(
            timestamp=datetime.now(),
            side='buy',
            price=50000,
            quantity=0.1,
            value=5000,
            commission=5
        ))
        
        backtester.reset()
        assert backtester.balance == 10000
        assert backtester.position == 0.0
        assert len(backtester.trades) == 0
    
    def test_calculate_position_size(self):
        backtester = Backtester(initial_balance=10000)
        size = backtester._calculate_position_size(50000, 1.0)  # full allocation
        assert abs(size - 0.2) < 0.01  # approximately 0.2 (accounting for commission)
        
        size = backtester._calculate_position_size(50000, 0.5)  # half allocation
        assert abs(size - 0.1) < 0.01  # approximately 0.1
    
    def test_execute_trade_buy(self):
        backtester = Backtester(initial_balance=10000, commission_rate=0.001)
        timestamp = datetime(2024, 1, 1)
        price = 50000.0
        signal_strength = 1.0
        
        backtester._execute_trade(timestamp, price, signal_strength)
        
        assert len(backtester.trades) == 1
        trade = backtester.trades[0]
        assert trade.side == 'buy'
        assert trade.price == price
        assert backtester.position > 0
        assert backtester.balance < 10000  # reduced by commission
    
    def test_execute_trade_sell(self):
        backtester = Backtester(initial_balance=10000, commission_rate=0.001)
        timestamp = datetime(2024, 1, 1)
        
        # First buy
        backtester._execute_trade(timestamp, 50000.0, 1.0)
        initial_position = backtester.position
        
        # Then sell
        backtester._execute_trade(timestamp, 52000.0, -1.0)
        
        assert len(backtester.trades) == 2
        sell_trade = backtester.trades[1]
        assert sell_trade.side == 'sell'
        assert sell_trade.pnl > 0  # profitable trade
        assert backtester.position == 0
    
    def test_calculate_metrics(self, sample_ohlcv_data, sample_strategy):
        backtester = Backtester(initial_balance=10000)
        result = backtester.run(sample_strategy, sample_ohlcv_data)
        
        assert isinstance(result, BacktestResult)
        assert result.initial_balance == 10000
        assert result.final_balance > 0
        assert isinstance(result.total_return, float)
        assert isinstance(result.sharpe_ratio, float)
        assert isinstance(result.max_drawdown, float)
        assert isinstance(result.win_rate, (float, int))
        assert result.total_trades >= 0
        # Total trades counts completed round trips (buy + sell)
        # result.trades contains all individual trades
        assert len(result.trades) >= result.total_trades
    
    def test_generate_report(self, sample_ohlcv_data, sample_strategy):
        backtester = Backtester(initial_balance=10000)
        result = backtester.run(sample_strategy, sample_ohlcv_data)
        report = backtester.generate_report(result)
        
        assert isinstance(report, str)
        assert 'Total Return' in report
        assert 'Sharpe Ratio' in report
        assert 'Max Drawdown' in report
        assert 'Win Rate' in report
    
    def test_empty_data(self, sample_strategy):
        backtester = Backtester()
        empty_df = pd.DataFrame()
        result = backtester.run(sample_strategy, empty_df)
        
        assert result.total_trades == 0
        assert result.final_balance == result.initial_balance
        assert result.total_return == 0.0
    
    def test_no_signals(self, sample_strategy):
        backtester = Backtester()
        # Create data with no signals
        data = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=10, freq='1min'),
            'open': [50000] * 10,
            'high': [50100] * 10,
            'low': [49900] * 10,
            'close': [50000] * 10,
            'volume': [1000] * 10
        })
        
        # Mock strategy to return no signals
        class NoSignalStrategy:
            def __init__(self):
                self.parameters = {}
                self.version = '1.0'
            
            def get_parameters(self):
                return self.parameters
                
            def generate_signals(self, df):
                return pd.Series(0, index=df.index)
        
        result = backtester.run(NoSignalStrategy(), data)
        assert result.total_trades == 0
        assert result.final_balance == result.initial_balance