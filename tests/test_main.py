import pytest
import os
from click.testing import CliRunner
from unittest.mock import patch, MagicMock, mock_open
import sys

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from main import cli, load_strategy_from_file

class TestCLI:
    def setup_method(self):
        self.runner = CliRunner()
    
    def test_cli_help(self):
        result = self.runner.invoke(cli, ['--help'])
        assert result.exit_code == 0
        assert 'Automated Trading System CLI' in result.output
    
    def test_cli_verbose_flag(self):
        with patch('logging.basicConfig') as mock_logging:
            result = self.runner.invoke(cli, ['--verbose', '--help'])
            assert result.exit_code == 0
            mock_logging.assert_called_once()
    
    @patch('main.DataLoader')
    def test_download_command(self, mock_data_loader):
        # Mock the DataLoader
        mock_loader_instance = MagicMock()
        mock_data_loader.return_value = mock_loader_instance
        
        # Mock async method
        async def mock_download(*args, **kwargs):
            return MagicMock(len=lambda: 100)
        
        mock_loader_instance.download_historical_data = mock_download
        
        with patch('asyncio.run') as mock_run:
            result = self.runner.invoke(cli, [
                'download', 'BTCUSDT', '2024-01-01', '2024-04-01', '1m'
            ])
            
            assert 'Downloading BTCUSDT data' in result.output
            mock_run.assert_called_once()
    
    @patch('main.load_strategy_from_file')
    @patch('main.DataLoader')
    @patch('main.Backtester')
    @patch('main.Database')
    def test_backtest_command(self, mock_db, mock_backtester, mock_data_loader, mock_load_strategy):
        # Mock strategy
        mock_strategy = MagicMock()
        mock_strategy.__class__.__name__ = 'TestStrategy'
        mock_strategy.get_parameters.return_value = {}
        mock_load_strategy.return_value = mock_strategy
        
        # Mock data loader
        mock_loader_instance = MagicMock()
        mock_data_loader.return_value = mock_loader_instance
        import pandas as pd
        mock_loader_instance.get_historical_data_from_db.return_value = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=10, freq='1min'),
            'close': [50000] * 10
        })
        
        # Mock backtester
        mock_backtester_instance = MagicMock()
        mock_backtester.return_value = mock_backtester_instance
        mock_result = MagicMock()
        mock_result.total_return = 5.5
        mock_backtester_instance.run.return_value = mock_result
        mock_backtester_instance.generate_report.return_value = "Test Report"
        
        # Mock database
        mock_db_instance = MagicMock()
        mock_db.return_value = mock_db_instance
        mock_db_instance.save_strategy.return_value = 1
        mock_db_instance.save_backtest_result.return_value = 1
        
        result = self.runner.invoke(cli, [
            'backtest', 'test_strategy.py', 'BTCUSDT', '2024-03-01', '2024-04-01'
        ])
        
        assert 'Running backtest' in result.output
        assert 'Test Report' in result.output
    
    @patch('main.load_strategy_from_file')
    @patch('main.LiveTrader')
    def test_live_command_dry_run(self, mock_live_trader, mock_load_strategy):
        # Mock strategy
        mock_strategy = MagicMock()
        mock_load_strategy.return_value = mock_strategy
        
        result = self.runner.invoke(cli, [
            'live', 'test_strategy.py', 'BTCUSDT', '--dry-run'
        ])
        
        assert 'DRY RUN MODE' in result.output
        assert 'would start trading here' in result.output
    
    @patch('main.Database')
    def test_history_command(self, mock_db):
        mock_db_instance = MagicMock()
        mock_db.return_value = mock_db_instance
        
        # Mock session and query results
        mock_session = MagicMock()
        mock_db_instance.get_session.return_value = mock_session
        
        # Mock backtest and strategy objects
        mock_backtest = MagicMock()
        mock_backtest.id = 1
        mock_backtest.symbol = 'BTCUSDT'
        mock_backtest.timeframe = '1m'
        mock_backtest.metrics = {'total_return': 5.5}
        mock_backtest.created_at.strftime.return_value = '2024-01-01 12:00'
        
        mock_strategy = MagicMock()
        mock_strategy.name = 'TestStrategy'
        
        mock_session.query.return_value.join.return_value.order_by.return_value.limit.return_value.all.return_value = [
            (mock_backtest, mock_strategy)
        ]
        
        result = self.runner.invoke(cli, ['history'])
        
        assert 'backtests:' in result.output
        mock_session.close.assert_called()
    
    @patch('main.Database')
    def test_status_command(self, mock_db):
        # Mock database connection
        mock_db_instance = MagicMock()
        mock_db.return_value = mock_db_instance
        mock_session = MagicMock()
        mock_db_instance.get_session.return_value = mock_session
        
        # Mock DataLoader for API check
        with patch('main.DataLoader') as mock_data_loader:
            mock_loader_instance = MagicMock()
            mock_data_loader.return_value = mock_loader_instance
            mock_loader_instance.get_account_balance.return_value = {
                'total': {'USDT': 10000.0}
            }
            
            result = self.runner.invoke(cli, ['status'])
            
            assert 'System Status' in result.output
            assert 'Database: Connected' in result.output
    
    @patch('main.Database')
    def test_init_command(self, mock_db):
        mock_db_instance = MagicMock()
        mock_db.return_value = mock_db_instance
        
        result = self.runner.invoke(cli, ['init'])
        
        assert 'Initializing database' in result.output
        mock_db_instance.create_tables.assert_called_once()

class TestLoadStrategyFromFile:
    def test_load_strategy_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            load_strategy_from_file('nonexistent_file.py')
    
    @patch('builtins.open', new_callable=mock_open, read_data='''
class TestStrategy:
    def generate_signals(self, df):
        return df
''')
    @patch('os.path.exists')
    @patch('importlib.util.spec_from_file_location')
    @patch('importlib.util.module_from_spec')
    def test_load_strategy_success(self, mock_module_from_spec, mock_spec_from_file, mock_exists, mock_file):
        mock_exists.return_value = True
        
        # Mock strategy class
        class MockStrategy:
            def generate_signals(self, df):
                return df
        
        # Mock module
        mock_module = MagicMock()
        mock_module.TestStrategy = MockStrategy
        mock_module_from_spec.return_value = mock_module
        
        # Mock spec and loader
        mock_spec = MagicMock()
        mock_loader = MagicMock()
        mock_spec.loader = mock_loader
        mock_spec_from_file.return_value = mock_spec
        
        # Mock dir() to return our class
        with patch('builtins.dir', return_value=['TestStrategy']):
            with patch('builtins.getattr', return_value=MockStrategy):
                strategy = load_strategy_from_file('test_strategy.py')
                assert isinstance(strategy, MockStrategy)
    
    @patch('builtins.open', new_callable=mock_open, read_data='# Empty file')
    @patch('os.path.exists')
    @patch('importlib.util.spec_from_file_location')
    @patch('importlib.util.module_from_spec')
    def test_load_strategy_no_valid_class(self, mock_module_from_spec, mock_spec_from_file, mock_exists, mock_file):
        mock_exists.return_value = True
        
        # Mock module with no valid strategy class
        mock_module = MagicMock()
        mock_module_from_spec.return_value = mock_module
        
        # Mock spec and loader
        mock_spec = MagicMock()
        mock_loader = MagicMock()
        mock_spec.loader = mock_loader
        mock_spec_from_file.return_value = mock_spec
        
        # Mock dir() to return no valid classes
        with patch('builtins.dir', return_value=['some_function']):
            with pytest.raises(ValueError, match="No valid strategy class found"):
                load_strategy_from_file('test_strategy.py')
    
    @patch('builtins.open', new_callable=mock_open, read_data='')
    @patch('os.path.exists')
    @patch('importlib.util.spec_from_file_location')
    @patch('importlib.util.module_from_spec')
    def test_load_strategy_specific_name_not_found(self, mock_module_from_spec, mock_spec_from_file, mock_exists, mock_file):
        mock_exists.return_value = True
        
        # Mock module without the specified strategy
        mock_module = MagicMock()
        mock_module_from_spec.return_value = mock_module
        
        # Mock spec and loader
        mock_spec = MagicMock()
        mock_loader = MagicMock()
        mock_spec.loader = mock_loader
        mock_spec_from_file.return_value = mock_spec
        
        # Mock hasattr to return False
        with patch('builtins.hasattr', return_value=False):
            with pytest.raises(AttributeError, match="Strategy 'NonExistentStrategy' not found"):
                load_strategy_from_file('test_strategy.py', 'NonExistentStrategy')