#!/usr/bin/env python3
import click
import asyncio
import importlib.util
import sys
import os
from datetime import datetime
from pathlib import Path
import json

from core.data_loader import DataLoader
from core.backtest import Backtester
from core.live import LiveTrader
from core.db import Database

def load_strategy_from_file(strategy_file: str, strategy_name: str = None):
    """Load strategy class from Python file"""
    if not os.path.exists(strategy_file):
        raise FileNotFoundError(f"Strategy file not found: {strategy_file}")
    
    # Load module from file
    spec = importlib.util.spec_from_file_location("strategy_module", strategy_file)
    strategy_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(strategy_module)
    
    # Try to find strategy class
    if strategy_name:
        if hasattr(strategy_module, strategy_name):
            return getattr(strategy_module, strategy_name)()
        else:
            raise AttributeError(f"Strategy '{strategy_name}' not found in {strategy_file}")
    
    # Auto-detect strategy class
    for attr_name in dir(strategy_module):
        attr = getattr(strategy_module, attr_name)
        if (isinstance(attr, type) and 
            hasattr(attr, 'generate_signals') and 
            attr_name != 'object'):
            return attr()
    
    raise ValueError(f"No valid strategy class found in {strategy_file}")

@click.group()
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging')
def cli(verbose):
    """Automated Trading System CLI"""
    if verbose:
        import logging
        logging.basicConfig(level=logging.DEBUG)

@cli.command()
@click.argument('symbol')
@click.argument('start_date')
@click.argument('end_date')
@click.argument('timeframe', default='1m')
@click.option('--force', is_flag=True, help='Force re-download even if data exists')
def download(symbol, start_date, end_date, timeframe, force):
    """
    Download historical data from Bybit
    
    Example: python main.py download BTCUSDT 2024-01-01 2024-04-01 1m
    """
    click.echo(f"Downloading {symbol} data from {start_date} to {end_date} ({timeframe})")
    
    data_loader = DataLoader()
    
    # Convert symbol format
    api_symbol = symbol if '/' in symbol else f"{symbol[:-4]}/{symbol[-4:]}"
    
    async def download_task():
        try:
            df = await data_loader.download_historical_data(
                symbol=api_symbol,
                start_date=start_date,
                end_date=end_date,
                timeframe=timeframe,
                save_to_db=True
            )
            click.echo(f"Successfully downloaded {len(df)} candles")
        except Exception as e:
            click.echo(f"Error downloading data: {e}", err=True)
            sys.exit(1)
    
    asyncio.run(download_task())

@cli.command()
@click.argument('strategy_file')
@click.argument('symbol')
@click.argument('start_date')
@click.argument('end_date')
@click.option('--timeframe', default='1m', help='Timeframe for backtest')
@click.option('--initial-balance', default=10000.0, help='Initial balance for backtest')
@click.option('--commission', default=0.001, help='Commission rate')
@click.option('--strategy-name', help='Specific strategy class name')
@click.option('--parameters', help='Strategy parameters as JSON string')
def backtest(strategy_file, symbol, start_date, end_date, timeframe, 
            initial_balance, commission, strategy_name, parameters):
    """
    Run backtest with specified strategy
    
    Example: python main.py backtest strategies/sample_meanrev.py BTCUSDT 2024-03-01 2024-04-01
    """
    click.echo(f"Running backtest: {strategy_file} on {symbol}")
    
    try:
        # Load strategy
        strategy = load_strategy_from_file(strategy_file, strategy_name)
        
        # Set custom parameters if provided
        if parameters:
            params = json.loads(parameters)
            strategy.set_parameters(params)
        
        # Load data
        data_loader = DataLoader()
        db_symbol = symbol.replace('/', '')
        
        df = data_loader.get_historical_data_from_db(
            symbol=db_symbol,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date
        )
        
        if df.empty:
            click.echo(f"No data found for {symbol}. Try downloading first.", err=True)
            sys.exit(1)
        
        # Run backtest
        backtester = Backtester(initial_balance=initial_balance, commission_rate=commission)
        result = backtester.run(strategy, df)
        
        # Print results
        click.echo("\n" + backtester.generate_report(result))
        
        # Save to database
        db = Database()
        strategy_id = db.save_strategy(
            name=strategy.__class__.__name__,
            version=getattr(strategy, 'version', '1.0'),
            parameters=strategy.get_parameters(),
            file=strategy_file
        )
        
        metrics = {
            'total_return': result.total_return,
            'sharpe_ratio': result.sharpe_ratio,
            'max_drawdown': result.max_drawdown,
            'win_rate': result.win_rate,
            'total_trades': result.total_trades,
            'profit_factor': result.profit_factor
        }
        
        backtest_id = db.save_backtest_result(
            strategy_id=strategy_id,
            symbol=db_symbol,
            timeframe=timeframe,
            period_start=datetime.strptime(start_date, '%Y-%m-%d'),
            period_end=datetime.strptime(end_date, '%Y-%m-%d'),
            initial_balance=initial_balance,
            metrics=metrics
        )
        
        click.echo(f"\nBacktest saved with ID: {backtest_id}")
        
    except Exception as e:
        click.echo(f"Error running backtest: {e}", err=True)
        sys.exit(1)

@cli.command()
@click.argument('strategy_file')
@click.argument('symbol')
@click.option('--qty', default=0.01, help='Quantity to trade')
@click.option('--timeframe', default='1m', help='Timeframe for signals')
@click.option('--strategy-name', help='Specific strategy class name')
@click.option('--parameters', help='Strategy parameters as JSON string')
@click.option('--dry-run', is_flag=True, help='Run in simulation mode (no real trades)')
def live(strategy_file, symbol, qty, timeframe, strategy_name, parameters, dry_run):
    """
    Start live trading
    
    Example: python main.py live strategies/sample_meanrev.py BTCUSDT --qty 0.01
    """
    if dry_run:
        click.echo("⚠️  DRY RUN MODE - No real trades will be executed")
    
    click.echo(f"Starting live trading: {strategy_file} on {symbol}")
    
    try:
        # Load strategy
        strategy = load_strategy_from_file(strategy_file, strategy_name)
        
        # Set custom parameters if provided
        if parameters:
            params = json.loads(parameters)
            strategy.set_parameters(params)
        
        # Convert symbol format
        api_symbol = symbol if '/' in symbol else f"{symbol[:-4]}/{symbol[-4:]}"
        
        # Start live trader
        trader = LiveTrader(strategy, api_symbol, qty)
        
        if dry_run:
            # In dry run mode, just simulate
            click.echo("Dry run mode - would start trading here")
            return
        
        # Confirm before starting real trading
        if not click.confirm(f"Start live trading with {qty} {symbol}?"):
            click.echo("Cancelled")
            return
        
        asyncio.run(trader.start(timeframe))
        
    except Exception as e:
        click.echo(f"Error in live trading: {e}", err=True)
        sys.exit(1)

@cli.command()
@click.option('--limit', default=10, help='Number of backtests to show')
def history(limit):
    """Show backtest history"""
    db = Database()
    session = db.get_session()
    
    try:
        from core.db import Backtest, Strategy
        
        results = session.query(Backtest, Strategy).join(Strategy).order_by(
            Backtest.created_at.desc()
        ).limit(limit).all()
        
        if not results:
            click.echo("No backtest history found")
            return
        
        click.echo(f"\nLast {len(results)} backtests:")
        click.echo("-" * 80)
        
        for backtest, strategy in results:
            metrics = backtest.metrics or {}
            click.echo(
                f"ID: {backtest.id:3d} | "
                f"{strategy.name:20s} | "
                f"{backtest.symbol:8s} | "
                f"{backtest.timeframe:4s} | "
                f"Return: {metrics.get('total_return', 0):6.2f}% | "
                f"Sharpe: {metrics.get('sharpe_ratio', 0):5.2f} | "
                f"MDD: {metrics.get('max_drawdown', 0):5.2f}% | "
                f"{backtest.created_at.strftime('%Y-%m-%d %H:%M')}"
            )
    
    finally:
        session.close()

@cli.command()
@click.argument('backtest_id', type=int)
def show(backtest_id):
    """Show detailed backtest results"""
    db = Database()
    session = db.get_session()
    
    try:
        from core.db import Backtest, Strategy
        
        result = session.query(Backtest, Strategy).join(Strategy).filter(
            Backtest.id == backtest_id
        ).first()
        
        if not result:
            click.echo(f"Backtest {backtest_id} not found")
            return
        
        backtest, strategy = result
        metrics = backtest.metrics or {}
        
        click.echo(f"""
Backtest Details (ID: {backtest.id})
{'=' * 40}
Strategy: {strategy.name} v{strategy.version}
Symbol: {backtest.symbol}
Timeframe: {backtest.timeframe}
Period: {backtest.period_start} to {backtest.period_end}
Initial Balance: ${backtest.initial_balance}

Performance Metrics:
- Total Return: {metrics.get('total_return', 0):.2f}%
- Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}
- Max Drawdown: {metrics.get('max_drawdown', 0):.2f}%
- Win Rate: {metrics.get('win_rate', 0):.2f}%
- Total Trades: {metrics.get('total_trades', 0)}
- Profit Factor: {metrics.get('profit_factor', 0):.2f}

Strategy Parameters:
{json.dumps(strategy.parameters, indent=2)}

Created: {backtest.created_at}
""")
    
    finally:
        session.close()

@cli.command()
def status():
    """Show system status"""
    click.echo("System Status")
    click.echo("=" * 20)
    
    # Check database connection
    try:
        db = Database()
        session = db.get_session()
        session.execute("SELECT 1")
        session.close()
        click.echo("✅ Database: Connected")
    except Exception as e:
        click.echo(f"❌ Database: {e}")
    
    # Check API connection
    try:
        data_loader = DataLoader()
        balance = data_loader.get_account_balance()
        click.echo("✅ Bybit API: Connected")
        click.echo(f"   USDT Balance: {balance.get('total', {}).get('USDT', 0):.2f}")
    except Exception as e:
        click.echo(f"❌ Bybit API: {e}")

@cli.command()
def init():
    """Initialize database"""
    click.echo("Initializing database...")
    
    try:
        db = Database()
        db.create_tables()
        click.echo("✅ Database initialized successfully")
        
        # Check if alembic is available and create initial migration
        try:
            import subprocess
            result = subprocess.run(['alembic', 'revision', '--autogenerate', '-m', 'Initial migration'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                click.echo("✅ Initial migration created")
            else:
                click.echo(f"⚠️  Migration warning: {result.stderr}")
        except FileNotFoundError:
            click.echo("⚠️  Alembic not found, skipping migration creation")
        
    except Exception as e:
        click.echo(f"❌ Error initializing database: {e}")
        sys.exit(1)

if __name__ == '__main__':
    cli()