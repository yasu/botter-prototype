import asyncio
import ccxt
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
import os
import signal
import sys
import logging
from .db import Database
from .data_loader import DataLoader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LiveTrader:
    def __init__(self, strategy: Any, symbol: str, quantity: float, 
                 api_key: Optional[str] = None, api_secret: Optional[str] = None):
        self.strategy = strategy
        self.symbol = symbol
        self.quantity = quantity
        self.api_key = api_key or os.environ.get('BYBIT_API_KEY')
        self.api_secret = api_secret or os.environ.get('BYBIT_API_SECRET')
        
        # Initialize exchange
        self.exchange = ccxt.bybit({
            'apiKey': self.api_key,
            'secret': self.api_secret,
            'enableRateLimit': True,
            'options': {
                'defaultType': 'spot',  # Can be 'spot', 'future', 'swap'
            }
        })
        
        # Data loader and database
        self.data_loader = DataLoader(self.api_key, self.api_secret)
        self.db = Database()
        
        # Trading state
        self.is_running = False
        self.position = 0.0
        self.entry_price = 0.0
        self.current_signal = 0
        self.historical_data = pd.DataFrame()
        
        # Risk management
        self.max_position_size = quantity
        self.stop_loss_pct = 0.02  # 2% stop loss
        self.take_profit_pct = 0.04  # 4% take profit
        
        # Performance tracking
        self.total_pnl = 0.0
        self.trade_count = 0
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        logger.info(f"Received signal {signum}, shutting down...")
        self.is_running = False
    
    async def start(self, timeframe: str = '1m', lookback_periods: int = 100):
        """
        Start live trading
        
        Args:
            timeframe: Candle timeframe for signals
            lookback_periods: Number of historical periods to load for indicators
        """
        logger.info(f"Starting live trading for {self.symbol} with strategy {self.strategy.__class__.__name__}")
        
        self.is_running = True
        
        try:
            # Load initial historical data
            await self._load_historical_data(timeframe, lookback_periods)
            
            # Start trading loop
            await self._trading_loop(timeframe)
            
        except Exception as e:
            logger.error(f"Error in live trading: {e}")
        finally:
            logger.info("Live trading stopped")
            # Close any open positions
            await self._close_all_positions()
    
    async def _load_historical_data(self, timeframe: str, lookback_periods: int):
        """Load historical data for indicators"""
        logger.info(f"Loading {lookback_periods} periods of historical data...")
        
        # Calculate start date
        end_date = datetime.now()
        # Rough calculation - improve based on timeframe
        if timeframe == '1m':
            start_date = end_date - timedelta(minutes=lookback_periods)
        elif timeframe == '5m':
            start_date = end_date - timedelta(minutes=lookback_periods * 5)
        elif timeframe == '1h':
            start_date = end_date - timedelta(hours=lookback_periods)
        elif timeframe == '1d':
            start_date = end_date - timedelta(days=lookback_periods)
        else:
            start_date = end_date - timedelta(minutes=lookback_periods)
        
        # Try to get data from database first
        db_symbol = self.symbol.replace('/', '')
        self.historical_data = self.data_loader.get_historical_data_from_db(
            db_symbol, timeframe, 
            start_date.strftime('%Y-%m-%d'), 
            end_date.strftime('%Y-%m-%d')
        )
        
        # If not enough data in DB, fetch from API
        if len(self.historical_data) < lookback_periods:
            logger.info("Insufficient data in database, fetching from API...")
            df = await self.data_loader.download_historical_data(
                self.symbol,
                start_date.strftime('%Y-%m-%d'),
                end_date.strftime('%Y-%m-%d'),
                timeframe,
                save_to_db=True
            )
            self.historical_data = df.set_index('timestamp')
        
        logger.info(f"Loaded {len(self.historical_data)} candles")
    
    async def _trading_loop(self, timeframe: str):
        """Main trading loop"""
        logger.info("Starting trading loop...")
        
        # Calculate sleep time based on timeframe
        if timeframe == '1m':
            sleep_time = 60
        elif timeframe == '5m':
            sleep_time = 300
        elif timeframe == '15m':
            sleep_time = 900
        elif timeframe == '1h':
            sleep_time = 3600
        else:
            sleep_time = 60
        
        while self.is_running:
            try:
                # Get latest candle
                await self._update_data(timeframe)
                
                # Generate signal
                signal = await self._generate_signal()
                
                # Execute trades based on signal
                await self._execute_signal(signal)
                
                # Check for stop loss/take profit
                await self._check_risk_management()
                
                # Log status
                await self._log_status()
                
                # Wait for next candle
                await asyncio.sleep(sleep_time)
                
            except Exception as e:
                logger.error(f"Error in trading loop: {e}")
                await asyncio.sleep(10)  # Short pause before retrying
    
    async def _update_data(self, timeframe: str):
        """Update historical data with latest candle"""
        try:
            # Get latest candles
            candles = self.exchange.fetch_ohlcv(
                symbol=self.symbol,
                timeframe=timeframe,
                limit=2
            )
            
            if len(candles) >= 1:
                latest_candle = candles[-1]
                new_row = pd.DataFrame({
                    'open': [latest_candle[1]],
                    'high': [latest_candle[2]],
                    'low': [latest_candle[3]],
                    'close': [latest_candle[4]],
                    'volume': [latest_candle[5]]
                }, index=[pd.to_datetime(latest_candle[0], unit='ms')])
                
                # Update historical data
                self.historical_data = pd.concat([self.historical_data, new_row])
                self.historical_data = self.historical_data[~self.historical_data.index.duplicated(keep='last')]
                self.historical_data = self.historical_data.tail(200)  # Keep last 200 candles
                
        except Exception as e:
            logger.error(f"Error updating data: {e}")
    
    async def _generate_signal(self) -> int:
        """Generate trading signal using strategy"""
        try:
            if len(self.historical_data) < 20:  # Need minimum data
                return 0
            
            # Generate signals using strategy
            signals = self.strategy.generate_signals(self.historical_data)
            latest_signal = signals.iloc[-1] if len(signals) > 0 else 0
            
            return int(latest_signal)
        
        except Exception as e:
            logger.error(f"Error generating signal: {e}")
            return 0
    
    async def _execute_signal(self, signal: int):
        """Execute trades based on signal"""
        try:
            current_price = float(self.historical_data['close'].iloc[-1])
            
            # Buy signal
            if signal == 1 and self.position == 0:
                await self._place_buy_order(current_price)
            
            # Sell signal
            elif signal == -1 and self.position > 0:
                await self._place_sell_order(current_price)
            
            self.current_signal = signal
            
        except Exception as e:
            logger.error(f"Error executing signal: {e}")
    
    async def _place_buy_order(self, price: float):
        """Place buy order"""
        try:
            order = self.exchange.create_market_buy_order(
                symbol=self.symbol,
                amount=self.quantity
            )
            
            if order['status'] == 'closed':
                self.position = self.quantity
                self.entry_price = float(order['average'] or price)
                self.trade_count += 1
                
                # Save to database
                self.db.save_live_order(
                    strategy_id=1,  # You might want to get this dynamically
                    bybit_order_id=order['id'],
                    side='buy',
                    qty=self.quantity,
                    price=self.entry_price,
                    status=order['status'],
                    filled_qty=float(order['filled'])
                )
                
                logger.info(f"BUY order executed: {self.quantity} @ {self.entry_price}")
            
        except Exception as e:
            logger.error(f"Error placing buy order: {e}")
    
    async def _place_sell_order(self, price: float):
        """Place sell order"""
        try:
            order = self.exchange.create_market_sell_order(
                symbol=self.symbol,
                amount=self.position
            )
            
            if order['status'] == 'closed':
                exit_price = float(order['average'] or price)
                pnl = (exit_price - self.entry_price) * self.position
                self.total_pnl += pnl
                
                # Save to database
                self.db.save_live_order(
                    strategy_id=1,  # You might want to get this dynamically
                    bybit_order_id=order['id'],
                    side='sell',
                    qty=self.position,
                    price=exit_price,
                    status=order['status'],
                    filled_qty=float(order['filled'])
                )
                
                logger.info(f"SELL order executed: {self.position} @ {exit_price}, PnL: {pnl:.2f}")
                
                self.position = 0.0
                self.entry_price = 0.0
            
        except Exception as e:
            logger.error(f"Error placing sell order: {e}")
    
    async def _check_risk_management(self):
        """Check stop loss and take profit conditions"""
        if self.position == 0:
            return
        
        try:
            current_price = float(self.historical_data['close'].iloc[-1])
            price_change = (current_price - self.entry_price) / self.entry_price
            
            # Stop loss
            if price_change <= -self.stop_loss_pct:
                logger.warning(f"Stop loss triggered at {current_price}")
                await self._place_sell_order(current_price)
            
            # Take profit
            elif price_change >= self.take_profit_pct:
                logger.info(f"Take profit triggered at {current_price}")
                await self._place_sell_order(current_price)
                
        except Exception as e:
            logger.error(f"Error in risk management: {e}")
    
    async def _close_all_positions(self):
        """Close all open positions before shutdown"""
        if self.position > 0:
            try:
                current_price = self.data_loader.get_latest_price(self.symbol)
                await self._place_sell_order(current_price)
                logger.info("Closed all positions before shutdown")
            except Exception as e:
                logger.error(f"Error closing positions: {e}")
    
    async def _log_status(self):
        """Log current trading status"""
        try:
            current_price = float(self.historical_data['close'].iloc[-1])
            balance = self.data_loader.get_account_balance()
            
            if self.position > 0:
                unrealized_pnl = (current_price - self.entry_price) * self.position
                logger.info(
                    f"Status - Price: {current_price:.2f}, "
                    f"Position: {self.position:.4f}, "
                    f"Entry: {self.entry_price:.2f}, "
                    f"Unrealized PnL: {unrealized_pnl:.2f}, "
                    f"Total PnL: {self.total_pnl:.2f}"
                )
            else:
                logger.info(
                    f"Status - Price: {current_price:.2f}, "
                    f"No position, "
                    f"Total PnL: {self.total_pnl:.2f}, "
                    f"Balance: {balance.get('total', {}).get('USDT', 0):.2f} USDT"
                )
                
        except Exception as e:
            logger.debug(f"Error logging status: {e}")
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary"""
        return {
            'total_pnl': self.total_pnl,
            'trade_count': self.trade_count,
            'current_position': self.position,
            'entry_price': self.entry_price,
            'is_running': self.is_running
        }