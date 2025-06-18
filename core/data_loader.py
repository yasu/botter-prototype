import ccxt
import asyncio
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, List, Dict
import os
import time
from .db import Database
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataLoader:
    def __init__(self, api_key: Optional[str] = None, api_secret: Optional[str] = None):
        self.api_key = api_key or os.environ.get('BYBIT_API_KEY')
        self.api_secret = api_secret or os.environ.get('BYBIT_API_SECRET')
        
        # Initialize Bybit exchange
        self.exchange = ccxt.bybit({
            'apiKey': self.api_key,
            'secret': self.api_secret,
            'enableRateLimit': True,
            'options': {
                'defaultType': 'spot',  # Can be 'spot', 'future', 'swap'
            }
        })
        
        # Database connection
        self.db = Database()
    
    def _timeframe_to_minutes(self, timeframe: str) -> int:
        """Convert timeframe string to minutes"""
        mapping = {
            '1m': 1,
            '3m': 3,
            '5m': 5,
            '15m': 15,
            '30m': 30,
            '1h': 60,
            '2h': 120,
            '4h': 240,
            '6h': 360,
            '12h': 720,
            '1d': 1440,
            '1w': 10080,
        }
        return mapping.get(timeframe, 60)
    
    def _parse_timestamp(self, timestamp) -> datetime:
        """Convert timestamp to datetime"""
        if isinstance(timestamp, int):
            return datetime.fromtimestamp(timestamp / 1000)
        elif isinstance(timestamp, str):
            return datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
        else:
            return timestamp
    
    async def download_historical_data(self, symbol: str, start_date: str, end_date: str, 
                                     timeframe: str = '1m', save_to_db: bool = True) -> pd.DataFrame:
        """
        Download historical OHLCV data from Bybit
        
        Args:
            symbol: Trading pair (e.g., 'BTC/USDT')
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
            timeframe: Candle timeframe (1m, 5m, 15m, 30m, 1h, 4h, 1d)
            save_to_db: Whether to save data to database
            
        Returns:
            DataFrame with OHLCV data
        """
        logger.info(f"Downloading {symbol} data from {start_date} to {end_date} with {timeframe} timeframe")
        
        # Convert dates to timestamps
        start_ts = int(datetime.strptime(start_date, '%Y-%m-%d').timestamp() * 1000)
        end_ts = int(datetime.strptime(end_date, '%Y-%m-%d').timestamp() * 1000)
        
        all_candles = []
        
        # Bybit returns max 200 candles per request
        limit = 200
        timeframe_ms = self._timeframe_to_minutes(timeframe) * 60 * 1000
        
        current_start = start_ts
        
        while current_start < end_ts:
            try:
                # Fetch candles
                candles = self.exchange.fetch_ohlcv(
                    symbol=symbol,
                    timeframe=timeframe,
                    since=current_start,
                    limit=limit
                )
                
                if not candles:
                    break
                
                all_candles.extend(candles)
                
                # Update start time for next batch
                last_candle_time = candles[-1][0]
                current_start = last_candle_time + timeframe_ms
                
                # Rate limiting
                time.sleep(self.exchange.rateLimit / 1000)
                
                logger.info(f"Downloaded {len(candles)} candles, total: {len(all_candles)}")
                
            except Exception as e:
                logger.error(f"Error downloading data: {e}")
                break
        
        # Convert to DataFrame
        df = pd.DataFrame(all_candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        
        # Save to database if requested
        if save_to_db and not df.empty:
            candles_data = []
            for _, row in df.iterrows():
                candles_data.append({
                    'timestamp': row['timestamp'],
                    'open': row['open'],
                    'high': row['high'],
                    'low': row['low'],
                    'close': row['close'],
                    'volume': row['volume']
                })
            
            # Convert symbol format for DB (BTC/USDT -> BTCUSDT)
            db_symbol = symbol.replace('/', '')
            self.db.save_candles(candles_data, db_symbol, timeframe)
            logger.info(f"Saved {len(candles_data)} candles to database")
        
        return df
    
    async def stream_live_data(self, symbol: str, timeframe: str = '1m', 
                             callback=None, save_to_db: bool = True):
        """
        Stream live candle data using WebSocket
        
        Args:
            symbol: Trading pair (e.g., 'BTC/USDT')
            timeframe: Candle timeframe
            callback: Function to call with new candle data
            save_to_db: Whether to save data to database
        """
        logger.info(f"Starting live data stream for {symbol} with {timeframe} timeframe")
        
        # Convert symbol format
        ws_symbol = symbol.replace('/', '')
        
        while True:
            try:
                # For simplicity, we'll use REST API polling instead of WebSocket
                # In production, you'd want to use proper WebSocket connection
                
                # Get latest candle
                candles = self.exchange.fetch_ohlcv(
                    symbol=symbol,
                    timeframe=timeframe,
                    limit=1
                )
                
                if candles:
                    latest_candle = candles[0]
                    candle_data = {
                        'timestamp': self._parse_timestamp(latest_candle[0]),
                        'open': latest_candle[1],
                        'high': latest_candle[2],
                        'low': latest_candle[3],
                        'close': latest_candle[4],
                        'volume': latest_candle[5]
                    }
                    
                    # Save to database
                    if save_to_db:
                        self.db.save_candles([candle_data], ws_symbol, timeframe)
                    
                    # Call callback if provided
                    if callback:
                        await callback(candle_data)
                    
                    logger.debug(f"Received candle: {candle_data['timestamp']} - Close: {candle_data['close']}")
                
                # Wait for next candle period
                await asyncio.sleep(self._timeframe_to_minutes(timeframe) * 60)
                
            except Exception as e:
                logger.error(f"Error in live data stream: {e}")
                await asyncio.sleep(5)  # Retry after 5 seconds
    
    def get_historical_data_from_db(self, symbol: str, timeframe: str, 
                                  start_date: str, end_date: str) -> pd.DataFrame:
        """
        Retrieve historical data from database
        
        Args:
            symbol: Trading pair (e.g., 'BTCUSDT')
            timeframe: Candle timeframe
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
            
        Returns:
            DataFrame with OHLCV data
        """
        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')
        
        candles = self.db.get_candles(symbol, timeframe, start_dt, end_dt)
        
        if not candles:
            return pd.DataFrame()
        
        data = []
        for candle in candles:
            data.append({
                'timestamp': candle.time,
                'open': float(candle.open),
                'high': float(candle.high),
                'low': float(candle.low),
                'close': float(candle.close),
                'volume': float(candle.volume)
            })
        
        df = pd.DataFrame(data)
        df.set_index('timestamp', inplace=True)
        return df
    
    def get_latest_price(self, symbol: str) -> float:
        """Get latest price for a symbol"""
        ticker = self.exchange.fetch_ticker(symbol)
        return ticker['last']
    
    def get_account_balance(self) -> Dict[str, float]:
        """Get account balance"""
        balance = self.exchange.fetch_balance()
        return {
            'total': balance['total'],
            'free': balance['free'],
            'used': balance['used']
        }
    
    def fetch_live_data(self, symbol: str, timeframe: str = '1m', limit: int = 1) -> pd.DataFrame:
        """Fetch live data via REST API"""
        candles = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        
        if not candles:
            return pd.DataFrame()
        
        df = pd.DataFrame(candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        return df
    
    def _format_symbol_for_api(self, symbol: str) -> str:
        """Format symbol for API (BTCUSDT -> BTC/USDT)"""
        if '/' in symbol:
            return symbol
        # Assume USDT pairs for now
        if symbol.endswith('USDT'):
            base = symbol[:-4]
            return f"{base}/USDT"
        return symbol
    
    def _format_symbol_for_db(self, symbol: str) -> str:
        """Format symbol for database (BTC/USDT -> BTCUSDT)"""
        return symbol.replace('/', '')
    
    def _validate_ohlcv_data(self, df: pd.DataFrame) -> bool:
        """Validate OHLCV data"""
        required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        
        # Check required columns
        if not all(col in df.columns for col in required_columns):
            return False
        
        # Check for negative prices
        price_columns = ['open', 'high', 'low', 'close']
        for col in price_columns:
            if (df[col] < 0).any():
                return False
        
        # Check for negative volume
        if (df['volume'] < 0).any():
            return False
        
        return True