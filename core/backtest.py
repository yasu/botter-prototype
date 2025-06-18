import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
from dataclasses import dataclass
import logging
from numba import jit

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class Trade:
    timestamp: datetime
    side: str  # 'buy' or 'sell'
    price: float
    quantity: float
    value: float
    commission: float
    pnl: float = 0.0

@dataclass
class BacktestResult:
    initial_balance: float
    final_balance: float
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    total_trades: int
    profitable_trades: int
    avg_profit: float
    avg_loss: float
    profit_factor: float
    equity_curve: pd.Series
    trades: List[Trade]

class Backtester:
    def __init__(self, initial_balance: float = 10000, commission_rate: float = 0.001):
        self.initial_balance = initial_balance
        self.commission_rate = commission_rate
        self.reset()
    
    def reset(self):
        """Reset backtester state"""
        self.balance = self.initial_balance
        self.position = 0.0
        self.trades: List[Trade] = []
        self.equity_curve = []
        self.entry_price = 0.0
    
    def _calculate_position_size(self, price: float, allocation: float) -> float:
        """Calculate position size based on available balance and allocation"""
        available_cash = self.balance * allocation
        commission = available_cash * self.commission_rate
        return (available_cash - commission) / price
    
    def _execute_trade(self, timestamp: datetime, price: float, signal_strength: float):
        """Execute a trade based on signal"""
        if signal_strength > 0 and self.position == 0:
            # Buy signal
            size = self._calculate_position_size(price, abs(signal_strength))
            if size > 0:
                value = size * price
                commission = value * self.commission_rate
                
                self.position = size
                self.balance -= (value + commission)
                self.entry_price = price
                
                trade = Trade(
                    timestamp=timestamp,
                    side='buy',
                    price=price,
                    quantity=size,
                    value=value,
                    commission=commission
                )
                self.trades.append(trade)
                
        elif signal_strength < 0 and self.position > 0:
            # Sell signal
            value = self.position * price
            commission = value * self.commission_rate
            pnl = (price - self.entry_price) * self.position - commission
            
            self.balance += (value - commission)
            
            trade = Trade(
                timestamp=timestamp,
                side='sell',
                price=price,
                quantity=self.position,
                value=value,
                commission=commission,
                pnl=pnl
            )
            self.trades.append(trade)
            
            self.position = 0.0
            self.entry_price = 0.0
    
    def run(self, strategy: Any, data: pd.DataFrame) -> BacktestResult:
        """
        Run backtest with given strategy and data
        
        Args:
            strategy: Strategy object with generate_signals method
            data: DataFrame with OHLCV data
            
        Returns:
            BacktestResult object
        """
        logger.info(f"Starting backtest with {len(data)} candles")
        self.reset()
        
        # Generate signals
        signals = strategy.generate_signals(data)
        
        # Ensure signals are properly aligned with data
        if len(signals) != len(data):
            raise ValueError("Signals length must match data length")
        
        # Vectorized backtest for performance
        results = self._vectorized_backtest(
            data['close'].values,
            signals,
            data.index
        )
        
        # Calculate metrics
        metrics = self._calculate_metrics(results['equity_curve'])
        
        return BacktestResult(
            initial_balance=self.initial_balance,
            final_balance=results['final_balance'],
            total_return=metrics['total_return'],
            sharpe_ratio=metrics['sharpe_ratio'],
            max_drawdown=metrics['max_drawdown'],
            win_rate=metrics['win_rate'],
            total_trades=metrics['total_trades'],
            profitable_trades=metrics['profitable_trades'],
            avg_profit=metrics['avg_profit'],
            avg_loss=metrics['avg_loss'],
            profit_factor=metrics['profit_factor'],
            equity_curve=results['equity_curve'],
            trades=results['trades']
        )
    
    def _vectorized_backtest(self, prices: np.ndarray, signals: np.ndarray, 
                           timestamps: pd.DatetimeIndex) -> Dict:
        """
        Vectorized backtest implementation for performance
        """
        n = len(prices)
        positions = np.zeros(n)
        cash = np.zeros(n)
        equity = np.zeros(n)
        trades = []
        
        # Initial values
        cash[0] = self.initial_balance
        equity[0] = self.initial_balance
        current_position = 0.0
        entry_price = 0.0
        
        for i in range(1, n):
            cash[i] = cash[i-1]
            positions[i] = current_position
            
            # Check for signal
            if signals[i] != signals[i-1]:
                if signals[i] == 1 and current_position == 0:  # Buy signal
                    # Calculate position size (use 95% of available cash)
                    position_value = cash[i] * 0.95
                    commission = position_value * self.commission_rate
                    position_size = (position_value - commission) / prices[i]
                    
                    if position_size > 0:
                        current_position = position_size
                        cash[i] -= (position_size * prices[i] + commission)
                        entry_price = prices[i]
                        
                        trades.append(Trade(
                            timestamp=timestamps[i],
                            side='buy',
                            price=prices[i],
                            quantity=position_size,
                            value=position_size * prices[i],
                            commission=commission
                        ))
                
                elif signals[i] == -1 and current_position > 0:  # Sell signal
                    # Close position
                    sell_value = current_position * prices[i]
                    commission = sell_value * self.commission_rate
                    pnl = (prices[i] - entry_price) * current_position - commission
                    
                    cash[i] += (sell_value - commission)
                    
                    trades.append(Trade(
                        timestamp=timestamps[i],
                        side='sell',
                        price=prices[i],
                        quantity=current_position,
                        value=sell_value,
                        commission=commission,
                        pnl=pnl
                    ))
                    
                    current_position = 0.0
                    entry_price = 0.0
            
            # Update positions
            positions[i] = current_position
            
            # Calculate equity
            equity[i] = cash[i] + positions[i] * prices[i]
        
        # Create equity curve
        equity_series = pd.Series(equity, index=timestamps)
        
        return {
            'equity_curve': equity_series,
            'final_balance': equity[-1],
            'trades': trades
        }
    
    def _calculate_metrics(self, equity_curve: pd.Series) -> Dict[str, float]:
        """Calculate performance metrics"""
        returns = equity_curve.pct_change().dropna()
        
        # Total return
        total_return = (equity_curve.iloc[-1] / equity_curve.iloc[0] - 1) * 100
        
        # Sharpe ratio (annualized, assuming daily data)
        if len(returns) > 0 and returns.std() > 0:
            sharpe_ratio = np.sqrt(252) * returns.mean() / returns.std()
        else:
            sharpe_ratio = 0.0
        
        # Maximum drawdown
        running_max = equity_curve.cummax()
        drawdown = (equity_curve - running_max) / running_max
        max_drawdown = abs(drawdown.min()) * 100
        
        # Trade statistics
        profitable_trades = [t for t in self.trades if t.pnl > 0]
        losing_trades = [t for t in self.trades if t.pnl < 0]
        
        total_trades = len([t for t in self.trades if t.side == 'sell'])
        profitable_count = len(profitable_trades)
        
        win_rate = (profitable_count / total_trades * 100) if total_trades > 0 else 0
        
        avg_profit = np.mean([t.pnl for t in profitable_trades]) if profitable_trades else 0
        avg_loss = np.mean([t.pnl for t in losing_trades]) if losing_trades else 0
        
        # Profit factor
        gross_profit = sum([t.pnl for t in profitable_trades]) if profitable_trades else 0
        gross_loss = abs(sum([t.pnl for t in losing_trades])) if losing_trades else 1
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
        
        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'total_trades': total_trades,
            'profitable_trades': profitable_count,
            'avg_profit': avg_profit,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor
        }
    
    def generate_report(self, result: BacktestResult) -> str:
        """Generate a text report of backtest results"""
        report = f"""
Backtest Results
================
Initial Balance: ${result.initial_balance:,.2f}
Final Balance: ${result.final_balance:,.2f}
Total Return: {result.total_return:.2f}%
Sharpe Ratio: {result.sharpe_ratio:.2f}
Max Drawdown: {result.max_drawdown:.2f}%

Trade Statistics
================
Total Trades: {result.total_trades}
Profitable Trades: {result.profitable_trades}
Win Rate: {result.win_rate:.2f}%
Average Profit: ${result.avg_profit:,.2f}
Average Loss: ${result.avg_loss:,.2f}
Profit Factor: {result.profit_factor:.2f}
"""
        return report

# Numba-optimized functions for performance
@jit(nopython=True)
def calculate_sma(prices: np.ndarray, period: int) -> np.ndarray:
    """Calculate Simple Moving Average"""
    sma = np.empty(len(prices))
    sma[:period-1] = np.nan
    
    for i in range(period-1, len(prices)):
        sma[i] = np.mean(prices[i-period+1:i+1])
    
    return sma

@jit(nopython=True)
def calculate_ema(prices: np.ndarray, period: int) -> np.ndarray:
    """Calculate Exponential Moving Average"""
    ema = np.empty(len(prices))
    ema[0] = prices[0]
    alpha = 2.0 / (period + 1)
    
    for i in range(1, len(prices)):
        ema[i] = alpha * prices[i] + (1 - alpha) * ema[i-1]
    
    return ema

@jit(nopython=True)
def calculate_rsi(prices: np.ndarray, period: int = 14) -> np.ndarray:
    """Calculate Relative Strength Index"""
    n = len(prices)
    rsi = np.empty(n)
    rsi[:period] = np.nan
    
    # Calculate price changes
    deltas = np.diff(prices)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    
    # Initial averages
    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])
    
    for i in range(period, n-1):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        
        if avg_loss == 0:
            rsi[i+1] = 100
        else:
            rs = avg_gain / avg_loss
            rsi[i+1] = 100 - (100 / (1 + rs))
    
    return rsi