import os
from datetime import datetime
from decimal import Decimal
from typing import Optional
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Numeric, JSON, ForeignKey, TIMESTAMP, Index
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from sqlalchemy.dialects.postgresql import TSRANGE
import json

Base = declarative_base()

class Strategy(Base):
    __tablename__ = 'strategies'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String, unique=True, nullable=False)
    version = Column(String, nullable=False)
    parameters = Column(JSON)
    file = Column(String, nullable=False)
    created_at = Column(TIMESTAMP(timezone=True), default=datetime.utcnow)
    
    backtests = relationship("Backtest", back_populates="strategy")
    live_orders = relationship("LiveOrder", back_populates="strategy")

class Backtest(Base):
    __tablename__ = 'backtests'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    strategy_id = Column(Integer, ForeignKey('strategies.id'))
    symbol = Column(String)
    timeframe = Column(String)
    period_start = Column(DateTime)
    period_end = Column(DateTime)
    initial_balance = Column(Numeric)
    metrics = Column(JSON)  # {"sharpe": 1.5, "mdd": 0.07, ...}
    created_at = Column(TIMESTAMP(timezone=True), default=datetime.utcnow)
    
    strategy = relationship("Strategy", back_populates="backtests")

class LiveOrder(Base):
    __tablename__ = 'live_orders'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    strategy_id = Column(Integer, ForeignKey('strategies.id'))
    bybit_order_id = Column(String)
    side = Column(String)
    qty = Column(Numeric)
    price = Column(Numeric)
    status = Column(String)
    filled_qty = Column(Numeric)
    created_at = Column(TIMESTAMP(timezone=True), default=datetime.utcnow)
    
    strategy = relationship("Strategy", back_populates="live_orders")

class Candle(Base):
    __tablename__ = 'candles'
    
    symbol = Column(String, primary_key=True)
    timeframe = Column(String, primary_key=True)
    time = Column(TIMESTAMP(timezone=True), primary_key=True)
    open = Column(Numeric)
    high = Column(Numeric)
    low = Column(Numeric)
    close = Column(Numeric)
    volume = Column(Numeric)
    
    __table_args__ = (
        Index('idx_candles_symbol_timeframe_time', 'symbol', 'timeframe', 'time'),
    )

class Database:
    def __init__(self, db_url: Optional[str] = None):
        self.db_url = db_url or os.environ.get("DATABASE_URL", os.environ.get("DB_URL", "sqlite:///trading_bot.db"))
        self.engine = create_engine(self.db_url)
        self.Session = sessionmaker(bind=self.engine)
    
    def create_tables(self):
        Base.metadata.create_all(self.engine)
    
    def get_session(self):
        return self.Session()
    
    def save_candles(self, candles_data: list, symbol: str, timeframe: str):
        session = self.get_session()
        try:
            for candle in candles_data:
                db_candle = Candle(
                    symbol=symbol,
                    timeframe=timeframe,
                    time=candle['timestamp'],
                    open=Decimal(str(candle['open'])),
                    high=Decimal(str(candle['high'])),
                    low=Decimal(str(candle['low'])),
                    close=Decimal(str(candle['close'])),
                    volume=Decimal(str(candle['volume']))
                )
                session.merge(db_candle)
            session.commit()
        except Exception as e:
            session.rollback()
            raise e
        finally:
            session.close()
    
    def get_candles(self, symbol: str, timeframe: str, start_time: datetime, end_time: datetime):
        session = self.get_session()
        try:
            candles = session.query(Candle).filter(
                Candle.symbol == symbol,
                Candle.timeframe == timeframe,
                Candle.time >= start_time,
                Candle.time <= end_time
            ).order_by(Candle.time).all()
            return candles
        finally:
            session.close()
    
    def save_strategy(self, name: str, version: str, parameters: dict, file: str):
        session = self.get_session()
        try:
            strategy = Strategy(
                name=name,
                version=version,
                parameters=parameters,
                file=file
            )
            session.add(strategy)
            session.commit()
            return strategy.id
        except Exception as e:
            session.rollback()
            raise e
        finally:
            session.close()
    
    def save_backtest_result(self, strategy_id: int, symbol: str, timeframe: str, 
                           period_start: datetime, period_end: datetime, 
                           initial_balance: float, metrics: dict):
        session = self.get_session()
        try:
            backtest = Backtest(
                strategy_id=strategy_id,
                symbol=symbol,
                timeframe=timeframe,
                period_start=period_start,
                period_end=period_end,
                initial_balance=Decimal(str(initial_balance)),
                metrics=metrics
            )
            session.add(backtest)
            session.commit()
            return backtest.id
        except Exception as e:
            session.rollback()
            raise e
        finally:
            session.close()
    
    def save_live_order(self, strategy_id: int, bybit_order_id: str, side: str, 
                       qty: float, price: float, status: str, filled_qty: float = 0):
        session = self.get_session()
        try:
            order = LiveOrder(
                strategy_id=strategy_id,
                bybit_order_id=bybit_order_id,
                side=side,
                qty=Decimal(str(qty)),
                price=Decimal(str(price)),
                status=status,
                filled_qty=Decimal(str(filled_qty))
            )
            session.add(order)
            session.commit()
            return order.id
        except Exception as e:
            session.rollback()
            raise e
        finally:
            session.close()