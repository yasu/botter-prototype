import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import json
import subprocess
import time
import os
import sys

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.db import Database
from core.data_loader import DataLoader

# Page config
st.set_page_config(
    page_title="Automated Trading System",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize database and data loader
@st.cache_resource
def init_database():
    return Database()

@st.cache_resource
def init_data_loader():
    return DataLoader()

db = init_database()
data_loader = init_data_loader()

# Sidebar navigation
st.sidebar.title("🤖 Auto Trader")
page = st.sidebar.selectbox(
    "Navigate",
    ["Dashboard", "Strategies", "Backtests", "Live Trading", "Data Management", "Settings"]
)

# Dashboard page
if page == "Dashboard":
    st.title("📊 Trading Dashboard")
    
    col1, col2, col3, col4 = st.columns(4)
    
    # Get recent statistics
    session = db.get_session()
    try:
        from core.db import Backtest, Strategy, LiveOrder
        
        # Count statistics
        total_backtests = session.query(Backtest).count()
        total_strategies = session.query(Strategy).count()
        total_orders = session.query(LiveOrder).count()
        
        # Recent performance
        recent_backtest = session.query(Backtest).order_by(Backtest.created_at.desc()).first()
        recent_return = 0
        if recent_backtest and recent_backtest.metrics:
            recent_return = recent_backtest.metrics.get('total_return', 0)
    
    finally:
        session.close()
    
    with col1:
        st.metric("Total Strategies", total_strategies)
    
    with col2:
        st.metric("Backtests Run", total_backtests)
    
    with col3:
        st.metric("Live Orders", total_orders)
    
    with col4:
        st.metric("Recent Return", f"{recent_return:.2f}%")
    
    # Recent backtests table
    st.subheader("Recent Backtests")
    session = db.get_session()
    try:
        results = session.query(Backtest, Strategy).join(Strategy).order_by(
            Backtest.created_at.desc()
        ).limit(10).all()
        
        if results:
            data = []
            for backtest, strategy in results:
                metrics = backtest.metrics or {}
                data.append({
                    "ID": backtest.id,
                    "Strategy": strategy.name,
                    "Symbol": backtest.symbol,
                    "Return": f"{metrics.get('total_return', 0):.2f}%",
                    "Sharpe": f"{metrics.get('sharpe_ratio', 0):.2f}",
                    "Max DD": f"{metrics.get('max_drawdown', 0):.2f}%",
                    "Date": backtest.created_at.strftime("%Y-%m-%d %H:%M")
                })
            
            df = pd.DataFrame(data)
            st.dataframe(df, use_container_width=True)
        else:
            st.info("No backtests found. Run your first backtest!")
    
    finally:
        session.close()

# Strategies page
elif page == "Strategies":
    st.title("⚙️ Trading Strategies")
    
    tab1, tab2 = st.tabs(["Available Strategies", "Create New"])
    
    with tab1:
        st.subheader("Strategy Library")
        
        # List available strategies
        strategies_dir = "strategies"
        if os.path.exists(strategies_dir):
            strategy_files = [f for f in os.listdir(strategies_dir) if f.endswith('.py') and f != '__init__.py']
            
            for strategy_file in strategy_files:
                with st.expander(f"📄 {strategy_file}"):
                    file_path = os.path.join(strategies_dir, strategy_file)
                    
                    # Try to load and describe strategy
                    try:
                        import importlib.util
                        spec = importlib.util.spec_from_file_location("strategy", file_path)
                        module = importlib.util.module_from_spec(spec)
                        spec.loader.exec_module(module)
                        
                        # Find strategy classes
                        strategy_classes = []
                        for attr_name in dir(module):
                            attr = getattr(module, attr_name)
                            if (isinstance(attr, type) and 
                                hasattr(attr, 'generate_signals') and 
                                attr_name != 'object'):
                                strategy_classes.append(attr_name)
                        
                        st.write(f"**Classes found:** {', '.join(strategy_classes)}")
                        
                        # Show strategy description if available
                        if strategy_classes:
                            strategy_class = getattr(module, strategy_classes[0])
                            strategy_instance = strategy_class()
                            if hasattr(strategy_instance, 'describe'):
                                st.code(strategy_instance.describe())
                        
                        # Quick backtest button
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            symbol = st.text_input(f"Symbol for {strategy_file}", "BTCUSDT", key=f"symbol_{strategy_file}")
                        with col2:
                            start_date = st.date_input(f"Start Date", datetime.now() - timedelta(days=30), key=f"start_{strategy_file}")
                        with col3:
                            end_date = st.date_input(f"End Date", datetime.now(), key=f"end_{strategy_file}")
                        
                        if st.button(f"Run Backtest", key=f"bt_{strategy_file}"):
                            with st.spinner("Running backtest..."):
                                cmd = [
                                    "python", "main.py", "backtest",
                                    file_path, symbol,
                                    start_date.strftime("%Y-%m-%d"),
                                    end_date.strftime("%Y-%m-%d")
                                ]
                                
                                try:
                                    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
                                    if result.returncode == 0:
                                        st.success("Backtest completed!")
                                        st.text(result.stdout)
                                    else:
                                        st.error(f"Backtest failed: {result.stderr}")
                                except subprocess.TimeoutExpired:
                                    st.error("Backtest timed out")
                                except Exception as e:
                                    st.error(f"Error: {e}")
                    
                    except Exception as e:
                        st.error(f"Error loading strategy: {e}")
        else:
            st.warning("Strategies directory not found")
    
    with tab2:
        st.subheader("Create New Strategy")
        st.info("Use the CLI or create a new Python file in the strategies/ directory")
        
        strategy_template = '''
import pandas as pd
import numpy as np
from typing import Dict, Any

class MyStrategy:
    def __init__(self, parameters: Dict[str, Any] = None):
        self.params = {
            'param1': 20,
            'param2': 0.02
        }
        if parameters:
            self.params.update(parameters)
        
        self.name = "My Strategy"
        self.version = "1.0"
    
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Generate trading signals"""
        signals = pd.Series(0, index=data.index)
        
        # Your strategy logic here
        # Return 1 for buy, -1 for sell, 0 for hold
        
        return signals
    
    def get_parameters(self) -> Dict[str, Any]:
        return self.params.copy()
    
    def set_parameters(self, parameters: Dict[str, Any]):
        self.params.update(parameters)
'''
        
        st.code(strategy_template, language='python')

# Backtests page
elif page == "Backtests":
    st.title("📈 Backtest Results")
    
    # Filters
    col1, col2, col3 = st.columns(3)
    with col1:
        symbol_filter = st.selectbox("Filter by Symbol", ["All"] + ["BTCUSDT", "ETHUSDT", "ADAUSDT"])
    with col2:
        limit = st.selectbox("Show Last", [10, 25, 50, 100])
    with col3:
        sort_by = st.selectbox("Sort by", ["Date", "Return", "Sharpe Ratio"])
    
    # Get backtest data
    session = db.get_session()
    try:
        from core.db import Backtest, Strategy
        
        query = session.query(Backtest, Strategy).join(Strategy)
        
        if symbol_filter != "All":
            query = query.filter(Backtest.symbol == symbol_filter)
        
        if sort_by == "Date":
            query = query.order_by(Backtest.created_at.desc())
        elif sort_by == "Return":
            query = query.order_by(Backtest.metrics['total_return'].desc())
        elif sort_by == "Sharpe Ratio":
            query = query.order_by(Backtest.metrics['sharpe_ratio'].desc())
        
        results = query.limit(limit).all()
        
        if results:
            # Create detailed table
            data = []
            for backtest, strategy in results:
                metrics = backtest.metrics or {}
                data.append({
                    "ID": backtest.id,
                    "Strategy": strategy.name,
                    "Symbol": backtest.symbol,
                    "Timeframe": backtest.timeframe,
                    "Start": backtest.period_start.strftime("%Y-%m-%d") if backtest.period_start else "N/A",
                    "End": backtest.period_end.strftime("%Y-%m-%d") if backtest.period_end else "N/A",
                    "Initial": f"${backtest.initial_balance:,.0f}",
                    "Return %": metrics.get('total_return', 0),
                    "Sharpe": metrics.get('sharpe_ratio', 0),
                    "Max DD %": metrics.get('max_drawdown', 0),
                    "Win Rate %": metrics.get('win_rate', 0),
                    "Trades": metrics.get('total_trades', 0),
                    "Profit Factor": metrics.get('profit_factor', 0),
                    "Created": backtest.created_at.strftime("%Y-%m-%d %H:%M")
                })
            
            df = pd.DataFrame(data)
            
            # Display metrics
            if not df.empty:
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    avg_return = df['Return %'].mean()
                    st.metric("Avg Return", f"{avg_return:.2f}%")
                with col2:
                    avg_sharpe = df['Sharpe'].mean()
                    st.metric("Avg Sharpe", f"{avg_sharpe:.2f}")
                with col3:
                    max_dd = df['Max DD %'].max()
                    st.metric("Max Drawdown", f"{max_dd:.2f}%")
                with col4:
                    total_trades = df['Trades'].sum()
                    st.metric("Total Trades", f"{total_trades:,.0f}")
            
            # Interactive table
            selected_row = st.dataframe(
                df,
                use_container_width=True,
                selection_mode="single-row",
                on_select="rerun"
            )
            
            # Show details for selected backtest
            if hasattr(selected_row, 'selection') and selected_row.selection.rows:
                selected_idx = selected_row.selection.rows[0]
                selected_id = df.iloc[selected_idx]['ID']
                
                st.subheader(f"Backtest Details - ID: {selected_id}")
                
                # Get detailed backtest info
                backtest_detail = session.query(Backtest, Strategy).join(Strategy).filter(
                    Backtest.id == selected_id
                ).first()
                
                if backtest_detail:
                    backtest, strategy = backtest_detail
                    
                    # Strategy parameters
                    st.write("**Strategy Parameters:**")
                    if strategy.parameters:
                        st.json(strategy.parameters)
                    
                    # Performance chart (placeholder - would need equity curve data)
                    st.write("**Performance Chart:**")
                    st.info("Equity curve visualization would be implemented here with stored backtest data")
        
        else:
            st.info("No backtests found matching the criteria")
    
    finally:
        session.close()

# Live Trading page  
elif page == "Live Trading":
    st.title("🚀 Live Trading")
    
    st.warning("⚠️ Live trading involves real money. Use at your own risk!")
    
    tab1, tab2 = st.tabs(["Start Trading", "Active Positions"])
    
    with tab1:
        st.subheader("Configure Live Trading")
        
        col1, col2 = st.columns(2)
        
        with col1:
            strategy_file = st.selectbox(
                "Select Strategy",
                ["strategies/sample_meanrev.py"]  # Would list available strategies
            )
            
            symbol = st.selectbox(
                "Trading Pair",
                ["BTC/USDT", "ETH/USDT", "ADA/USDT"]
            )
            
            quantity = st.number_input("Quantity", min_value=0.001, value=0.01, step=0.001)
            
        with col2:
            timeframe = st.selectbox("Timeframe", ["1m", "5m", "15m", "1h"])
            
            dry_run = st.checkbox("Dry Run (Simulation)", value=True)
            
            # Risk management settings
            st.subheader("Risk Management")
            stop_loss = st.number_input("Stop Loss %", min_value=0.1, max_value=10.0, value=2.0)
            take_profit = st.number_input("Take Profit %", min_value=0.1, max_value=20.0, value=4.0)
        
        if st.button("Start Live Trading", type="primary"):
            if dry_run:
                st.info("Starting dry run mode...")
            else:
                st.error("Live trading not implemented in UI. Use CLI: `python main.py live`")
    
    with tab2:
        st.subheader("Live Orders")
        
        # Get live orders from database
        session = db.get_session()
        try:
            from core.db import LiveOrder
            
            orders = session.query(LiveOrder).order_by(LiveOrder.created_at.desc()).limit(50).all()
            
            if orders:
                order_data = []
                for order in orders:
                    order_data.append({
                        "ID": order.id,
                        "Bybit ID": order.bybit_order_id,
                        "Side": order.side,
                        "Quantity": float(order.qty),
                        "Price": float(order.price),
                        "Status": order.status,
                        "Filled": float(order.filled_qty),
                        "Created": order.created_at.strftime("%Y-%m-%d %H:%M:%S")
                    })
                
                df = pd.DataFrame(order_data)
                st.dataframe(df, use_container_width=True)
            else:
                st.info("No live orders found")
        
        finally:
            session.close()

# Data Management page
elif page == "Data Management":
    st.title("💾 Data Management")
    
    tab1, tab2 = st.tabs(["Download Data", "View Data"])
    
    with tab1:
        st.subheader("Download Historical Data")
        
        col1, col2 = st.columns(2)
        
        with col1:
            symbol = st.text_input("Symbol", "BTCUSDT")
            timeframe = st.selectbox("Timeframe", ["1m", "5m", "15m", "30m", "1h", "4h", "1d"])
        
        with col2:
            start_date = st.date_input("Start Date", datetime.now() - timedelta(days=7))
            end_date = st.date_input("End Date", datetime.now())
        
        if st.button("Download Data"):
            if start_date >= end_date:
                st.error("Start date must be before end date")
            else:
                with st.spinner("Downloading data..."):
                    cmd = [
                        "python", "main.py", "download",
                        symbol,
                        start_date.strftime("%Y-%m-%d"),
                        end_date.strftime("%Y-%m-%d"),
                        timeframe
                    ]
                    
                    try:
                        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
                        if result.returncode == 0:
                            st.success("Data downloaded successfully!")
                            st.text(result.stdout)
                        else:
                            st.error(f"Download failed: {result.stderr}")
                    except subprocess.TimeoutExpired:
                        st.error("Download timed out")
                    except Exception as e:
                        st.error(f"Error: {e}")
    
    with tab2:
        st.subheader("View Historical Data")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            view_symbol = st.text_input("Symbol to View", "BTCUSDT")
        with col2:
            view_timeframe = st.selectbox("Timeframe to View", ["1m", "5m", "15m", "30m", "1h"])
        with col3:
            days_back = st.number_input("Days Back", min_value=1, max_value=365, value=7)
        
        if st.button("Load Data"):
            try:
                end_date = datetime.now()
                start_date = end_date - timedelta(days=days_back)
                
                df = data_loader.get_historical_data_from_db(
                    view_symbol, 
                    view_timeframe,
                    start_date.strftime("%Y-%m-%d"),
                    end_date.strftime("%Y-%m-%d")
                )
                
                if not df.empty:
                    st.success(f"Loaded {len(df)} candles")
                    
                    # Chart
                    fig = go.Figure(data=go.Candlestick(
                        x=df.index,
                        open=df['open'],
                        high=df['high'],
                        low=df['low'],
                        close=df['close']
                    ))
                    fig.update_layout(title=f"{view_symbol} {view_timeframe} Chart")
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Data table
                    st.subheader("Recent Data")
                    st.dataframe(df.tail(20), use_container_width=True)
                else:
                    st.warning("No data found. Try downloading data first.")
            
            except Exception as e:
                st.error(f"Error loading data: {e}")

# Settings page
elif page == "Settings":
    st.title("⚙️ Settings")
    
    tab1, tab2, tab3 = st.tabs(["API Configuration", "Database", "System"])
    
    with tab1:
        st.subheader("Bybit API Configuration")
        
        # Check current API status
        try:
            balance = data_loader.get_account_balance()
            st.success("✅ API Connected")
            st.write(f"USDT Balance: {balance.get('total', {}).get('USDT', 0):.2f}")
        except Exception as e:
            st.error(f"❌ API Error: {e}")
        
        st.info("API keys are configured via environment variables (BYBIT_API_KEY, BYBIT_API_SECRET)")
    
    with tab2:
        st.subheader("Database Status")
        
        try:
            session = db.get_session()
            session.execute("SELECT 1")
            session.close()
            st.success("✅ Database Connected")
            
            # Database statistics
            session = db.get_session()
            try:
                from core.db import Backtest, Strategy, LiveOrder, Candle
                
                backtest_count = session.query(Backtest).count()
                strategy_count = session.query(Strategy).count()
                order_count = session.query(LiveOrder).count()
                candle_count = session.query(Candle).count()
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Backtests", backtest_count)
                with col2:
                    st.metric("Strategies", strategy_count)
                with col3:
                    st.metric("Orders", order_count)
                with col4:
                    st.metric("Candles", f"{candle_count:,}")
            
            finally:
                session.close()
        
        except Exception as e:
            st.error(f"❌ Database Error: {e}")
            
            if st.button("Initialize Database"):
                try:
                    db.create_tables()
                    st.success("Database initialized!")
                    st.experimental_rerun()
                except Exception as e:
                    st.error(f"Initialization failed: {e}")
    
    with tab3:
        st.subheader("System Information")
        
        st.write("**Environment Variables:**")
        env_vars = ["DB_URL", "BYBIT_API_KEY", "BYBIT_API_SECRET"]
        for var in env_vars:
            value = os.environ.get(var, "Not set")
            if "KEY" in var or "SECRET" in var:
                value = "***" if value != "Not set" else "Not set"
            st.write(f"- {var}: {value}")

# Auto-refresh for live data
if page in ["Dashboard", "Live Trading"]:
    time.sleep(1)
    st.experimental_rerun()