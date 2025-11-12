"""
Moving Average Crossover Strategy - Original Implementation

This is the exact implementation from mvg_avg_co_bt_stgy.py
embedded as a strategy class for the strategy builder framework.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional


class MovingAverageCrossoverOriginal:
    """
    Original Moving Average Crossover Strategy
    
    This strategy matches the exact logic from mvg_avg_co_bt_stgy.py:
    - SMA50, EMA100, SMA200 for trend identification
    - MACD filter for momentum confirmation
    - Volume filter for entry validation
    - Swing-based stop loss for risk management
    """
    
    def __init__(self, params: Optional[Dict[str, Any]] = None):
        """
        Initialize strategy with parameters
        
        Args:
            params: Dictionary with strategy parameters:
                - short_sma_period: int (default: 50)
                - mid_ema_period: int (default: 100)
                - long_sma_period: int (default: 200)
                - swing_lookback: int (default: 5)
                - volume_filter: bool (default: True)
                - macd_filter: bool (default: True)
        """
        self.name = "Moving Average Crossover (Original)"
        self.params = params or {}
        
        # Set defaults
        self.short_sma_period = self.params.get('short_sma_period', 50)
        self.mid_ema_period = self.params.get('mid_ema_period', 100)
        self.long_sma_period = self.params.get('long_sma_period', 200)
        self.swing_lookback = self.params.get('swing_lookback', 5)
        self.volume_filter = self.params.get('volume_filter', True)
        self.macd_filter = self.params.get('macd_filter', True)
    
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate technical indicators (exact logic from original script)
        
        Args:
            df: DataFrame with columns: open, high, low, close, volume
        
        Returns:
            DataFrame with added indicator columns
        """
        df = df.copy()
        
        # Normalize column names - data service returns lowercase, original uses capitalized
        # Create capitalized versions for compatibility with original logic
        if 'close' in df.columns:
            df['Close'] = df['close']
        elif 'Close' not in df.columns:
            raise ValueError("DataFrame must have 'close' or 'Close' column")
        
        if 'high' in df.columns:
            df['High'] = df['high']
        elif 'High' not in df.columns:
            raise ValueError("DataFrame must have 'high' or 'High' column")
        
        if 'low' in df.columns:
            df['Low'] = df['low']
        elif 'Low' not in df.columns:
            raise ValueError("DataFrame must have 'low' or 'Low' column")
        
        if 'volume' in df.columns:
            df['Volume'] = df['volume']
        elif 'Volume' not in df.columns:
            df['Volume'] = 0  # Default to 0 if volume not available
        
        # Calculate indicators (exact from original)
        df['SMA50'] = df['Close'].rolling(self.short_sma_period).mean()
        df['EMA100'] = df['Close'].ewm(span=self.mid_ema_period, adjust=False).mean()
        df['SMA200'] = df['Close'].rolling(self.long_sma_period).mean()
        
        if self.macd_filter:
            ema12 = df['Close'].ewm(span=12, adjust=False).mean()
            ema26 = df['Close'].ewm(span=26, adjust=False).mean()
            df['MACD'] = ema12 - ema26
            df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        
        return df
    
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals (exact logic from original script)
        
        Args:
            df: DataFrame with indicators calculated
        
        Returns:
            DataFrame with 'Signal' column (1=Buy, -1=Sell, 0=Hold)
        """
        df = df.copy()
        df['Signal'] = 0
        
        # Exact logic from original script
        for i in range(self.long_sma_period, len(df)):
            buy_cond = df['SMA50'].iloc[i] > df['SMA200'].iloc[i] and df['SMA50'].iloc[i-1] <= df['SMA200'].iloc[i-1]
            sell_cond = df['SMA50'].iloc[i] < df['SMA200'].iloc[i] and df['SMA50'].iloc[i-1] >= df['SMA200'].iloc[i-1]
            
            if df['Close'].iloc[i] <= df['EMA100'].iloc[i]:
                buy_cond = False
            if df['Close'].iloc[i] >= df['EMA100'].iloc[i]:
                sell_cond = False
            
            if self.volume_filter and df['Volume'].iloc[i] < df['Volume'].iloc[i-1]:
                buy_cond = False
                sell_cond = False
            
            if self.macd_filter:
                if df['MACD'].iloc[i] < df['MACD_signal'].iloc[i]:
                    buy_cond = False
                if df['MACD'].iloc[i] > df['MACD_signal'].iloc[i]:
                    sell_cond = False
            
            if buy_cond:
                df.loc[df.index[i], 'Signal'] = 1
            elif sell_cond:
                df.loc[df.index[i], 'Signal'] = -1
        
        return df
    
    def run_backtest(self, df: pd.DataFrame, initial_capital: float = 100000.0) -> Dict[str, Any]:
        """
        Run backtest (exact logic from original script)
        
        Args:
            df: DataFrame with signals
            initial_capital: Starting capital
        
        Returns:
            Dictionary with backtest results
        """
        position = 0
        entry_price = 0
        stop_loss = 0
        capital = initial_capital
        trades = []
        
        # Exact backtest logic from original script
        for i in range(self.long_sma_period, len(df)):
            swing_low = df['Low'].iloc[i-self.swing_lookback:i].min()
            swing_high = df['High'].iloc[i-self.swing_lookback:i].max()
            
            if position == 0:
                if df['Signal'].iloc[i] == 1:
                    position = 1
                    entry_price = df['Close'].iloc[i]
                    stop_loss = swing_low
                elif df['Signal'].iloc[i] == -1:
                    position = -1
                    entry_price = df['Close'].iloc[i]
                    stop_loss = swing_high
            elif position == 1:
                stop_loss = max(stop_loss, swing_low)
                if df['Close'].iloc[i] < stop_loss or df['Signal'].iloc[i] == -1:
                    trade_pnl = df['Close'].iloc[i] - entry_price
                    capital += trade_pnl
                    trades.append({
                        'Type': 'Long',
                        'Entry': entry_price,
                        'Exit': df['Close'].iloc[i],
                        'PnL': trade_pnl
                    })
                    position = 0
            elif position == -1:
                stop_loss = min(stop_loss, swing_high)
                if df['Close'].iloc[i] > stop_loss or df['Signal'].iloc[i] == 1:
                    trade_pnl = entry_price - df['Close'].iloc[i]
                    capital += trade_pnl
                    trades.append({
                        'Type': 'Short',
                        'Entry': entry_price,
                        'Exit': df['Close'].iloc[i],
                        'PnL': trade_pnl
                    })
                    position = 0
        
        # Calculate results (exact from original script)
        total_trades = len(trades)
        winning_trades = [t for t in trades if t['PnL'] > 0]
        losing_trades = [t for t in trades if t['PnL'] <= 0]
        win_rate = len(winning_trades) / total_trades * 100 if total_trades > 0 else 0
        total_profit = sum(t['PnL'] for t in trades)
        
        return {
            'total_trades': total_trades,
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': win_rate,
            'total_pnl': total_profit,
            'final_capital': capital,
            'trades': trades
        }
    
    def prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare data by calculating indicators and generating signals
        
        Args:
            df: Raw price data
        
        Returns:
            DataFrame with indicators and signals
        """
        df = self.calculate_indicators(df)
        df = self.generate_signals(df)
        return df

