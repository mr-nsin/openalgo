"""
Moving Average Crossover Strategy

A trend-following strategy that uses:
- SMA50, EMA100, SMA200 for trend identification
- MACD filter for momentum confirmation
- Volume filter for entry validation
- Swing-based stop loss for risk management
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class MovingAverageCrossoverParams:
    """Parameters for Moving Average Crossover Strategy"""
    short_sma_period: int = 50
    mid_ema_period: int = 100
    long_sma_period: int = 200
    swing_lookback: int = 5
    volume_filter: bool = True
    macd_filter: bool = True
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9


class MovingAverageCrossoverStrategy:
    """
    Moving Average Crossover Strategy
    
    Entry Rules:
    - Long: SMA50 crosses above SMA200, price above EMA100, volume increasing, MACD bullish
    - Short: SMA50 crosses below SMA200, price below EMA100, volume increasing, MACD bearish
    
    Exit Rules:
    - Stop loss based on swing high/low
    - Opposite signal
    """
    
    def __init__(self, params: Optional[MovingAverageCrossoverParams] = None):
        self.params = params or MovingAverageCrossoverParams()
        self.name = "Moving Average Crossover"
    
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate technical indicators
        
        Args:
            df: DataFrame with columns: open, high, low, close, volume
        
        Returns:
            DataFrame with added indicator columns
        """
        df = df.copy()
        
        # Ensure we have the required columns
        if 'close' not in df.columns:
            raise ValueError("DataFrame must have 'close' column")
        if 'volume' not in df.columns:
            df['volume'] = 0
        
        # Moving Averages
        df['SMA50'] = df['close'].rolling(self.params.short_sma_period).mean()
        df['EMA100'] = df['close'].ewm(span=self.params.mid_ema_period, adjust=False).mean()
        df['SMA200'] = df['close'].rolling(self.params.long_sma_period).mean()
        
        # MACD (if filter is enabled)
        if self.params.macd_filter:
            ema12 = df['close'].ewm(span=self.params.macd_fast, adjust=False).mean()
            ema26 = df['close'].ewm(span=self.params.macd_slow, adjust=False).mean()
            df['MACD'] = ema12 - ema26
            df['MACD_signal'] = df['MACD'].ewm(span=self.params.macd_signal, adjust=False).mean()
        
        return df
    
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals
        
        Args:
            df: DataFrame with indicators calculated
        
        Returns:
            DataFrame with 'Signal' column (1=Buy, -1=Sell, 0=Hold)
        """
        df = df.copy()
        df['Signal'] = 0
        
        # Need at least long_sma_period rows for signals
        min_period = self.params.long_sma_period
        
        for i in range(min_period, len(df)):
            # Check for crossover
            buy_cond = (
                df['SMA50'].iloc[i] > df['SMA200'].iloc[i] and
                df['SMA50'].iloc[i-1] <= df['SMA200'].iloc[i-1]
            )
            sell_cond = (
                df['SMA50'].iloc[i] < df['SMA200'].iloc[i] and
                df['SMA50'].iloc[i-1] >= df['SMA200'].iloc[i-1]
            )
            
            # EMA100 filter
            if buy_cond and df['close'].iloc[i] <= df['EMA100'].iloc[i]:
                buy_cond = False
            if sell_cond and df['close'].iloc[i] >= df['EMA100'].iloc[i]:
                sell_cond = False
            
            # Volume filter
            if self.params.volume_filter:
                if i > 0 and df['volume'].iloc[i] < df['volume'].iloc[i-1]:
                    buy_cond = False
                    sell_cond = False
            
            # MACD filter
            if self.params.macd_filter:
                if buy_cond and df['MACD'].iloc[i] < df['MACD_signal'].iloc[i]:
                    buy_cond = False
                if sell_cond and df['MACD'].iloc[i] > df['MACD_signal'].iloc[i]:
                    sell_cond = False
            
            # Set signal
            if buy_cond:
                df.loc[df.index[i], 'Signal'] = 1
            elif sell_cond:
                df.loc[df.index[i], 'Signal'] = -1
        
        return df
    
    def get_stop_loss(self, df: pd.DataFrame, index: int, position: int) -> float:
        """
        Calculate stop loss based on swing high/low
        
        Args:
            df: DataFrame with price data
            index: Current index
            position: Current position (1=Long, -1=Short)
        
        Returns:
            Stop loss price
        """
        lookback = self.params.swing_lookback
        start_idx = max(0, index - lookback)
        
        if position == 1:  # Long position
            return df['low'].iloc[start_idx:index].min()
        elif position == -1:  # Short position
            return df['high'].iloc[start_idx:index].max()
        else:
            return 0.0
    
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

