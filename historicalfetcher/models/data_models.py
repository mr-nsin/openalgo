"""
Data models for OpenAlgo Historical Data Fetcher

This module contains shared data models to avoid circular imports.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Dict, List
from enum import IntEnum


@dataclass
class HistoricalCandle:
    """Historical OHLCV data point"""
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int
    oi: int = 0  # Open Interest (0 for equity)


@dataclass
class SymbolInfo:
    """Symbol information container"""
    symbol: str
    exchange: str
    instrument_type: str
    token: str
    name: Optional[str] = None
    expiry: Optional[str] = None
    strike: Optional[float] = None
    lotsize: Optional[int] = None
    tick_size: Optional[float] = None


class TimeFrameCode(IntEnum):
    """Numeric encoding for timeframes for fast filtering"""
    MINUTE_1 = 1
    MINUTE_3 = 3
    MINUTE_5 = 5
    MINUTE_15 = 15
    MINUTE_30 = 30
    HOUR_1 = 60
    DAILY = 1440
    
    @classmethod
    def from_timeframe(cls, timeframe):
        """Convert TimeFrame enum to TimeFrameCode (for backward compatibility)"""
        # Import here to avoid circular imports
        from historicalfetcher.config.openalgo_settings import TimeFrame
        
        mapping = {
            TimeFrame.MINUTE_1: cls.MINUTE_1,
            TimeFrame.MINUTE_3: cls.MINUTE_3,
            TimeFrame.MINUTE_5: cls.MINUTE_5,
            TimeFrame.MINUTE_15: cls.MINUTE_15,
            TimeFrame.MINUTE_30: cls.MINUTE_30,
            TimeFrame.HOUR_1: cls.HOUR_1,
            TimeFrame.DAILY: cls.DAILY,
        }
        
        # Handle string values as well
        if isinstance(timeframe, str):
            string_mapping = {
                '1m': cls.MINUTE_1,
                '3m': cls.MINUTE_3,
                '5m': cls.MINUTE_5,
                '15m': cls.MINUTE_15,
                '30m': cls.MINUTE_30,
                '1h': cls.HOUR_1,
                'D': cls.DAILY,
            }
            return string_mapping.get(timeframe, cls.MINUTE_1)
        
        return mapping.get(timeframe, cls.MINUTE_1)
    
    @classmethod
    def to_string(cls, timeframe):
        """Convert TimeFrame enum to string for QuestDB SYMBOL storage"""
        # Import here to avoid circular imports
        from historicalfetcher.config.openalgo_settings import TimeFrame
        
        # Handle TimeFrame enum
        if hasattr(timeframe, 'value'):
            return timeframe.value
        
        # Handle string values (pass through)
        if isinstance(timeframe, str):
            return timeframe
            
        # Handle numeric codes (convert to string)
        numeric_mapping = {
            cls.MINUTE_1: '1m',
            cls.MINUTE_3: '3m',
            cls.MINUTE_5: '5m',
            cls.MINUTE_15: '15m',
            cls.MINUTE_30: '30m',
            cls.HOUR_1: '1h',
            cls.DAILY: 'D',
        }
        return numeric_mapping.get(timeframe, '1m')


class OptionTypeCode(IntEnum):
    """Numeric encoding for option types"""
    CALL = 1  # CE
    PUT = 2   # PE


@dataclass
class IndicatorResult:
    """Container for calculated indicators"""
    symbol: str
    timeframe: str  # Changed to string for SYMBOL storage ('1m', '5m', 'D', etc.)
    timestamp: datetime
    
    # Basic OHLCV
    open: float
    high: float
    low: float
    close: float
    volume: int
    
    # Technical Indicators
    indicators: Dict[str, float]
    
    # Options Greeks (if applicable)
    greeks: Optional[Dict[str, float]] = None
    
    # Market Depth (if available)
    market_depth: Optional[Dict[str, float]] = None
    
    # Derived Metrics
    derived_metrics: Optional[Dict[str, float]] = None


@dataclass
class CalculationConfig:
    """Configuration for indicator calculations"""
    
    # Grouped Technical Indicators (matching indicator engine expectations)
    calculate_trend_indicators: bool = True      # EMA, SMA
    calculate_momentum_indicators: bool = True   # RSI, MACD, Stochastic
    calculate_volatility_indicators: bool = True # ATR, Bollinger Bands
    calculate_volume_indicators: bool = True     # VWAP, OBV
    calculate_greeks: bool = False              # Options Greeks
    calculate_iv: bool = False                  # Implied Volatility
    calculate_advanced_greeks: bool = False     # Advanced Greeks
    
    # Individual Technical Indicators (for backward compatibility)
    enable_sma: bool = True
    enable_ema: bool = True
    enable_rsi: bool = True
    enable_macd: bool = True
    enable_bollinger: bool = True
    enable_stochastic: bool = True
    enable_atr: bool = True
    enable_adx: bool = True
    enable_williams_r: bool = True
    enable_cci: bool = True
    
    # SMA periods
    sma_periods: List[int] = None
    
    # EMA periods
    ema_periods: List[int] = None
    
    # RSI configuration
    rsi_period: int = 14
    
    # MACD configuration
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    
    # Bollinger Bands
    bb_period: int = 20
    bb_std_dev: float = 2.0
    
    # Stochastic
    stoch_k_period: int = 14
    stoch_d_period: int = 3
    
    # ATR
    atr_period: int = 14
    
    # ADX
    adx_period: int = 14
    
    # Williams %R
    williams_r_period: int = 14
    
    # CCI
    cci_period: int = 20
    
    # Options Greeks (if applicable)
    enable_greeks: bool = False
    risk_free_rate: float = 0.05
    
    # Performance optimization
    enable_parallel_processing: bool = True
    max_workers: int = 4
    
    # Caching configuration
    enable_caching: bool = True
    
    def __post_init__(self):
        """Set default values for list fields"""
        if self.sma_periods is None:
            self.sma_periods = [5, 10, 20, 50, 200]
        
        if self.ema_periods is None:
            self.ema_periods = [9, 12, 21, 26, 50]
