"""
Optimized Data Models for Symbol-Specific Tables

Enhanced models that support the new optimized schema with numeric timeframes
and symbol-specific table structures.
"""

from dataclasses import dataclass
from datetime import datetime, date
from typing import Optional, Dict, Any, List
from enum import IntEnum

class TimeFrameCode(IntEnum):
    """Numeric encoding for timeframes"""
    MINUTE_1 = 1
    MINUTE_3 = 3
    MINUTE_5 = 5
    MINUTE_15 = 15
    MINUTE_30 = 30
    HOUR_1 = 60
    DAILY = 1440

class OptionTypeCode(IntEnum):
    """Numeric encoding for option types"""
    CALL = 1
    PUT = 2

@dataclass
class OptimizedEquityData:
    """Optimized model for equity historical data"""
    tf: int                    # Timeframe code
    open: float
    high: float
    low: float
    close: float
    volume: int
    timestamp: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'tf': self.tf,
            'open': self.open,
            'high': self.high,
            'low': self.low,
            'close': self.close,
            'volume': self.volume,
            'timestamp': self.timestamp
        }

@dataclass
class OptimizedFuturesData:
    """Optimized model for futures historical data"""
    contract_token: str
    expiry_date: Optional[date]
    tf: int                    # Timeframe code
    open: float
    high: float
    low: float
    close: float
    volume: int
    oi: int                    # Open Interest
    timestamp: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'contract_token': self.contract_token,
            'expiry_date': self.expiry_date,
            'tf': self.tf,
            'open': self.open,
            'high': self.high,
            'low': self.low,
            'close': self.close,
            'volume': self.volume,
            'oi': self.oi,
            'timestamp': self.timestamp
        }

@dataclass
class OptimizedOptionsData:
    """Optimized model for options historical data"""
    contract_token: str
    option_type: int           # 1=CE, 2=PE
    strike: int                # Strike * 100 for precision
    tf: int                    # Timeframe code
    open: float
    high: float
    low: float
    close: float
    volume: int
    oi: int                    # Open Interest
    # Greeks (optional)
    iv: Optional[float] = None      # Implied Volatility
    delta: Optional[float] = None
    gamma: Optional[float] = None
    theta: Optional[float] = None
    vega: Optional[float] = None
    timestamp: Optional[datetime] = None
    
    @property
    def strike_price(self) -> float:
        """Get actual strike price from integer representation"""
        return self.strike / 100.0
    
    @property
    def option_type_str(self) -> str:
        """Get option type as string"""
        return 'CE' if self.option_type == OptionTypeCode.CALL else 'PE'
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'contract_token': self.contract_token,
            'option_type': self.option_type,
            'option_type_str': self.option_type_str,
            'strike': self.strike,
            'strike_price': self.strike_price,
            'tf': self.tf,
            'open': self.open,
            'high': self.high,
            'low': self.low,
            'close': self.close,
            'volume': self.volume,
            'oi': self.oi,
            'iv': self.iv,
            'delta': self.delta,
            'gamma': self.gamma,
            'theta': self.theta,
            'vega': self.vega,
            'timestamp': self.timestamp
        }

@dataclass
class OptimizedIndexData:
    """Optimized model for index historical data"""
    tf: int                    # Timeframe code
    open: float
    high: float
    low: float
    close: float
    timestamp: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'tf': self.tf,
            'open': self.open,
            'high': self.high,
            'low': self.low,
            'close': self.close,
            'timestamp': self.timestamp
        }

@dataclass
class TableMetadata:
    """Metadata for dynamically created tables"""
    table_name: str
    symbol: str
    instrument_type: str
    exchange: str
    created_at: datetime
    last_updated: Optional[datetime] = None
    record_count: int = 0
    size_mb: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'table_name': self.table_name,
            'symbol': self.symbol,
            'instrument_type': self.instrument_type,
            'exchange': self.exchange,
            'created_at': self.created_at,
            'last_updated': self.last_updated,
            'record_count': self.record_count,
            'size_mb': self.size_mb
        }

@dataclass
class OptionsChainSnapshot:
    """Complete options chain snapshot at a point in time"""
    underlying: str
    exchange: str
    expiry: str
    timestamp: datetime
    call_options: List[OptimizedOptionsData]
    put_options: List[OptimizedOptionsData]
    
    def get_strikes(self) -> List[float]:
        """Get all unique strikes in the chain"""
        strikes = set()
        for option in self.call_options + self.put_options:
            strikes.add(option.strike_price)
        return sorted(list(strikes))
    
    def get_atm_strike(self, spot_price: float) -> Optional[float]:
        """Get the at-the-money strike closest to spot price"""
        strikes = self.get_strikes()
        if not strikes:
            return None
        
        return min(strikes, key=lambda x: abs(x - spot_price))
    
    def get_option_by_strike(self, strike: float, option_type: str) -> Optional[OptimizedOptionsData]:
        """Get specific option by strike and type"""
        options = self.call_options if option_type.upper() == 'CE' else self.put_options
        
        for option in options:
            if abs(option.strike_price - strike) < 0.01:  # Allow small floating point differences
                return option
        
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'underlying': self.underlying,
            'exchange': self.exchange,
            'expiry': self.expiry,
            'timestamp': self.timestamp,
            'strikes': self.get_strikes(),
            'call_options': [opt.to_dict() for opt in self.call_options],
            'put_options': [opt.to_dict() for opt in self.put_options]
        }

# Schema definitions for optimized tables
OPTIMIZED_TABLE_SCHEMAS = {
    'equity': """
        CREATE TABLE IF NOT EXISTS {table_name} (
            tf BYTE,                    -- Timeframe code (1,3,5,15,30,60,1440)
            open DOUBLE,
            high DOUBLE,
            low DOUBLE,
            close DOUBLE,
            volume LONG,
            timestamp TIMESTAMP
        ) TIMESTAMP(timestamp) PARTITION BY DAY;
        
        -- Create indexes for fast queries
        -- QuestDB automatically creates indexes on SYMBOL columns and timestamp
    """,
    
    'futures': """
        CREATE TABLE IF NOT EXISTS {table_name} (
            contract_token SYMBOL CAPACITY 1000 CACHE,
            expiry_date DATE,
            tf BYTE,
            open DOUBLE,
            high DOUBLE,
            low DOUBLE,
            close DOUBLE,
            volume LONG,
            oi LONG,
            timestamp TIMESTAMP
        ) TIMESTAMP(timestamp) PARTITION BY DAY;
    """,
    
    'options': """
        CREATE TABLE IF NOT EXISTS {table_name} (
            contract_token SYMBOL CAPACITY 2000 CACHE,
            option_type BYTE,           -- 1=CE, 2=PE
            strike INT,                 -- Strike * 100
            tf BYTE,
            open DOUBLE,
            high DOUBLE,
            low DOUBLE,
            close DOUBLE,
            volume LONG,
            oi LONG,
            iv DOUBLE,                  -- Implied Volatility
            delta DOUBLE,
            gamma DOUBLE,
            theta DOUBLE,
            vega DOUBLE,
            timestamp TIMESTAMP
        ) TIMESTAMP(timestamp) PARTITION BY DAY;
    """,
    
    'index': """
        CREATE TABLE IF NOT EXISTS {table_name} (
            tf BYTE,
            open DOUBLE,
            high DOUBLE,
            low DOUBLE,
            close DOUBLE,
            timestamp TIMESTAMP
        ) TIMESTAMP(timestamp) PARTITION BY DAY;
    """
}

# Query templates for common operations
OPTIMIZED_QUERY_TEMPLATES = {
    'latest_ohlc': """
        SELECT tf, open, high, low, close, volume, timestamp
        FROM {table_name}
        WHERE tf = {timeframe_code}
        ORDER BY timestamp DESC
        LIMIT {limit}
    """,
    
    'ohlc_range': """
        SELECT tf, open, high, low, close, volume, timestamp
        FROM {table_name}
        WHERE tf = {timeframe_code}
        AND timestamp BETWEEN '{start_date}' AND '{end_date}'
        ORDER BY timestamp ASC
    """,
    
    'options_chain_latest': """
        SELECT 
            contract_token,
            option_type,
            strike / 100.0 as strike_price,
            open, high, low, close, volume, oi,
            iv, delta, gamma, theta, vega,
            timestamp
        FROM {table_name}
        WHERE tf = {timeframe_code}
        AND timestamp = (SELECT MAX(timestamp) FROM {table_name} WHERE tf = {timeframe_code})
        ORDER BY strike, option_type
    """,
    
    'futures_oi_analysis': """
        SELECT 
            contract_token,
            expiry_date,
            FIRST(oi) as opening_oi,
            LAST(oi) as closing_oi,
            LAST(oi) - FIRST(oi) as oi_change,
            SUM(volume) as total_volume
        FROM {table_name}
        WHERE tf = {timeframe_code}
        AND timestamp >= '{start_date}'
        GROUP BY contract_token, expiry_date
        ORDER BY oi_change DESC
    """,
    
    'volume_profile': """
        SELECT 
            ROUND(close / {price_bucket}) * {price_bucket} as price_level,
            SUM(volume) as total_volume,
            COUNT(*) as candle_count
        FROM {table_name}
        WHERE tf = {timeframe_code}
        AND timestamp >= '{start_date}'
        GROUP BY price_level
        ORDER BY total_volume DESC
    """
}

# Utility functions for data conversion
class DataConverter:
    """Utility functions for converting between formats"""
    
    @staticmethod
    def timeframe_to_code(timeframe_str: str) -> int:
        """Convert timeframe string to numeric code"""
        mapping = {
            '1m': TimeFrameCode.MINUTE_1,
            '3m': TimeFrameCode.MINUTE_3,
            '5m': TimeFrameCode.MINUTE_5,
            '15m': TimeFrameCode.MINUTE_15,
            '30m': TimeFrameCode.MINUTE_30,
            '1h': TimeFrameCode.HOUR_1,
            'D': TimeFrameCode.DAILY
        }
        return int(mapping.get(timeframe_str, TimeFrameCode.MINUTE_1))
    
    @staticmethod
    def code_to_timeframe(code: int) -> str:
        """Convert numeric code to timeframe string"""
        mapping = {
            1: '1m',
            3: '3m',
            5: '5m',
            15: '15m',
            30: '30m',
            60: '1h',
            1440: 'D'
        }
        return mapping.get(code, '1m')
    
    @staticmethod
    def option_type_to_code(option_type_str: str) -> int:
        """Convert option type string to numeric code"""
        return OptionTypeCode.CALL if option_type_str.upper() == 'CE' else OptionTypeCode.PUT
    
    @staticmethod
    def code_to_option_type(code: int) -> str:
        """Convert numeric code to option type string"""
        return 'CE' if code == OptionTypeCode.CALL else 'PE'
    
    @staticmethod
    def strike_to_int(strike_price: float) -> int:
        """Convert strike price to integer representation"""
        return int(strike_price * 100)
    
    @staticmethod
    def int_to_strike(strike_int: int) -> float:
        """Convert integer representation to strike price"""
        return strike_int / 100.0

# Performance optimization helpers
class QueryOptimizer:
    """Helper class for query optimization"""
    
    @staticmethod
    def get_optimal_partition_filter(start_date: datetime, end_date: datetime) -> str:
        """Generate optimal partition filter for date range queries"""
        
        # For queries spanning multiple days, use partition pruning
        if (end_date - start_date).days > 1:
            return f"timestamp IN '{start_date.strftime('%Y-%m-%d')}' TO '{end_date.strftime('%Y-%m-%d')}'"
        else:
            return f"timestamp >= '{start_date.isoformat()}' AND timestamp <= '{end_date.isoformat()}'"
    
    @staticmethod
    def get_sample_by_clause(timeframe_code: int, aggregation_period: str = 'auto') -> str:
        """Generate SAMPLE BY clause based on timeframe"""
        
        if aggregation_period == 'auto':
            if timeframe_code <= 5:  # 1m, 3m, 5m
                return "SAMPLE BY 1h"
            elif timeframe_code <= 60:  # 15m, 30m, 1h
                return "SAMPLE BY 1d"
            else:  # Daily
                return "SAMPLE BY 1M"  # Monthly
        
        return f"SAMPLE BY {aggregation_period}"
    
    @staticmethod
    def build_efficient_where_clause(filters: Dict[str, Any]) -> str:
        """Build efficient WHERE clause with proper indexing"""
        
        conditions = []
        
        # Always put timeframe filter first (most selective)
        if 'tf' in filters:
            conditions.append(f"tf = {filters['tf']}")
        
        # Then timestamp filters (for partition pruning)
        if 'start_date' in filters:
            conditions.append(f"timestamp >= '{filters['start_date']}'")
        if 'end_date' in filters:
            conditions.append(f"timestamp <= '{filters['end_date']}'")
        
        # Then other filters
        for key, value in filters.items():
            if key not in ['tf', 'start_date', 'end_date']:
                if isinstance(value, str):
                    conditions.append(f"{key} = '{value}'")
                else:
                    conditions.append(f"{key} = {value}")
        
        return " AND ".join(conditions) if conditions else "1=1"
