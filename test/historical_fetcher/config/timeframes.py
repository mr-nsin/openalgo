"""
Timeframe configuration and mapping utilities
"""

from typing import Dict
from .settings import TimeFrame

class TimeFrameConfig:
    """Timeframe configuration and mapping utilities"""
    
    # Zerodha API timeframe mapping
    ZERODHA_TIMEFRAME_MAP = {
        TimeFrame.MINUTE_1: 'minute',
        TimeFrame.MINUTE_3: '3minute',
        TimeFrame.MINUTE_5: '5minute',
        TimeFrame.MINUTE_15: '15minute',
        TimeFrame.MINUTE_30: '30minute',
        TimeFrame.HOUR_1: '60minute',
        TimeFrame.DAILY: 'day'
    }
    
    # Expected records per day for each timeframe (approximate)
    EXPECTED_RECORDS_PER_DAY = {
        TimeFrame.MINUTE_1: 375,    # 6.25 hours * 60 minutes
        TimeFrame.MINUTE_3: 125,    # 375 / 3
        TimeFrame.MINUTE_5: 75,     # 375 / 5
        TimeFrame.MINUTE_15: 25,    # 375 / 15
        TimeFrame.MINUTE_30: 13,    # 375 / 30 (rounded)
        TimeFrame.HOUR_1: 6,        # 6.25 hours (rounded)
        TimeFrame.DAILY: 1          # 1 record per day
    }
    
    # Timeframe priorities for processing (higher number = higher priority)
    PROCESSING_PRIORITY = {
        TimeFrame.DAILY: 5,         # Process daily data first
        TimeFrame.HOUR_1: 4,        # Then hourly
        TimeFrame.MINUTE_15: 3,     # Then 15-minute
        TimeFrame.MINUTE_5: 2,      # Then 5-minute
        TimeFrame.MINUTE_3: 1,      # Then 3-minute
        TimeFrame.MINUTE_1: 0       # Process 1-minute data last (most volume)
    }
    
    @classmethod
    def get_zerodha_interval(cls, timeframe: TimeFrame) -> str:
        """Get Zerodha API interval string for timeframe"""
        return cls.ZERODHA_TIMEFRAME_MAP.get(timeframe, 'minute')
    
    @classmethod
    def get_expected_records(cls, timeframe: TimeFrame, days: int = 1) -> int:
        """Get expected number of records for timeframe and days"""
        per_day = cls.EXPECTED_RECORDS_PER_DAY.get(timeframe, 1)
        return per_day * days
    
    @classmethod
    def get_processing_order(cls, timeframes: list) -> list:
        """Get timeframes sorted by processing priority"""
        return sorted(timeframes, key=lambda tf: cls.PROCESSING_PRIORITY.get(tf, 0), reverse=True)
    
    @classmethod
    def is_intraday_timeframe(cls, timeframe: TimeFrame) -> bool:
        """Check if timeframe is intraday (less than daily)"""
        return timeframe != TimeFrame.DAILY
    
    @classmethod
    def get_storage_table_suffix(cls, timeframe: TimeFrame) -> str:
        """Get table suffix for timeframe-based partitioning"""
        if timeframe == TimeFrame.DAILY:
            return "daily"
        elif timeframe in [TimeFrame.HOUR_1, TimeFrame.MINUTE_30]:
            return "hourly"
        else:
            return "minute"
