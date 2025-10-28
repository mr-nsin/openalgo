"""
Configuration settings for Historical Data Fetcher
"""

import os
from pydantic import BaseSettings, Field, validator
from typing import List, Dict, Optional
from enum import Enum

class InstrumentType(str, Enum):
    """Supported instrument types"""
    EQUITY = "EQ"
    FUTURES = "FUT"
    CALL_OPTION = "CE"
    PUT_OPTION = "PE"
    INDEX = "INDEX"

class TimeFrame(str, Enum):
    """Supported timeframes"""
    MINUTE_1 = "1m"
    MINUTE_3 = "3m"
    MINUTE_5 = "5m"
    MINUTE_15 = "15m"
    MINUTE_30 = "30m"
    HOUR_1 = "1h"
    DAILY = "D"

class Settings(BaseSettings):
    """Configuration settings with environment variable support"""
    
    # QuestDB Configuration
    questdb_host: str = "localhost"
    questdb_port: int = 9000
    questdb_username: Optional[str] = None
    questdb_password: Optional[str] = None
    questdb_database: str = "qdb"
    
    # OpenAlgo Database Configuration (for symtoken table)
    openalgo_database_url: str = Field(default_factory=lambda: os.getenv('DATABASE_URL', 'sqlite:///db/openalgo.db'))
    
    # Zerodha API Configuration
    zerodha_api_key: str = Field(..., description="Zerodha API Key")
    zerodha_access_token: str = Field(..., description="Zerodha Access Token")
    zerodha_base_url: str = "https://api.kite.trade"
    
    # Processing Configuration
    api_requests_per_second: int = 3
    max_concurrent_requests: int = 5
    batch_size: int = 50
    chunk_days: int = 60  # Zerodha API limit
    
    # Data Configuration
    enabled_timeframes: List[str] = [
        "1m", "5m", "15m", "1h", "D"
    ]
    
    enabled_instrument_types: List[str] = [
        "EQ", "FUT", "CE", "PE", "INDEX"
    ]
    
    # Exchange Configuration
    enabled_exchanges: List[str] = ["NSE", "BSE", "NFO", "BFO", "NSE_INDEX", "BSE_INDEX"]
    
    # Historical Data Configuration
    historical_days_limit: int = 365
    start_date_override: Optional[str] = None  # Format: YYYY-MM-DD
    
    # Notification Configuration
    telegram_bot_token: Optional[str] = None
    telegram_chat_ids: List[str] = []
    
    smtp_host: Optional[str] = None
    smtp_port: int = 587
    smtp_username: Optional[str] = None
    smtp_password: Optional[str] = None
    email_recipients: List[str] = []
    
    # Logging Configuration
    log_level: str = "INFO"
    log_file_path: str = "logs/historical_fetcher.log"
    log_rotation: str = "100 MB"
    log_retention: str = "30 days"
    
    # Performance Configuration
    enable_performance_monitoring: bool = True
    memory_limit_mb: int = 4096
    
    @validator('enabled_timeframes', pre=True)
    def parse_timeframes(cls, v):
        """Parse timeframes from string or list"""
        if isinstance(v, str):
            return [tf.strip() for tf in v.split(',')]
        return v
    
    @validator('enabled_instrument_types', pre=True)
    def parse_instrument_types(cls, v):
        """Parse instrument types from string or list"""
        if isinstance(v, str):
            return [it.strip() for it in v.split(',')]
        return v
    
    @validator('enabled_exchanges', pre=True)
    def parse_exchanges(cls, v):
        """Parse exchanges from string or list"""
        if isinstance(v, str):
            return [ex.strip() for ex in v.split(',')]
        return v
    
    @validator('telegram_chat_ids', pre=True)
    def parse_chat_ids(cls, v):
        """Parse Telegram chat IDs from string or list"""
        if isinstance(v, str):
            # Handle JSON-like string format
            v = v.strip('[]"\'')
            if ',' in v:
                return [chat_id.strip().strip('"\'') for chat_id in v.split(',')]
            return [v] if v else []
        return v
    
    @validator('email_recipients', pre=True)
    def parse_email_recipients(cls, v):
        """Parse email recipients from string or list"""
        if isinstance(v, str):
            # Handle JSON-like string format
            v = v.strip('[]"\'')
            if ',' in v:
                return [email.strip().strip('"\'') for email in v.split(',')]
            return [v] if v else []
        return v
    
    def get_timeframe_objects(self) -> List[TimeFrame]:
        """Get TimeFrame enum objects from enabled timeframes"""
        timeframe_map = {
            '1m': TimeFrame.MINUTE_1,
            '3m': TimeFrame.MINUTE_3,
            '5m': TimeFrame.MINUTE_5,
            '15m': TimeFrame.MINUTE_15,
            '30m': TimeFrame.MINUTE_30,
            '1h': TimeFrame.HOUR_1,
            'D': TimeFrame.DAILY
        }
        return [timeframe_map[tf] for tf in self.enabled_timeframes if tf in timeframe_map]
    
    def get_instrument_type_objects(self) -> List[InstrumentType]:
        """Get InstrumentType enum objects from enabled instrument types"""
        instrument_map = {
            'EQ': InstrumentType.EQUITY,
            'FUT': InstrumentType.FUTURES,
            'CE': InstrumentType.CALL_OPTION,
            'PE': InstrumentType.PUT_OPTION,
            'INDEX': InstrumentType.INDEX
        }
        return [instrument_map[it] for it in self.enabled_instrument_types if it in instrument_map]
    
    class Config:
        env_file = ".env"
        env_prefix = "HIST_FETCHER_"
        case_sensitive = False
