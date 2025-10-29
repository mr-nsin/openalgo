"""
OpenAlgo-Integrated Configuration Settings

This configuration integrates with OpenAlgo's existing environment variable system
and uses the same database and authentication mechanisms.
"""

import os
from typing import List, Optional
from pydantic import BaseSettings, Field, validator
from enum import Enum

class TimeFrame(str, Enum):
    """Supported timeframes for historical data"""
    MINUTE_1 = "1m"
    MINUTE_3 = "3m"
    MINUTE_5 = "5m"
    MINUTE_15 = "15m"
    MINUTE_30 = "30m"
    HOUR_1 = "1h"
    DAILY = "D"

class InstrumentType(str, Enum):
    """Supported instrument types"""
    EQ = "EQ"
    FUT = "FUT"
    CE = "CE"
    PE = "PE"
    INDEX = "INDEX"

class OpenAlgoSettings(BaseSettings):
    """Configuration settings integrated with OpenAlgo environment variables"""
    
    # OpenAlgo Integration - Use OpenAlgo's authentication system
    # These are read from OpenAlgo's main .env file
    openalgo_database_url: str = Field(default_factory=lambda: os.getenv('DATABASE_URL', 'sqlite:///db/openalgo.db'))
    openalgo_broker_api_key: str = Field(default_factory=lambda: os.getenv('BROKER_API_KEY', ''))
    openalgo_broker_api_secret: str = Field(default_factory=lambda: os.getenv('BROKER_API_SECRET', ''))
    openalgo_user_id: str = Field(default="default_user", description="User ID for OpenAlgo authentication")
    
    # QuestDB Configuration (for historical data storage)
    questdb_host: str = Field(default_factory=lambda: os.getenv('HIST_FETCHER_QUESTDB_HOST', 'localhost'))
    questdb_port: int = Field(default_factory=lambda: int(os.getenv('HIST_FETCHER_QUESTDB_PORT', '9000')))
    questdb_username: Optional[str] = Field(default_factory=lambda: os.getenv('HIST_FETCHER_QUESTDB_USERNAME'))
    questdb_password: Optional[str] = Field(default_factory=lambda: os.getenv('HIST_FETCHER_QUESTDB_PASSWORD'))
    questdb_database: str = Field(default_factory=lambda: os.getenv('HIST_FETCHER_QUESTDB_DATABASE', 'qdb'))
    
    # Processing Configuration
    api_requests_per_second: int = Field(default_factory=lambda: int(os.getenv('HIST_FETCHER_API_REQUESTS_PER_SECOND', '3')))
    max_concurrent_requests: int = Field(default_factory=lambda: int(os.getenv('HIST_FETCHER_MAX_CONCURRENT_REQUESTS', '5')))
    batch_size: int = Field(default_factory=lambda: int(os.getenv('HIST_FETCHER_BATCH_SIZE', '50')))
    chunk_days: int = 60  # Zerodha API limit
    
    # Data Configuration
    enabled_timeframes: List[str] = Field(
        default_factory=lambda: os.getenv('HIST_FETCHER_ENABLED_TIMEFRAMES', '1m,5m,15m,1h,D').split(',')
    )
    
    enabled_instrument_types: List[str] = Field(
        default_factory=lambda: os.getenv('HIST_FETCHER_ENABLED_INSTRUMENT_TYPES', 'EQ,FUT,CE,PE,INDEX').split(',')
    )
    
    enabled_exchanges: List[str] = Field(
        default_factory=lambda: os.getenv('HIST_FETCHER_ENABLED_EXCHANGES', 'NSE,BSE,NFO,BFO,NSE_INDEX,BSE_INDEX').split(',')
    )
    
    # Historical Data Configuration
    historical_days_limit: int = Field(default_factory=lambda: int(os.getenv('HIST_FETCHER_HISTORICAL_DAYS_LIMIT', '365')))
    start_date_override: Optional[str] = Field(default_factory=lambda: os.getenv('HIST_FETCHER_START_DATE_OVERRIDE'))
    
    # Notification Configuration (reuse OpenAlgo's notification system)
    telegram_bot_token: Optional[str] = Field(default_factory=lambda: os.getenv('TELEGRAM_BOT_TOKEN'))
    telegram_chat_ids: List[str] = Field(
        default_factory=lambda: os.getenv('TELEGRAM_CHAT_IDS', '[]').replace('[', '').replace(']', '').replace('"', '').split(',') if os.getenv('TELEGRAM_CHAT_IDS') else []
    )
    
    # Email Configuration
    smtp_host: Optional[str] = Field(default_factory=lambda: os.getenv('HIST_FETCHER_SMTP_HOST'))
    smtp_port: int = Field(default_factory=lambda: int(os.getenv('HIST_FETCHER_SMTP_PORT', '587')))
    smtp_username: Optional[str] = Field(default_factory=lambda: os.getenv('HIST_FETCHER_SMTP_USERNAME'))
    smtp_password: Optional[str] = Field(default_factory=lambda: os.getenv('HIST_FETCHER_SMTP_PASSWORD'))
    email_recipients: List[str] = Field(
        default_factory=lambda: os.getenv('HIST_FETCHER_EMAIL_RECIPIENTS', '[]').replace('[', '').replace(']', '').replace('"', '').split(',') if os.getenv('HIST_FETCHER_EMAIL_RECIPIENTS') else []
    )
    
    # Logging Configuration (reuse OpenAlgo's logging)
    log_level: str = Field(default_factory=lambda: os.getenv('HIST_FETCHER_LOG_LEVEL', os.getenv('LOG_LEVEL', 'INFO')))
    log_file_path: str = Field(default_factory=lambda: os.getenv('HIST_FETCHER_LOG_FILE_PATH', 'logs/historical_fetcher.log'))
    log_rotation: str = Field(default_factory=lambda: os.getenv('HIST_FETCHER_LOG_ROTATION', '100 MB'))
    log_retention: str = Field(default_factory=lambda: os.getenv('HIST_FETCHER_LOG_RETENTION', '30 days'))
    
    # Performance Configuration
    enable_performance_monitoring: bool = Field(default_factory=lambda: os.getenv('HIST_FETCHER_ENABLE_PERFORMANCE_MONITORING', 'true').lower() == 'true')
    memory_limit_mb: int = Field(default_factory=lambda: int(os.getenv('HIST_FETCHER_MEMORY_LIMIT_MB', '4096')))
    
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
    def parse_telegram_chat_ids(cls, v):
        """Parse telegram chat IDs from string or list"""
        if isinstance(v, str):
            if v == '[]' or not v:
                return []
            return [cid.strip().replace('"', '').replace("'", '') for cid in v.split(',')]
        return v
    
    @validator('email_recipients', pre=True)
    def parse_email_recipients(cls, v):
        """Parse email recipients from string or list"""
        if isinstance(v, str):
            if v == '[]' or not v:
                return []
            return [email.strip().replace('"', '').replace("'", '') for email in v.split(',')]
        return v
    
    def get_timeframe_objects(self) -> List[TimeFrame]:
        """Get TimeFrame enum objects for enabled timeframes"""
        timeframe_objects = []
        for tf_str in self.enabled_timeframes:
            try:
                timeframe_objects.append(TimeFrame(tf_str))
            except ValueError:
                print(f"Warning: Unknown timeframe '{tf_str}' - skipping")
        return timeframe_objects
    
    def get_instrument_type_objects(self) -> List[InstrumentType]:
        """Get InstrumentType enum objects for enabled instrument types"""
        instrument_type_objects = []
        for it_str in self.enabled_instrument_types:
            try:
                instrument_type_objects.append(InstrumentType(it_str))
            except ValueError:
                print(f"Warning: Unknown instrument type '{it_str}' - skipping")
        return instrument_type_objects
    
    class Config:
        env_file = '.env'
        env_file_encoding = 'utf-8'
        case_sensitive = False
