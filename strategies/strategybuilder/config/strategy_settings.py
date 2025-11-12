"""
Strategy Builder Configuration Settings

This configuration integrates with OpenAlgo's existing environment variable system
and uses the same database and authentication mechanisms.
"""

import os
from typing import Optional
from enum import Enum

try:
    # Pydantic v2+ (BaseSettings moved to pydantic-settings)
    from pydantic_settings import BaseSettings
    from pydantic import Field
    PYDANTIC_V2 = True
except ImportError:
    try:
        # Pydantic v1 (BaseSettings in main pydantic package)
        from pydantic import BaseSettings, Field
        PYDANTIC_V2 = False
    except ImportError:
        raise ImportError(
            "Neither pydantic-settings nor pydantic BaseSettings could be imported. "
            "Please install pydantic-settings: pip install pydantic-settings"
        )


class TimeFrame(str, Enum):
    """Supported timeframes for historical data"""
    MINUTE_1 = "1m"
    MINUTE_3 = "3m"
    MINUTE_5 = "5m"
    MINUTE_15 = "15m"
    MINUTE_30 = "30m"
    HOUR_1 = "1h"
    DAILY = "D"


class DataSource(str, Enum):
    """Data source options"""
    QUESTDB = "questdb"  # Fetch from QuestDB (faster, requires historicalfetcher)
    OPENALGO_API = "openalgo_api"  # Fetch from OpenAlgo API (slower but always available)


class StrategySettings(BaseSettings):
    """Configuration settings for strategy builder"""
    
    # OpenAlgo Integration - Use OpenAlgo's API system
    openalgo_database_url: str = Field(
        default_factory=lambda: os.getenv('DATABASE_URL', 'sqlite:///db/openalgo.db')
    )
    openalgo_api_key: str = Field(
        default_factory=lambda: os.getenv('OPENALGO_API_KEY', '')
    )
    openalgo_api_host: str = Field(
        default_factory=lambda: os.getenv('OPENALGO_API_HOST', 'http://127.0.0.1:5000')
    )
    
    # QuestDB Configuration (for historical data storage)
    questdb_host: str = Field(
        default_factory=lambda: os.getenv('STRATEGY_QUESTDB_HOST', os.getenv('HIST_FETCHER_QUESTDB_HOST', 'localhost'))
    )
    questdb_port: int = Field(
        default_factory=lambda: int(os.getenv('STRATEGY_QUESTDB_PORT', os.getenv('HIST_FETCHER_QUESTDB_PORT', '9000')))
    )
    questdb_username: Optional[str] = Field(
        default_factory=lambda: os.getenv('STRATEGY_QUESTDB_USERNAME', os.getenv('HIST_FETCHER_QUESTDB_USERNAME'))
    )
    questdb_password: Optional[str] = Field(
        default_factory=lambda: os.getenv('STRATEGY_QUESTDB_PASSWORD', os.getenv('HIST_FETCHER_QUESTDB_PASSWORD'))
    )
    questdb_database: str = Field(
        default_factory=lambda: os.getenv('STRATEGY_QUESTDB_DATABASE', os.getenv('HIST_FETCHER_QUESTDB_DATABASE', 'qdb'))
    )
    
    # Data Source Configuration
    data_source: DataSource = Field(
        default_factory=lambda: DataSource(os.getenv('STRATEGY_DATA_SOURCE', 'questdb').lower())
    )
    
    # Logging Configuration
    log_level: str = Field(
        default_factory=lambda: os.getenv('STRATEGY_LOG_LEVEL', 'INFO')
    )
    log_file_path: str = Field(
        default_factory=lambda: os.getenv('STRATEGY_LOG_FILE_PATH', 'logs/strategy_builder.log')
    )
    
    # Strategy Configuration
    strategy_name: str = Field(
        default_factory=lambda: os.getenv('STRATEGY_NAME', 'mvg_avg_crossover_original')
    )
    
    # Backtest Configuration
    symbol: str = Field(
        default_factory=lambda: os.getenv('STRATEGY_SYMBOL', 'RELIANCE')
    )
    exchange: str = Field(
        default_factory=lambda: os.getenv('STRATEGY_EXCHANGE', 'NSE')
    )
    interval: str = Field(
        default_factory=lambda: os.getenv('STRATEGY_INTERVAL', 'D')
    )
    start_date: str = Field(
        default_factory=lambda: os.getenv('STRATEGY_START_DATE', '')
    )
    end_date: str = Field(
        default_factory=lambda: os.getenv('STRATEGY_END_DATE', '')
    )
    initial_capital: float = Field(
        default_factory=lambda: float(os.getenv('STRATEGY_INITIAL_CAPITAL', '100000'))
    )
    
    # Moving Average Crossover Strategy Parameters
    short_sma_period: int = Field(
        default_factory=lambda: int(os.getenv('STRATEGY_SHORT_SMA_PERIOD', '50'))
    )
    mid_ema_period: int = Field(
        default_factory=lambda: int(os.getenv('STRATEGY_MID_EMA_PERIOD', '100'))
    )
    long_sma_period: int = Field(
        default_factory=lambda: int(os.getenv('STRATEGY_LONG_SMA_PERIOD', '200'))
    )
    swing_lookback: int = Field(
        default_factory=lambda: int(os.getenv('STRATEGY_SWING_LOOKBACK', '5'))
    )
    volume_filter: bool = Field(
        default_factory=lambda: os.getenv('STRATEGY_VOLUME_FILTER', 'true').lower() == 'true'
    )
    macd_filter: bool = Field(
        default_factory=lambda: os.getenv('STRATEGY_MACD_FILTER', 'true').lower() == 'true'
    )
    
    class Config:
        env_file = '.env'
        env_file_encoding = 'utf-8'
        case_sensitive = False

