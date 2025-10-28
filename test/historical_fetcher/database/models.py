"""
Data models for Historical Data Fetcher

Defines data structures for different instrument types and database operations.
"""

from dataclasses import dataclass
from datetime import datetime, date
from typing import Optional, Dict, Any
from enum import Enum

class TableType(Enum):
    """Database table types"""
    EQUITY = "equity_historical_data"
    FUTURES = "futures_historical_data"
    OPTIONS = "options_historical_data"
    INDEX = "index_historical_data"
    FETCH_STATUS = "fetch_status"
    FETCH_SUMMARY = "fetch_summary"

@dataclass
class HistoricalDataModel:
    """Base model for historical data"""
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int
    timeframe: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for database insertion"""
        return {
            'timestamp': self.timestamp,
            'open': self.open,
            'high': self.high,
            'low': self.low,
            'close': self.close,
            'volume': self.volume,
            'timeframe': self.timeframe
        }

@dataclass
class EquityDataModel(HistoricalDataModel):
    """Model for equity historical data"""
    symbol: str
    exchange: str
    instrument_token: str
    
    def to_dict(self) -> Dict[str, Any]:
        data = super().to_dict()
        data.update({
            'symbol': self.symbol,
            'exchange': self.exchange,
            'instrument_token': self.instrument_token
        })
        return data

@dataclass
class FuturesDataModel(HistoricalDataModel):
    """Model for futures historical data"""
    underlying_symbol: str
    exchange: str
    instrument_token: str
    expiry_date: Optional[date]
    contract_symbol: str
    oi: int = 0  # Open Interest
    
    def to_dict(self) -> Dict[str, Any]:
        data = super().to_dict()
        data.update({
            'underlying_symbol': self.underlying_symbol,
            'exchange': self.exchange,
            'instrument_token': self.instrument_token,
            'expiry_date': self.expiry_date,
            'contract_symbol': self.contract_symbol,
            'oi': self.oi
        })
        return data

@dataclass
class OptionsDataModel(HistoricalDataModel):
    """Model for options historical data"""
    underlying_symbol: str
    exchange: str
    instrument_token: str
    option_type: str  # CE or PE
    strike_price: float
    expiry_date: Optional[date]
    contract_symbol: str
    oi: int = 0  # Open Interest
    
    def to_dict(self) -> Dict[str, Any]:
        data = super().to_dict()
        data.update({
            'underlying_symbol': self.underlying_symbol,
            'exchange': self.exchange,
            'instrument_token': self.instrument_token,
            'option_type': self.option_type,
            'strike_price': self.strike_price,
            'expiry_date': self.expiry_date,
            'contract_symbol': self.contract_symbol,
            'oi': self.oi
        })
        return data

@dataclass
class IndexDataModel(HistoricalDataModel):
    """Model for index historical data"""
    index_name: str
    exchange: str
    instrument_token: str
    
    def to_dict(self) -> Dict[str, Any]:
        data = super().to_dict()
        # Remove volume for indices (not applicable)
        data.pop('volume', None)
        data.update({
            'index_name': self.index_name,
            'exchange': self.exchange,
            'instrument_token': self.instrument_token
        })
        return data

@dataclass
class FetchStatusModel:
    """Model for fetch status tracking"""
    symbol: str
    exchange: str
    instrument_token: str
    instrument_type: str
    timeframe: str
    last_fetch_date: datetime
    last_successful_fetch: Optional[datetime]
    total_records: int
    status: str  # success/error/pending/running
    error_message: Optional[str]
    retry_count: int
    created_at: datetime
    updated_at: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'symbol': self.symbol,
            'exchange': self.exchange,
            'instrument_token': self.instrument_token,
            'instrument_type': self.instrument_type,
            'timeframe': self.timeframe,
            'last_fetch_date': self.last_fetch_date,
            'last_successful_fetch': self.last_successful_fetch,
            'total_records': self.total_records,
            'status': self.status,
            'error_message': self.error_message,
            'retry_count': self.retry_count,
            'created_at': self.created_at,
            'updated_at': self.updated_at
        }

@dataclass
class FetchSummaryModel:
    """Model for daily fetch summary"""
    fetch_date: date
    total_symbols: int
    successful_symbols: int
    failed_symbols: int
    total_records_inserted: int
    processing_time_minutes: int
    equity_symbols: int
    futures_symbols: int
    options_symbols: int
    index_symbols: int
    minute_data_records: int
    daily_data_records: int
    created_at: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'fetch_date': self.fetch_date,
            'total_symbols': self.total_symbols,
            'successful_symbols': self.successful_symbols,
            'failed_symbols': self.failed_symbols,
            'total_records_inserted': self.total_records_inserted,
            'processing_time_minutes': self.processing_time_minutes,
            'equity_symbols': self.equity_symbols,
            'futures_symbols': self.futures_symbols,
            'options_symbols': self.options_symbols,
            'index_symbols': self.index_symbols,
            'minute_data_records': self.minute_data_records,
            'daily_data_records': self.daily_data_records,
            'created_at': self.created_at
        }

# Schema definitions for QuestDB table creation
QUESTDB_SCHEMAS = {
    TableType.EQUITY: """
        CREATE TABLE IF NOT EXISTS equity_historical_data (
            symbol SYMBOL CAPACITY 10000 CACHE,
            exchange SYMBOL CAPACITY 10 CACHE,
            instrument_token SYMBOL CAPACITY 50000 CACHE,
            open DOUBLE,
            high DOUBLE,
            low DOUBLE,
            close DOUBLE,
            volume LONG,
            timeframe SYMBOL CAPACITY 15 CACHE,
            timestamp TIMESTAMP
        ) TIMESTAMP(timestamp) PARTITION BY DAY;
    """,
    
    TableType.FUTURES: """
        CREATE TABLE IF NOT EXISTS futures_historical_data (
            underlying_symbol SYMBOL CAPACITY 5000 CACHE,
            exchange SYMBOL CAPACITY 10 CACHE,
            instrument_token SYMBOL CAPACITY 50000 CACHE,
            expiry_date DATE,
            contract_symbol SYMBOL CAPACITY 20000 CACHE,
            open DOUBLE,
            high DOUBLE,
            low DOUBLE,
            close DOUBLE,
            volume LONG,
            oi LONG,
            timeframe SYMBOL CAPACITY 15 CACHE,
            timestamp TIMESTAMP
        ) TIMESTAMP(timestamp) PARTITION BY DAY;
    """,
    
    TableType.OPTIONS: """
        CREATE TABLE IF NOT EXISTS options_historical_data (
            underlying_symbol SYMBOL CAPACITY 5000 CACHE,
            exchange SYMBOL CAPACITY 10 CACHE,
            instrument_token SYMBOL CAPACITY 100000 CACHE,
            option_type SYMBOL CAPACITY 2 CACHE,
            strike_price DOUBLE,
            expiry_date DATE,
            contract_symbol SYMBOL CAPACITY 30000 CACHE,
            open DOUBLE,
            high DOUBLE,
            low DOUBLE,
            close DOUBLE,
            volume LONG,
            oi LONG,
            timeframe SYMBOL CAPACITY 15 CACHE,
            timestamp TIMESTAMP
        ) TIMESTAMP(timestamp) PARTITION BY DAY;
    """,
    
    TableType.INDEX: """
        CREATE TABLE IF NOT EXISTS index_historical_data (
            index_name SYMBOL CAPACITY 100 CACHE,
            exchange SYMBOL CAPACITY 10 CACHE,
            instrument_token SYMBOL CAPACITY 1000 CACHE,
            open DOUBLE,
            high DOUBLE,
            low DOUBLE,
            close DOUBLE,
            timeframe SYMBOL CAPACITY 15 CACHE,
            timestamp TIMESTAMP
        ) TIMESTAMP(timestamp) PARTITION BY DAY;
    """,
    
    TableType.FETCH_STATUS: """
        CREATE TABLE IF NOT EXISTS fetch_status (
            symbol SYMBOL CAPACITY 50000 CACHE,
            exchange SYMBOL CAPACITY 20 CACHE,
            instrument_token SYMBOL CAPACITY 200000 CACHE,
            instrument_type SYMBOL CAPACITY 10 CACHE,
            timeframe SYMBOL CAPACITY 15 CACHE,
            last_fetch_date TIMESTAMP,
            last_successful_fetch TIMESTAMP,
            total_records LONG,
            status SYMBOL CAPACITY 10 CACHE,
            error_message STRING,
            retry_count INT,
            created_at TIMESTAMP,
            updated_at TIMESTAMP
        ) TIMESTAMP(updated_at) PARTITION BY DAY;
    """,
    
    TableType.FETCH_SUMMARY: """
        CREATE TABLE IF NOT EXISTS fetch_summary (
            fetch_date DATE,
            total_symbols INT,
            successful_symbols INT,
            failed_symbols INT,
            total_records_inserted LONG,
            processing_time_minutes INT,
            equity_symbols INT,
            futures_symbols INT,
            options_symbols INT,
            index_symbols INT,
            minute_data_records LONG,
            daily_data_records LONG,
            created_at TIMESTAMP
        ) TIMESTAMP(created_at) PARTITION BY MONTH;
    """
}
