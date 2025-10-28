"""
QuestDB Client for Historical Data Storage

Handles connections, table creation, and data operations for QuestDB with
instrument-specific optimizations.
"""

import asyncio
import asyncpg
import sys
import os
from typing import List, Dict, Any, Optional, Union
from datetime import datetime, date
import time

# Add the OpenAlgo root directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from utils.logging import get_logger
from config.settings import Settings, InstrumentType
from fetchers.symbol_manager import SymbolInfo
from fetchers.zerodha_fetcher import HistoricalCandle
from database.models import (
    TableType, QUESTDB_SCHEMAS,
    EquityDataModel, FuturesDataModel, OptionsDataModel, IndexDataModel,
    FetchStatusModel, FetchSummaryModel
)

logger = get_logger(__name__)

class QuestDBError(Exception):
    """Custom exception for QuestDB operations"""
    pass

class QuestDBClient:
    """
    QuestDB client with instrument-specific table operations
    """
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.pool: Optional[asyncpg.Pool] = None
        
        # Table mapping for different instrument types
        self.table_mapping = {
            InstrumentType.EQUITY: TableType.EQUITY,
            InstrumentType.FUTURES: TableType.FUTURES,
            InstrumentType.CALL_OPTION: TableType.OPTIONS,
            InstrumentType.PUT_OPTION: TableType.OPTIONS,
            InstrumentType.INDEX: TableType.INDEX
        }
        
        # Statistics
        self.stats = {
            'total_inserts': 0,
            'successful_inserts': 0,
            'failed_inserts': 0,
            'total_records_inserted': 0,
            'connection_errors': 0
        }
    
    async def connect(self):
        """Initialize connection pool to QuestDB"""
        try:
            # QuestDB connection parameters
            connection_params = {
                'host': self.settings.questdb_host,
                'port': self.settings.questdb_port,
                'database': self.settings.questdb_database,
                'min_size': 5,
                'max_size': 20,
                'command_timeout': 60,
                'server_settings': {
                    'application_name': 'historical_data_fetcher'
                }
            }
            
            # Add authentication if provided
            if self.settings.questdb_username:
                connection_params['user'] = self.settings.questdb_username
            if self.settings.questdb_password:
                connection_params['password'] = self.settings.questdb_password
            
            self.pool = await asyncpg.create_pool(**connection_params)
            
            # Test connection
            async with self.pool.acquire() as conn:
                await conn.execute('SELECT 1')
            
            logger.info(f"Connected to QuestDB at {self.settings.questdb_host}:{self.settings.questdb_port}")
            
        except Exception as e:
            self.stats['connection_errors'] += 1
            logger.error(f"Failed to connect to QuestDB: {e}")
            raise QuestDBError(f"Connection failed: {e}")
    
    async def create_tables(self):
        """Create all required tables with optimized schemas"""
        
        if not self.pool:
            raise QuestDBError("Not connected to QuestDB")
        
        async with self.pool.acquire() as conn:
            for table_type, schema in QUESTDB_SCHEMAS.items():
                try:
                    await conn.execute(schema)
                    logger.info(f"Created/verified table: {table_type.value}")
                except Exception as e:
                    logger.error(f"Error creating table {table_type.value}: {e}")
                    raise QuestDBError(f"Table creation failed for {table_type.value}: {e}")
    
    async def upsert_historical_data(
        self,
        symbol_info: SymbolInfo,
        timeframe: str,
        candles: List[HistoricalCandle]
    ) -> int:
        """
        Upsert historical data based on instrument type
        
        Returns:
            Number of records inserted
        """
        
        if not candles:
            return 0
        
        table_type = self.table_mapping.get(symbol_info.instrument_type)
        if not table_type:
            logger.warning(f"Unknown instrument type: {symbol_info.instrument_type}")
            return 0
        
        try:
            # Route to appropriate handler based on instrument type
            if symbol_info.instrument_type == InstrumentType.EQUITY:
                return await self._upsert_equity_data(symbol_info, timeframe, candles)
            elif symbol_info.instrument_type == InstrumentType.FUTURES:
                return await self._upsert_futures_data(symbol_info, timeframe, candles)
            elif symbol_info.instrument_type in [InstrumentType.CALL_OPTION, InstrumentType.PUT_OPTION]:
                return await self._upsert_options_data(symbol_info, timeframe, candles)
            elif symbol_info.instrument_type == InstrumentType.INDEX:
                return await self._upsert_index_data(symbol_info, timeframe, candles)
            else:
                logger.warning(f"Unsupported instrument type: {symbol_info.instrument_type}")
                return 0
                
        except Exception as e:
            self.stats['failed_inserts'] += 1
            logger.error(f"Error upserting data for {symbol_info.symbol}: {e}")
            raise QuestDBError(f"Data upsert failed: {e}")
    
    async def _upsert_equity_data(
        self,
        symbol_info: SymbolInfo,
        timeframe: str,
        candles: List[HistoricalCandle]
    ) -> int:
        """Upsert equity historical data"""
        
        table_name = TableType.EQUITY.value
        
        # Prepare batch insert data
        insert_data = []
        for candle in candles:
            insert_data.append((
                symbol_info.symbol,
                symbol_info.exchange,
                symbol_info.instrument_token,
                candle.open,
                candle.high,
                candle.low,
                candle.close,
                candle.volume,
                timeframe,
                candle.timestamp
            ))
        
        # Batch insert query
        insert_query = f"""
            INSERT INTO {table_name} (
                symbol, exchange, instrument_token, open, high, low, close, 
                volume, timeframe, timestamp
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
        """
        
        return await self._execute_batch_insert(insert_query, insert_data, symbol_info.symbol)
    
    async def _upsert_futures_data(
        self,
        symbol_info: SymbolInfo,
        timeframe: str,
        candles: List[HistoricalCandle]
    ) -> int:
        """Upsert futures historical data"""
        
        table_name = TableType.FUTURES.value
        underlying_symbol = symbol_info.extract_underlying_symbol()
        expiry_date = self._parse_expiry_date(symbol_info.expiry) if symbol_info.expiry else None
        
        insert_data = []
        for candle in candles:
            insert_data.append((
                underlying_symbol,
                symbol_info.exchange,
                symbol_info.instrument_token,
                expiry_date,
                symbol_info.symbol,  # contract_symbol
                candle.open,
                candle.high,
                candle.low,
                candle.close,
                candle.volume,
                candle.oi,
                timeframe,
                candle.timestamp
            ))
        
        insert_query = f"""
            INSERT INTO {table_name} (
                underlying_symbol, exchange, instrument_token, expiry_date, contract_symbol,
                open, high, low, close, volume, oi, timeframe, timestamp
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)
        """
        
        return await self._execute_batch_insert(insert_query, insert_data, symbol_info.symbol)
    
    async def _upsert_options_data(
        self,
        symbol_info: SymbolInfo,
        timeframe: str,
        candles: List[HistoricalCandle]
    ) -> int:
        """Upsert options historical data"""
        
        table_name = TableType.OPTIONS.value
        underlying_symbol = symbol_info.extract_underlying_symbol()
        expiry_date = self._parse_expiry_date(symbol_info.expiry) if symbol_info.expiry else None
        option_type = symbol_info.instrument_type.value  # CE or PE
        
        insert_data = []
        for candle in candles:
            insert_data.append((
                underlying_symbol,
                symbol_info.exchange,
                symbol_info.instrument_token,
                option_type,
                symbol_info.strike or 0.0,
                expiry_date,
                symbol_info.symbol,  # contract_symbol
                candle.open,
                candle.high,
                candle.low,
                candle.close,
                candle.volume,
                candle.oi,
                timeframe,
                candle.timestamp
            ))
        
        insert_query = f"""
            INSERT INTO {table_name} (
                underlying_symbol, exchange, instrument_token, option_type, strike_price,
                expiry_date, contract_symbol, open, high, low, close, volume, oi, 
                timeframe, timestamp
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15)
        """
        
        return await self._execute_batch_insert(insert_query, insert_data, symbol_info.symbol)
    
    async def _upsert_index_data(
        self,
        symbol_info: SymbolInfo,
        timeframe: str,
        candles: List[HistoricalCandle]
    ) -> int:
        """Upsert index historical data"""
        
        table_name = TableType.INDEX.value
        
        insert_data = []
        for candle in candles:
            insert_data.append((
                symbol_info.symbol,  # index_name
                symbol_info.exchange,
                symbol_info.instrument_token,
                candle.open,
                candle.high,
                candle.low,
                candle.close,
                timeframe,
                candle.timestamp
            ))
        
        insert_query = f"""
            INSERT INTO {table_name} (
                index_name, exchange, instrument_token, open, high, low, close, 
                timeframe, timestamp
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
        """
        
        return await self._execute_batch_insert(insert_query, insert_data, symbol_info.symbol)
    
    async def _execute_batch_insert(
        self,
        query: str,
        data: List[tuple],
        symbol: str
    ) -> int:
        """Execute batch insert with error handling"""
        
        if not self.pool:
            raise QuestDBError("Not connected to QuestDB")
        
        start_time = time.monotonic()
        
        try:
            async with self.pool.acquire() as conn:
                # Use executemany for batch insert
                await conn.executemany(query, data)
                
                # Update statistics
                self.stats['successful_inserts'] += 1
                self.stats['total_records_inserted'] += len(data)
                
                execution_time = time.monotonic() - start_time
                logger.debug(
                    f"Inserted {len(data)} records for {symbol} in {execution_time:.2f}s"
                )
                
                return len(data)
                
        except Exception as e:
            self.stats['failed_inserts'] += 1
            execution_time = time.monotonic() - start_time
            logger.error(
                f"Error inserting {len(data)} records for {symbol} after {execution_time:.2f}s: {e}"
            )
            raise
    
    async def update_fetch_status(
        self,
        symbol_info: SymbolInfo,
        timeframe: str,
        status: str,
        records_count: int = 0,
        error_message: Optional[str] = None
    ):
        """Update fetch status for a symbol-timeframe combination"""
        
        if not self.pool:
            raise QuestDBError("Not connected to QuestDB")
        
        now = datetime.now()
        last_successful_fetch = now if status == 'success' else None
        
        # Use INSERT for QuestDB (it handles duplicates automatically with timestamp)
        upsert_query = f"""
            INSERT INTO {TableType.FETCH_STATUS.value} (
                symbol, exchange, instrument_token, instrument_type, timeframe,
                last_fetch_date, last_successful_fetch, total_records, status,
                error_message, retry_count, created_at, updated_at
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, 0, $11, $12)
        """
        
        try:
            async with self.pool.acquire() as conn:
                await conn.execute(
                    upsert_query,
                    symbol_info.symbol,
                    symbol_info.exchange,
                    symbol_info.instrument_token,
                    symbol_info.instrument_type.value,
                    timeframe,
                    now,  # last_fetch_date
                    last_successful_fetch,
                    records_count,
                    status,
                    error_message,
                    now,  # created_at
                    now   # updated_at
                )
                
        except Exception as e:
            logger.error(f"Error updating fetch status for {symbol_info.symbol}: {e}")
    
    async def get_last_fetch_date(
        self,
        symbol_info: SymbolInfo,
        timeframe: str
    ) -> Optional[datetime]:
        """Get last successful fetch date for a symbol-timeframe combination"""
        
        if not self.pool:
            raise QuestDBError("Not connected to QuestDB")
        
        query = f"""
            SELECT last_successful_fetch 
            FROM {TableType.FETCH_STATUS.value} 
            WHERE symbol = $1 AND exchange = $2 AND timeframe = $3 AND status = 'success'
            ORDER BY updated_at DESC 
            LIMIT 1
        """
        
        try:
            async with self.pool.acquire() as conn:
                result = await conn.fetchval(
                    query,
                    symbol_info.symbol,
                    symbol_info.exchange,
                    timeframe
                )
                return result
        except Exception as e:
            logger.error(f"Error getting last fetch date for {symbol_info.symbol}: {e}")
            return None
    
    async def insert_fetch_summary(self, summary: FetchSummaryModel):
        """Insert daily fetch summary"""
        
        if not self.pool:
            raise QuestDBError("Not connected to QuestDB")
        
        insert_query = f"""
            INSERT INTO {TableType.FETCH_SUMMARY.value} (
                fetch_date, total_symbols, successful_symbols, failed_symbols,
                total_records_inserted, processing_time_minutes, equity_symbols,
                futures_symbols, options_symbols, index_symbols, minute_data_records,
                daily_data_records, created_at
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)
        """
        
        try:
            async with self.pool.acquire() as conn:
                await conn.execute(
                    insert_query,
                    summary.fetch_date,
                    summary.total_symbols,
                    summary.successful_symbols,
                    summary.failed_symbols,
                    summary.total_records_inserted,
                    summary.processing_time_minutes,
                    summary.equity_symbols,
                    summary.futures_symbols,
                    summary.options_symbols,
                    summary.index_symbols,
                    summary.minute_data_records,
                    summary.daily_data_records,
                    summary.created_at
                )
                
        except Exception as e:
            logger.error(f"Error inserting fetch summary: {e}")
    
    async def get_data_statistics(self) -> Dict[str, Any]:
        """Get statistics about stored data"""
        
        if not self.pool:
            raise QuestDBError("Not connected to QuestDB")
        
        stats = {}
        
        try:
            async with self.pool.acquire() as conn:
                # Get record counts for each table
                for table_type in [TableType.EQUITY, TableType.FUTURES, TableType.OPTIONS, TableType.INDEX]:
                    try:
                        count = await conn.fetchval(f"SELECT COUNT(*) FROM {table_type.value}")
                        stats[f'{table_type.value}_records'] = count
                    except Exception as e:
                        logger.warning(f"Error getting count for {table_type.value}: {e}")
                        stats[f'{table_type.value}_records'] = 0
                
                # Get latest data timestamps
                for table_type in [TableType.EQUITY, TableType.FUTURES, TableType.OPTIONS, TableType.INDEX]:
                    try:
                        latest = await conn.fetchval(
                            f"SELECT MAX(timestamp) FROM {table_type.value}"
                        )
                        stats[f'{table_type.value}_latest'] = latest
                    except Exception as e:
                        logger.warning(f"Error getting latest timestamp for {table_type.value}: {e}")
                        stats[f'{table_type.value}_latest'] = None
                
        except Exception as e:
            logger.error(f"Error getting data statistics: {e}")
        
        return stats
    
    def _parse_expiry_date(self, expiry_str: str) -> Optional[date]:
        """Parse expiry date from DD-MMM-YY format"""
        try:
            return datetime.strptime(expiry_str, "%d-%b-%y").date()
        except Exception as e:
            logger.warning(f"Could not parse expiry date '{expiry_str}': {e}")
            return None
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get client statistics"""
        stats = self.stats.copy()
        
        if stats['total_inserts'] > 0:
            stats['success_rate'] = (stats['successful_inserts'] / stats['total_inserts']) * 100
        else:
            stats['success_rate'] = 0
        
        return stats
    
    async def test_connection(self) -> bool:
        """Test database connection"""
        try:
            if not self.pool:
                return False
            
            async with self.pool.acquire() as conn:
                await conn.execute('SELECT 1')
                return True
                
        except Exception as e:
            logger.error(f"Connection test failed: {e}")
            return False
    
    async def cleanup(self):
        """Cleanup database connections"""
        if self.pool:
            await self.pool.close()
            logger.info("QuestDB connection pool closed")
        
        # Log final statistics
        stats = self.get_statistics()
        logger.info(f"QuestDB Client Statistics: {stats}")

# Utility functions for database operations
class DatabaseUtils:
    """Utility functions for database operations"""
    
    @staticmethod
    def validate_candle_data(candles: List[HistoricalCandle]) -> List[HistoricalCandle]:
        """Validate and clean candle data"""
        valid_candles = []
        
        for candle in candles:
            # Basic validation
            if (candle.open > 0 and candle.high > 0 and candle.low > 0 and candle.close > 0 and
                candle.low <= candle.open <= candle.high and
                candle.low <= candle.close <= candle.high and
                candle.volume >= 0):
                valid_candles.append(candle)
        
        return valid_candles
    
    @staticmethod
    def deduplicate_candles(candles: List[HistoricalCandle]) -> List[HistoricalCandle]:
        """Remove duplicate candles based on timestamp"""
        seen_timestamps = set()
        unique_candles = []
        
        for candle in candles:
            if candle.timestamp not in seen_timestamps:
                seen_timestamps.add(candle.timestamp)
                unique_candles.append(candle)
        
        return unique_candles
    
    @staticmethod
    def sort_candles_by_timestamp(candles: List[HistoricalCandle]) -> List[HistoricalCandle]:
        """Sort candles by timestamp"""
        return sorted(candles, key=lambda c: c.timestamp)
