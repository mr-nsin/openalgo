"""
Dynamic Table Manager for Optimized Historical Data Storage

Creates and manages symbol-specific tables with optimized schemas for
different instrument types and timeframes.
"""

import asyncio
import re
from typing import Dict, List, Optional, Set, Tuple
from datetime import datetime, date
from dataclasses import dataclass
from enum import IntEnum
import sys
import os

# Add the OpenAlgo root directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

# Import configuration - works with both direct script execution and package imports
try:
    from historicalfetcher.config.openalgo_settings import TimeFrame
    from historicalfetcher.utils.async_logger import get_async_logger
    from historicalfetcher.models.data_models import SymbolInfo, TimeFrameCode
    from historicalfetcher.database.enhanced_schemas import EnhancedTableSchemas
except ImportError:
    from ..config.openalgo_settings import TimeFrame
    from ..utils.async_logger import get_async_logger
    from ..models.data_models import SymbolInfo, TimeFrameCode
    from .enhanced_schemas import EnhancedTableSchemas

_async_logger = get_async_logger()
logger = _async_logger.get_logger()
from enum import Enum

class InstrumentType(str, Enum):
    """Instrument types for compatibility"""
    EQUITY = "EQ"
    FUTURES = "FUT"
    CALL_OPTION = "CE"
    PUT_OPTION = "PE"
    INDEX = "INDEX"

# Logger is imported from loguru above

# TimeFrameCode is now imported from historicalfetcher.models.data_models

@dataclass
class TableInfo:
    """Information about a created table"""
    table_name: str
    symbol: str
    instrument_type: InstrumentType
    exchange: str
    created_at: datetime
    record_count: int = 0
    last_updated: Optional[datetime] = None

class TableNamingStrategy:
    """Handles table naming conventions for different instruments"""
    
    @staticmethod
    def sanitize_symbol(symbol: str) -> str:
        """Sanitize symbol name for table naming"""
        # Remove special characters and replace with underscore
        sanitized = re.sub(r'[^a-zA-Z0-9]', '_', symbol.upper())
        # Remove consecutive underscores
        sanitized = re.sub(r'_+', '_', sanitized)
        # Remove leading/trailing underscores
        sanitized = sanitized.strip('_')
        return sanitized
    
    @staticmethod
    def get_equity_table_name(symbol: str, exchange: str) -> str:
        """Generate table name for equity"""
        clean_symbol = TableNamingStrategy.sanitize_symbol(symbol)
        clean_exchange = exchange.lower()
        return f"eq_{clean_exchange}_{clean_symbol}"
    
    @staticmethod
    def get_futures_table_name(underlying: str, exchange: str) -> str:
        """Generate table name for futures (by underlying)"""
        clean_underlying = TableNamingStrategy.sanitize_symbol(underlying)
        clean_exchange = exchange.lower()
        return f"fut_{clean_exchange}_{clean_underlying}"
    
    @staticmethod
    def get_options_table_name(underlying: str, exchange: str, expiry: str) -> str:
        """Generate table name for options (by underlying and expiry)"""
        clean_underlying = TableNamingStrategy.sanitize_symbol(underlying)
        clean_exchange = exchange.lower()
        # Convert expiry date to compact format (YYMMDD)
        expiry_compact = "000000"
        try:
            # Try DD-MMM-YY format first (e.g., "25-Nov-25")
            expiry_date = datetime.strptime(expiry, "%d-%b-%y")
            expiry_compact = expiry_date.strftime("%y%m%d")
        except:
            try:
                # Try DDMMMYY format (e.g., "25NOV25")
                if len(expiry) == 7 and expiry[2:5].isalpha():
                    day = expiry[:2]
                    month_str = expiry[2:5].upper()
                    year = expiry[5:7]
                    month_map = {
                        'JAN': 1, 'FEB': 2, 'MAR': 3, 'APR': 4, 'MAY': 5, 'JUN': 6,
                        'JUL': 7, 'AUG': 8, 'SEP': 9, 'OCT': 10, 'NOV': 11, 'DEC': 12
                    }
                    month = month_map.get(month_str, 1)
                    expiry_date = datetime(int('20' + year), month, int(day))
                    expiry_compact = expiry_date.strftime("%y%m%d")
            except Exception as e:
                logger.warning(f"Could not parse expiry '{expiry}', using default: {e}")
                expiry_compact = "000000"
        
        return f"opt_{clean_exchange}_{clean_underlying}_{expiry_compact}"
    
    @staticmethod
    def get_index_table_name(index_name: str, exchange: str) -> str:
        """Generate table name for index"""
        clean_index = TableNamingStrategy.sanitize_symbol(index_name)
        clean_exchange = exchange.lower()
        return f"idx_{clean_exchange}_{clean_index}"

class OptimizedTableSchemas:
    """Optimized table schemas for different instrument types"""
    
    @staticmethod
    def get_equity_schema(table_name: str) -> str:
        """Optimized schema for equity tables"""
        return f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            tf SHORT,                   -- Timeframe code (1,3,5,15,30,60,1440)
            open DOUBLE,
            high DOUBLE,
            low DOUBLE,
            close DOUBLE,
            volume LONG,
            timestamp TIMESTAMP
        ) TIMESTAMP(timestamp) PARTITION BY DAY;
        """
    
    @staticmethod
    def get_futures_schema(table_name: str) -> str:
        """Optimized schema for futures tables"""
        return f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            contract_token SYMBOL CAPACITY 1000 CACHE,  -- Specific contract token
            expiry_date DATE,
            tf SHORT,                   -- Timeframe code
            open DOUBLE,
            high DOUBLE,
            low DOUBLE,
            close DOUBLE,
            volume LONG,
            oi LONG,                    -- Open Interest
            timestamp TIMESTAMP
        ) TIMESTAMP(timestamp) PARTITION BY DAY;
        """
    
    @staticmethod
    def get_options_schema(table_name: str) -> str:
        """Optimized schema for options tables (per underlying-expiry)"""
        return f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            contract_token SYMBOL CAPACITY 2000 CACHE,  -- Specific option contract
            option_type BYTE,           -- 1=CE, 2=PE (faster than string)
            strike INT,                 -- Strike price as integer (multiply by 100)
            tf BYTE,                    -- Timeframe code
            open DOUBLE,
            high DOUBLE,
            low DOUBLE,
            close DOUBLE,
            volume LONG,
            oi LONG,                    -- Open Interest
            -- Option Greeks (for future enhancement)
            iv DOUBLE,                  -- Implied Volatility
            delta DOUBLE,
            gamma DOUBLE,
            theta DOUBLE,
            vega DOUBLE,
            timestamp TIMESTAMP
        ) TIMESTAMP(timestamp) PARTITION BY DAY;
        """
    
    @staticmethod
    def get_index_schema(table_name: str) -> str:
        """Optimized schema for index tables"""
        return f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            tf BYTE,                    -- Timeframe code
            open DOUBLE,
            high DOUBLE,
            low DOUBLE,
            close DOUBLE,
            timestamp TIMESTAMP
        ) TIMESTAMP(timestamp) PARTITION BY DAY;
        """

class DynamicTableManager:
    """Manages dynamic table creation and optimization"""
    
    def __init__(self, questdb_client):
        self.questdb_client = questdb_client
        self.created_tables: Dict[str, TableInfo] = {}
        self.table_cache: Set[str] = set()
        
        # Performance optimization settings
        self.max_tables_per_batch = 50
        self.table_creation_semaphore = asyncio.Semaphore(10)
    
    async def initialize(self):
        """Initialize table manager and load existing tables"""
        await self._load_existing_tables()
        logger.info(f"Table manager initialized with {len(self.table_cache)} existing tables")
    
    async def get_or_create_table(self, symbol_info: SymbolInfo) -> str:
        """Get existing table or create new one for symbol"""
        
        table_name = self._generate_table_name(symbol_info)
        
        if table_name in self.table_cache:
            return table_name
        
        # Create table if it doesn't exist
        async with self.table_creation_semaphore:
            # Double-check after acquiring semaphore
            if table_name not in self.table_cache:
                await self._create_table(symbol_info, table_name)
                self.table_cache.add(table_name)
        
        return table_name
    
    def _extract_underlying_from_symbol(self, symbol: str, instrument_type: str) -> str:
        """
        Extract underlying symbol from option/future symbol.
        
        Format examples:
        - Option: "360ONE25NOV251000CE" -> "360ONE"
        - Option: "NIFTY28NOV2424000CE" -> "NIFTY"
        - Future: "NIFTY28NOV24FUT" -> "NIFTY"
        - Future: "RELIANCE31JAN25FUT" -> "RELIANCE"
        """
        import re
        
        if instrument_type in [InstrumentType.CALL_OPTION, InstrumentType.PUT_OPTION]:
            # Pattern: SYMBOL + DD + MMM + YY + STRIKE + CE/PE
            # Match: ([A-Z]+)(\d{2})([A-Z]{3})(\d{2})([\d.]+)(CE|PE)
            match = re.match(r"^([A-Z]+)(\d{2}[A-Z]{3}\d{2}[\d.]+)(CE|PE)$", symbol.upper())
            if match:
                return match.group(1)
        
        elif instrument_type == InstrumentType.FUTURES:
            # Pattern: SYMBOL + DD + MMM + YY + FUT
            # Match: ([A-Z]+)(\d{2}[A-Z]{3}\d{2})FUT
            match = re.match(r"^([A-Z]+)(\d{2}[A-Z]{3}\d{2})FUT$", symbol.upper())
            if match:
                return match.group(1)
        
        # If no pattern matches, return symbol as-is (fallback)
        return symbol
    
    def _generate_table_name(self, symbol_info: SymbolInfo) -> str:
        """Generate appropriate table name based on instrument type"""
        
        if symbol_info.instrument_type == InstrumentType.EQUITY:
            return TableNamingStrategy.get_equity_table_name(
                symbol_info.symbol, symbol_info.exchange
            )
        
        elif symbol_info.instrument_type == InstrumentType.FUTURES:
            underlying = self._extract_underlying_from_symbol(
                symbol_info.symbol, symbol_info.instrument_type
            )
            return TableNamingStrategy.get_futures_table_name(
                underlying, symbol_info.exchange
            )
        
        elif symbol_info.instrument_type in [InstrumentType.CALL_OPTION, InstrumentType.PUT_OPTION]:
            underlying = self._extract_underlying_from_symbol(
                symbol_info.symbol, symbol_info.instrument_type
            )
            # Extract expiry from symbol if not provided
            expiry = symbol_info.expiry
            if not expiry:
                # Try to extract from symbol: SYMBOL + DDMMMYY + STRIKE + CE/PE
                import re
                match = re.match(r"^[A-Z]+(\d{2}[A-Z]{3}\d{2})[\d.]+(CE|PE)$", symbol_info.symbol.upper())
                if match:
                    expiry_str = match.group(1)  # e.g., "25NOV25"
                    # Convert to DD-MMM-YY format for table naming
                    try:
                        day = expiry_str[:2]
                        month_map = {
                            'JAN': 'Jan', 'FEB': 'Feb', 'MAR': 'Mar', 'APR': 'Apr',
                            'MAY': 'May', 'JUN': 'Jun', 'JUL': 'Jul', 'AUG': 'Aug',
                            'SEP': 'Sep', 'OCT': 'Oct', 'NOV': 'Nov', 'DEC': 'Dec'
                        }
                        month = expiry_str[2:5]
                        year = expiry_str[5:7]
                        expiry = f"{day}-{month_map.get(month, 'Jan')}-{year}"
                    except:
                        expiry = "000000"
                else:
                    expiry = "000000"
            
            return TableNamingStrategy.get_options_table_name(
                underlying, symbol_info.exchange, expiry or "000000"
            )
        
        elif symbol_info.instrument_type == InstrumentType.INDEX:
            return TableNamingStrategy.get_index_table_name(
                symbol_info.symbol, symbol_info.exchange
            )
        
        else:
            # Fallback to equity naming
            return TableNamingStrategy.get_equity_table_name(
                symbol_info.symbol, symbol_info.exchange
            )
    
    async def _create_table(self, symbol_info: SymbolInfo, table_name: str):
        """Create optimized table for specific instrument type"""
        
        try:
            # Get appropriate enhanced schema with full analytics
            if symbol_info.instrument_type == InstrumentType.EQUITY:
                schema = EnhancedTableSchemas.get_enhanced_equity_schema(table_name)
            elif symbol_info.instrument_type == InstrumentType.FUTURES:
                schema = EnhancedTableSchemas.get_enhanced_futures_schema(table_name)
            elif symbol_info.instrument_type in [InstrumentType.CALL_OPTION, InstrumentType.PUT_OPTION]:
                schema = EnhancedTableSchemas.get_enhanced_options_schema(table_name)
            elif symbol_info.instrument_type == InstrumentType.INDEX:
                schema = EnhancedTableSchemas.get_enhanced_index_schema(table_name)
            else:
                schema = EnhancedTableSchemas.get_enhanced_equity_schema(table_name)
            
            # Create table
            async with self.questdb_client.pool.acquire() as conn:
                await conn.execute(schema)
            
            # Store table info
            table_info = TableInfo(
                table_name=table_name,
                symbol=symbol_info.symbol,
                instrument_type=symbol_info.instrument_type,
                exchange=symbol_info.exchange,
                created_at=datetime.now()
            )
            self.created_tables[table_name] = table_info
            
            logger.info(f"Created optimized table: {table_name} for {symbol_info.instrument_type}")
            
        except Exception as e:
            logger.error(f"Error creating table {table_name}: {e}")
            raise
    
    async def _load_existing_tables(self):
        """Load list of existing tables from QuestDB"""
        try:
            async with self.questdb_client.pool.acquire() as conn:
                # Query QuestDB system tables to get existing table names
                result = await conn.fetch("SELECT table_name FROM tables() WHERE table_name LIKE 'eq_%' OR table_name LIKE 'fut_%' OR table_name LIKE 'opt_%' OR table_name LIKE 'idx_%'")
                
                for row in result:
                    self.table_cache.add(row['table_name'])
                
        except Exception as e:
            logger.warning(f"Could not load existing tables: {e}")
    
    async def get_table_statistics(self) -> Dict[str, Dict]:
        """Get statistics for all managed tables"""
        
        stats = {}
        
        for table_name in self.table_cache:
            try:
                async with self.questdb_client.pool.acquire() as conn:
                    # Get record count
                    count_result = await conn.fetchval(f"SELECT COUNT(*) FROM {table_name}")
                    
                    # Get date range
                    date_result = await conn.fetchrow(
                        f"SELECT MIN(timestamp) as min_date, MAX(timestamp) as max_date FROM {table_name}"
                    )
                    
                    stats[table_name] = {
                        'record_count': count_result,
                        'min_date': date_result['min_date'] if date_result else None,
                        'max_date': date_result['max_date'] if date_result else None
                    }
                    
            except Exception as e:
                logger.warning(f"Error getting stats for table {table_name}: {e}")
                stats[table_name] = {'error': str(e)}
        
        return stats
    
    async def cleanup_old_tables(self, days_threshold: int = 365):
        """Clean up tables with no recent data"""
        
        cutoff_date = datetime.now() - timedelta(days=days_threshold)
        tables_to_drop = []
        
        for table_name in self.table_cache:
            try:
                async with self.questdb_client.pool.acquire() as conn:
                    max_date = await conn.fetchval(f"SELECT MAX(timestamp) FROM {table_name}")
                    
                    if max_date and max_date < cutoff_date:
                        tables_to_drop.append(table_name)
                        
            except Exception as e:
                logger.warning(f"Error checking table {table_name} for cleanup: {e}")
        
        # Drop old tables
        for table_name in tables_to_drop:
            try:
                async with self.questdb_client.pool.acquire() as conn:
                    await conn.execute(f"DROP TABLE {table_name}")
                
                self.table_cache.discard(table_name)
                self.created_tables.pop(table_name, None)
                
                logger.info(f"Dropped old table: {table_name}")
                
            except Exception as e:
                logger.error(f"Error dropping table {table_name}: {e}")
    
    def get_created_tables_summary(self) -> Dict[str, int]:
        """Get summary of created tables by instrument type"""
        
        summary = {}
        
        for table_info in self.created_tables.values():
            instrument_type = table_info.instrument_type.value
            summary[instrument_type] = summary.get(instrument_type, 0) + 1
        
        return summary

class OptionsTableOptimizer:
    """Specialized optimizer for options tables"""
    
    @staticmethod
    def should_create_separate_expiry_table(underlying: str, expiry: str, strike_count: int) -> bool:
        """Determine if options should have separate table per expiry"""
        
        # Create separate table if:
        # 1. More than 50 strikes for the expiry
        # 2. Major indices (NIFTY, BANKNIFTY)
        # 3. High-volume underlyings
        
        major_indices = {'NIFTY', 'BANKNIFTY', 'FINNIFTY'}
        high_volume_stocks = {'RELIANCE', 'TCS', 'INFY', 'HDFC', 'ICICIBANK'}
        
        if underlying in major_indices:
            return True
        
        if underlying in high_volume_stocks and strike_count > 20:
            return True
        
        if strike_count > 50:
            return True
        
        return False
    
    @staticmethod
    def get_strike_range_table_name(underlying: str, exchange: str, expiry: str, strike_range: Tuple[int, int]) -> str:
        """Generate table name for specific strike range"""
        
        clean_underlying = TableNamingStrategy.sanitize_symbol(underlying)
        clean_exchange = exchange.lower()
        
        try:
            expiry_date = datetime.strptime(expiry, "%d-%b-%y")
            expiry_compact = expiry_date.strftime("%y%m%d")
        except:
            expiry_compact = "000000"
        
        strike_min, strike_max = strike_range
        return f"opt_{clean_exchange}_{clean_underlying}_{expiry_compact}_{strike_min}_{strike_max}"

# Utility functions for table management
class TableMaintenanceUtils:
    """Utilities for table maintenance and optimization"""
    
    @staticmethod
    async def optimize_table_partitions(questdb_client, table_name: str):
        """Optimize table partitions for better performance"""
        
        try:
            async with questdb_client.pool.acquire() as conn:
                # QuestDB automatically optimizes partitions, but we can trigger compaction
                await conn.execute(f"ALTER TABLE {table_name} RESUME WAL")
                
        except Exception as e:
            logger.warning(f"Could not optimize partitions for {table_name}: {e}")
    
    @staticmethod
    async def get_table_size_info(questdb_client, table_name: str) -> Dict[str, any]:
        """Get detailed size information for a table"""
        
        try:
            async with questdb_client.pool.acquire() as conn:
                # Get record count and date range
                stats = await conn.fetchrow(f"""
                    SELECT 
                        COUNT(*) as record_count,
                        MIN(timestamp) as min_date,
                        MAX(timestamp) as max_date
                    FROM {table_name}
                """)
                
                return {
                    'record_count': stats['record_count'],
                    'min_date': stats['min_date'],
                    'max_date': stats['max_date'],
                    'days_span': (stats['max_date'] - stats['min_date']).days if stats['max_date'] and stats['min_date'] else 0
                }
                
        except Exception as e:
            logger.error(f"Error getting size info for {table_name}: {e}")
            return {'error': str(e)}
