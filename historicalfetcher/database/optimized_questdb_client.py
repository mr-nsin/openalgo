"""
Optimized QuestDB Client with Dynamic Table Management

Enhanced client that creates symbol-specific tables with optimized schemas
and numeric timeframe encoding for maximum performance.
"""

import asyncio
import asyncpg
import sys
import os
from typing import List, Dict, Any, Optional, Union, Tuple
from datetime import datetime, date, timedelta
import time

# Add the OpenAlgo root directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

# Import configuration - works with both direct script execution and package imports
try:
    from historicalfetcher.config.openalgo_settings import OpenAlgoSettings, TimeFrame
except ImportError:
    from ..config.openalgo_settings import OpenAlgoSettings, TimeFrame
from enum import Enum

class InstrumentType(str, Enum):
    """Instrument types for compatibility"""
    EQUITY = "EQ"
    FUTURES = "FUT"
    CALL_OPTION = "CE"
    PUT_OPTION = "PE"
    INDEX = "INDEX"
# Import models and utilities - works with both direct script execution and package imports
try:
    from historicalfetcher.models.data_models import SymbolInfo, HistoricalCandle, IndicatorResult, CalculationConfig, TimeFrameCode
    from historicalfetcher.database.table_manager import DynamicTableManager, TableNamingStrategy, OptionsTableOptimizer, TableMaintenanceUtils
    from historicalfetcher.database.enhanced_schemas import EnhancedTableSchemas
    from historicalfetcher.utils.async_logger import AsyncLogger, get_async_logger
except ImportError:
    from ..models.data_models import SymbolInfo, HistoricalCandle, IndicatorResult, CalculationConfig, TimeFrameCode
    from .table_manager import DynamicTableManager, TableNamingStrategy, OptionsTableOptimizer, TableMaintenanceUtils
    from .enhanced_schemas import EnhancedTableSchemas
    from ..utils.async_logger import AsyncLogger, get_async_logger

# Initialize async logger for module-level logging
_module_async_logger = get_async_logger()
logger = _module_async_logger.get_logger()

class OptimizedQuestDBClient:
    """
    Optimized QuestDB client with symbol-specific tables and numeric timeframes
    """
    
    def __init__(self, settings: OpenAlgoSettings, async_logger: Optional[AsyncLogger] = None):
        self.settings = settings
        self.pool: Optional[asyncpg.Pool] = None
        self._async_logger = async_logger
        
        # Table management
        self.table_manager: Optional[DynamicTableManager] = None
        
        # Indicator calculation engine (lazy loaded to avoid circular imports)
        self.indicator_engine = None
        self._indicator_config = CalculationConfig(
            enable_greeks=True,
            enable_parallel_processing=True,
            max_workers=4
        )
        
        # Performance optimization - increased batch sizes for better throughput
        self.batch_insert_size = getattr(settings, 'batch_size', 2000) if settings else 2000  # Use from settings
        self.connection_pool_size = 30  # Increased from 20
        self.preferred_batch_size = getattr(settings, 'batch_size', 5000) * 10 if settings else 5000  # 10x batch_size for large inserts
        
        # Connection pool optimization
        self.pool_min_size = 10
        self.pool_max_size = self.connection_pool_size
        
        # Statistics
        self.stats = {
            'total_inserts': 0,
            'successful_inserts': 0,
            'failed_inserts': 0,
            'total_records_inserted': 0,
            'tables_created': 0,
            'connection_errors': 0,
            'cache_hits': 0,
            'batch_operations': 0
        }
        
        # Cache for table existence checks (with TTL)
        self.table_existence_cache: Dict[str, Tuple[bool, float]] = {}  # (exists, timestamp)
        self.cache_ttl = 300  # 5 minutes
    
    def _get_indicator_engine(self):
        """Lazy load indicator engine to avoid circular imports"""
        if self.indicator_engine is None:
            try:
                from historicalfetcher.indicators.indicator_engine import IndicatorEngine
                self.indicator_engine = IndicatorEngine(self._indicator_config)
            except ImportError:
                logger.warning("IndicatorEngine not available, indicators will be disabled")
                self.indicator_engine = None
        return self.indicator_engine
    
    async def connect(self):
        """Initialize connection pool and table manager"""
        try:
            # QuestDB connection parameters
            # Note: QuestDB PostgreSQL wire protocol runs on port 8812 by default, not 9000
            questdb_port = 8812 if self.settings.questdb_port == 9000 else self.settings.questdb_port
            
            connection_params = {
                'host': self.settings.questdb_host,
                'port': questdb_port,
                'database': self.settings.questdb_database,
                'min_size': self.pool_min_size,
                'max_size': self.pool_max_size,
                'command_timeout': 180,  # Increased timeout for large operations
                'max_queries': 50000,  # Increase query cache
                # Remove server_settings for now to avoid the parameter error
                # 'server_settings': {
                #     'application_name': 'optimized_historical_fetcher',
                # }
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
            
            # Initialize table manager
            self.table_manager = DynamicTableManager(self)
            await self.table_manager.initialize()
            
            logger.info(f"Connected to QuestDB at {self.settings.questdb_host}:{self.settings.questdb_port}")
            
            if self._async_logger:
                await self._async_logger.log_database_operation(
                    operation='connect',
                    table_name='questdb',
                    records_count=0,
                    execution_time=0.0,
                    status='success'
                )
            
        except Exception as e:
            self.stats['connection_errors'] += 1
            logger.exception(f"Failed to connect to QuestDB: {e}")
            
            if self._async_logger:
                await self._async_logger.log_error_with_context(
                    error=e,
                    context={'operation': 'connect_questdb'},
                    operation='connect'
                )
            raise
    
    async def upsert_historical_data_with_indicators(
        self,
        symbol_info: SymbolInfo,
        timeframe: TimeFrame,
        candles: List[HistoricalCandle],
        spot_price: Optional[float] = None,
        market_depth_data: Optional[Dict] = None
    ) -> int:
        """
        Upsert historical data with calculated indicators using optimized symbol-specific tables
        """
        
        if not candles:
            return 0
        
        try:
            # Get or initialize indicator engine
            indicator_engine = self._get_indicator_engine()
            
            # Calculate indicators for the data (with fallback for missing engine)
            indicator_results = []
            if indicator_engine:
                try:
                    if symbol_info.instrument_type == InstrumentType.EQUITY:
                        indicator_results = await indicator_engine.calculate_equity_indicators(
                            symbol_info, candles, timeframe, market_depth_data
                        )
                    elif symbol_info.instrument_type == InstrumentType.FUTURES:
                        indicator_results = await indicator_engine.calculate_futures_indicators(
                            symbol_info, candles, timeframe, spot_price, market_depth_data
                        )
                    elif symbol_info.instrument_type in [InstrumentType.CALL_OPTION, InstrumentType.PUT_OPTION]:
                        if spot_price is None:
                            logger.warning(f"Spot price required for options {symbol_info.symbol}")
                            spot_price = 0.0
                        indicator_results = await indicator_engine.calculate_options_indicators(
                            symbol_info, candles, timeframe, spot_price, market_depth_data=market_depth_data
                        )
                    elif symbol_info.instrument_type == InstrumentType.INDEX:
                        indicator_results = await indicator_engine.calculate_equity_indicators(
                            symbol_info, candles, timeframe, market_depth_data
                        )
                    else:
                        logger.warning(f"Unsupported instrument type: {symbol_info.instrument_type}")
                        # Continue without indicators instead of returning 0
                except Exception as indicator_error:
                    logger.warning(f"Indicator calculation failed for {symbol_info.symbol}: {indicator_error}")
                    # Continue with basic data insertion without indicators
                    indicator_results = []
            else:
                logger.debug(f"Indicator engine not available, inserting basic data for {symbol_info.symbol}")
                # Create basic indicator results from candles
                indicator_results = []
                tf_string = TimeFrameCode.to_string(timeframe)
                for candle in candles:
                    from historicalfetcher.models.data_models import IndicatorResult
                    basic_result = IndicatorResult(
                        symbol=symbol_info.symbol,
                        timeframe=tf_string,
                        timestamp=candle.timestamp,
                        open=candle.open,
                        high=candle.high,
                        low=candle.low,
                        close=candle.close,
                        volume=candle.volume,
                        indicators={},
                        greeks={},
                        market_depth={},
                        derived_metrics={}
                    )
                    indicator_results.append(basic_result)
            
            # Get or create appropriate table
            table_name = await self.table_manager.get_or_create_table(symbol_info)
            
            # Insert data with indicators (enhanced schema)
            if indicator_results and len(indicator_results) > 0:
                return await self._insert_enhanced_data(table_name, symbol_info, indicator_results)
            else:
                # Fallback: insert basic OHLCV data without indicators
                return await self._insert_basic_data(table_name, symbol_info, timeframe, candles)
                
        except Exception as e:
            self.stats['failed_inserts'] += 1
            logger.error(f"Error upserting data for {symbol_info.symbol}: {e}")
            raise
    
    async def _insert_basic_data(
        self, 
        table_name: str, 
        symbol_info: SymbolInfo, 
        timeframe: TimeFrame, 
        candles: List[HistoricalCandle]
    ) -> int:
        """Insert basic OHLCV data without complex indicators"""
        
        if not candles:
            return 0
        
        try:
            # Convert timeframe to string for SYMBOL storage
            from historicalfetcher.models.data_models import TimeFrameCode
            tf_string = TimeFrameCode.to_string(timeframe)
            
            # Prepare basic insert data
            insert_data = []
            for candle in candles:
                data_tuple = (
                    tf_string,
                    float(candle.open),
                    float(candle.high),
                    float(candle.low),
                    float(candle.close),
                    int(candle.volume),
                    candle.timestamp
                )
                insert_data.append(data_tuple)
            
            # Basic insert query for OHLCV data
            insert_query = f"""
                INSERT INTO {table_name} (
                    tf, open, high, low, close, volume, timestamp
                ) VALUES ($1, $2, $3, $4, $5, $6, $7)
            """
            
            # Execute batch insert
            async with self.pool.acquire() as conn:
                await conn.executemany(insert_query, insert_data)
            
            # Update statistics
            self.stats['total_inserts'] += 1
            self.stats['successful_inserts'] += 1
            self.stats['total_records_inserted'] += len(candles)
            
            return len(candles)
            
        except Exception as e:
            self.stats['failed_inserts'] += 1
            logger.error(f"Error inserting basic data for {symbol_info.symbol}: {e}")
            raise
    
    async def _insert_enhanced_data(
        self,
        table_name: str,
        symbol_info: SymbolInfo,
        indicator_results: List[IndicatorResult]
    ) -> int:
        """Insert enhanced data with all indicators and analytics"""
        
        if not indicator_results:
            return 0
        
        try:
            # Convert timeframe to numeric code
            from historicalfetcher.models.data_models import TimeFrameCode
            
            # Prepare enhanced insert data based on instrument type
            if symbol_info.instrument_type == InstrumentType.EQUITY:
                return await self._insert_enhanced_equity_data(table_name, indicator_results)
            elif symbol_info.instrument_type == InstrumentType.FUTURES:
                return await self._insert_enhanced_futures_data(table_name, indicator_results)
            elif symbol_info.instrument_type in [InstrumentType.CALL_OPTION, InstrumentType.PUT_OPTION]:
                return await self._insert_enhanced_options_data(table_name, indicator_results)
            elif symbol_info.instrument_type == InstrumentType.INDEX:
                return await self._insert_enhanced_index_data(table_name, indicator_results)
            else:
                # Default to equity
                return await self._insert_enhanced_equity_data(table_name, indicator_results)
            
        except Exception as e:
            self.stats['failed_inserts'] += 1
            logger.error(f"Error inserting enhanced data for {symbol_info.symbol}: {e}")
            raise
    
    async def _insert_enhanced_equity_data(
        self,
        table_name: str,
        indicator_results: List[IndicatorResult]
    ) -> int:
        """Insert enhanced equity data with full analytics"""
        
        insert_data = []
        for result in indicator_results:
            # Build comprehensive data tuple for enhanced equity schema
            data_tuple = (
                # Basic OHLCV
                result.timeframe,
                float(result.open),
                float(result.high),
                float(result.low),
                float(result.close),
                int(result.volume),
                
                # Technical Indicators
                result.indicators.get('ema_9'),
                result.indicators.get('ema_21'),
                result.indicators.get('ema_50'),
                result.indicators.get('ema_200'),
                result.indicators.get('sma_20'),
                result.indicators.get('sma_50'),
                
                # Momentum Indicators
                result.indicators.get('rsi_14'),
                result.indicators.get('macd_line'),
                result.indicators.get('macd_signal'),
                result.indicators.get('macd_histogram'),
                result.indicators.get('stoch_k'),
                result.indicators.get('stoch_d'),
                
                # Volatility Indicators
                result.indicators.get('atr_14'),
                result.indicators.get('bb_upper'),
                result.indicators.get('bb_middle'),
                result.indicators.get('bb_lower'),
                result.indicators.get('bb_width'),
                result.indicators.get('bb_percent'),
                
                # Trend Following Indicators
                result.indicators.get('supertrend_7_3'),
                result.indicators.get('supertrend_signal_7_3'),
                result.indicators.get('supertrend_10_3'),
                result.indicators.get('supertrend_signal_10_3'),
                result.indicators.get('parabolic_sar'),
                
                # Volume Indicators
                result.indicators.get('volume_sma_20'),
                result.indicators.get('vwap'),
                result.indicators.get('obv'),
                
                # Price Action Indicators
                result.indicators.get('pivot_point'),
                result.indicators.get('resistance_1'),
                result.indicators.get('resistance_2'),
                result.indicators.get('resistance_3'),
                result.indicators.get('support_1'),
                result.indicators.get('support_2'),
                result.indicators.get('support_3'),
                
                # Market Microstructure (5 levels) - from market_depth if available
                result.market_depth.get('bid_1') if result.market_depth else None,
                result.market_depth.get('bid_qty_1') if result.market_depth else None,
                result.market_depth.get('bid_2') if result.market_depth else None,
                result.market_depth.get('bid_qty_2') if result.market_depth else None,
                result.market_depth.get('bid_3') if result.market_depth else None,
                result.market_depth.get('bid_qty_3') if result.market_depth else None,
                result.market_depth.get('bid_4') if result.market_depth else None,
                result.market_depth.get('bid_qty_4') if result.market_depth else None,
                result.market_depth.get('bid_5') if result.market_depth else None,
                result.market_depth.get('bid_qty_5') if result.market_depth else None,
                
                result.market_depth.get('ask_1') if result.market_depth else None,
                result.market_depth.get('ask_qty_1') if result.market_depth else None,
                result.market_depth.get('ask_2') if result.market_depth else None,
                result.market_depth.get('ask_qty_2') if result.market_depth else None,
                result.market_depth.get('ask_3') if result.market_depth else None,
                result.market_depth.get('ask_qty_3') if result.market_depth else None,
                result.market_depth.get('ask_4') if result.market_depth else None,
                result.market_depth.get('ask_qty_4') if result.market_depth else None,
                result.market_depth.get('ask_5') if result.market_depth else None,
                result.market_depth.get('ask_qty_5') if result.market_depth else None,
                
                # Derived Market Data
                result.market_depth.get('bid_ask_spread') if result.market_depth else None,
                result.market_depth.get('bid_ask_spread_pct') if result.market_depth else None,
                result.market_depth.get('mid_price') if result.market_depth else None,
                result.market_depth.get('total_bid_qty') if result.market_depth else None,
                result.market_depth.get('total_ask_qty') if result.market_depth else None,
                
                # Additional Analytics
                result.derived_metrics.get('price_change') if result.derived_metrics else None,
                result.derived_metrics.get('price_change_pct') if result.derived_metrics else None,
                result.derived_metrics.get('high_low_pct') if result.derived_metrics else None,
                
                result.timestamp
            )
            
            insert_data.append(data_tuple)
        
        # Comprehensive equity insert query
        insert_query = f"""
            INSERT INTO {table_name} (
                tf, open, high, low, close, volume,
                ema_9, ema_21, ema_50, ema_200, sma_20, sma_50,
                rsi_14, macd_line, macd_signal, macd_histogram, stoch_k, stoch_d,
                atr_14, bb_upper, bb_middle, bb_lower, bb_width, bb_percent,
                supertrend_7_3, supertrend_signal_7_3, supertrend_10_3, supertrend_signal_10_3, parabolic_sar,
                volume_sma_20, vwap, obv,
                pivot_point, resistance_1, resistance_2, resistance_3, support_1, support_2, support_3,
                bid_1, bid_qty_1, bid_2, bid_qty_2, bid_3, bid_qty_3, bid_4, bid_qty_4, bid_5, bid_qty_5,
                ask_1, ask_qty_1, ask_2, ask_qty_2, ask_3, ask_qty_3, ask_4, ask_qty_4, ask_5, ask_qty_5,
                bid_ask_spread, bid_ask_spread_pct, mid_price, total_bid_qty, total_ask_qty,
                price_change, price_change_pct, high_low_pct,
                timestamp
            ) VALUES (
                $1, $2, $3, $4, $5, $6,
                $7, $8, $9, $10, $11, $12,
                $13, $14, $15, $16, $17, $18,
                $19, $20, $21, $22, $23, $24,
                $25, $26, $27, $28, $29,
                $30, $31, $32,
                $33, $34, $35, $36, $37, $38, $39,
                $40, $41, $42, $43, $44, $45, $46, $47, $48, $49,
                $50, $51, $52, $53, $54, $55, $56, $57, $58, $59,
                $60, $61, $62, $63, $64,
                $65, $66, $67,
                $68
            )
        """
        
        # Execute batch insert
        async with self.pool.acquire() as conn:
            await conn.executemany(insert_query, insert_data)
        
        # Update statistics
        self.stats['total_inserts'] += 1
        self.stats['successful_inserts'] += 1
        self.stats['total_records_inserted'] += len(indicator_results)
        
        return len(indicator_results)
    
    async def _insert_enhanced_futures_data(
        self,
        table_name: str,
        indicator_results: List[IndicatorResult]
    ) -> int:
        """Insert enhanced futures data - simplified for now"""
        # For now, use basic insertion for futures until full implementation
        return await self._insert_enhanced_equity_data(table_name, indicator_results)
    
    async def _insert_enhanced_options_data(
        self,
        table_name: str,
        indicator_results: List[IndicatorResult]
    ) -> int:
        """Insert enhanced options data - simplified for now"""
        # For now, use basic insertion for options until full implementation
        return await self._insert_enhanced_equity_data(table_name, indicator_results)
    
    async def _insert_enhanced_index_data(
        self,
        table_name: str,
        indicator_results: List[IndicatorResult]
    ) -> int:
        """Insert enhanced index data - simplified for now"""
        # For now, use equity insertion for indices
        return await self._insert_enhanced_equity_data(table_name, indicator_results)
    
    async def _insert_enhanced_equity_data(
        self,
        table_name: str,
        indicator_results: List[IndicatorResult]
    ) -> int:
        """Insert enhanced equity data with all indicators"""
        
        insert_data = []
        
        for result in indicator_results:
            # Build comprehensive data tuple
            data_tuple = (
                # Basic OHLCV
                result.timeframe,
                result.open, result.high, result.low, result.close, result.volume,
                
                # Technical Indicators
                result.indicators.get('ema_9'),
                result.indicators.get('ema_21'),
                result.indicators.get('ema_50'),
                result.indicators.get('ema_200'),
                result.indicators.get('sma_20'),
                result.indicators.get('sma_50'),
                
                # Momentum Indicators
                result.indicators.get('rsi_14'),
                result.indicators.get('macd_line'),
                result.indicators.get('macd_signal'),
                result.indicators.get('macd_histogram'),
                result.indicators.get('stoch_k'),
                result.indicators.get('stoch_d'),
                
                # Volatility Indicators
                result.indicators.get('atr_14'),
                result.indicators.get('bb_upper'),
                result.indicators.get('bb_middle'),
                result.indicators.get('bb_lower'),
                result.indicators.get('bb_width'),
                result.indicators.get('bb_percent'),
                
                # Trend Following
                result.indicators.get('supertrend_7_3'),
                result.indicators.get('supertrend_signal_7_3'),
                result.indicators.get('supertrend_10_3'),
                result.indicators.get('supertrend_signal_10_3'),
                result.indicators.get('parabolic_sar'),
                
                # Volume Indicators
                result.indicators.get('volume_sma_20'),
                result.indicators.get('vwap'),
                result.indicators.get('obv'),
                
                # Support/Resistance
                result.derived_metrics.get('pivot_point') if result.derived_metrics else None,
                result.derived_metrics.get('resistance_1') if result.derived_metrics else None,
                result.derived_metrics.get('resistance_2') if result.derived_metrics else None,
                result.derived_metrics.get('resistance_3') if result.derived_metrics else None,
                result.derived_metrics.get('support_1') if result.derived_metrics else None,
                result.derived_metrics.get('support_2') if result.derived_metrics else None,
                result.derived_metrics.get('support_3') if result.derived_metrics else None,
                
                # Market Depth (5 levels)
                result.market_depth.get('bid_1') if result.market_depth else None,
                result.market_depth.get('bid_qty_1') if result.market_depth else None,
                result.market_depth.get('bid_2') if result.market_depth else None,
                result.market_depth.get('bid_qty_2') if result.market_depth else None,
                result.market_depth.get('bid_3') if result.market_depth else None,
                result.market_depth.get('bid_qty_3') if result.market_depth else None,
                result.market_depth.get('bid_4') if result.market_depth else None,
                result.market_depth.get('bid_qty_4') if result.market_depth else None,
                result.market_depth.get('bid_5') if result.market_depth else None,
                result.market_depth.get('bid_qty_5') if result.market_depth else None,
                
                result.market_depth.get('ask_1') if result.market_depth else None,
                result.market_depth.get('ask_qty_1') if result.market_depth else None,
                result.market_depth.get('ask_2') if result.market_depth else None,
                result.market_depth.get('ask_qty_2') if result.market_depth else None,
                result.market_depth.get('ask_3') if result.market_depth else None,
                result.market_depth.get('ask_qty_3') if result.market_depth else None,
                result.market_depth.get('ask_4') if result.market_depth else None,
                result.market_depth.get('ask_qty_4') if result.market_depth else None,
                result.market_depth.get('ask_5') if result.market_depth else None,
                result.market_depth.get('ask_qty_5') if result.market_depth else None,
                
                # Derived Market Data
                result.market_depth.get('bid_ask_spread') if result.market_depth else None,
                result.market_depth.get('bid_ask_spread_pct') if result.market_depth else None,
                result.market_depth.get('mid_price') if result.market_depth else None,
                result.market_depth.get('total_bid_qty') if result.market_depth else None,
                result.market_depth.get('total_ask_qty') if result.market_depth else None,
                
                # Additional Analytics
                result.derived_metrics.get('price_change') if result.derived_metrics else None,
                result.derived_metrics.get('price_change_pct') if result.derived_metrics else None,
                result.derived_metrics.get('high_low_pct') if result.derived_metrics else None,
                
                result.timestamp
            )
            
            insert_data.append(data_tuple)
        
        # Build comprehensive insert query
        insert_query = f"""
            INSERT INTO {table_name} (
                tf, open, high, low, close, volume,
                ema_9, ema_21, ema_50, ema_200, sma_20, sma_50,
                rsi_14, macd_line, macd_signal, macd_histogram, stoch_k, stoch_d,
                atr_14, bb_upper, bb_middle, bb_lower, bb_width, bb_percent,
                supertrend_7_3, supertrend_signal_7_3, supertrend_10_3, supertrend_signal_10_3, parabolic_sar,
                volume_sma_20, vwap, obv,
                pivot_point, resistance_1, resistance_2, resistance_3, support_1, support_2, support_3,
                bid_1, bid_qty_1, bid_2, bid_qty_2, bid_3, bid_qty_3, bid_4, bid_qty_4, bid_5, bid_qty_5,
                ask_1, ask_qty_1, ask_2, ask_qty_2, ask_3, ask_qty_3, ask_4, ask_qty_4, ask_5, ask_qty_5,
                bid_ask_spread, bid_ask_spread_pct, mid_price, total_bid_qty, total_ask_qty,
                price_change, price_change_pct, high_low_pct,
                timestamp
            ) VALUES (
                $1, $2, $3, $4, $5, $6,
                $7, $8, $9, $10, $11, $12,
                $13, $14, $15, $16, $17, $18,
                $19, $20, $21, $22, $23, $24,
                $25, $26, $27, $28, $29,
                $30, $31, $32,
                $33, $34, $35, $36, $37, $38, $39,
                $40, $41, $42, $43, $44, $45, $46, $47, $48, $49,
                $50, $51, $52, $53, $54, $55, $56, $57, $58, $59,
                $60, $61, $62, $63, $64,
                $65, $66, $67,
                $68
            )
        """
        
        return await self._execute_batch_insert(insert_query, insert_data, table_name)
    
    async def _insert_enhanced_options_data(
        self,
        table_name: str,
        indicator_results: List[IndicatorResult]
    ) -> int:
        """Insert enhanced options data with Greeks and indicators"""
        
        insert_data = []
        
        for result in indicator_results:
            # Extract options-specific data
            contract_token = result.symbol  # Use symbol as contract token for now
            option_type = 1 if 'CE' in result.symbol else 2  # 1=CE, 2=PE
            
            # Extract strike from symbol (this would need proper parsing)
            strike = 0  # This should be extracted from symbol_info
            
            data_tuple = (
                # Contract Information
                contract_token, option_type, strike, result.timeframe,
                
                # Basic OHLCV + OI
                result.open, result.high, result.low, result.close, result.volume, 0,  # OI placeholder
                
                # Options Greeks
                result.greeks.get('delta') if result.greeks else None,
                result.greeks.get('gamma') if result.greeks else None,
                result.greeks.get('theta') if result.greeks else None,
                result.greeks.get('vega') if result.greeks else None,
                result.greeks.get('rho') if result.greeks else None,
                
                # Volatility Metrics
                result.greeks.get('implied_volatility') if result.greeks else None,
                None,  # historical_volatility placeholder
                None,  # iv_rank placeholder
                None,  # iv_percentile placeholder
                
                # Value Components
                result.greeks.get('intrinsic_value') if result.greeks else None,
                result.greeks.get('time_value') if result.greeks else None,
                result.greeks.get('moneyness') if result.greeks else None,
                
                # Advanced Greeks
                result.greeks.get('lambda_greek') if result.greeks else None,
                None,  # epsilon placeholder
                None,  # vera placeholder
                
                # Risk Metrics
                result.greeks.get('probability_itm') if result.greeks else None,
                None,  # probability_profit placeholder
                None,  # max_pain placeholder
                
                # Technical Indicators (subset for options)
                result.indicators.get('rsi_14'),
                result.indicators.get('ema_9'),
                result.indicators.get('ema_21'),
                result.indicators.get('atr_14'),
                result.indicators.get('bb_upper'),
                result.indicators.get('bb_lower'),
                
                # Market Depth (5 levels) - same as equity
                result.market_depth.get('bid_1') if result.market_depth else None,
                result.market_depth.get('bid_qty_1') if result.market_depth else None,
                result.market_depth.get('bid_2') if result.market_depth else None,
                result.market_depth.get('bid_qty_2') if result.market_depth else None,
                result.market_depth.get('bid_3') if result.market_depth else None,
                result.market_depth.get('bid_qty_3') if result.market_depth else None,
                result.market_depth.get('bid_4') if result.market_depth else None,
                result.market_depth.get('bid_qty_4') if result.market_depth else None,
                result.market_depth.get('bid_5') if result.market_depth else None,
                result.market_depth.get('bid_qty_5') if result.market_depth else None,
                
                result.market_depth.get('ask_1') if result.market_depth else None,
                result.market_depth.get('ask_qty_1') if result.market_depth else None,
                result.market_depth.get('ask_2') if result.market_depth else None,
                result.market_depth.get('ask_qty_2') if result.market_depth else None,
                result.market_depth.get('ask_3') if result.market_depth else None,
                result.market_depth.get('ask_qty_3') if result.market_depth else None,
                result.market_depth.get('ask_4') if result.market_depth else None,
                result.market_depth.get('ask_qty_4') if result.market_depth else None,
                result.market_depth.get('ask_5') if result.market_depth else None,
                result.market_depth.get('ask_qty_5') if result.market_depth else None,
                
                # Derived Market Data
                result.market_depth.get('bid_ask_spread') if result.market_depth else None,
                result.market_depth.get('bid_ask_spread_pct') if result.market_depth else None,
                result.market_depth.get('mid_price') if result.market_depth else None,
                result.market_depth.get('total_bid_qty') if result.market_depth else None,
                result.market_depth.get('total_ask_qty') if result.market_depth else None,
                
                # Options-Specific Market Data
                None,  # put_call_ratio placeholder
                None,  # max_pain_distance placeholder
                None,  # skew placeholder
                
                # Change Metrics
                result.derived_metrics.get('price_change') if result.derived_metrics else None,
                result.derived_metrics.get('price_change_pct') if result.derived_metrics else None,
                None,  # oi_change placeholder
                None,  # oi_change_pct placeholder
                None,  # iv_change placeholder
                None,  # delta_change placeholder
                
                result.timestamp
            )
            
            insert_data.append(data_tuple)
        
        # Comprehensive options insert query (simplified for now)
        insert_query = f"""
            INSERT INTO {table_name} (
                contract_token, option_type, strike, tf,
                open, high, low, close, volume, oi,
                delta, gamma, theta, vega, rho,
                implied_volatility, historical_volatility, iv_rank, iv_percentile,
                intrinsic_value, time_value, moneyness,
                lambda_greek, epsilon, vera,
                probability_itm, probability_profit, max_pain,
                rsi_14, ema_9, ema_21, atr_14, bb_upper, bb_lower,
                bid_1, bid_qty_1, bid_2, bid_qty_2, bid_3, bid_qty_3, bid_4, bid_qty_4, bid_5, bid_qty_5,
                ask_1, ask_qty_1, ask_2, ask_qty_2, ask_3, ask_qty_3, ask_4, ask_qty_4, ask_5, ask_qty_5,
                bid_ask_spread, bid_ask_spread_pct, mid_price, total_bid_qty, total_ask_qty,
                put_call_ratio, max_pain_distance, skew,
                price_change, price_change_pct, oi_change, oi_change_pct, iv_change, delta_change,
                timestamp
            ) VALUES (
                $1, $2, $3, $4,
                $5, $6, $7, $8, $9, $10,
                $11, $12, $13, $14, $15,
                $16, $17, $18, $19,
                $20, $21, $22,
                $23, $24, $25,
                $26, $27, $28,
                $29, $30, $31, $32, $33, $34,
                $35, $36, $37, $38, $39, $40, $41, $42, $43, $44,
                $45, $46, $47, $48, $49, $50, $51, $52, $53, $54,
                $55, $56, $57, $58, $59,
                $60, $61, $62,
                $63, $64, $65, $66, $67, $68,
                $69
            )
        """
        
        return await self._execute_batch_insert(insert_query, insert_data, table_name)
    
    async def _insert_enhanced_futures_data(self, table_name: str, indicator_results: List[IndicatorResult]) -> int:
        """Insert enhanced futures data - placeholder implementation"""
        # Similar to equity but with futures-specific fields
        return await self._insert_enhanced_equity_data(table_name, indicator_results)
    
    async def _get_table_columns(self, table_name: str) -> set:
        """Get list of columns that exist in the table by trying to query it"""
        try:
            async with self.pool.acquire() as conn:
                # Try to get one row to see what columns exist
                # Even if table is empty, this will return column names
                try:
                    result = await conn.fetchrow(f"SELECT * FROM {table_name} LIMIT 1")
                    if result:
                        columns = set(result.keys())
                        return columns
                except Exception as query_error:
                    # If table is empty or doesn't exist, try to check if it exists
                    # by attempting a count query
                    try:
                        await conn.fetchval(f"SELECT COUNT(*) FROM {table_name}")
                        # Table exists but is empty - try to get schema another way
                        # For empty tables, we'll need to check the actual schema
                        # QuestDB doesn't have easy DESCRIBE, so we'll use a fallback
                        return set()  # Return empty to use all columns (table will be created with new schema)
                    except:
                        # Table doesn't exist
                        return set()
        except Exception as e:
            logger.debug(f"Could not get columns for table {table_name}: {e}")
            # Return empty set - will use all columns for new table
            return set()
    
    async def _insert_enhanced_index_data(self, table_name: str, indicator_results: List[IndicatorResult]) -> int:
        """Insert enhanced index data - dynamically builds query based on existing table columns"""
        
        if not indicator_results:
            return 0
        
        from historicalfetcher.models.data_models import TimeFrameCode
        
        # Get existing table columns
        existing_columns = await self._get_table_columns(table_name)
        
        # If table doesn't exist or has no columns, it will be created with new schema
        # For now, build query based on what columns exist
        # Define all possible columns in order
        all_columns = [
            'tf', 'open', 'high', 'low', 'close',
            'ema_9', 'ema_21', 'ema_50', 'ema_200', 'sma_20', 'sma_50', 'sma_100',
            'rsi_14', 'macd_line', 'macd_signal', 'macd_histogram', 'stoch_k', 'stoch_d',
            'atr_14', 'bb_upper', 'bb_middle', 'bb_lower', 'bb_width', 'bb_percent',
            'supertrend_7_3', 'supertrend_signal_7_3', 'supertrend_10_3', 'supertrend_signal_10_3', 'parabolic_sar',
            'pivot_point', 'resistance_1', 'resistance_2', 'resistance_3', 'support_1', 'support_2', 'support_3',
            'bid_1', 'bid_qty_1', 'bid_2', 'bid_qty_2', 'bid_3', 'bid_qty_3', 'bid_4', 'bid_qty_4', 'bid_5', 'bid_qty_5',
            'ask_1', 'ask_qty_1', 'ask_2', 'ask_qty_2', 'ask_3', 'ask_qty_3', 'ask_4', 'ask_qty_4', 'ask_5', 'ask_qty_5',
            'bid_ask_spread', 'bid_ask_spread_pct', 'mid_price', 'total_bid_qty', 'total_ask_qty',
            'price_change', 'price_change_pct', 'high_low_pct',
            'advance_decline_ratio', 'high_low_index', 'mcclellan_oscillator',
            'realized_volatility', 'garch_volatility',
            'timestamp'
        ]
        
        # Filter to only columns that exist in the table (or if table is new, use all)
        if existing_columns:
            # Table exists - only use columns that exist
            columns_to_use = [col for col in all_columns if col in existing_columns]
            if not columns_to_use:
                # Fallback: use basic columns if nothing matches
                columns_to_use = ['tf', 'open', 'high', 'low', 'close', 'timestamp']
        else:
            # Table doesn't exist or is new - use all columns (will be created with new schema)
            columns_to_use = all_columns
        
        insert_data = []
        
        for result in indicator_results:
            # Convert timeframe to string for SYMBOL storage
            tf_string = TimeFrameCode.to_string(result.timeframe)
            
            # Build data tuple dynamically based on columns_to_use
            data_values = []
            column_mapping = {
                'tf': tf_string,
                'open': result.open,
                'high': result.high,
                'low': result.low,
                'close': result.close,
                'ema_9': result.indicators.get('ema_9'),
                'ema_21': result.indicators.get('ema_21'),
                'ema_50': result.indicators.get('ema_50'),
                'ema_200': result.indicators.get('ema_200'),
                'sma_20': result.indicators.get('sma_20'),
                'sma_50': result.indicators.get('sma_50'),
                'sma_100': result.indicators.get('sma_100'),
                'rsi_14': result.indicators.get('rsi_14'),
                'macd_line': result.indicators.get('macd_line'),
                'macd_signal': result.indicators.get('macd_signal'),
                'macd_histogram': result.indicators.get('macd_histogram'),
                'stoch_k': result.indicators.get('stoch_k'),
                'stoch_d': result.indicators.get('stoch_d'),
                'atr_14': result.indicators.get('atr_14'),
                'bb_upper': result.indicators.get('bb_upper'),
                'bb_middle': result.indicators.get('bb_middle'),
                'bb_lower': result.indicators.get('bb_lower'),
                'bb_width': result.indicators.get('bb_width'),
                'bb_percent': result.indicators.get('bb_percent'),
                'supertrend_7_3': result.indicators.get('supertrend_7_3'),
                'supertrend_signal_7_3': result.indicators.get('supertrend_signal_7_3'),
                'supertrend_10_3': result.indicators.get('supertrend_10_3'),
                'supertrend_signal_10_3': result.indicators.get('supertrend_signal_10_3'),
                'parabolic_sar': result.indicators.get('parabolic_sar'),
                'pivot_point': result.derived_metrics.get('pivot_point') if result.derived_metrics else None,
                'resistance_1': result.derived_metrics.get('resistance_1') if result.derived_metrics else None,
                'resistance_2': result.derived_metrics.get('resistance_2') if result.derived_metrics else None,
                'resistance_3': result.derived_metrics.get('resistance_3') if result.derived_metrics else None,
                'support_1': result.derived_metrics.get('support_1') if result.derived_metrics else None,
                'support_2': result.derived_metrics.get('support_2') if result.derived_metrics else None,
                'support_3': result.derived_metrics.get('support_3') if result.derived_metrics else None,
                'bid_1': result.market_depth.get('bid_1') if result.market_depth else None,
                'bid_qty_1': result.market_depth.get('bid_qty_1') if result.market_depth else None,
                'bid_2': result.market_depth.get('bid_2') if result.market_depth else None,
                'bid_qty_2': result.market_depth.get('bid_qty_2') if result.market_depth else None,
                'bid_3': result.market_depth.get('bid_3') if result.market_depth else None,
                'bid_qty_3': result.market_depth.get('bid_qty_3') if result.market_depth else None,
                'bid_4': result.market_depth.get('bid_4') if result.market_depth else None,
                'bid_qty_4': result.market_depth.get('bid_qty_4') if result.market_depth else None,
                'bid_5': result.market_depth.get('bid_5') if result.market_depth else None,
                'bid_qty_5': result.market_depth.get('bid_qty_5') if result.market_depth else None,
                'ask_1': result.market_depth.get('ask_1') if result.market_depth else None,
                'ask_qty_1': result.market_depth.get('ask_qty_1') if result.market_depth else None,
                'ask_2': result.market_depth.get('ask_2') if result.market_depth else None,
                'ask_qty_2': result.market_depth.get('ask_qty_2') if result.market_depth else None,
                'ask_3': result.market_depth.get('ask_3') if result.market_depth else None,
                'ask_qty_3': result.market_depth.get('ask_qty_3') if result.market_depth else None,
                'ask_4': result.market_depth.get('ask_4') if result.market_depth else None,
                'ask_qty_4': result.market_depth.get('ask_qty_4') if result.market_depth else None,
                'ask_5': result.market_depth.get('ask_5') if result.market_depth else None,
                'ask_qty_5': result.market_depth.get('ask_qty_5') if result.market_depth else None,
                'bid_ask_spread': result.market_depth.get('bid_ask_spread') if result.market_depth else None,
                'bid_ask_spread_pct': result.market_depth.get('bid_ask_spread_pct') if result.market_depth else None,
                'mid_price': result.market_depth.get('mid_price') if result.market_depth else None,
                'total_bid_qty': result.market_depth.get('total_bid_qty') if result.market_depth else None,
                'total_ask_qty': result.market_depth.get('total_ask_qty') if result.market_depth else None,
                'price_change': result.derived_metrics.get('price_change') if result.derived_metrics else None,
                'price_change_pct': result.derived_metrics.get('price_change_pct') if result.derived_metrics else None,
                'high_low_pct': result.derived_metrics.get('high_low_pct') if result.derived_metrics else None,
                'advance_decline_ratio': result.indicators.get('advance_decline_ratio'),
                'high_low_index': result.indicators.get('high_low_index'),
                'mcclellan_oscillator': result.indicators.get('mcclellan_oscillator'),
                'realized_volatility': result.indicators.get('realized_volatility'),
                'garch_volatility': result.indicators.get('garch_volatility'),
                'timestamp': result.timestamp
            }
            
            # Build tuple with only columns that exist in table
            data_tuple = tuple(column_mapping[col] for col in columns_to_use)
            insert_data.append(data_tuple)
        
        # Build insert query dynamically based on columns_to_use
        columns_str = ', '.join(columns_to_use)
        placeholders = ', '.join([f'${i+1}' for i in range(len(columns_to_use))])
        
        insert_query = f"""
            INSERT INTO {table_name} ({columns_str})
            VALUES ({placeholders})
        """
        
        try:
            return await self._execute_batch_insert(insert_query, insert_data, table_name)
        except Exception as e:
            error_str = str(e)
            if "Invalid column" in error_str or "column" in error_str.lower():
                # Table has old schema - get actual columns and retry with only those
                logger.warning(f"Table {table_name} has old schema (missing columns), detecting actual columns...")
                
                # Try to get actual columns by querying the table
                actual_columns = await self._get_table_columns(table_name)
                
                if actual_columns:
                    # Use only columns that actually exist
                    safe_columns = [col for col in all_columns if col in actual_columns]
                    if not safe_columns:
                        # Fallback to basic columns
                        safe_columns = ['tf', 'open', 'high', 'low', 'close', 'timestamp']
                else:
                    # Can't detect columns, use basic set
                    safe_columns = ['tf', 'open', 'high', 'low', 'close', 'timestamp']
                
                logger.info(f"Retrying insert for {table_name} with columns: {safe_columns}")
                
                # Rebuild data with only safe columns
                safe_insert_data = []
                for result in indicator_results:
                    tf_string = TimeFrameCode.to_string(result.timeframe)
                    # Rebuild mapping for this result
                    safe_mapping = {
                        'tf': tf_string,
                        'open': result.open,
                        'high': result.high,
                        'low': result.low,
                        'close': result.close,
                        'timestamp': result.timestamp
                    }
                    # Add other columns if they exist in safe_columns
                    for col in safe_columns:
                        if col not in safe_mapping:
                            if col.startswith('ema_') or col.startswith('sma_') or col in ['rsi_14', 'macd_line', 'macd_signal', 'macd_histogram', 'stoch_k', 'stoch_d', 'atr_14', 'bb_upper', 'bb_middle', 'bb_lower', 'bb_width', 'bb_percent', 'supertrend_7_3', 'supertrend_signal_7_3', 'supertrend_10_3', 'supertrend_signal_10_3', 'parabolic_sar', 'advance_decline_ratio', 'high_low_index', 'mcclellan_oscillator', 'realized_volatility', 'garch_volatility']:
                                safe_mapping[col] = result.indicators.get(col)
                            elif col in ['pivot_point', 'resistance_1', 'resistance_2', 'resistance_3', 'support_1', 'support_2', 'support_3', 'price_change', 'price_change_pct', 'high_low_pct']:
                                safe_mapping[col] = result.derived_metrics.get(col) if result.derived_metrics else None
                            elif col.startswith('bid_') or col.startswith('ask_') or col in ['bid_ask_spread', 'bid_ask_spread_pct', 'mid_price', 'total_bid_qty', 'total_ask_qty']:
                                safe_mapping[col] = result.market_depth.get(col) if result.market_depth else None
                    
                    safe_tuple = tuple(safe_mapping[col] for col in safe_columns)
                    safe_insert_data.append(safe_tuple)
                
                safe_columns_str = ', '.join(safe_columns)
                safe_placeholders = ', '.join([f'${i+1}' for i in range(len(safe_columns))])
                safe_insert_query = f"""
                    INSERT INTO {table_name} ({safe_columns_str})
                    VALUES ({safe_placeholders})
                """
                
                return await self._execute_batch_insert(safe_insert_query, safe_insert_data, table_name)
            else:
                # Re-raise if it's not a column error
                raise
    
    async def _insert_equity_data(
        self,
        table_name: str,
        symbol_info: SymbolInfo,
        tf_code: TimeFrameCode,
        candles: List[HistoricalCandle]
    ) -> int:
        """Insert equity data with optimized schema"""
        
        from historicalfetcher.models.data_models import TimeFrameCode
        tf_string = TimeFrameCode.to_string(tf_code)
        
        # Prepare batch data
        insert_data = []
        for candle in candles:
            insert_data.append((
                tf_string,              # String timeframe
                candle.open,
                candle.high,
                candle.low,
                candle.close,
                candle.volume,
                candle.timestamp
            ))
        
        insert_query = f"""
            INSERT INTO {table_name} (tf, open, high, low, close, volume, timestamp)
            VALUES ($1, $2, $3, $4, $5, $6, $7)
        """
        
        return await self._execute_batch_insert(insert_query, insert_data, symbol_info.symbol)
    
    async def _insert_futures_data(
        self,
        table_name: str,
        symbol_info: SymbolInfo,
        tf_code: TimeFrameCode,
        candles: List[HistoricalCandle]
    ) -> int:
        """Insert futures data with contract tracking"""
        
        from historicalfetcher.models.data_models import TimeFrameCode
        tf_string = TimeFrameCode.to_string(tf_code)
        
        expiry_date = self._parse_expiry_date(symbol_info.expiry) if symbol_info.expiry else None
        
        insert_data = []
        for candle in candles:
            insert_data.append((
                symbol_info.instrument_token,  # Contract-specific token
                expiry_date,
                tf_string,
                candle.open,
                candle.high,
                candle.low,
                candle.close,
                candle.volume,
                candle.oi,
                candle.timestamp
            ))
        
        insert_query = f"""
            INSERT INTO {table_name} (contract_token, expiry_date, tf, open, high, low, close, volume, oi, timestamp)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
        """
        
        return await self._execute_batch_insert(insert_query, insert_data, symbol_info.symbol)
    
    async def _insert_options_data(
        self,
        table_name: str,
        symbol_info: SymbolInfo,
        tf_code: TimeFrameCode,
        candles: List[HistoricalCandle]
    ) -> int:
        """Insert options data with optimized encoding"""
        
        from historicalfetcher.models.data_models import TimeFrameCode
        tf_string = TimeFrameCode.to_string(tf_code)
        
        # Convert option type to numeric (1=CE, 2=PE)
        option_type_code = 1 if symbol_info.instrument_type == InstrumentType.CALL_OPTION else 2
        
        # Convert strike to integer (multiply by 100 for precision)
        strike_int = int((symbol_info.strike or 0) * 100)
        
        insert_data = []
        for candle in candles:
            insert_data.append((
                symbol_info.instrument_token,  # Contract-specific token
                option_type_code,              # Numeric option type
                strike_int,                    # Integer strike
                tf_string,
                candle.open,
                candle.high,
                candle.low,
                candle.close,
                candle.volume,
                candle.oi,
                None,  # iv (implied volatility) - placeholder
                None,  # delta - placeholder
                None,  # gamma - placeholder
                None,  # theta - placeholder
                None,  # vega - placeholder
                candle.timestamp
            ))
        
        insert_query = f"""
            INSERT INTO {table_name} (contract_token, option_type, strike, tf, open, high, low, close, volume, oi, iv, delta, gamma, theta, vega, timestamp)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16)
        """
        
        return await self._execute_batch_insert(insert_query, insert_data, symbol_info.symbol)
    
    async def _insert_index_data(
        self,
        table_name: str,
        symbol_info: SymbolInfo,
        tf_code: TimeFrameCode,
        candles: List[HistoricalCandle]
    ) -> int:
        """Insert index data (no volume/OI)"""
        
        from historicalfetcher.models.data_models import TimeFrameCode
        tf_string = TimeFrameCode.to_string(tf_code)
        
        insert_data = []
        for candle in candles:
            insert_data.append((
                tf_string,
                candle.open,
                candle.high,
                candle.low,
                candle.close,
                candle.timestamp
            ))
        
        insert_query = f"""
            INSERT INTO {table_name} (tf, open, high, low, close, timestamp)
            VALUES ($1, $2, $3, $4, $5, $6)
        """
        
        return await self._execute_batch_insert(insert_query, insert_data, symbol_info.symbol)
    
    async def _execute_batch_insert(
        self,
        query: str,
        data: List[tuple],
        symbol: str
    ) -> int:
        """Execute optimized batch insert with chunking"""
        
        if not self.pool:
            raise Exception("Not connected to QuestDB")
        
        start_time = time.monotonic()
        total_inserted = 0
        
        try:
            # Process in chunks for better memory management
            chunk_size = self.batch_insert_size
            
            for i in range(0, len(data), chunk_size):
                chunk = data[i:i + chunk_size]
                
                async with self.pool.acquire() as conn:
                    # Use executemany for batch insert
                    await conn.executemany(query, chunk)
                    total_inserted += len(chunk)
            
            # Update statistics
            self.stats['successful_inserts'] += 1
            self.stats['total_records_inserted'] += total_inserted
            
            execution_time = time.monotonic() - start_time
            logger.debug(
                f"Inserted {total_inserted} records for {symbol} in {execution_time:.2f}s "
                f"({total_inserted/execution_time:.0f} records/sec)"
            )
            
            return total_inserted
            
        except Exception as e:
            self.stats['failed_inserts'] += 1
            execution_time = time.monotonic() - start_time
            logger.error(
                f"Error inserting {len(data)} records for {symbol} after {execution_time:.2f}s: {e}"
            )
            raise
    
    async def get_symbol_data(
        self,
        symbol_info: SymbolInfo,
        timeframe: TimeFrame,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve historical data for a symbol with optimized queries
        """
        
        try:
            table_name = await self.table_manager.get_or_create_table(symbol_info)
            tf_string = TimeFrameCode.to_string(timeframe)
            
            # Build query based on instrument type
            if symbol_info.instrument_type == InstrumentType.EQUITY:
                base_query = f"SELECT tf, open, high, low, close, volume, timestamp FROM {table_name}"
            elif symbol_info.instrument_type == InstrumentType.FUTURES:
                base_query = f"SELECT contract_token, expiry_date, tf, open, high, low, close, volume, oi, timestamp FROM {table_name}"
            elif symbol_info.instrument_type in [InstrumentType.CALL_OPTION, InstrumentType.PUT_OPTION]:
                base_query = f"SELECT contract_token, option_type, strike, tf, open, high, low, close, volume, oi, timestamp FROM {table_name}"
            elif symbol_info.instrument_type == InstrumentType.INDEX:
                base_query = f"SELECT tf, open, high, low, close, timestamp FROM {table_name}"
            else:
                base_query = f"SELECT * FROM {table_name}"
            
            # Add WHERE conditions
            conditions = [f"tf = '{tf_string}'"]
            
            if start_date:
                conditions.append(f"timestamp >= '{start_date.isoformat()}'")
            if end_date:
                conditions.append(f"timestamp <= '{end_date.isoformat()}'")
            
            where_clause = " AND ".join(conditions)
            query = f"{base_query} WHERE {where_clause} ORDER BY timestamp DESC"
            
            if limit:
                query += f" LIMIT {limit}"
            
            async with self.pool.acquire() as conn:
                result = await conn.fetch(query)
                return [dict(row) for row in result]
                
        except Exception as e:
            logger.error(f"Error retrieving data for {symbol_info.symbol}: {e}")
            return []
    
    async def get_options_chain_data(
        self,
        underlying: str,
        exchange: str,
        expiry: str,
        timeframe: TimeFrame,
        timestamp: Optional[datetime] = None
    ) -> Dict[str, List[Dict]]:
        """
        Get complete options chain data for an underlying at specific time
        """
        
        try:
            table_name = TableNamingStrategy.get_options_table_name(underlying, exchange, expiry)
            tf_string = TimeFrameCode.to_string(timeframe)
            
            # Get latest data if no timestamp specified
            time_condition = ""
            if timestamp:
                time_condition = f"AND timestamp <= '{timestamp.isoformat()}'"
            
            query = f"""
                SELECT 
                    contract_token,
                    option_type,
                    strike / 100.0 as strike_price,
                    open, high, low, close, volume, oi,
                    timestamp
                FROM {table_name}
                WHERE tf = '{tf_string}' {time_condition}
                ORDER BY strike, option_type, timestamp DESC
            """
            
            async with self.pool.acquire() as conn:
                result = await conn.fetch(query)
            
            # Organize by strike and option type
            options_chain = {'CE': [], 'PE': []}
            
            for row in result:
                option_type = 'CE' if row['option_type'] == 1 else 'PE'
                options_chain[option_type].append(dict(row))
            
            return options_chain
            
        except Exception as e:
            logger.error(f"Error retrieving options chain for {underlying}: {e}")
            return {'CE': [], 'PE': []}
    
    async def get_table_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics about all tables"""
        
        if not self.table_manager:
            return {}
        
        try:
            table_stats = await self.table_manager.get_table_statistics()
            
            # Aggregate statistics
            total_records = sum(
                stats.get('record_count', 0) 
                for stats in table_stats.values() 
                if isinstance(stats.get('record_count'), int)
            )
            
            total_tables = len(table_stats)
            
            # Group by instrument type
            instrument_stats = {}
            for table_name, stats in table_stats.items():
                if table_name.startswith('eq_'):
                    instrument_type = 'equity'
                elif table_name.startswith('fut_'):
                    instrument_type = 'futures'
                elif table_name.startswith('opt_'):
                    instrument_type = 'options'
                elif table_name.startswith('idx_'):
                    instrument_type = 'index'
                else:
                    instrument_type = 'other'
                
                if instrument_type not in instrument_stats:
                    instrument_stats[instrument_type] = {'tables': 0, 'records': 0}
                
                instrument_stats[instrument_type]['tables'] += 1
                instrument_stats[instrument_type]['records'] += stats.get('record_count', 0)
            
            return {
                'total_tables': total_tables,
                'total_records': total_records,
                'instrument_breakdown': instrument_stats,
                'table_details': table_stats,
                'client_stats': self.get_statistics()
            }
            
        except Exception as e:
            logger.error(f"Error getting table statistics: {e}")
            return {'error': str(e)}
    
    async def optimize_tables(self):
        """Run optimization on all tables"""
        
        if not self.table_manager:
            return
        
        logger.info("Starting table optimization...")
        
        for table_name in self.table_manager.table_cache:
            try:
                await TableMaintenanceUtils.optimize_table_partitions(self, table_name)
                logger.debug(f"Optimized table: {table_name}")
            except Exception as e:
                logger.warning(f"Could not optimize table {table_name}: {e}")
        
        logger.info("Table optimization completed")
    
    async def cleanup_old_data(self, days_to_keep: int = 365):
        """Clean up old data beyond retention period"""
        
        if not self.table_manager:
            return
        
        cutoff_date = datetime.now() - timedelta(days=days_to_keep)
        
        logger.info(f"Cleaning up data older than {cutoff_date.date()}")
        
        for table_name in self.table_manager.table_cache:
            try:
                async with self.pool.acquire() as conn:
                    result = await conn.execute(
                        f"DELETE FROM {table_name} WHERE timestamp < '{cutoff_date.isoformat()}'"
                    )
                    
                    if result and 'DELETE' in result:
                        deleted_count = int(result.split()[1])
                        if deleted_count > 0:
                            logger.info(f"Deleted {deleted_count} old records from {table_name}")
                            
            except Exception as e:
                logger.warning(f"Could not cleanup table {table_name}: {e}")
    
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
        
        # Add table manager stats
        if self.table_manager:
            stats['tables_managed'] = len(self.table_manager.table_cache)
            stats['tables_created_this_session'] = len(self.table_manager.created_tables)
        
        return stats
    
    async def get_last_fetch_date(self, symbol_info, timeframe: str) -> Optional[datetime]:
        """Get the last successful fetch date for a symbol and timeframe"""
        try:
            table_name = self.table_manager._generate_table_name(symbol_info)
            
            async with self.pool.acquire() as conn:
                # Get the latest timestamp from the table for this timeframe
                # Use string timeframe (tf is now SYMBOL type with string values)
                from historicalfetcher.models.data_models import TimeFrameCode
                tf_string = TimeFrameCode.to_string(timeframe)
                
                query = f"""
                    SELECT MAX(timestamp) as last_date 
                    FROM {table_name} 
                    WHERE tf = '{tf_string}'
                """
                result = await conn.fetchval(query)
                
                if result:
                    return result
                return None
                
        except Exception as e:
            logger.debug(f"Could not get last fetch date for {symbol_info.symbol} ({timeframe}): {e}")
            return None
    
    async def update_fetch_status(self, symbol_info, timeframe: str, status: str, records_count: int = 0):
        """Update fetch status for a symbol and timeframe"""
        try:
            # Create fetch_status table if it doesn't exist
            async with self.pool.acquire() as conn:
                await conn.execute("""
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
                """)
                
                # Insert or update fetch status
                now = datetime.now()
                last_successful = now if status == 'success' else None
                
                await conn.execute("""
                    INSERT INTO fetch_status (
                        symbol, exchange, instrument_token, instrument_type, timeframe,
                        last_fetch_date, last_successful_fetch, total_records, status,
                        error_message, retry_count, created_at, updated_at
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)
                """, 
                    symbol_info.symbol,
                    symbol_info.exchange,
                    str(symbol_info.token),
                    symbol_info.instrument_type,
                    timeframe,
                    now,
                    last_successful,
                    records_count,
                    status,
                    None,  # error_message
                    0,     # retry_count
                    now,   # created_at
                    now    # updated_at
                )
                
        except Exception as e:
            logger.warning(f"Could not update fetch status for {symbol_info.symbol} ({timeframe}): {e}")

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
        """Cleanup database connections and resources"""
        if self.pool:
            await self.pool.close()
            logger.info("QuestDB connection pool closed")
        
        # Log final statistics
        stats = self.get_statistics()
        logger.info(f"Optimized QuestDB Client Statistics: {stats}")

# Query builder for optimized data retrieval
class OptimizedQueryBuilder:
    """Build optimized queries for different data retrieval patterns"""
    
    @staticmethod
    def build_ohlc_query(
        table_name: str,
        timeframe_code: int,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: Optional[int] = None
    ) -> str:
        """Build OHLC query with optimal performance"""
        
        query = f"""
            SELECT 
                timestamp,
                open,
                high,
                low,
                close,
                volume
            FROM {table_name}
            WHERE tf = {timeframe_code}
        """
        
        if start_date:
            query += f" AND timestamp >= '{start_date.isoformat()}'"
        if end_date:
            query += f" AND timestamp <= '{end_date.isoformat()}'"
        
        query += " ORDER BY timestamp DESC"
        
        if limit:
            query += f" LIMIT {limit}"
        
        return query
    
    @staticmethod
    def build_options_analytics_query(
        table_name: str,
        timeframe_code: int,
        strike_range: Optional[Tuple[int, int]] = None
    ) -> str:
        """Build options analytics query"""
        
        query = f"""
            SELECT 
                strike / 100.0 as strike_price,
                option_type,
                AVG(close) as avg_price,
                SUM(volume) as total_volume,
                LAST(oi) as open_interest,
                COUNT(*) as data_points
            FROM {table_name}
            WHERE tf = {timeframe_code}
        """
        
        if strike_range:
            min_strike, max_strike = strike_range
            query += f" AND strike BETWEEN {min_strike * 100} AND {max_strike * 100}"
        
        query += """
            GROUP BY strike, option_type
            ORDER BY strike, option_type
        """
        
        return query
