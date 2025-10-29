"""
OpenAlgo-Integrated Zerodha Historical Data Fetcher

This fetcher uses OpenAlgo's existing Zerodha API implementation
to fetch historical data with proper integration to the OpenAlgo ecosystem.
"""

import asyncio
import sys
import os
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

# Add the OpenAlgo root directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from loguru import logger
from broker.zerodha.api.data import BrokerData
from database.auth_db import get_auth_token
from database.symbol import SymToken, db_session
from config.openalgo_settings import OpenAlgoSettings, TimeFrame
from typing import Optional, Dict
from utils.async_logger import setup_async_logger

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

class OpenAlgoZerodhaHistoricalFetcher:
    """Fetches historical data using OpenAlgo's Zerodha API integration"""
    
    def __init__(self, settings: OpenAlgoSettings):
        self.settings = settings
        self.broker_data = None
        self.user_id = "historical_fetcher"  # Default user for historical data
        self._async_logger = None
        
        # Rate limiting - optimized with better semaphore management
        self._request_semaphore = asyncio.Semaphore(settings.api_requests_per_second)
        self._last_request_time = 0.0
        self._min_request_interval = 1.0 / settings.api_requests_per_second
        
        # Caching for better performance
        self._broker_data_cache = None
        self._interval_cache = {}
        
        # Statistics
        self.stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'total_candles_fetched': 0,
            'cache_hits': 0
        }
    
    async def initialize(self, async_logger=None):
        """Initialize the fetcher with OpenAlgo's Zerodha API"""
        self._async_logger = async_logger
        
        try:
            # Get auth token from OpenAlgo database
            auth_token = get_auth_token(self.user_id)
            if not auth_token:
                # If no auth token exists, create a default one for historical fetching
                logger.warning("No auth token found for historical fetcher, using API key directly")
                auth_token = f"{self.settings.openalgo_broker_api_key}:{self.settings.openalgo_broker_api_secret}"
            
            # Cache broker data initialization for better performance
            if not self._broker_data_cache:
                # Initialize OpenAlgo's Zerodha data handler
                self.broker_data = BrokerData(auth_token)
                self._broker_data_cache = self.broker_data
            else:
                self.broker_data = self._broker_data_cache
                self.stats['cache_hits'] += 1
            
            if self._async_logger:
                await self._async_logger.log_symbol_processing(
                    symbol="SYSTEM",
                    instrument_type="INIT",
                    timeframe="N/A",
                    records_count=0,
                    processing_time=0.0,
                    status="success"
                )
            
            logger.info("OpenAlgo Zerodha historical fetcher initialized successfully")
            
        except Exception as e:
            logger.exception(f"Failed to initialize OpenAlgo Zerodha fetcher: {e}")
            if self._async_logger:
                await self._async_logger.log_error_with_context(
                    error=e,
                    context={'operation': 'initialize_fetcher'},
                    operation='initialize'
                )
            raise
    
    async def fetch_historical_data(
        self,
        symbol_info,
        timeframe: TimeFrame,
        from_date: datetime,
        to_date: datetime
    ) -> List[HistoricalCandle]:
        """
        Fetch historical data for a symbol and timeframe using OpenAlgo's Zerodha API
        """
        
        if not self.broker_data:
            await self.initialize()
        
        # Apply rate limiting
        async with self._request_semaphore:
            await self._enforce_rate_limit()
            
            try:
                # Convert symbol info to required format
                symbol = symbol_info.symbol
                exchange = symbol_info.exchange
                
                # Format dates for API
                from_str = from_date.strftime('%Y-%m-%d')
                to_str = to_date.strftime('%Y-%m-%d')
                
                # Get Zerodha interval format (with caching)
                timeframe_key = timeframe.value
                if timeframe_key not in self._interval_cache:
                    self._interval_cache[timeframe_key] = self._get_zerodha_interval(timeframe)
                zerodha_interval = self._interval_cache[timeframe_key]
                
                logger.debug(f"Fetching {timeframe.value} data for {symbol} ({exchange}) from {from_str} to {to_str}")
                
                # Use OpenAlgo's Zerodha API to fetch historical data
                # Run in thread pool to avoid blocking
                import concurrent.futures
                loop = asyncio.get_event_loop()
                
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    df = await loop.run_in_executor(
                        executor,
                        lambda: self.broker_data.get_history(
                            symbol=symbol,
                            exchange=exchange,
                            timeframe=zerodha_interval,
                            from_date=from_str,
                            to_date=to_str
                        )
                    )
                
                # Convert DataFrame to HistoricalCandle objects (optimized)
                candles = await self._dataframe_to_candles_async(df)
                
                self.stats['total_requests'] += 1
                self.stats['successful_requests'] += 1
                self.stats['total_candles_fetched'] += len(candles)
                
                logger.debug(f"Fetched {len(candles)} candles for {symbol} ({timeframe.value})")
                
                return candles
                
            except Exception as e:
                self.stats['total_requests'] += 1
                self.stats['failed_requests'] += 1
                logger.exception(f"Error fetching historical data for {symbol_info.symbol}: {e}")
                
                if self._async_logger:
                    await self._async_logger.log_error_with_context(
                        error=e,
                        context={
                            'symbol': symbol_info.symbol,
                            'timeframe': timeframe.value,
                            'from_date': str(from_date),
                            'to_date': str(to_date)
                        },
                        operation='fetch_historical_data'
                    )
                raise
    
    def _get_zerodha_interval(self, timeframe: TimeFrame) -> str:
        """Convert TimeFrame enum to Zerodha interval format"""
        
        timeframe_map = {
            TimeFrame.MINUTE_1: '1m',
            TimeFrame.MINUTE_3: '3m',
            TimeFrame.MINUTE_5: '5m',
            TimeFrame.MINUTE_15: '15m',
            TimeFrame.MINUTE_30: '30m',
            TimeFrame.HOUR_1: '1h',
            TimeFrame.DAILY: 'D'
        }
        
        return timeframe_map.get(timeframe, '1m')
    
    async def _dataframe_to_candles_async(self, df) -> List[HistoricalCandle]:
        """Convert pandas DataFrame to HistoricalCandle objects (optimized async version)"""
        
        if df.empty:
            return []
        
        candles = []
        
        # Optimize: use vectorized operations where possible
        import pandas as pd
        
        try:
            # Batch process in chunks for better memory efficiency
            chunk_size = 1000
            total_rows = len(df)
            
            for start_idx in range(0, total_rows, chunk_size):
                end_idx = min(start_idx + chunk_size, total_rows)
                chunk = df.iloc[start_idx:end_idx]
                
                # Process chunk
                for idx, row in chunk.iterrows():
                    try:
                        # Convert timestamp from epoch to datetime
                        if isinstance(row['timestamp'], (int, float)):
                            timestamp = datetime.fromtimestamp(row['timestamp'])
                        else:
                            timestamp = pd.to_datetime(row['timestamp']).to_pydatetime()
                        
                        candle = HistoricalCandle(
                            timestamp=timestamp,
                            open=float(row['open']),
                            high=float(row['high']),
                            low=float(row['low']),
                            close=float(row['close']),
                            volume=int(row['volume']),
                            oi=int(row.get('oi', 0))
                        )
                        
                        candles.append(candle)
                        
                    except (ValueError, TypeError, KeyError) as e:
                        logger.warning(f"Error parsing candle data at index {idx}: {e}")
                        continue
                
                # Yield control to event loop periodically
                if start_idx % (chunk_size * 5) == 0:
                    await asyncio.sleep(0)  # Yield to event loop
        
        except Exception as e:
            logger.exception(f"Error in batch candle conversion: {e}")
            raise
        
        return candles
    
    async def _enforce_rate_limit(self):
        """Enforce rate limiting between requests (optimized)"""
        current_time = asyncio.get_event_loop().time()
        time_since_last_request = current_time - self._last_request_time
        
        if time_since_last_request < self._min_request_interval:
            sleep_time = self._min_request_interval - time_since_last_request
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)
        
        self._last_request_time = asyncio.get_event_loop().time()
    
    async def fetch_multiple_symbols(
        self,
        symbols: List,
        timeframe: TimeFrame,
        from_date: datetime,
        to_date: datetime,
        max_concurrent: int = None
    ) -> Dict[str, List[HistoricalCandle]]:
        """Fetch historical data for multiple symbols concurrently"""
        
        if max_concurrent is None:
            max_concurrent = self.settings.max_concurrent_requests
        
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def fetch_with_semaphore(symbol_info):
            async with semaphore:
                try:
                    candles = await self.fetch_historical_data(
                        symbol_info, timeframe, from_date, to_date
                    )
                    return symbol_info.symbol, candles
                except Exception as e:
                    logger.exception(f"Error fetching data for {symbol_info.symbol}: {e}")
                    if self._async_logger:
                        await self._async_logger.log_error_with_context(
                            error=e,
                            context={'symbol': symbol_info.symbol},
                            operation='fetch_multiple_symbols'
                        )
                    return symbol_info.symbol, []
        
        # Execute all requests concurrently
        tasks = [fetch_with_semaphore(symbol) for symbol in symbols]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        symbol_data = {}
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Task failed with exception: {result}")
                continue
            
            symbol, candles = result
            symbol_data[symbol] = candles
        
        return symbol_data
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get fetcher statistics"""
        stats = self.stats.copy()
        
        if stats['total_requests'] > 0:
            stats['success_rate'] = (stats['successful_requests'] / stats['total_requests']) * 100
        else:
            stats['success_rate'] = 0
        
        return stats
    
    async def cleanup(self):
        """Cleanup resources"""
        if self._async_logger:
            stats = self.get_statistics()
            await self._async_logger.log_api_metrics(stats)
        
        logger.info("OpenAlgo Zerodha fetcher cleanup completed")
        
        # Log final statistics
        stats = self.get_statistics()
        logger.info(f"OpenAlgo Zerodha Fetcher Statistics: {stats}")
        
        # Clear caches
        self._broker_data_cache = None
        self._interval_cache.clear()

class OpenAlgoSymbolManager:
    """Symbol manager that integrates with OpenAlgo's database"""
    
    def __init__(self, settings: OpenAlgoSettings):
        self.settings = settings
        self._symbol_cache: Optional[Dict[str, List]] = None
        self._cache_timestamp: Optional[datetime] = None
        self._cache_ttl = timedelta(hours=1)  # Cache for 1 hour
    
    async def get_all_active_symbols(self, force_refresh: bool = False) -> Dict[str, List]:
        """Get all active symbols from OpenAlgo database categorized by instrument type (cached)"""
        
        # Check cache first
        if not force_refresh and self._symbol_cache and self._cache_timestamp:
            if datetime.now() - self._cache_timestamp < self._cache_ttl:
                logger.debug(f"Returning cached symbols (cached at {self._cache_timestamp})")
                return self._symbol_cache
        
        symbols_by_type = {
            'EQ': [],
            'FUT': [],
            'CE': [],
            'PE': [],
            'INDEX': []
        }
        
        try:
            # Run database query in thread pool to avoid blocking
            import concurrent.futures
            loop = asyncio.get_event_loop()
            
            def fetch_symbols():
                with db_session() as session:
                    # Query symbols based on enabled instrument types and exchanges
                    query = session.query(SymToken).filter(
                        SymToken.instrumenttype.in_(self.settings.enabled_instrument_types),
                        SymToken.exchange.in_(self.settings.enabled_exchanges)
                    )
                    return query.all()
            
            symbols = await loop.run_in_executor(None, fetch_symbols)
            
            logger.info(f"Found {len(symbols)} symbols in OpenAlgo database")
            
            # Process symbols asynchronously in batches
            batch_size = 1000
            for i in range(0, len(symbols), batch_size):
                batch = symbols[i:i + batch_size]
                
                for symbol in batch:
                    # Create symbol info object
                    symbol_info = SymbolInfo(
                        symbol=symbol.symbol,
                        exchange=symbol.exchange,
                        instrument_type=symbol.instrumenttype,
                        token=symbol.token,
                        name=symbol.name,
                        expiry=symbol.expiry,
                        strike=symbol.strike,
                        lotsize=symbol.lotsize,
                        tick_size=symbol.tick_size
                    )
                    
                    # Categorize by instrument type
                    instrument_type = symbol.instrumenttype
                    if instrument_type in symbols_by_type:
                        symbols_by_type[instrument_type].append(symbol_info)
                    elif instrument_type in ['CE', 'PE']:
                        # Options
                        symbols_by_type[instrument_type].append(symbol_info)
                
                # Yield control periodically
                if i % (batch_size * 5) == 0:
                    await asyncio.sleep(0)
            
            # Update cache
            self._symbol_cache = symbols_by_type
            self._cache_timestamp = datetime.now()
            
            # Log breakdown
            for inst_type, symbol_list in symbols_by_type.items():
                if symbol_list:
                    logger.info(f"  {inst_type}: {len(symbol_list)} symbols")
            
            return symbols_by_type
                
        except Exception as e:
            logger.exception(f"Error fetching symbols from OpenAlgo database: {e}")
            raise

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
