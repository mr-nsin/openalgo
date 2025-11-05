"""
OpenAlgo-Integrated Zerodha Historical Data Fetcher

This fetcher uses OpenAlgo's existing Zerodha API implementation
to fetch historical data with proper integration to the OpenAlgo ecosystem.
"""

import asyncio
import sys
import os
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

# Add the OpenAlgo root directory to Python path
openalgo_root = os.path.join(os.path.dirname(__file__), '..', '..')
openalgo_root = os.path.abspath(openalgo_root)
sys.path.insert(0, openalgo_root)

from openalgo import api

# Import from main OpenAlgo database (not local database)
# We need to ensure the utils module is available when loading symbol.py
import importlib.util

# First, load the utils.logging module to make it available
utils_logging_path = os.path.join(openalgo_root, 'utils', 'logging.py')
spec_utils = importlib.util.spec_from_file_location("utils.logging", utils_logging_path)
utils_logging_module = importlib.util.module_from_spec(spec_utils)
sys.modules['utils.logging'] = utils_logging_module
spec_utils.loader.exec_module(utils_logging_module)

# Now load the database symbol module
database_symbol_path = os.path.join(openalgo_root, 'database', 'symbol.py')
spec = importlib.util.spec_from_file_location("openalgo_database_symbol", database_symbol_path)
openalgo_database_symbol = importlib.util.module_from_spec(spec)
spec.loader.exec_module(openalgo_database_symbol)
SymToken = openalgo_database_symbol.SymToken
db_session = openalgo_database_symbol.db_session

from database.auth_db import get_auth_token_broker
from historicalfetcher.config.openalgo_settings import OpenAlgoSettings, TimeFrame
from typing import Optional, Dict
# Import utilities from historicalfetcher package
from historicalfetcher.utils.async_logger import get_async_logger
from historicalfetcher.models.data_models import HistoricalCandle, SymbolInfo

# Initialize async logger for this module
_async_logger = get_async_logger()
logger = _async_logger.get_logger()

class OpenAlgoZerodhaHistoricalFetcher:
    """Fetches historical data using OpenAlgo's Zerodha API integration"""
    
    def __init__(self, settings: OpenAlgoSettings):
        self.settings = settings
        self.openalgo_client = None
        self.user_id = settings.openalgo_user_id
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
            # Initialize OpenAlgo API client
            if not self.openalgo_client:
                self.openalgo_client = api(api_key=self.settings.openalgo_api_key, host=self.settings.openalgo_api_host)
                logger.info(f"Initialized OpenAlgo client for historical data fetching")
            
            # Cache client for better performance
            if not self._broker_data_cache:
                self._broker_data_cache = self.openalgo_client
            else:
                self.openalgo_client = self._broker_data_cache
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
        
        if not self.openalgo_client:
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
                
                # 🔥 USE SAME SIMPLE APPROACH AS test_history_format.py
                # Skip complex auth token logic and use OpenAlgo API client directly
                logger.debug(f"Using simplified OpenAlgo API client approach for {symbol}")
                
                
                # Run in thread pool to avoid blocking with retry logic
                import concurrent.futures
                loop = asyncio.get_event_loop()
                
                max_retries = getattr(self.settings, 'max_retries', 3)  # Use from settings
                retry_delay = 1.0
                df = pd.DataFrame()  # Initialize df outside the loop to avoid UnboundLocalError
                
                for attempt in range(max_retries):
                    try:
                        # 🔥 USE EXACT SAME APPROACH AS test_history_format.py
                        with concurrent.futures.ThreadPoolExecutor() as executor:
                            def make_simple_api_call():
                                try:
                                    # 🔥 USE DEDICATED HISTORICALFETCHER API SERVICE (no changes to main OpenAlgo code)
                                    from historicalfetcher.services.openalgo_api_service import OpenAlgoAPIService
                                    
                                    # Create API service instance
                                    api_service = OpenAlgoAPIService(
                                        api_key=self.settings.openalgo_api_key,
                                        api_host=self.settings.openalgo_api_host
                                    )
                                    
                                    # Call through dedicated service
                                    success, response_data, status_code = api_service.get_historical_data(
                                        symbol=symbol,
                                        exchange=exchange,
                                        interval=zerodha_interval,
                                        start_date=from_str,
                                        end_date=to_str
                                    )
                                    
                                    return success, response_data, status_code
                                        
                                except Exception as e:
                                    error_msg = str(e)
                                    if "too many requests" in error_msg.lower() or "429" in error_msg:
                                        return False, {
                                            'status': 'error',
                                            'message': 'API Error: Too many requests'
                                        }, 429
                                    else:
                                        return False, {
                                            'status': 'error',
                                            'message': str(e)
                                        }, 500
                            
                            success, response_data, status_code = await loop.run_in_executor(
                                executor, make_simple_api_call
                            )
                            # Log the actual API response for debugging
                            logger.debug(f"API call successful for {symbol}: success={success}, status_code={status_code}")
                            if not success:
                                logger.warning(f"API returned failure for {symbol}: {response_data}")
                            else:
                                # 🔍 LOG FIRST RECORD OF EVERY SUCCESSFUL API RESPONSE
                                data = response_data.get('data', [])
                                if data and len(data) > 0:
                                    first_record = data[0]
                                    logger.info(f"📊 FIRST RECORD for {symbol} ({exchange}, {zerodha_interval}): {first_record}")
                                    logger.info(f"📈 Total records received for {symbol}: {len(data)}")
                                else:
                                    logger.warning(f"⚠️ No data records received for {symbol}")
                            break  # Success, exit retry loop
                    except Exception as retry_error:
                        error_msg = str(retry_error)
                        
                        # Log the actual exception details for debugging
                        logger.error(f"API call exception for {symbol} (attempt {attempt + 1}/{max_retries}): {type(retry_error).__name__}: {retry_error}")
                        
                        # Check if it's a retryable error
                        is_retryable = (
                            "WinError 10035" in error_msg or 
                            "non-blocking socket operation could not be completed" in error_msg or
                            "timeout" in error_msg.lower() or
                            "ConnectionError" in error_msg or
                            "Too many requests" in error_msg or  # 🔥 HANDLE RATE LIMITING ERRORS
                            "429" in error_msg or  # HTTP 429 Too Many Requests
                            "rate limit" in error_msg.lower()
                        )
                        
                        if is_retryable:
                            if attempt < max_retries - 1:
                                # Special handling for rate limiting errors
                                if "Too many requests" in error_msg or "429" in error_msg or "rate limit" in error_msg.lower():
                                    backoff_time = retry_delay * (3 ** attempt)  # More aggressive backoff for rate limits
                                    logger.warning(f"🚨 RATE LIMIT ERROR on attempt {attempt + 1}/{max_retries} for {symbol}. Backing off for {backoff_time:.1f}s: {retry_error}")
                                else:
                                    backoff_time = retry_delay * (2 ** attempt)  # Standard exponential backoff
                                    logger.warning(f"Retryable error on attempt {attempt + 1}/{max_retries} for {symbol}. Backing off for {backoff_time:.1f}s: {retry_error}")
                                
                                await asyncio.sleep(backoff_time)
                                continue
                            else:
                                logger.error(f"❌ Max retries exceeded for {symbol} after {max_retries} attempts: {retry_error}")
                                raise retry_error
                        else:
                            # Non-retryable error, raise immediately
                            logger.error(f"❌ Non-retryable error for {symbol}: {retry_error}")
                            raise retry_error
                
                if not success:
                    raise Exception(f"Failed to fetch historical data: {response_data}")
                
                # 🔍 DEBUG: Log full response structure
                # logger.info(f"🔍 FULL RESPONSE for {symbol}: {response_data}")
                logger.info(f"🔍 RESPONSE TYPE: {type(response_data)}")
                logger.info(f"🔍 RESPONSE KEYS: {list(response_data.keys()) if isinstance(response_data, dict) else 'Not a dict'}")
                
                # Convert response to DataFrame
                data = response_data.get('data', [])
                logger.info(f"🔍 EXTRACTED DATA TYPE: {type(data)}, LENGTH: {len(data) if hasattr(data, '__len__') else 'N/A'}")
                
                # 🔍 DEBUG: Log data conversion process
                logger.info(f"🔍 DATA CONVERSION for {symbol}: data type = {type(data)}, length = {len(data) if hasattr(data, '__len__') else 'N/A'}")
                
                # Handle case where data might be a list instead of DataFrame
                if isinstance(data, list):
                    if data:  # If list has data, convert to DataFrame
                        logger.info(f"🔍 SAMPLE DATA for {symbol}: {data[0] if data else 'None'}")
                        df = pd.DataFrame(data)
                        logger.info(f"✅ Converted {len(data)} records to DataFrame for {symbol} - DataFrame shape: {df.shape}")
                    else:  # If empty list, create empty DataFrame
                        df = pd.DataFrame()
                        logger.warning(f"⚠️ Empty data list for {symbol}")
                elif isinstance(data, pd.DataFrame):
                    df = data
                    logger.info(f"✅ Using existing DataFrame with {len(df)} rows for {symbol}")
                else:
                    # Fallback: create empty DataFrame
                    df = pd.DataFrame()
                    logger.error(f"❌ Unknown data type {type(data)} for {symbol}, creating empty DataFrame")
                
                # 🔍 DEBUG: Log before candle conversion
                logger.info(f"🔍 ABOUT TO CONVERT TO CANDLES: DataFrame shape = {df.shape if not df.empty else 'EMPTY'}")
                
                # Convert DataFrame to HistoricalCandle objects (optimized) 
                if df.empty:
                    # Enhanced error handling for empty dataframes
                    if timeframe in [TimeFrame.MINUTE_1, TimeFrame.MINUTE_3, TimeFrame.MINUTE_5, 
                                   TimeFrame.MINUTE_15, TimeFrame.MINUTE_30, TimeFrame.HOUR_1]:
                        logger.warning(f"⚠️ No intraday data available for {symbol} ({timeframe.value}) from {from_str} to {to_str}")
                        logger.info(f"💡 This could be due to: market hours, weekends, or data availability")
                    else:
                        logger.warning(f"⚠️ No daily data available for {symbol} ({timeframe.value}) from {from_str} to {to_str}")
                    candles = []
                else:
                    logger.info(f"🔄 Converting DataFrame with {len(df)} rows to candles for {symbol}")
                    candles = await self._dataframe_to_candles_async(df)
                    logger.info(f"✅ CONVERSION SUCCESS: {len(candles)} candles created for {symbol}")
                
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
        
        # 🔍 DEBUG: Log input type and size
        logger.info(f"🔍 CANDLE CONVERSION INPUT: type = {type(df)}, size = {len(df) if hasattr(df, '__len__') else 'N/A'}")
        
        # Handle case where df might be a list instead of DataFrame
        if isinstance(df, list):
            if not df:  # Empty list
                logger.warning(f"⚠️ Empty list provided for candle conversion")
                return []
            # Convert list to DataFrame
            df = pd.DataFrame(df)
            logger.info(f"✅ Converted list to DataFrame: {df.shape}")
        elif not isinstance(df, pd.DataFrame):
            # If it's neither list nor DataFrame, return empty
            logger.error(f"❌ Invalid input type for candle conversion: {type(df)}")
            return []
        
        if df.empty:
            logger.warning(f"⚠️ Empty DataFrame provided for candle conversion")
            return []
        
        # 🔍 DEBUG: Log DataFrame details
        logger.info(f"🔍 DataFrame details: shape = {df.shape}, columns = {list(df.columns)}")
        
        # Check required columns (timestamp might be missing from API response)
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            logger.error(f"❌ Missing required columns: {missing_columns}")
            logger.error(f"Available columns: {list(df.columns)}")
            return []
        
        # Verify timestamp column exists (should be fixed by OpenAlgo API service)
        if 'timestamp' not in df.columns:
            logger.error(f"❌ CRITICAL: Still missing timestamp column after API service fix!")
            logger.error(f"Available columns: {list(df.columns)}")
            logger.error(f"DataFrame index: {df.index.name}")
            return []
        
        candles = []
        
        # Optimize: use vectorized operations where possible
        
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
                            # Fallback: try to parse as datetime string
                            from datetime import datetime as dt
                            timestamp = dt.fromisoformat(str(row['timestamp']).replace('Z', '+00:00'))
                        
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
        """Enforce rate limiting between requests with improved backoff"""
        current_time = asyncio.get_event_loop().time()
        time_since_last_request = current_time - self._last_request_time
        
        # Minimum interval between requests (more conservative)
        min_interval = self._min_request_interval
        
        if time_since_last_request < min_interval:
            sleep_time = min_interval - time_since_last_request
            if sleep_time > 0:
                logger.debug(f"⏱️ Rate limiting: sleeping for {sleep_time:.2f}s")
                await asyncio.sleep(sleep_time)
        
        # Add extra buffer for broker API stability (increased for rate limiting)
        await asyncio.sleep(2.0)  # 2 second extra buffer to prevent rate limiting
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
                    # 🔥 PRIORITIZE LIQUID STOCKS - Start with proven liquid NSE stocks
                    liquid_nse_stocks = [
                        'RELIANCE', 'TCS', 'INFY', 'HDFCBANK', 'ICICIBANK',
                        'BHARTIARTL', 'SBIN', 'LT', 'ITC', 'KOTAKBANK',
                        'HINDUNILVR', 'ASIANPAINT', 'MARUTI', 'BAJFINANCE', 'HCLTECH',
                        'AXISBANK', 'WIPRO', 'ULTRACEMCO', 'NESTLEIND', 'TITAN',
                        'ADANIPORTS', 'COALINDIA', 'NTPC', 'ONGC', 'POWERGRID',
                        'SUNPHARMA', 'TECHM', 'GRASIM', 'JSWSTEEL', 'TATAMOTORS'
                    ]
                    
                    # First, get liquid NSE stocks that we know work
                    liquid_query = session.query(SymToken).filter(
                        SymToken.instrumenttype.in_(self.settings.enabled_instrument_types),
                        SymToken.exchange == 'NSE',  # NSE only for liquid stocks
                        SymToken.symbol.in_(liquid_nse_stocks)
                    ).order_by(SymToken.symbol.asc())
                    
                    liquid_symbols = liquid_query.all()
                    
                    # 🔥 FETCH ALL SYMBOLS - No artificial limits!
                    # Get ALL symbols that match the criteria (instrument types and exchanges)
                    all_query = session.query(SymToken).filter(
                        SymToken.instrumenttype.in_(self.settings.enabled_instrument_types),
                        SymToken.exchange.in_(self.settings.enabled_exchanges),
                        # 🔥 EXCLUDE ETF/NAV SYMBOLS that have limited historical data
                        ~SymToken.symbol.like('%NAV%'),
                        ~SymToken.symbol.like('%ETF%'),
                        ~SymToken.symbol.contains('#'),
                        ~SymToken.symbol.like('%GOLD%'),
                        ~SymToken.symbol.like('%SILVER%')
                    ).order_by(
                        # Prioritize liquid stocks first, then others
                        SymToken.symbol.in_(liquid_nse_stocks).desc(),
                        SymToken.exchange.desc(),  # NSE before BSE
                        SymToken.symbol.asc()
                    )
                    
                    all_symbols = all_query.all()
                    liquid_count = len([s for s in all_symbols if s.symbol in liquid_nse_stocks])
                    other_count = len(all_symbols) - liquid_count
                    
                    logger.info(f"📊 Symbol fetching: {liquid_count} liquid + {other_count} others = {len(all_symbols)} total (NO LIMITS)")
                    return all_symbols
            
            symbols = await loop.run_in_executor(None, fetch_symbols)
            
            logger.info(f"Found {len(symbols)} symbols in OpenAlgo database")
            
            # Process symbols asynchronously in batches
            batch_size = self.settings.batch_size  # Use from settings
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

