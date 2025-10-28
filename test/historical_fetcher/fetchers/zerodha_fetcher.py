"""
Zerodha Historical Data Fetcher

Handles fetching historical OHLCV data from Zerodha API with proper rate limiting,
chunking, and error handling.
"""

import asyncio
import aiohttp
import pandas as pd
import sys
import os
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

# Add the OpenAlgo root directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from utils.logging import get_logger
from config.settings import Settings, TimeFrame
from config.timeframes import TimeFrameConfig
from fetchers.symbol_manager import SymbolInfo

logger = get_logger(__name__)

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

class ZerodhaAPIError(Exception):
    """Custom exception for Zerodha API errors"""
    pass

class ZerodhaHistoricalFetcher:
    """Fetches historical data from Zerodha API with async processing"""
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.base_url = settings.zerodha_base_url
        self.api_key = settings.zerodha_api_key
        self.access_token = settings.zerodha_access_token
        
        # Session management
        self.session: Optional[aiohttp.ClientSession] = None
        
        # Rate limiting
        self._request_semaphore = asyncio.Semaphore(settings.api_requests_per_second)
        self._last_request_time = 0
        self._min_request_interval = 1.0 / settings.api_requests_per_second
        
        # Statistics
        self.stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'total_candles_fetched': 0
        }
    
    async def initialize(self):
        """Initialize HTTP session with proper configuration"""
        
        headers = {
            'X-Kite-Version': '3',
            'Authorization': f'token {self.api_key}:{self.access_token}',
            'Content-Type': 'application/json'
        }
        
        # Configure connector for optimal performance
        connector = aiohttp.TCPConnector(
            limit=100,
            limit_per_host=20,
            ttl_dns_cache=300,
            use_dns_cache=True,
            keepalive_timeout=30,
            enable_cleanup_closed=True
        )
        
        # Configure timeout
        timeout = aiohttp.ClientTimeout(
            total=60,
            connect=15,
            sock_read=30
        )
        
        self.session = aiohttp.ClientSession(
            headers=headers,
            connector=connector,
            timeout=timeout
        )
        
        logger.info("Zerodha historical fetcher initialized")
    
    async def fetch_historical_data(
        self,
        symbol_info: SymbolInfo,
        timeframe: TimeFrame,
        from_date: datetime,
        to_date: datetime
    ) -> List[HistoricalCandle]:
        """
        Fetch historical data for a symbol and timeframe
        Handles 60-day chunking as per Zerodha API limits
        """
        
        if not self.session:
            await self.initialize()
        
        # Get Zerodha interval string
        zerodha_interval = TimeFrameConfig.get_zerodha_interval(timeframe)
        
        all_candles = []
        current_start = from_date
        
        logger.debug(f"Fetching {timeframe.value} data for {symbol_info.symbol} from {from_date.date()} to {to_date.date()}")
        
        while current_start <= to_date:
            # Calculate chunk end date (60 days max as per Zerodha limit)
            current_end = min(current_start + timedelta(days=self.settings.chunk_days - 1), to_date)
            
            try:
                chunk_candles = await self._fetch_data_chunk(
                    symbol_info,
                    zerodha_interval,
                    current_start,
                    current_end
                )
                
                all_candles.extend(chunk_candles)
                self.stats['total_candles_fetched'] += len(chunk_candles)
                
                logger.debug(
                    f"Fetched {len(chunk_candles)} candles for {symbol_info.symbol} "
                    f"({timeframe.value}) from {current_start.date()} to {current_end.date()}"
                )
                
            except Exception as e:
                logger.error(
                    f"Error fetching chunk for {symbol_info.symbol} "
                    f"({timeframe.value}) from {current_start.date()} to {current_end.date()}: {e}"
                )
                # Continue with next chunk instead of failing completely
                self.stats['failed_requests'] += 1
            
            # Move to next chunk
            current_start = current_end + timedelta(days=1)
        
        logger.info(f"Total {len(all_candles)} candles fetched for {symbol_info.symbol} ({timeframe.value})")
        return all_candles
    
    async def _fetch_data_chunk(
        self,
        symbol_info: SymbolInfo,
        interval: str,
        from_date: datetime,
        to_date: datetime
    ) -> List[HistoricalCandle]:
        """Fetch a single chunk of historical data with rate limiting"""
        
        # Format dates for API (Zerodha expects specific format)
        from_str = from_date.strftime('%Y-%m-%d+00:00:00')
        to_str = to_date.strftime('%Y-%m-%d+23:59:59')
        
        # Construct endpoint
        endpoint = f"/instruments/historical/{symbol_info.instrument_token}/{interval}"
        params = {
            'from': from_str,
            'to': to_str,
            'oi': '1'  # Include Open Interest
        }
        
        url = f"{self.base_url}{endpoint}"
        
        # Apply rate limiting
        async with self._request_semaphore:
            await self._enforce_rate_limit()
            
            try:
                response_data = await self._make_api_request(url, params)
                self.stats['successful_requests'] += 1
                return self._parse_candles_response(response_data)
                
            except Exception as e:
                self.stats['failed_requests'] += 1
                raise
    
    async def _enforce_rate_limit(self):
        """Enforce rate limiting between requests"""
        current_time = asyncio.get_event_loop().time()
        time_since_last_request = current_time - self._last_request_time
        
        if time_since_last_request < self._min_request_interval:
            sleep_time = self._min_request_interval - time_since_last_request
            await asyncio.sleep(sleep_time)
        
        self._last_request_time = asyncio.get_event_loop().time()
    
    async def _make_api_request(self, url: str, params: Dict) -> Dict:
        """Make API request with comprehensive error handling"""
        
        self.stats['total_requests'] += 1
        
        try:
            async with self.session.get(url, params=params) as response:
                # Log request details for debugging
                logger.debug(f"API Request: {url} with params: {params}")
                logger.debug(f"Response status: {response.status}")
                
                if response.status == 200:
                    data = await response.json()
                    
                    # Check API response status
                    if data.get('status') != 'success':
                        error_msg = data.get('message', 'Unknown API error')
                        error_type = data.get('error_type', 'APIError')
                        
                        if 'permission' in error_msg.lower() or error_type == 'PermissionException':
                            raise ZerodhaAPIError(f"Permission denied: {error_msg}")
                        else:
                            raise ZerodhaAPIError(f"API Error: {error_msg}")
                    
                    return data
                    
                elif response.status == 429:
                    # Rate limit exceeded
                    retry_after = response.headers.get('Retry-After', '60')
                    raise ZerodhaAPIError(f"Rate limit exceeded. Retry after {retry_after} seconds")
                    
                elif response.status == 403:
                    # Permission denied
                    error_text = await response.text()
                    raise ZerodhaAPIError(f"Permission denied: {error_text}")
                    
                else:
                    # Other HTTP errors
                    error_text = await response.text()
                    raise ZerodhaAPIError(f"HTTP {response.status}: {error_text}")
                    
        except aiohttp.ClientError as e:
            raise ZerodhaAPIError(f"Network error: {e}")
        except asyncio.TimeoutError:
            raise ZerodhaAPIError("Request timeout")
        except Exception as e:
            if isinstance(e, ZerodhaAPIError):
                raise
            raise ZerodhaAPIError(f"Unexpected error: {e}")
    
    def _parse_candles_response(self, response_data: Dict) -> List[HistoricalCandle]:
        """Parse Zerodha API response to HistoricalCandle objects"""
        
        candles_data = response_data.get('data', {}).get('candles', [])
        candles = []
        
        for candle in candles_data:
            try:
                # Zerodha format: [timestamp, open, high, low, close, volume, oi]
                if len(candle) >= 6:
                    # Parse timestamp (ISO format)
                    timestamp = pd.to_datetime(candle[0])
                    
                    candles.append(HistoricalCandle(
                        timestamp=timestamp,
                        open=float(candle[1]),
                        high=float(candle[2]),
                        low=float(candle[3]),
                        close=float(candle[4]),
                        volume=int(candle[5]),
                        oi=int(candle[6]) if len(candle) > 6 else 0
                    ))
                else:
                    logger.warning(f"Invalid candle data format: {candle}")
                    
            except (ValueError, TypeError, IndexError) as e:
                logger.warning(f"Error parsing candle data {candle}: {e}")
                continue
        
        return candles
    
    async def fetch_multiple_symbols(
        self,
        symbols: List[SymbolInfo],
        timeframe: TimeFrame,
        from_date: datetime,
        to_date: datetime,
        max_concurrent: int = None
    ) -> Dict[str, List[HistoricalCandle]]:
        """Fetch historical data for multiple symbols concurrently"""
        
        if max_concurrent is None:
            max_concurrent = self.settings.max_concurrent_requests
        
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def fetch_with_semaphore(symbol_info: SymbolInfo):
            async with semaphore:
                try:
                    candles = await self.fetch_historical_data(
                        symbol_info, timeframe, from_date, to_date
                    )
                    return symbol_info.symbol, candles
                except Exception as e:
                    logger.error(f"Error fetching data for {symbol_info.symbol}: {e}")
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
        if self.session:
            await self.session.close()
            logger.info("Zerodha fetcher session closed")
        
        # Log final statistics
        stats = self.get_statistics()
        logger.info(f"Zerodha Fetcher Statistics: {stats}")

# Utility functions for data validation and processing
class DataValidator:
    """Validates historical data for quality and completeness"""
    
    @staticmethod
    def validate_candles(candles: List[HistoricalCandle]) -> Dict[str, Any]:
        """Validate candle data and return quality metrics"""
        
        if not candles:
            return {
                'valid': False,
                'total_candles': 0,
                'issues': ['No data available']
            }
        
        issues = []
        valid_candles = 0
        
        for i, candle in enumerate(candles):
            # Check for valid OHLC relationships
            if not (candle.low <= candle.open <= candle.high and 
                   candle.low <= candle.close <= candle.high):
                issues.append(f"Invalid OHLC at index {i}")
                continue
            
            # Check for zero or negative values
            if any(val <= 0 for val in [candle.open, candle.high, candle.low, candle.close]):
                issues.append(f"Zero/negative price at index {i}")
                continue
            
            # Check for negative volume
            if candle.volume < 0:
                issues.append(f"Negative volume at index {i}")
                continue
            
            valid_candles += 1
        
        return {
            'valid': len(issues) == 0,
            'total_candles': len(candles),
            'valid_candles': valid_candles,
            'issues': issues[:10],  # Limit to first 10 issues
            'data_quality_score': (valid_candles / len(candles)) * 100 if candles else 0
        }
    
    @staticmethod
    def detect_data_gaps(candles: List[HistoricalCandle], timeframe: TimeFrame) -> List[Dict]:
        """Detect gaps in historical data"""
        
        if len(candles) < 2:
            return []
        
        # Sort candles by timestamp
        sorted_candles = sorted(candles, key=lambda c: c.timestamp)
        
        # Calculate expected interval
        interval_minutes = {
            TimeFrame.MINUTE_1: 1,
            TimeFrame.MINUTE_3: 3,
            TimeFrame.MINUTE_5: 5,
            TimeFrame.MINUTE_15: 15,
            TimeFrame.MINUTE_30: 30,
            TimeFrame.HOUR_1: 60,
            TimeFrame.DAILY: 1440  # 24 hours
        }.get(timeframe, 1)
        
        expected_interval = timedelta(minutes=interval_minutes)
        gaps = []
        
        for i in range(1, len(sorted_candles)):
            current_time = sorted_candles[i].timestamp
            previous_time = sorted_candles[i-1].timestamp
            actual_interval = current_time - previous_time
            
            # Allow some tolerance for market holidays and weekends
            tolerance = expected_interval * 2
            
            if actual_interval > tolerance:
                gaps.append({
                    'start_time': previous_time,
                    'end_time': current_time,
                    'gap_duration': actual_interval,
                    'expected_interval': expected_interval
                })
        
        return gaps
