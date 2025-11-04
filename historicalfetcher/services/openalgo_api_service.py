"""
OpenAlgo API Service for Historical Fetcher

This service handles API calls to OpenAlgo without modifying the main OpenAlgo codebase.
It provides a clean interface for the historical fetcher to get data from OpenAlgo.
"""

import pandas as pd
from typing import Tuple, Dict, Any
from loguru import logger


class OpenAlgoAPIService:
    """Service to interact with OpenAlgo API for historical data"""
    
    def __init__(self, api_key: str, api_host: str = 'http://127.0.0.1:5000'):
        self.api_key = api_key
        self.api_host = api_host
        self._client = None
    
    def _get_client(self):
        """Get or create OpenAlgo API client"""
        if self._client is None:
            from openalgo import api
            self._client = api(api_key=self.api_key, host=self.api_host)
        return self._client
    
    def get_historical_data(
        self,
        symbol: str, 
        exchange: str, 
        interval: str, 
        start_date: str, 
        end_date: str
    ) -> Tuple[bool, Dict[str, Any], int]:
        """
        Get historical data using OpenAlgo API client
        
        Args:
            symbol: Trading symbol
            exchange: Exchange (NSE, BSE, etc.)
            interval: Time interval (1m, 5m, 15m, 1h, D)
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            
        Returns:
            Tuple containing:
            - Success status (bool)
            - Response data (dict)
            - HTTP status code (int)
        """
        try:
            client = self._get_client()
            
            # Call OpenAlgo API
            response = client.history(
                symbol=symbol,
                exchange=exchange,
                interval=interval,
                start_date=start_date,
                end_date=end_date
            )
            
            # Handle DataFrame response
            if isinstance(response, pd.DataFrame) and not response.empty:
                # 🔥 FIX: OpenAlgo API sets timestamp as index, we need to reset it to column
                if response.index.name == 'timestamp' or 'timestamp' in str(response.index.name):
                    # Reset index to convert timestamp from index back to column
                    response = response.reset_index()
                    logger.info(f"✅ Reset timestamp index to column for {symbol}")
                
                # Ensure all responses include 'oi' field, set to 0 if not present
                if 'oi' not in response.columns:
                    response['oi'] = 0
                
                # 🔍 DEBUG: Log DataFrame structure after reset
                logger.info(f"🔍 DataFrame columns after reset: {list(response.columns)}")
                
                # 🔥 FIX: Convert Timestamp objects to numeric timestamps for candle conversion
                if 'timestamp' in response.columns:
                    # Check if timestamp is Timestamp object and convert to numeric
                    if hasattr(response['timestamp'].iloc[0], 'timestamp'):
                        response['timestamp'] = response['timestamp'].apply(lambda x: int(x.timestamp()))
                        logger.info(f"✅ Converted Timestamp objects to numeric timestamps for {symbol}")
                
                # Convert to records format
                data_records = response.to_dict(orient='records')
                
                # Log first record for debugging
                if data_records:
                    first_record = data_records[0]
                    logger.info(f"🎯 OPENALGO API SUCCESS - First record for {symbol} ({exchange}, {interval}): {first_record}")
                    logger.info(f"📊 OPENALGO API SUCCESS - Total records for {symbol}: {len(data_records)}")
                
                return True, {
                    'status': 'success',
                    'data': data_records
                }, 200
            else:
                logger.warning(f"⚠️ OPENALGO API - No records returned for {symbol} ({exchange}, {interval})")
                return True, {
                    'status': 'success',
                    'data': []
                }, 200
                
        except Exception as e:
            error_msg = str(e)
            logger.error(f"❌ OPENALGO API ERROR for {symbol}: {e}")
            
            # Handle specific error types
            if "too many requests" in error_msg.lower() or "429" in error_msg:
                return False, {
                    'status': 'error',
                    'message': 'API Error: Too many requests'
                }, 429
            elif "timeout" in error_msg.lower() or "WinError 10035" in error_msg:
                return False, {
                    'status': 'error',
                    'message': 'Network timeout - please try again later'
                }, 503
            else:
                return False, {
                    'status': 'error',
                    'message': str(e)
                }, 500
