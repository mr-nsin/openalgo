"""
Data Service for Strategy Builder

Fetches historical data from QuestDB or OpenAlgo API
"""

import pandas as pd
import asyncio
import asyncpg
from typing import Optional, List, Dict, Any
from datetime import datetime, date
from loguru import logger

from ..config.strategy_settings import StrategySettings, TimeFrame, DataSource


class DataService:
    """Service to fetch historical data from QuestDB or OpenAlgo API"""
    
    def __init__(self, settings: StrategySettings):
        self.settings = settings
        self.pool: Optional[asyncpg.Pool] = None
        self._openalgo_client = None
    
    async def initialize(self):
        """Initialize the data service"""
        if self.settings.data_source == DataSource.QUESTDB:
            await self._initialize_questdb()
        else:
            await self._initialize_openalgo_api()
    
    async def _initialize_questdb(self):
        """Initialize QuestDB connection pool"""
        try:
            # QuestDB PostgreSQL wire protocol runs on port 8812 by default
            questdb_port = 8812 if self.settings.questdb_port == 9000 else self.settings.questdb_port
            
            connection_params = {
                'host': self.settings.questdb_host,
                'port': questdb_port,
                'database': self.settings.questdb_database,
                'min_size': 2,
                'max_size': 10,
                'command_timeout': 60,
            }
            
            if self.settings.questdb_username:
                connection_params['user'] = self.settings.questdb_username
            if self.settings.questdb_password:
                connection_params['password'] = self.settings.questdb_password
            
            self.pool = await asyncpg.create_pool(**connection_params)
            
            # Test connection
            async with self.pool.acquire() as conn:
                await conn.execute('SELECT 1')
            
            logger.info(f"Connected to QuestDB at {self.settings.questdb_host}:{questdb_port}")
            
        except Exception as e:
            logger.error(f"Failed to connect to QuestDB: {e}")
            logger.warning("Falling back to OpenAlgo API")
            self.settings.data_source = DataSource.OPENALGO_API
            await self._initialize_openalgo_api()
    
    async def _initialize_openalgo_api(self):
        """Initialize OpenAlgo API client"""
        try:
            from openalgo import api
            self._openalgo_client = api(
                api_key=self.settings.openalgo_api_key,
                host=self.settings.openalgo_api_host
            )
            logger.info(f"Initialized OpenAlgo API client at {self.settings.openalgo_api_host}")
        except Exception as e:
            logger.error(f"Failed to initialize OpenAlgo API client: {e}")
            raise
    
    async def get_historical_data(
        self,
        symbol: str,
        exchange: str,
        interval: str,
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """
        Get historical data for a symbol
        
        Args:
            symbol: Trading symbol (e.g., 'RELIANCE')
            exchange: Exchange (e.g., 'NSE')
            interval: Time interval (e.g., 'D' for daily, '1m', '5m', etc.)
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
        
        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume
        """
        if self.settings.data_source == DataSource.QUESTDB:
            return await self._fetch_from_questdb(symbol, exchange, interval, start_date, end_date)
        else:
            return await self._fetch_from_openalgo_api(symbol, exchange, interval, start_date, end_date)
    
    async def _fetch_from_questdb(
        self,
        symbol: str,
        exchange: str,
        interval: str,
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """Fetch historical data from QuestDB"""
        try:
            if not self.pool:
                await self._initialize_questdb()
            
            # QuestDB table naming: symbol_exchange (e.g., RELIANCE_NSE)
            table_name = f"{symbol}_{exchange}".upper()
            
            # Convert interval to QuestDB timeframe format
            tf_map = {
                '1m': '1m', '3m': '3m', '5m': '5m',
                '15m': '15m', '30m': '30m',
                '1h': '1h', 'D': 'D'
            }
            tf = tf_map.get(interval, interval)
            
            query = f"""
                SELECT timestamp, open, high, low, close, volume
                FROM {table_name}
                WHERE tf = '{tf}'
                AND timestamp >= '{start_date}'
                AND timestamp <= '{end_date}'
                ORDER BY timestamp ASC
            """
            
            async with self.pool.acquire() as conn:
                rows = await conn.fetch(query)
            
            if not rows:
                logger.warning(f"No data found in QuestDB for {symbol} {exchange}. Falling back to OpenAlgo API.")
                return await self._fetch_from_openalgo_api(symbol, exchange, interval, start_date, end_date)
            
            # Convert to DataFrame
            data = [dict(row) for row in rows]
            df = pd.DataFrame(data)
            
            # Ensure timestamp is datetime
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df = df.set_index('timestamp')
            
            # Rename columns to standard format
            df.columns = [col.lower() for col in df.columns]
            
            # Ensure required columns exist
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            for col in required_cols:
                if col not in df.columns:
                    if col == 'volume':
                        df[col] = 0
                    else:
                        raise ValueError(f"Missing required column: {col}")
            
            logger.info(f"Fetched {len(df)} records from QuestDB for {symbol} {exchange}")
            return df
            
        except Exception as e:
            logger.error(f"Error fetching from QuestDB: {e}")
            logger.warning(f"Falling back to OpenAlgo API for {symbol} {exchange}")
            return await self._fetch_from_openalgo_api(symbol, exchange, interval, start_date, end_date)
    
    async def _fetch_from_openalgo_api(
        self,
        symbol: str,
        exchange: str,
        interval: str,
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """Fetch historical data from OpenAlgo API"""
        try:
            if not self._openalgo_client:
                await self._initialize_openalgo_api()
            
            # Run in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            df = await loop.run_in_executor(
                None,
                lambda: self._openalgo_client.history(
                    symbol=symbol,
                    exchange=exchange,
                    interval=interval,
                    start_date=start_date,
                    end_date=end_date
                )
            )
            
            if df is None or df.empty:
                raise ValueError(f"No data returned for {symbol} {exchange}")
            
            # Ensure timestamp is datetime and set as index
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df = df.set_index('timestamp')
            elif df.index.name == 'timestamp' or 'timestamp' in str(df.index.name):
                df.index = pd.to_datetime(df.index)
                df.index.name = 'timestamp'
            
            # Ensure required columns exist
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            for col in required_cols:
                if col not in df.columns:
                    if col == 'volume':
                        df[col] = 0
                    else:
                        raise ValueError(f"Missing required column: {col}")
            
            # Sort by timestamp
            df = df.sort_index()
            
            logger.info(f"Fetched {len(df)} records from OpenAlgo API for {symbol} {exchange}")
            return df
            
        except Exception as e:
            logger.error(f"Error fetching from OpenAlgo API: {e}")
            raise
    
    async def close(self):
        """Close connections"""
        if self.pool:
            await self.pool.close()
            logger.info("Closed QuestDB connection pool")

