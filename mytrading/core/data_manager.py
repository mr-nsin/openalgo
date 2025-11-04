"""
Data Manager
============

Manages all market data flows including real-time WebSocket feeds,
historical data integration, and data fusion for the trading system.
"""

import asyncio
import time
from typing import Dict, List, Optional, Any, Set
from datetime import datetime, timedelta
from collections import defaultdict, deque

from ..config.settings import TradingSettings
from ..config.symbols import SymbolConfig, TimeFrame, Exchange, InstrumentType
from ..utils.logging_config import get_logger
from ..utils.performance_monitor import performance_timer, get_performance_monitor
from ..communication.zmq_publisher import ZMQPublisher
from ..communication.message_types import create_market_data_message, MessageType

logger = get_logger(__name__)


class DataManager:
    """
    Manages all market data flows for the trading system
    
    Responsibilities:
    - Real-time WebSocket data feeds
    - Historical data integration
    - Data validation and cleaning
    - Data fusion and synchronization
    - Publishing processed data via ZMQ
    """
    
    def __init__(self, settings: TradingSettings):
        self.settings = settings
        self.is_running = False
        self.start_time = time.time()
        
        # Symbol configuration
        self.symbol_config = SymbolConfig.create_default_config()
        
        # Data storage
        self.market_data: Dict[str, Dict] = defaultdict(dict)
        self.historical_data: Dict[str, Dict] = defaultdict(dict)
        self.data_buffer: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        
        # WebSocket integration (will integrate with existing nifty_atm_websocket_tracker.py)
        self.websocket_feed = None
        self.subscribed_symbols: Set[str] = set()
        
        # Communication
        self.market_data_publisher: Optional[ZMQPublisher] = None
        
        # Performance tracking
        self.performance_monitor = get_performance_monitor()
        self.data_stats = {
            "messages_received": 0,
            "messages_published": 0,
            "symbols_active": 0,
            "last_update_time": 0,
            "data_quality_score": 1.0
        }
        
        # Data validation
        self.validation_enabled = True
        self.max_price_change = 0.10  # 10% max price change
        
        logger.info("📊 DataManager initialized")
    
    async def initialize(self):
        """Initialize the data manager"""
        try:
            logger.info("📊 Initializing Data Manager...")
            
            # Initialize communication
            await self._initialize_communication()
            
            # Load symbol configuration
            await self._load_symbol_configuration()
            
            # Initialize WebSocket feed if enabled
            if self.settings.enable_websocket:
                await self._initialize_websocket_feed()
            
            # Initialize historical data if enabled
            if self.settings.enable_historical_data:
                await self._initialize_historical_data()
            
            logger.success("✅ Data Manager initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize Data Manager: {e}")
            raise
    
    async def run(self):
        """Main data manager loop"""
        try:
            logger.info("📊 Starting Data Manager...")
            self.is_running = True
            
            # Start data processing tasks
            tasks = []
            
            if self.settings.enable_websocket:
                tasks.append(asyncio.create_task(self._websocket_data_loop()))
            
            if self.settings.enable_historical_data:
                tasks.append(asyncio.create_task(self._historical_data_loop()))
            
            # Start data validation and publishing
            tasks.append(asyncio.create_task(self._data_processing_loop()))
            tasks.append(asyncio.create_task(self._statistics_loop()))
            
            # Wait for all tasks
            await asyncio.gather(*tasks, return_exceptions=True)
            
        except Exception as e:
            logger.error(f"❌ Error in Data Manager: {e}")
        finally:
            self.is_running = False
    
    async def shutdown(self):
        """Shutdown the data manager"""
        logger.info("📊 Shutting down Data Manager...")
        self.is_running = False
        
        # Disconnect WebSocket
        if self.websocket_feed:
            try:
                self.websocket_feed.disconnect()
            except Exception as e:
                logger.error(f"Error disconnecting WebSocket: {e}")
        
        # Disconnect publisher
        if self.market_data_publisher:
            self.market_data_publisher.disconnect()
        
        logger.success("✅ Data Manager shutdown complete")
    
    async def _initialize_communication(self):
        """Initialize ZMQ communication"""
        logger.info("📡 Initializing data communication...")
        
        self.market_data_publisher = ZMQPublisher(
            port=self.settings.messaging.market_data_port,
            high_water_mark=self.settings.messaging.zmq_high_water_mark
        )
        await self.market_data_publisher.connect()
        
        logger.success("✅ Data communication initialized")
    
    async def _load_symbol_configuration(self):
        """Load symbol configuration from environment variables"""
        logger.info("📋 Loading symbol configuration...")
        
        # This is a placeholder - in a full implementation, you would:
        # 1. Parse ENABLED_SYMBOLS from environment
        # 2. Parse ENABLED_TIMEFRAMES from environment  
        # 3. Parse ENABLED_EXCHANGES from environment
        # 4. Create SymbolInfo objects for each symbol
        
        # For now, use default configuration
        logger.info(f"📊 Loaded {len(self.symbol_config.symbols)} symbols")
        logger.success("✅ Symbol configuration loaded")
    
    async def _initialize_websocket_feed(self):
        """Initialize WebSocket feed (integration point for nifty_atm_websocket_tracker.py)"""
        logger.info("📡 Initializing WebSocket feed...")
        
        try:
            # This is where we would integrate the existing WebSocket tracker
            # from test/nifty_atm_websocket_tracker.py
            
            # For now, create a placeholder
            logger.info("📡 WebSocket feed integration placeholder")
            logger.info("🔗 Integration point for nifty_atm_websocket_tracker.py")
            
            # TODO: Integrate actual WebSocket feed
            # self.websocket_feed = WebSocketFeed(
            #     host=self.settings.openalgo.websocket_host,
            #     port=self.settings.openalgo.websocket_port,
            #     api_key=self.settings.openalgo.api_key
            # )
            # await self.websocket_feed.connect()
            
            logger.success("✅ WebSocket feed initialized (placeholder)")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize WebSocket feed: {e}")
            raise
    
    async def _initialize_historical_data(self):
        """Initialize historical data integration"""
        logger.info("📈 Initializing historical data integration...")
        
        try:
            # This is where we would integrate with the historical fetcher
            # or database to get historical data
            
            logger.info("📈 Historical data integration placeholder")
            logger.info("🔗 Integration point for historical data fetcher")
            
            # TODO: Integrate actual historical data
            # self.historical_fetcher = HistoricalDataFetcher(self.settings)
            # await self.historical_fetcher.initialize()
            
            logger.success("✅ Historical data initialized (placeholder)")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize historical data: {e}")
            raise
    
    async def _websocket_data_loop(self):
        """Process real-time WebSocket data"""
        logger.info("📡 Starting WebSocket data processing...")
        
        try:
            while self.is_running:
                # This is where we would process incoming WebSocket data
                # and integrate with the existing nifty_atm_websocket_tracker.py
                
                # Placeholder for WebSocket data processing
                await asyncio.sleep(0.1)  # 100ms processing cycle
                
                # TODO: Implement actual WebSocket data processing
                # market_data = await self.websocket_feed.get_next_data()
                # await self._process_market_data(market_data)
                
        except Exception as e:
            logger.error(f"Error in WebSocket data loop: {e}")
    
    async def _historical_data_loop(self):
        """Process historical data integration"""
        logger.info("📈 Starting historical data processing...")
        
        try:
            while self.is_running:
                # This is where we would fetch and process historical data
                # to fill gaps in real-time data
                
                # Placeholder for historical data processing
                await asyncio.sleep(60)  # Check every minute
                
                # TODO: Implement actual historical data processing
                # await self._fetch_missing_historical_data()
                # await self._merge_historical_with_realtime()
                
        except Exception as e:
            logger.error(f"Error in historical data loop: {e}")
    
    async def _data_processing_loop(self):
        """Main data processing and publishing loop"""
        logger.info("🔄 Starting data processing loop...")
        
        try:
            while self.is_running:
                # Process and validate market data
                await self._process_pending_data()
                
                # Publish processed data
                await self._publish_market_data()
                
                # Small delay to prevent busy waiting
                await asyncio.sleep(0.01)  # 10ms processing cycle
                
        except Exception as e:
            logger.error(f"Error in data processing loop: {e}")
    
    async def _statistics_loop(self):
        """Statistics and monitoring loop"""
        logger.info("📊 Starting statistics loop...")
        
        try:
            while self.is_running:
                await self._update_statistics()
                await asyncio.sleep(5)  # Update stats every 5 seconds
                
        except Exception as e:
            logger.error(f"Error in statistics loop: {e}")
    
    @performance_timer("data_manager", "process_market_data")
    async def _process_market_data(self, symbol: str, data: Dict[str, Any]):
        """Process incoming market data"""
        try:
            # Validate data
            if self.validation_enabled:
                if not self._validate_market_data(symbol, data):
                    return
            
            # Store in buffer
            self.data_buffer[symbol].append({
                'timestamp': time.time(),
                'data': data
            })
            
            # Update market data store
            self.market_data[symbol].update(data)
            self.market_data[symbol]['last_update'] = time.time()
            
            # Update statistics
            self.data_stats["messages_received"] += 1
            self.data_stats["last_update_time"] = time.time()
            
        except Exception as e:
            logger.error(f"Error processing market data for {symbol}: {e}")
    
    def _validate_market_data(self, symbol: str, data: Dict[str, Any]) -> bool:
        """Validate incoming market data"""
        try:
            # Check if LTP exists and is valid
            ltp = data.get('ltp', 0)
            if ltp <= 0:
                return False
            
            # Check for unrealistic price changes
            if symbol in self.market_data:
                last_ltp = self.market_data[symbol].get('ltp', 0)
                if last_ltp > 0:
                    price_change = abs(ltp - last_ltp) / last_ltp
                    if price_change > self.max_price_change:
                        logger.warning(f"Large price change detected for {symbol}: {price_change:.2%}")
                        return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating data for {symbol}: {e}")
            return False
    
    async def _process_pending_data(self):
        """Process any pending data in buffers"""
        # Placeholder for data processing logic
        pass
    
    async def _publish_market_data(self):
        """Publish processed market data via ZMQ"""
        try:
            if not self.market_data_publisher:
                return
            
            # Publish data for each symbol that has updates
            current_time = time.time()
            
            for symbol, data in self.market_data.items():
                last_update = data.get('last_update', 0)
                
                # Only publish if data is recent (within last 5 seconds)
                if current_time - last_update < 5.0:
                    # Create market data message
                    message = create_market_data_message(
                        symbol=symbol,
                        exchange="NSE",  # Default exchange
                        data=data,
                        source="data_manager"
                    )
                    
                    # Publish message
                    await self.market_data_publisher.publish_message(message)
                    self.data_stats["messages_published"] += 1
            
        except Exception as e:
            logger.error(f"Error publishing market data: {e}")
    
    async def _update_statistics(self):
        """Update data manager statistics"""
        try:
            current_time = time.time()
            uptime = current_time - self.start_time
            
            # Update active symbols count
            active_symbols = sum(
                1 for data in self.market_data.values()
                if current_time - data.get('last_update', 0) < 60  # Active within last minute
            )
            
            self.data_stats.update({
                "symbols_active": active_symbols,
                "uptime": uptime,
                "messages_per_second": self.data_stats["messages_received"] / uptime if uptime > 0 else 0
            })
            
            # Log statistics periodically
            if int(current_time) % 30 == 0:  # Every 30 seconds
                logger.info(
                    f"📊 Data Stats: {active_symbols} active symbols, "
                    f"{self.data_stats['messages_per_second']:.1f} msg/sec, "
                    f"{self.data_stats['messages_published']} published"
                )
            
        except Exception as e:
            logger.error(f"Error updating statistics: {e}")
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on data manager"""
        current_time = time.time()
        
        # Check if we're receiving data
        time_since_last_update = current_time - self.data_stats.get("last_update_time", 0)
        receiving_data = time_since_last_update < 30  # Should receive data within 30 seconds
        
        # Check active symbols
        active_symbols = self.data_stats.get("symbols_active", 0)
        
        # Determine health status
        if receiving_data and active_symbols > 0:
            status = "healthy"
        elif active_symbols > 0:
            status = "warning"
        else:
            status = "unhealthy"
        
        return {
            "status": status,
            "timestamp": datetime.utcnow().isoformat(),
            "component": "data_manager",
            "metrics": {
                "active_symbols": active_symbols,
                "messages_received": self.data_stats["messages_received"],
                "messages_published": self.data_stats["messages_published"],
                "time_since_last_update": time_since_last_update,
                "uptime": current_time - self.start_time
            },
            "issues": [] if status == "healthy" else [
                "No recent data updates" if not receiving_data else "",
                "No active symbols" if active_symbols == 0 else ""
            ]
        }
    
    def get_market_data(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get current market data for a symbol"""
        return self.market_data.get(symbol)
    
    def get_symbols(self) -> List[str]:
        """Get list of all tracked symbols"""
        return list(self.market_data.keys())
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get data manager statistics"""
        return dict(self.data_stats)
