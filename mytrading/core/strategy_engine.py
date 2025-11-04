"""
Strategy Engine
===============

Executes trading strategies, calculates signals, and manages strategy lifecycle.
Subscribes to market data and publishes trading signals.
"""

import asyncio
import time
from typing import Dict, List, Optional, Any, Set
from datetime import datetime
from collections import defaultdict, deque

from ..config.settings import TradingSettings
from ..config.strategies import StrategyManager, StrategyConfig
from ..utils.logging_config import get_logger
from ..utils.performance_monitor import performance_timer, get_performance_monitor
from ..communication.zmq_subscriber import ZMQSubscriber
from ..communication.zmq_publisher import ZMQPublisher
from ..communication.message_types import (
    create_signal_message, parse_message, MessageType, MarketDataMessage
)

logger = get_logger(__name__)


class StrategyEngine:
    """
    Strategy execution engine
    
    Responsibilities:
    - Load and manage trading strategies
    - Subscribe to market data feeds
    - Execute strategy calculations
    - Generate and publish trading signals
    - Track strategy performance
    """
    
    def __init__(self, settings: TradingSettings):
        self.settings = settings
        self.is_running = False
        self.start_time = time.time()
        
        # Strategy management
        self.strategy_manager = StrategyManager.create_default_strategies()
        self.active_strategies: Dict[str, Any] = {}  # Will hold actual strategy instances
        
        # Market data subscription
        self.market_data_subscriber: Optional[ZMQSubscriber] = None
        self.subscribed_symbols: Set[str] = set()
        
        # Signal publishing
        self.signal_publisher: Optional[ZMQPublisher] = None
        
        # Data storage for strategies
        self.market_data_buffer: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.strategy_data: Dict[str, Dict] = defaultdict(dict)
        
        # Performance tracking
        self.performance_monitor = get_performance_monitor()
        self.strategy_stats = {
            "signals_generated": 0,
            "strategies_active": 0,
            "last_calculation_time": 0,
            "calculation_count": 0,
            "average_calculation_time": 0.0
        }
        
        logger.info("🧠 StrategyEngine initialized")
    
    async def initialize(self):
        """Initialize the strategy engine"""
        try:
            logger.info("🧠 Initializing Strategy Engine...")
            
            # Initialize communication
            await self._initialize_communication()
            
            # Load and initialize strategies
            await self._load_strategies()
            
            # Setup market data subscriptions
            await self._setup_subscriptions()
            
            logger.success("✅ Strategy Engine initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize Strategy Engine: {e}")
            raise
    
    async def run(self):
        """Main strategy engine loop"""
        try:
            logger.info("🧠 Starting Strategy Engine...")
            self.is_running = True
            
            # Start processing tasks
            tasks = [
                asyncio.create_task(self._market_data_processing_loop()),
                asyncio.create_task(self._strategy_calculation_loop()),
                asyncio.create_task(self._statistics_loop())
            ]
            
            # Wait for all tasks
            await asyncio.gather(*tasks, return_exceptions=True)
            
        except Exception as e:
            logger.error(f"❌ Error in Strategy Engine: {e}")
        finally:
            self.is_running = False
    
    async def shutdown(self):
        """Shutdown the strategy engine"""
        logger.info("🧠 Shutting down Strategy Engine...")
        self.is_running = False
        
        # Disconnect subscribers and publishers
        if self.market_data_subscriber:
            await self.market_data_subscriber.disconnect()
        
        if self.signal_publisher:
            self.signal_publisher.disconnect()
        
        logger.success("✅ Strategy Engine shutdown complete")
    
    async def _initialize_communication(self):
        """Initialize ZMQ communication"""
        logger.info("📡 Initializing strategy communication...")
        
        # Market data subscriber
        self.market_data_subscriber = ZMQSubscriber(
            host="localhost",
            port=self.settings.messaging.market_data_port
        )
        await self.market_data_subscriber.connect()
        
        # Signal publisher
        self.signal_publisher = ZMQPublisher(
            port=self.settings.messaging.strategy_signals_port,
            high_water_mark=self.settings.messaging.zmq_high_water_mark
        )
        await self.signal_publisher.connect()
        
        logger.success("✅ Strategy communication initialized")
    
    async def _load_strategies(self):
        """Load and initialize trading strategies"""
        logger.info("📋 Loading trading strategies...")
        
        try:
            # Get enabled strategies from environment
            enabled_strategies = self._get_enabled_strategies()
            
            for strategy_name in enabled_strategies:
                strategy_config = self.strategy_manager.get_strategy(strategy_name)
                if strategy_config and strategy_config.enabled:
                    # Initialize strategy instance
                    strategy_instance = await self._create_strategy_instance(strategy_config)
                    if strategy_instance:
                        self.active_strategies[strategy_name] = strategy_instance
                        logger.info(f"✅ Loaded strategy: {strategy_name}")
                    else:
                        logger.warning(f"⚠️  Failed to create strategy instance: {strategy_name}")
                else:
                    logger.warning(f"⚠️  Strategy not found or disabled: {strategy_name}")
            
            logger.success(f"✅ Loaded {len(self.active_strategies)} strategies")
            
        except Exception as e:
            logger.error(f"❌ Failed to load strategies: {e}")
            raise
    
    def _get_enabled_strategies(self) -> List[str]:
        """Get enabled strategies from environment variables"""
        import os
        
        enabled_str = os.getenv('ENABLED_STRATEGIES', 'SMA_Crossover,RSI_MeanReversion')
        return [s.strip() for s in enabled_str.split(',') if s.strip()]
    
    async def _create_strategy_instance(self, config: StrategyConfig) -> Optional[Any]:
        """Create a strategy instance from configuration"""
        try:
            # This is a placeholder for actual strategy instantiation
            # In a full implementation, you would:
            # 1. Import the strategy class based on config.name
            # 2. Create an instance with the configuration
            # 3. Initialize the strategy with indicators and parameters
            
            logger.info(f"🔧 Creating strategy instance: {config.name}")
            
            # Placeholder strategy instance
            strategy_instance = {
                'name': config.name,
                'config': config,
                'initialized': True,
                'last_calculation': 0,
                'signals_generated': 0,
                'indicators': {},
                'state': {}
            }
            
            # TODO: Implement actual strategy instantiation
            # from ..strategies.technical_strategies import SMAStrategy, RSIStrategy
            # if config.name == "SMA_Crossover":
            #     strategy_instance = SMAStrategy(config)
            # elif config.name == "RSI_MeanReversion":
            #     strategy_instance = RSIStrategy(config)
            
            return strategy_instance
            
        except Exception as e:
            logger.error(f"Error creating strategy instance for {config.name}: {e}")
            return None
    
    async def _setup_subscriptions(self):
        """Setup market data subscriptions for strategies"""
        logger.info("📡 Setting up market data subscriptions...")
        
        try:
            # Collect all symbols needed by strategies
            required_symbols = set()
            
            for strategy_name, strategy in self.active_strategies.items():
                config = strategy['config']
                for symbol_pattern in config.symbols:
                    # For now, add the pattern directly
                    # In a full implementation, you would expand patterns like "NIFTY*"
                    required_symbols.add(symbol_pattern)
            
            # Subscribe to market data for required symbols
            for symbol in required_symbols:
                await self.market_data_subscriber.subscribe(f"market_data.{symbol}")
                self.subscribed_symbols.add(symbol)
            
            logger.success(f"✅ Subscribed to {len(required_symbols)} symbols")
            
        except Exception as e:
            logger.error(f"❌ Failed to setup subscriptions: {e}")
            raise
    
    async def _market_data_processing_loop(self):
        """Process incoming market data messages"""
        logger.info("📊 Starting market data processing...")
        
        try:
            while self.is_running:
                # Receive market data messages
                message_data = await self.market_data_subscriber.receive_message()
                
                if message_data:
                    await self._process_market_data_message(message_data)
                
                # Small delay to prevent busy waiting
                await asyncio.sleep(0.001)  # 1ms
                
        except Exception as e:
            logger.error(f"Error in market data processing loop: {e}")
    
    async def _process_market_data_message(self, message_data: bytes):
        """Process a single market data message"""
        try:
            # Parse message
            message = parse_message(message_data)
            
            if isinstance(message, MarketDataMessage):
                # Store market data for strategies
                symbol = message.symbol
                
                # Add to buffer
                self.market_data_buffer[symbol].append({
                    'timestamp': message.timestamp,
                    'ltp': message.ltp,
                    'volume': message.volume,
                    'open': message.open,
                    'high': message.high,
                    'low': message.low,
                    'close': message.close,
                    'data': message.data
                })
                
                # Update latest data
                self.strategy_data[symbol] = {
                    'ltp': message.ltp,
                    'volume': message.volume,
                    'timestamp': message.timestamp,
                    'data': message.data
                }
                
        except Exception as e:
            logger.error(f"Error processing market data message: {e}")
    
    async def _strategy_calculation_loop(self):
        """Main strategy calculation loop"""
        logger.info("🧠 Starting strategy calculations...")
        
        try:
            while self.is_running:
                # Calculate signals for all active strategies
                await self._calculate_all_strategies()
                
                # Delay between calculation cycles
                await asyncio.sleep(1.0)  # 1 second calculation cycle
                
        except Exception as e:
            logger.error(f"Error in strategy calculation loop: {e}")
    
    @performance_timer("strategy_engine", "calculate_strategies")
    async def _calculate_all_strategies(self):
        """Calculate signals for all active strategies"""
        try:
            calculation_start = time.time()
            
            for strategy_name, strategy in self.active_strategies.items():
                await self._calculate_strategy_signals(strategy_name, strategy)
            
            # Update statistics
            calculation_time = time.time() - calculation_start
            self.strategy_stats["last_calculation_time"] = calculation_time
            self.strategy_stats["calculation_count"] += 1
            
            # Update average calculation time
            total_time = (self.strategy_stats["average_calculation_time"] * 
                         (self.strategy_stats["calculation_count"] - 1) + calculation_time)
            self.strategy_stats["average_calculation_time"] = total_time / self.strategy_stats["calculation_count"]
            
        except Exception as e:
            logger.error(f"Error calculating strategies: {e}")
    
    async def _calculate_strategy_signals(self, strategy_name: str, strategy: Dict):
        """Calculate signals for a specific strategy"""
        try:
            config = strategy['config']
            
            # Get required market data for this strategy
            strategy_data = {}
            for symbol_pattern in config.symbols:
                # For now, use exact match
                # In full implementation, expand patterns
                if symbol_pattern in self.strategy_data:
                    strategy_data[symbol_pattern] = self.strategy_data[symbol_pattern]
            
            if not strategy_data:
                return  # No data available for this strategy
            
            # Placeholder for strategy calculation
            # In a full implementation, you would call the actual strategy logic
            signals = await self._execute_strategy_logic(strategy_name, strategy, strategy_data)
            
            # Publish generated signals
            for signal in signals:
                await self._publish_signal(signal)
                strategy['signals_generated'] += 1
                self.strategy_stats["signals_generated"] += 1
            
            strategy['last_calculation'] = time.time()
            
        except Exception as e:
            logger.error(f"Error calculating signals for {strategy_name}: {e}")
    
    async def _execute_strategy_logic(self, strategy_name: str, strategy: Dict, market_data: Dict) -> List[Dict]:
        """Execute the actual strategy logic (placeholder)"""
        signals = []
        
        try:
            config = strategy['config']
            
            # Placeholder strategy logic based on strategy type
            if strategy_name == "SMA_Crossover":
                signals = await self._sma_crossover_logic(config, market_data)
            elif strategy_name == "RSI_MeanReversion":
                signals = await self._rsi_mean_reversion_logic(config, market_data)
            
            # TODO: Implement actual strategy logic
            # signals = await strategy_instance.calculate_signals(market_data)
            
        except Exception as e:
            logger.error(f"Error executing strategy logic for {strategy_name}: {e}")
        
        return signals
    
    async def _sma_crossover_logic(self, config: StrategyConfig, market_data: Dict) -> List[Dict]:
        """Placeholder SMA crossover strategy logic"""
        signals = []
        
        # Placeholder logic - in reality, you would:
        # 1. Calculate SMA indicators
        # 2. Check for crossover conditions
        # 3. Generate buy/sell signals
        
        for symbol, data in market_data.items():
            if data.get('ltp', 0) > 0:
                # Placeholder signal generation
                if time.time() % 60 < 1:  # Generate signal once per minute
                    signals.append({
                        'strategy': config.name,
                        'symbol': symbol,
                        'signal_type': 'BUY',
                        'confidence': 0.7,
                        'price': data['ltp'],
                        'reasoning': 'SMA crossover detected (placeholder)'
                    })
        
        return signals
    
    async def _rsi_mean_reversion_logic(self, config: StrategyConfig, market_data: Dict) -> List[Dict]:
        """Placeholder RSI mean reversion strategy logic"""
        signals = []
        
        # Placeholder logic - in reality, you would:
        # 1. Calculate RSI indicator
        # 2. Check for oversold/overbought conditions
        # 3. Generate mean reversion signals
        
        for symbol, data in market_data.items():
            if data.get('ltp', 0) > 0:
                # Placeholder signal generation
                if time.time() % 120 < 1:  # Generate signal once per 2 minutes
                    signals.append({
                        'strategy': config.name,
                        'symbol': symbol,
                        'signal_type': 'SELL',
                        'confidence': 0.6,
                        'price': data['ltp'],
                        'reasoning': 'RSI overbought condition (placeholder)'
                    })
        
        return signals
    
    async def _publish_signal(self, signal: Dict):
        """Publish a trading signal"""
        try:
            if not self.signal_publisher:
                return
            
            # Create signal message
            message = create_signal_message(
                strategy_name=signal['strategy'],
                symbol=signal['symbol'],
                signal_type=signal['signal_type'],
                confidence=signal['confidence'],
                source="strategy_engine",
                entry_price=signal.get('price'),
                reasoning=signal.get('reasoning')
            )
            
            # Publish signal
            await self.signal_publisher.publish_message(message)
            
            logger.info(f"📈 Signal: {signal['strategy']} - {signal['symbol']} {signal['signal_type']} "
                       f"@ {signal.get('price', 0):.2f} (confidence: {signal['confidence']:.1%})")
            
        except Exception as e:
            logger.error(f"Error publishing signal: {e}")
    
    async def _statistics_loop(self):
        """Statistics and monitoring loop"""
        logger.info("📊 Starting strategy statistics loop...")
        
        try:
            while self.is_running:
                await self._update_statistics()
                await asyncio.sleep(10)  # Update stats every 10 seconds
                
        except Exception as e:
            logger.error(f"Error in statistics loop: {e}")
    
    async def _update_statistics(self):
        """Update strategy engine statistics"""
        try:
            current_time = time.time()
            uptime = current_time - self.start_time
            
            # Count active strategies
            active_strategies = len([s for s in self.active_strategies.values() if s.get('initialized', False)])
            
            self.strategy_stats.update({
                "strategies_active": active_strategies,
                "uptime": uptime,
                "signals_per_minute": self.strategy_stats["signals_generated"] / (uptime / 60) if uptime > 0 else 0
            })
            
            # Log statistics periodically
            if int(current_time) % 60 == 0:  # Every minute
                logger.info(
                    f"🧠 Strategy Stats: {active_strategies} active, "
                    f"{self.strategy_stats['signals_generated']} signals generated, "
                    f"{self.strategy_stats['average_calculation_time']*1000:.1f}ms avg calc time"
                )
            
        except Exception as e:
            logger.error(f"Error updating statistics: {e}")
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on strategy engine"""
        current_time = time.time()
        
        # Check if strategies are calculating
        time_since_last_calc = current_time - self.strategy_stats.get("last_calculation_time", 0)
        calculating = time_since_last_calc < 60  # Should calculate within last minute
        
        # Check active strategies
        active_strategies = self.strategy_stats.get("strategies_active", 0)
        
        # Determine health status
        if calculating and active_strategies > 0:
            status = "healthy"
        elif active_strategies > 0:
            status = "warning"
        else:
            status = "unhealthy"
        
        return {
            "status": status,
            "timestamp": datetime.utcnow().isoformat(),
            "component": "strategy_engine",
            "metrics": {
                "active_strategies": active_strategies,
                "signals_generated": self.strategy_stats["signals_generated"],
                "calculation_count": self.strategy_stats["calculation_count"],
                "average_calculation_time": self.strategy_stats["average_calculation_time"],
                "time_since_last_calculation": time_since_last_calc,
                "uptime": current_time - self.start_time
            },
            "issues": [] if status == "healthy" else [
                "No recent calculations" if not calculating else "",
                "No active strategies" if active_strategies == 0 else ""
            ]
        }
    
    def get_active_strategies(self) -> List[str]:
        """Get list of active strategy names"""
        return list(self.active_strategies.keys())
    
    def get_strategy_statistics(self) -> Dict[str, Any]:
        """Get strategy engine statistics"""
        return dict(self.strategy_stats)
