"""
Trading System Orchestrator
===========================

Main orchestrator that coordinates all components of the MyTrading system.
Manages the lifecycle of data feeds, strategy engines, and trade execution.
"""

import asyncio
import signal
import time
from typing import Dict, List, Optional, Any
from datetime import datetime

from ..config.settings import TradingSettings
from ..utils.logging_config import get_logger
from ..utils.performance_monitor import get_performance_monitor, performance_timer
from ..communication.zmq_publisher import ZMQPublisher
from ..communication.message_types import create_system_status_message, MessageType

from .data_manager import DataManager
from .strategy_engine import StrategyEngine
from .trade_manager import TradeManager

logger = get_logger(__name__)


class TradingOrchestrator:
    """
    Main orchestrator for the trading system
    
    Coordinates all system components and manages their lifecycle:
    - Data Manager: Real-time and historical data
    - Strategy Engine: Signal generation and calculations
    - Trade Manager: Order execution and position management
    - Communication: ZMQ messaging between components
    """
    
    def __init__(self, settings: TradingSettings):
        self.settings = settings
        self.start_time = datetime.utcnow()
        self.is_running = False
        self.shutdown_event = asyncio.Event()
        
        # Core components
        self.data_manager: Optional[DataManager] = None
        self.strategy_engine: Optional[StrategyEngine] = None
        self.trade_manager: Optional[TradeManager] = None
        
        # Communication
        self.system_publisher: Optional[ZMQPublisher] = None
        
        # Component tasks
        self.component_tasks: List[asyncio.Task] = []
        
        # Performance monitoring
        self.performance_monitor = get_performance_monitor()
        
        # Health check
        self.last_health_check = time.time()
        self.health_check_task: Optional[asyncio.Task] = None
        
        logger.info(f"🎯 TradingOrchestrator initialized - Mode: {settings.trading_mode.value}")
    
    async def start(self):
        """Start the trading system"""
        try:
            logger.info("🚀 Starting Trading System...")
            
            # Validate configuration
            await self._validate_configuration()
            
            # Initialize communication
            await self._initialize_communication()
            
            # Initialize core components
            await self._initialize_components()
            
            # Start all components
            await self._start_components()
            
            # Start health monitoring
            if self.settings.enable_health_checks:
                await self._start_health_monitoring()
            
            # Start performance monitoring
            if self.settings.enable_performance_monitoring:
                await self.performance_monitor.start_monitoring()
            
            self.is_running = True
            logger.success("✅ Trading System started successfully!")
            
            # Publish system status
            await self._publish_system_status("RUNNING", "Trading system started successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to start trading system: {e}")
            await self.shutdown()
            raise
    
    async def shutdown(self):
        """Gracefully shutdown the trading system"""
        if not self.is_running:
            return
        
        logger.info("🛑 Shutting down Trading System...")
        self.is_running = False
        self.shutdown_event.set()
        
        try:
            # Publish shutdown status
            if self.system_publisher:
                await self._publish_system_status("SHUTTING_DOWN", "Trading system shutting down")
            
            # Stop health monitoring
            if self.health_check_task:
                self.health_check_task.cancel()
                try:
                    await self.health_check_task
                except asyncio.CancelledError:
                    pass
            
            # Stop performance monitoring
            if self.settings.enable_performance_monitoring:
                await self.performance_monitor.stop_monitoring()
            
            # Shutdown components in reverse order
            await self._shutdown_components()
            
            # Cancel all component tasks
            for task in self.component_tasks:
                if not task.done():
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass
            
            # Shutdown communication
            await self._shutdown_communication()
            
            # Final status
            if self.system_publisher:
                await self._publish_system_status("STOPPED", "Trading system stopped")
            
            logger.success("✅ Trading System shutdown complete")
            
        except Exception as e:
            logger.error(f"❌ Error during shutdown: {e}")
    
    async def _validate_configuration(self):
        """Validate system configuration"""
        logger.info("📋 Validating configuration...")
        
        errors = self.settings.validate()
        if errors:
            error_msg = "Configuration validation failed:\n" + "\n".join(f"  - {error}" for error in errors)
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        logger.success("✅ Configuration validation passed")
    
    async def _initialize_communication(self):
        """Initialize ZMQ communication system"""
        logger.info("📡 Initializing communication system...")
        
        try:
            # System status publisher
            self.system_publisher = ZMQPublisher(
                port=self.settings.messaging.system_status_port,
                high_water_mark=self.settings.messaging.zmq_high_water_mark
            )
            await self.system_publisher.connect()
            
            logger.success("✅ Communication system initialized")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize communication: {e}")
            raise
    
    async def _initialize_components(self):
        """Initialize all core components"""
        logger.info("🔧 Initializing core components...")
        
        try:
            # Initialize Data Manager
            if self.settings.enable_websocket or self.settings.enable_historical_data:
                logger.info("📊 Initializing Data Manager...")
                self.data_manager = DataManager(self.settings)
                await self.data_manager.initialize()
                logger.success("✅ Data Manager initialized")
            
            # Initialize Strategy Engine
            if self.settings.enable_strategy_engine:
                logger.info("🧠 Initializing Strategy Engine...")
                self.strategy_engine = StrategyEngine(self.settings)
                await self.strategy_engine.initialize()
                logger.success("✅ Strategy Engine initialized")
            
            # Initialize Trade Manager
            if self.settings.enable_trade_execution:
                logger.info("💼 Initializing Trade Manager...")
                self.trade_manager = TradeManager(self.settings)
                await self.trade_manager.initialize()
                logger.success("✅ Trade Manager initialized")
            else:
                logger.info("💼 Trade execution disabled - Trade Manager not initialized")
            
            logger.success("✅ All core components initialized")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize components: {e}")
            raise
    
    @performance_timer("orchestrator", "start_components")
    async def _start_components(self):
        """Start all initialized components"""
        logger.info("▶️  Starting core components...")
        
        try:
            # Start Data Manager
            if self.data_manager:
                logger.info("📊 Starting Data Manager...")
                data_task = asyncio.create_task(self.data_manager.run())
                self.component_tasks.append(data_task)
                logger.success("✅ Data Manager started")
            
            # Start Strategy Engine
            if self.strategy_engine:
                logger.info("🧠 Starting Strategy Engine...")
                strategy_task = asyncio.create_task(self.strategy_engine.run())
                self.component_tasks.append(strategy_task)
                logger.success("✅ Strategy Engine started")
            
            # Start Trade Manager
            if self.trade_manager:
                logger.info("💼 Starting Trade Manager...")
                trade_task = asyncio.create_task(self.trade_manager.run())
                self.component_tasks.append(trade_task)
                logger.success("✅ Trade Manager started")
            
            logger.success("✅ All components started successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to start components: {e}")
            raise
    
    async def _shutdown_components(self):
        """Shutdown all components in reverse order"""
        logger.info("🛑 Shutting down core components...")
        
        # Shutdown Trade Manager first (most critical)
        if self.trade_manager:
            logger.info("💼 Shutting down Trade Manager...")
            await self.trade_manager.shutdown()
            logger.success("✅ Trade Manager shutdown complete")
        
        # Shutdown Strategy Engine
        if self.strategy_engine:
            logger.info("🧠 Shutting down Strategy Engine...")
            await self.strategy_engine.shutdown()
            logger.success("✅ Strategy Engine shutdown complete")
        
        # Shutdown Data Manager last
        if self.data_manager:
            logger.info("📊 Shutting down Data Manager...")
            await self.data_manager.shutdown()
            logger.success("✅ Data Manager shutdown complete")
        
        logger.success("✅ All components shutdown complete")
    
    async def _shutdown_communication(self):
        """Shutdown communication system"""
        logger.info("📡 Shutting down communication system...")
        
        if self.system_publisher:
            self.system_publisher.disconnect()
        
        logger.success("✅ Communication system shutdown complete")
    
    async def _start_health_monitoring(self):
        """Start health monitoring task"""
        logger.info("🏥 Starting health monitoring...")
        
        self.health_check_task = asyncio.create_task(self._health_check_loop())
        logger.success("✅ Health monitoring started")
    
    async def _health_check_loop(self):
        """Continuous health check loop"""
        try:
            while self.is_running:
                await self._perform_health_check()
                await asyncio.sleep(self.settings.health_check_interval)
        except asyncio.CancelledError:
            logger.debug("Health check loop cancelled")
        except Exception as e:
            logger.error(f"Error in health check loop: {e}")
    
    async def _perform_health_check(self):
        """Perform system health check"""
        try:
            current_time = time.time()
            health_status = {
                "timestamp": datetime.utcnow().isoformat(),
                "uptime": current_time - time.mktime(self.start_time.timetuple()),
                "components": {}
            }
            
            # Check Data Manager health
            if self.data_manager:
                health_status["components"]["data_manager"] = await self.data_manager.health_check()
            
            # Check Strategy Engine health
            if self.strategy_engine:
                health_status["components"]["strategy_engine"] = await self.strategy_engine.health_check()
            
            # Check Trade Manager health
            if self.trade_manager:
                health_status["components"]["trade_manager"] = await self.trade_manager.health_check()
            
            # Check communication health
            if self.system_publisher:
                health_status["components"]["communication"] = await self.system_publisher.health_check()
            
            # Determine overall health
            component_statuses = [comp.get("status", "unknown") for comp in health_status["components"].values()]
            if all(status == "healthy" for status in component_statuses):
                overall_status = "healthy"
            elif any(status == "unhealthy" for status in component_statuses):
                overall_status = "unhealthy"
            else:
                overall_status = "warning"
            
            health_status["overall_status"] = overall_status
            self.last_health_check = current_time
            
            # Log health status
            if overall_status == "healthy":
                logger.debug(f"🏥 System health check: {overall_status}")
            else:
                logger.warning(f"🏥 System health check: {overall_status}")
                logger.warning(f"Health details: {health_status}")
            
            # Publish health status
            await self._publish_system_status(overall_status.upper(), f"Health check: {overall_status}", health_status)
            
        except Exception as e:
            logger.error(f"Error performing health check: {e}")
    
    async def _publish_system_status(self, status: str, message: str, details: Optional[Dict] = None):
        """Publish system status message"""
        if not self.system_publisher:
            return
        
        try:
            status_message = create_system_status_message(
                component="orchestrator",
                status=status,
                source="trading_orchestrator",
                details=message
            )
            
            if details:
                status_message.metadata.update(details)
            
            await self.system_publisher.publish_message(status_message)
            
        except Exception as e:
            logger.error(f"Error publishing system status: {e}")
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get comprehensive system information"""
        current_time = time.time()
        
        return {
            "system": {
                "name": self.settings.system_name,
                "version": self.settings.version,
                "mode": self.settings.trading_mode.value,
                "start_time": self.start_time.isoformat(),
                "uptime": current_time - time.mktime(self.start_time.timetuple()),
                "is_running": self.is_running
            },
            "components": {
                "data_manager": self.data_manager is not None,
                "strategy_engine": self.strategy_engine is not None,
                "trade_manager": self.trade_manager is not None,
                "communication": self.system_publisher is not None
            },
            "settings": {
                "enable_websocket": self.settings.enable_websocket,
                "enable_historical_data": self.settings.enable_historical_data,
                "enable_strategy_engine": self.settings.enable_strategy_engine,
                "enable_trade_execution": self.settings.enable_trade_execution,
                "enable_performance_monitoring": self.settings.enable_performance_monitoring,
                "enable_health_checks": self.settings.enable_health_checks
            },
            "performance": self.performance_monitor.get_summary_report() if self.performance_monitor else None
        }
    
    async def wait_for_shutdown(self):
        """Wait for shutdown signal"""
        await self.shutdown_event.wait()
    
    def __repr__(self):
        return f"TradingOrchestrator(mode={self.settings.trading_mode.value}, running={self.is_running})"
