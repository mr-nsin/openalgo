"""
Trade Manager
=============

Manages trade execution, position tracking, and risk management.
Subscribes to trading signals and executes trades via OpenAlgo API.
"""

import asyncio
import time
from typing import Dict, List, Optional, Any, Set
from datetime import datetime
from collections import defaultdict
from enum import Enum

from ..config.settings import TradingSettings, TradingMode
from ..utils.logging_config import get_logger
from ..utils.performance_monitor import performance_timer, get_performance_monitor
from ..communication.zmq_subscriber import ZMQSubscriber
from ..communication.zmq_publisher import ZMQPublisher
from ..communication.message_types import (
    create_trade_message, parse_message, MessageType, SignalMessage
)

logger = get_logger(__name__)


class OrderStatus(Enum):
    """Order status enumeration"""
    PENDING = "PENDING"
    SUBMITTED = "SUBMITTED"
    FILLED = "FILLED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"


class PositionType(Enum):
    """Position type enumeration"""
    LONG = "LONG"
    SHORT = "SHORT"


class TradeManager:
    """
    Trade execution and position management
    
    Responsibilities:
    - Subscribe to trading signals
    - Execute trades via OpenAlgo API
    - Track positions and orders
    - Implement risk management
    - Publish trade updates
    """
    
    def __init__(self, settings: TradingSettings):
        self.settings = settings
        self.is_running = False
        self.start_time = time.time()
        
        # OpenAlgo client integration
        self.openalgo_client = None  # Will be initialized with actual client
        
        # Signal subscription
        self.signal_subscriber: Optional[ZMQSubscriber] = None
        
        # Trade publishing
        self.trade_publisher: Optional[ZMQPublisher] = None
        
        # Position and order tracking
        self.positions: Dict[str, Dict] = {}  # symbol -> position info
        self.orders: Dict[str, Dict] = {}     # order_id -> order info
        self.pending_signals: List[Dict] = []
        
        # Risk management
        self.daily_pnl = 0.0
        self.max_positions = settings.risk_management.max_positions
        self.max_daily_loss = settings.risk_management.max_daily_loss
        self.position_size_limit = settings.risk_management.position_size_limit
        
        # Performance tracking
        self.performance_monitor = get_performance_monitor()
        self.trade_stats = {
            "signals_received": 0,
            "trades_executed": 0,
            "orders_placed": 0,
            "positions_active": 0,
            "daily_pnl": 0.0,
            "total_pnl": 0.0,
            "win_rate": 0.0,
            "last_trade_time": 0
        }
        
        logger.info(f"💼 TradeManager initialized - Mode: {settings.trading_mode.value}")
    
    async def initialize(self):
        """Initialize the trade manager"""
        try:
            logger.info("💼 Initializing Trade Manager...")
            
            # Initialize OpenAlgo client
            await self._initialize_openalgo_client()
            
            # Initialize communication
            await self._initialize_communication()
            
            # Load existing positions (if any)
            await self._load_existing_positions()
            
            # Initialize risk management
            await self._initialize_risk_management()
            
            logger.success("✅ Trade Manager initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize Trade Manager: {e}")
            raise
    
    async def run(self):
        """Main trade manager loop"""
        try:
            logger.info("💼 Starting Trade Manager...")
            self.is_running = True
            
            # Start processing tasks
            tasks = [
                asyncio.create_task(self._signal_processing_loop()),
                asyncio.create_task(self._order_monitoring_loop()),
                asyncio.create_task(self._position_monitoring_loop()),
                asyncio.create_task(self._risk_monitoring_loop()),
                asyncio.create_task(self._statistics_loop())
            ]
            
            # Wait for all tasks
            await asyncio.gather(*tasks, return_exceptions=True)
            
        except Exception as e:
            logger.error(f"❌ Error in Trade Manager: {e}")
        finally:
            self.is_running = False
    
    async def shutdown(self):
        """Shutdown the trade manager"""
        logger.info("💼 Shutting down Trade Manager...")
        self.is_running = False
        
        # Close all positions if configured to do so
        if self.settings.close_positions_on_shutdown:
            await self._close_all_positions()
        
        # Cancel pending orders
        await self._cancel_pending_orders()
        
        # Disconnect communication
        if self.signal_subscriber:
            await self.signal_subscriber.disconnect()
        
        if self.trade_publisher:
            self.trade_publisher.disconnect()
        
        logger.success("✅ Trade Manager shutdown complete")
    
    async def _initialize_openalgo_client(self):
        """Initialize OpenAlgo API client"""
        logger.info("🔗 Initializing OpenAlgo client...")
        
        try:
            # This is where we would initialize the actual OpenAlgo client
            # from the existing codebase
            
            logger.info("🔗 OpenAlgo client integration placeholder")
            logger.info(f"🔗 API Host: {self.settings.openalgo.api_host}")
            logger.info(f"🔗 Trading Mode: {self.settings.trading_mode.value}")
            
            # TODO: Initialize actual OpenAlgo client
            # from openalgo_client import OpenAlgoClient
            # self.openalgo_client = OpenAlgoClient(
            #     api_key=self.settings.openalgo.api_key,
            #     api_host=self.settings.openalgo.api_host,
            #     timeout=self.settings.openalgo.timeout
            # )
            # await self.openalgo_client.authenticate()
            
            if self.settings.trading_mode == TradingMode.PAPER:
                logger.info("📝 Paper trading mode - trades will be simulated")
            else:
                logger.info("💰 Live trading mode - real trades will be executed")
            
            logger.success("✅ OpenAlgo client initialized (placeholder)")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize OpenAlgo client: {e}")
            raise
    
    async def _initialize_communication(self):
        """Initialize ZMQ communication"""
        logger.info("📡 Initializing trade communication...")
        
        # Signal subscriber
        self.signal_subscriber = ZMQSubscriber(
            host="localhost",
            port=self.settings.messaging.strategy_signals_port
        )
        await self.signal_subscriber.connect()
        await self.signal_subscriber.subscribe("signals.*")  # Subscribe to all signals
        
        # Trade publisher
        self.trade_publisher = ZMQPublisher(
            port=self.settings.messaging.trade_updates_port,
            high_water_mark=self.settings.messaging.zmq_high_water_mark
        )
        await self.trade_publisher.connect()
        
        logger.success("✅ Trade communication initialized")
    
    async def _load_existing_positions(self):
        """Load existing positions from OpenAlgo"""
        logger.info("📊 Loading existing positions...")
        
        try:
            # This is where we would load existing positions from OpenAlgo API
            # positions = await self.openalgo_client.get_positions()
            
            # Placeholder for position loading
            logger.info("📊 Position loading placeholder")
            
            # TODO: Load actual positions
            # for position in positions:
            #     self.positions[position['symbol']] = {
            #         'symbol': position['symbol'],
            #         'quantity': position['quantity'],
            #         'average_price': position['average_price'],
            #         'pnl': position['pnl'],
            #         'position_type': PositionType.LONG if position['quantity'] > 0 else PositionType.SHORT
            #     }
            
            logger.success(f"✅ Loaded {len(self.positions)} existing positions")
            
        except Exception as e:
            logger.error(f"❌ Failed to load existing positions: {e}")
            # Don't raise - continue with empty positions
    
    async def _initialize_risk_management(self):
        """Initialize risk management parameters"""
        logger.info("🛡️  Initializing risk management...")
        
        # Load daily PnL from database or API
        # self.daily_pnl = await self._get_daily_pnl()
        
        logger.info(f"🛡️  Max positions: {self.max_positions}")
        logger.info(f"🛡️  Max daily loss: ${self.max_daily_loss:,.2f}")
        logger.info(f"🛡️  Position size limit: ${self.position_size_limit:,.2f}")
        
        logger.success("✅ Risk management initialized")
    
    async def _signal_processing_loop(self):
        """Process incoming trading signals"""
        logger.info("📈 Starting signal processing...")
        
        try:
            while self.is_running:
                # Receive signal messages
                message_data = await self.signal_subscriber.receive_message()
                
                if message_data:
                    await self._process_signal_message(message_data)
                
                # Small delay to prevent busy waiting
                await asyncio.sleep(0.001)  # 1ms
                
        except Exception as e:
            logger.error(f"Error in signal processing loop: {e}")
    
    async def _process_signal_message(self, message_data: bytes):
        """Process a single trading signal message"""
        try:
            # Parse message
            message = parse_message(message_data)
            
            if isinstance(message, SignalMessage):
                self.trade_stats["signals_received"] += 1
                
                logger.info(f"📈 Received signal: {message.strategy_name} - {message.symbol} "
                           f"{message.signal_type} (confidence: {message.confidence:.1%})")
                
                # Validate signal
                if await self._validate_signal(message):
                    # Execute trade
                    await self._execute_signal(message)
                else:
                    logger.warning(f"⚠️  Signal validation failed: {message.symbol} {message.signal_type}")
                
        except Exception as e:
            logger.error(f"Error processing signal message: {e}")
    
    async def _validate_signal(self, signal: SignalMessage) -> bool:
        """Validate a trading signal against risk management rules"""
        try:
            # Check if trading is enabled
            if self.settings.trading_mode == TradingMode.DISABLED:
                return False
            
            # Check confidence threshold
            if signal.confidence < self.settings.risk_management.min_signal_confidence:
                logger.debug(f"Signal confidence too low: {signal.confidence:.1%}")
                return False
            
            # Check daily loss limit
            if self.daily_pnl <= -self.max_daily_loss:
                logger.warning(f"Daily loss limit reached: ${self.daily_pnl:,.2f}")
                return False
            
            # Check maximum positions
            if len(self.positions) >= self.max_positions:
                logger.warning(f"Maximum positions limit reached: {len(self.positions)}")
                return False
            
            # Check if we already have a position in this symbol
            if signal.symbol in self.positions:
                current_position = self.positions[signal.symbol]
                # Allow closing positions or adding to existing positions based on strategy
                logger.debug(f"Existing position in {signal.symbol}: {current_position}")
            
            # Check position size limit
            if signal.entry_price and signal.entry_price * 100 > self.position_size_limit:  # Assuming 100 quantity
                logger.warning(f"Position size too large: ${signal.entry_price * 100:,.2f}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating signal: {e}")
            return False
    
    @performance_timer("trade_manager", "execute_signal")
    async def _execute_signal(self, signal: SignalMessage):
        """Execute a trading signal"""
        try:
            # Determine order parameters
            order_params = await self._calculate_order_parameters(signal)
            
            if not order_params:
                logger.warning(f"Could not calculate order parameters for {signal.symbol}")
                return
            
            # Execute order based on trading mode
            if self.settings.trading_mode == TradingMode.PAPER:
                order_result = await self._execute_paper_trade(order_params)
            else:
                order_result = await self._execute_live_trade(order_params)
            
            if order_result:
                # Track the order
                self.orders[order_result['order_id']] = order_result
                self.trade_stats["orders_placed"] += 1
                self.trade_stats["last_trade_time"] = time.time()
                
                # Publish trade update
                await self._publish_trade_update(order_result)
                
                logger.success(f"✅ Order placed: {order_result['symbol']} {order_result['side']} "
                              f"{order_result['quantity']} @ {order_result['price']:.2f}")
            
        except Exception as e:
            logger.error(f"Error executing signal: {e}")
    
    async def _calculate_order_parameters(self, signal: SignalMessage) -> Optional[Dict]:
        """Calculate order parameters from signal"""
        try:
            # Basic order parameters
            order_params = {
                'symbol': signal.symbol,
                'side': 'BUY' if signal.signal_type == 'BUY' else 'SELL',
                'quantity': 100,  # Default quantity - should be calculated based on risk management
                'price': signal.entry_price or 0,
                'order_type': 'MARKET',  # Default to market order
                'strategy': signal.strategy_name,
                'confidence': signal.confidence
            }
            
            # Calculate position size based on risk management
            if signal.entry_price:
                max_quantity = int(self.position_size_limit / signal.entry_price)
                order_params['quantity'] = min(order_params['quantity'], max_quantity)
            
            # Validate quantity
            if order_params['quantity'] <= 0:
                return None
            
            return order_params
            
        except Exception as e:
            logger.error(f"Error calculating order parameters: {e}")
            return None
    
    async def _execute_paper_trade(self, order_params: Dict) -> Optional[Dict]:
        """Execute a paper trade (simulation)"""
        try:
            # Simulate order execution
            order_id = f"PAPER_{int(time.time() * 1000)}"
            
            order_result = {
                'order_id': order_id,
                'symbol': order_params['symbol'],
                'side': order_params['side'],
                'quantity': order_params['quantity'],
                'price': order_params['price'],
                'order_type': order_params['order_type'],
                'status': OrderStatus.FILLED.value,
                'filled_quantity': order_params['quantity'],
                'filled_price': order_params['price'],
                'timestamp': time.time(),
                'is_paper_trade': True
            }
            
            # Update position
            await self._update_position(order_result)
            
            return order_result
            
        except Exception as e:
            logger.error(f"Error executing paper trade: {e}")
            return None
    
    async def _execute_live_trade(self, order_params: Dict) -> Optional[Dict]:
        """Execute a live trade via OpenAlgo API"""
        try:
            # This is where we would place actual orders via OpenAlgo
            logger.info(f"🔴 LIVE TRADE: {order_params['symbol']} {order_params['side']} "
                       f"{order_params['quantity']} @ {order_params['price']:.2f}")
            
            # TODO: Implement actual OpenAlgo order placement
            # order_result = await self.openalgo_client.place_order(
            #     symbol=order_params['symbol'],
            #     side=order_params['side'],
            #     quantity=order_params['quantity'],
            #     price=order_params['price'],
            #     order_type=order_params['order_type']
            # )
            
            # Placeholder for live trading
            order_result = {
                'order_id': f"LIVE_{int(time.time() * 1000)}",
                'symbol': order_params['symbol'],
                'side': order_params['side'],
                'quantity': order_params['quantity'],
                'price': order_params['price'],
                'order_type': order_params['order_type'],
                'status': OrderStatus.SUBMITTED.value,
                'filled_quantity': 0,
                'filled_price': 0,
                'timestamp': time.time(),
                'is_paper_trade': False
            }
            
            return order_result
            
        except Exception as e:
            logger.error(f"Error executing live trade: {e}")
            return None
    
    async def _update_position(self, order_result: Dict):
        """Update position based on order result"""
        try:
            symbol = order_result['symbol']
            side = order_result['side']
            quantity = order_result['filled_quantity']
            price = order_result['filled_price']
            
            if symbol not in self.positions:
                # New position
                self.positions[symbol] = {
                    'symbol': symbol,
                    'quantity': quantity if side == 'BUY' else -quantity,
                    'average_price': price,
                    'total_cost': quantity * price,
                    'pnl': 0.0,
                    'position_type': PositionType.LONG if side == 'BUY' else PositionType.SHORT,
                    'created_time': time.time()
                }
            else:
                # Update existing position
                position = self.positions[symbol]
                current_quantity = position['quantity']
                current_cost = position['total_cost']
                
                new_quantity = quantity if side == 'BUY' else -quantity
                new_cost = quantity * price
                
                # Calculate new average price
                total_quantity = current_quantity + new_quantity
                total_cost = current_cost + (new_cost if side == 'BUY' else -new_cost)
                
                if total_quantity != 0:
                    position['quantity'] = total_quantity
                    position['average_price'] = abs(total_cost / total_quantity)
                    position['total_cost'] = total_cost
                    position['position_type'] = PositionType.LONG if total_quantity > 0 else PositionType.SHORT
                else:
                    # Position closed
                    del self.positions[symbol]
            
            self.trade_stats["trades_executed"] += 1
            self.trade_stats["positions_active"] = len(self.positions)
            
        except Exception as e:
            logger.error(f"Error updating position: {e}")
    
    async def _publish_trade_update(self, order_result: Dict):
        """Publish trade update message"""
        try:
            if not self.trade_publisher:
                return
            
            # Create trade message
            message = create_trade_message(
                order_id=order_result['order_id'],
                symbol=order_result['symbol'],
                side=order_result['side'],
                quantity=order_result['quantity'],
                price=order_result['price'],
                status=order_result['status'],
                source="trade_manager"
            )
            
            # Publish message
            await self.trade_publisher.publish_message(message)
            
        except Exception as e:
            logger.error(f"Error publishing trade update: {e}")
    
    async def _order_monitoring_loop(self):
        """Monitor order status updates"""
        logger.info("📋 Starting order monitoring...")
        
        try:
            while self.is_running:
                # Check order status updates
                await self._check_order_updates()
                await asyncio.sleep(5)  # Check every 5 seconds
                
        except Exception as e:
            logger.error(f"Error in order monitoring loop: {e}")
    
    async def _check_order_updates(self):
        """Check for order status updates"""
        try:
            # This is where we would check order status via OpenAlgo API
            # for order_id, order in self.orders.items():
            #     if order['status'] in [OrderStatus.SUBMITTED.value, OrderStatus.PARTIALLY_FILLED.value]:
            #         updated_order = await self.openalgo_client.get_order_status(order_id)
            #         if updated_order['status'] != order['status']:
            #             await self._handle_order_update(updated_order)
            
            pass  # Placeholder
            
        except Exception as e:
            logger.error(f"Error checking order updates: {e}")
    
    async def _position_monitoring_loop(self):
        """Monitor position updates and PnL"""
        logger.info("📊 Starting position monitoring...")
        
        try:
            while self.is_running:
                await self._update_position_pnl()
                await asyncio.sleep(10)  # Update every 10 seconds
                
        except Exception as e:
            logger.error(f"Error in position monitoring loop: {e}")
    
    async def _update_position_pnl(self):
        """Update position PnL based on current market prices"""
        try:
            # This is where we would get current market prices and calculate PnL
            # for symbol, position in self.positions.items():
            #     current_price = await self._get_current_price(symbol)
            #     if current_price:
            #         position['pnl'] = self._calculate_pnl(position, current_price)
            
            # Update daily PnL
            total_pnl = sum(pos.get('pnl', 0) for pos in self.positions.values())
            self.daily_pnl = total_pnl
            self.trade_stats["daily_pnl"] = self.daily_pnl
            
        except Exception as e:
            logger.error(f"Error updating position PnL: {e}")
    
    async def _risk_monitoring_loop(self):
        """Monitor risk management rules"""
        logger.info("🛡️  Starting risk monitoring...")
        
        try:
            while self.is_running:
                await self._check_risk_limits()
                await asyncio.sleep(30)  # Check every 30 seconds
                
        except Exception as e:
            logger.error(f"Error in risk monitoring loop: {e}")
    
    async def _check_risk_limits(self):
        """Check risk management limits"""
        try:
            # Check daily loss limit
            if self.daily_pnl <= -self.max_daily_loss:
                logger.critical(f"🚨 DAILY LOSS LIMIT BREACHED: ${self.daily_pnl:,.2f}")
                # Could trigger emergency position closure
            
            # Check position limits
            if len(self.positions) >= self.max_positions:
                logger.warning(f"⚠️  Position limit reached: {len(self.positions)}/{self.max_positions}")
            
            # Check individual position sizes
            for symbol, position in self.positions.items():
                position_value = abs(position['quantity'] * position['average_price'])
                if position_value > self.position_size_limit:
                    logger.warning(f"⚠️  Position size limit exceeded for {symbol}: ${position_value:,.2f}")
            
        except Exception as e:
            logger.error(f"Error checking risk limits: {e}")
    
    async def _close_all_positions(self):
        """Close all open positions"""
        logger.info("🔴 Closing all positions...")
        
        for symbol, position in self.positions.items():
            try:
                # Create closing order
                close_side = 'SELL' if position['quantity'] > 0 else 'BUY'
                close_quantity = abs(position['quantity'])
                
                logger.info(f"Closing position: {symbol} {close_side} {close_quantity}")
                
                # TODO: Execute closing order
                # await self._execute_closing_order(symbol, close_side, close_quantity)
                
            except Exception as e:
                logger.error(f"Error closing position {symbol}: {e}")
    
    async def _cancel_pending_orders(self):
        """Cancel all pending orders"""
        logger.info("❌ Cancelling pending orders...")
        
        for order_id, order in self.orders.items():
            if order['status'] in [OrderStatus.SUBMITTED.value, OrderStatus.PARTIALLY_FILLED.value]:
                try:
                    # TODO: Cancel order via OpenAlgo API
                    # await self.openalgo_client.cancel_order(order_id)
                    logger.info(f"Cancelled order: {order_id}")
                except Exception as e:
                    logger.error(f"Error cancelling order {order_id}: {e}")
    
    async def _statistics_loop(self):
        """Statistics and monitoring loop"""
        logger.info("📊 Starting trade statistics loop...")
        
        try:
            while self.is_running:
                await self._update_statistics()
                await asyncio.sleep(15)  # Update stats every 15 seconds
                
        except Exception as e:
            logger.error(f"Error in statistics loop: {e}")
    
    async def _update_statistics(self):
        """Update trade manager statistics"""
        try:
            current_time = time.time()
            uptime = current_time - self.start_time
            
            # Calculate win rate (placeholder)
            # In a full implementation, you would track wins/losses
            
            self.trade_stats.update({
                "positions_active": len(self.positions),
                "uptime": uptime,
                "trades_per_hour": self.trade_stats["trades_executed"] / (uptime / 3600) if uptime > 0 else 0
            })
            
            # Log statistics periodically
            if int(current_time) % 120 == 0:  # Every 2 minutes
                logger.info(
                    f"💼 Trade Stats: {len(self.positions)} positions, "
                    f"{self.trade_stats['trades_executed']} trades executed, "
                    f"Daily PnL: ${self.daily_pnl:,.2f}"
                )
            
        except Exception as e:
            logger.error(f"Error updating statistics: {e}")
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on trade manager"""
        current_time = time.time()
        
        # Check if we're processing signals
        time_since_last_trade = current_time - self.trade_stats.get("last_trade_time", 0)
        
        # Check risk limits
        risk_ok = (
            self.daily_pnl > -self.max_daily_loss and
            len(self.positions) <= self.max_positions
        )
        
        # Determine health status
        if risk_ok:
            status = "healthy"
        else:
            status = "unhealthy"
        
        return {
            "status": status,
            "timestamp": datetime.utcnow().isoformat(),
            "component": "trade_manager",
            "metrics": {
                "signals_received": self.trade_stats["signals_received"],
                "trades_executed": self.trade_stats["trades_executed"],
                "positions_active": len(self.positions),
                "daily_pnl": self.daily_pnl,
                "time_since_last_trade": time_since_last_trade,
                "uptime": current_time - self.start_time
            },
            "risk_status": {
                "daily_loss_limit_ok": self.daily_pnl > -self.max_daily_loss,
                "position_limit_ok": len(self.positions) <= self.max_positions,
                "overall_risk_ok": risk_ok
            },
            "issues": [] if status == "healthy" else [
                "Daily loss limit breached" if self.daily_pnl <= -self.max_daily_loss else "",
                "Position limit exceeded" if len(self.positions) > self.max_positions else ""
            ]
        }
    
    def get_positions(self) -> Dict[str, Dict]:
        """Get current positions"""
        return dict(self.positions)
    
    def get_orders(self) -> Dict[str, Dict]:
        """Get current orders"""
        return dict(self.orders)
    
    def get_trade_statistics(self) -> Dict[str, Any]:
        """Get trade manager statistics"""
        return dict(self.trade_stats)
