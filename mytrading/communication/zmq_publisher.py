"""
ZeroMQ Publisher
===============

High-performance message publisher using ZeroMQ for broadcasting messages
to multiple subscribers in the trading system.
"""

import zmq
import zmq.asyncio
import asyncio
import threading
import time
from typing import Optional, Dict, Any, List, Callable
from datetime import datetime
import json

from .message_types import BaseMessage, MessageType, Priority, get_message_topic
from ..utils.logging_config import get_logger
from ..utils.performance_monitor import performance_timer, get_performance_monitor

logger = get_logger(__name__)


class ZMQPublisher:
    """
    High-performance ZeroMQ publisher for broadcasting messages
    """
    
    def __init__(
        self,
        port: int,
        bind_address: str = "*",
        high_water_mark: int = 10000,
        linger: int = 1000,
        use_async: bool = True
    ):
        self.port = port
        self.bind_address = bind_address
        self.high_water_mark = high_water_mark
        self.linger = linger
        self.use_async = use_async
        
        # ZMQ context and socket
        if use_async:
            self.context = zmq.asyncio.Context()
        else:
            self.context = zmq.Context()
        
        self.socket: Optional[zmq.Socket] = None
        self.is_connected = False
        
        # Performance tracking
        self.message_count = 0
        self.bytes_sent = 0
        self.start_time = time.time()
        self.last_stats_time = time.time()
        
        # Message filtering and routing
        self.topic_filters: Dict[str, List[Callable]] = {}
        self.priority_queue: Dict[Priority, List[BaseMessage]] = {
            Priority.LOW: [],
            Priority.NORMAL: [],
            Priority.HIGH: [],
            Priority.CRITICAL: []
        }
        
        # Batch publishing
        self.batch_size = 100
        self.batch_timeout = 0.1  # 100ms
        self.pending_messages: List[BaseMessage] = []
        self.last_batch_time = time.time()
        
        # Thread safety
        self.lock = threading.RLock()
        
        logger.info(f"ZMQ Publisher initialized on port {port}")
    
    async def connect(self) -> bool:
        """
        Connect to ZMQ socket and start publishing
        
        Returns:
            True if connection successful
        """
        try:
            self.socket = self.context.socket(zmq.PUB)
            
            # Configure socket options
            self.socket.setsockopt(zmq.SNDHWM, self.high_water_mark)
            self.socket.setsockopt(zmq.LINGER, self.linger)
            self.socket.setsockopt(zmq.IMMEDIATE, 1)  # Don't queue messages for disconnected peers
            
            # Bind to address
            bind_url = f"tcp://{self.bind_address}:{self.port}"
            self.socket.bind(bind_url)
            
            self.is_connected = True
            self.start_time = time.time()
            
            logger.info(f"✅ ZMQ Publisher connected to {bind_url}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to connect ZMQ Publisher: {e}")
            return False
    
    def disconnect(self):
        """Disconnect from ZMQ socket"""
        try:
            if self.socket:
                self.socket.close()
                self.socket = None
            
            if self.context:
                self.context.term()
            
            self.is_connected = False
            logger.info("ZMQ Publisher disconnected")
            
        except Exception as e:
            logger.error(f"Error disconnecting ZMQ Publisher: {e}")
    
    @performance_timer("zmq_publisher", "publish_message")
    async def publish_message(
        self,
        message: BaseMessage,
        topic: Optional[str] = None
    ) -> bool:
        """
        Publish a message to subscribers
        
        Args:
            message: Message to publish
            topic: Optional topic override
            
        Returns:
            True if message published successfully
        """
        if not self.is_connected or not self.socket:
            logger.warning("Publisher not connected, cannot publish message")
            return False
        
        try:
            # Get topic for message routing
            if topic is None:
                topic = get_message_topic(message)
            
            # Apply topic filters
            if not self._should_publish_message(topic, message):
                return True  # Filtered out, but not an error
            
            # Serialize message
            message_data = message.to_json()
            
            # Publish message
            if self.use_async:
                await self.socket.send_multipart([
                    topic.encode('utf-8'),
                    message_data.encode('utf-8')
                ])
            else:
                self.socket.send_multipart([
                    topic.encode('utf-8'),
                    message_data.encode('utf-8')
                ])
            
            # Update statistics
            with self.lock:
                self.message_count += 1
                self.bytes_sent += len(message_data)
            
            # Log high-priority messages
            if message.priority in [Priority.HIGH, Priority.CRITICAL]:
                logger.info(f"Published {message.priority.name} message: {topic}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error publishing message: {e}")
            return False
    
    @performance_timer("zmq_publisher", "publish_batch")
    async def publish_batch(self, messages: List[BaseMessage]) -> int:
        """
        Publish multiple messages in batch
        
        Args:
            messages: List of messages to publish
            
        Returns:
            Number of messages published successfully
        """
        if not messages:
            return 0
        
        published_count = 0
        
        for message in messages:
            if await self.publish_message(message):
                published_count += 1
        
        logger.debug(f"Published batch: {published_count}/{len(messages)} messages")
        return published_count
    
    def publish_sync(self, message: BaseMessage, topic: Optional[str] = None) -> bool:
        """
        Synchronous message publishing for non-async contexts
        
        Args:
            message: Message to publish
            topic: Optional topic override
            
        Returns:
            True if published successfully
        """
        if self.use_async:
            # Run in event loop if available
            try:
                loop = asyncio.get_event_loop()
                return loop.run_until_complete(self.publish_message(message, topic))
            except RuntimeError:
                # No event loop, create new one
                return asyncio.run(self.publish_message(message, topic))
        else:
            # Direct synchronous publishing
            if not self.is_connected or not self.socket:
                return False
            
            try:
                topic = topic or get_message_topic(message)
                message_data = message.to_json()
                
                self.socket.send_multipart([
                    topic.encode('utf-8'),
                    message_data.encode('utf-8')
                ])
                
                with self.lock:
                    self.message_count += 1
                    self.bytes_sent += len(message_data)
                
                return True
                
            except Exception as e:
                logger.error(f"Error in sync publish: {e}")
                return False
    
    def add_topic_filter(self, topic_pattern: str, filter_func: Callable[[BaseMessage], bool]):
        """
        Add message filter for specific topic pattern
        
        Args:
            topic_pattern: Topic pattern to filter
            filter_func: Function that returns True if message should be published
        """
        if topic_pattern not in self.topic_filters:
            self.topic_filters[topic_pattern] = []
        
        self.topic_filters[topic_pattern].append(filter_func)
        logger.debug(f"Added topic filter for: {topic_pattern}")
    
    def _should_publish_message(self, topic: str, message: BaseMessage) -> bool:
        """
        Check if message should be published based on filters
        
        Args:
            topic: Message topic
            message: Message to check
            
        Returns:
            True if message should be published
        """
        # Check topic-specific filters
        for pattern, filters in self.topic_filters.items():
            if pattern in topic or pattern == "*":
                for filter_func in filters:
                    try:
                        if not filter_func(message):
                            return False
                    except Exception as e:
                        logger.error(f"Error in topic filter: {e}")
        
        return True
    
    async def start_batch_publisher(self):
        """Start background batch publisher for improved performance"""
        logger.info("Starting batch publisher")
        
        while self.is_connected:
            try:
                current_time = time.time()
                
                # Check if we should publish batch
                should_publish = (
                    len(self.pending_messages) >= self.batch_size or
                    (self.pending_messages and 
                     current_time - self.last_batch_time >= self.batch_timeout)
                )
                
                if should_publish:
                    with self.lock:
                        messages_to_publish = self.pending_messages.copy()
                        self.pending_messages.clear()
                        self.last_batch_time = current_time
                    
                    if messages_to_publish:
                        await self.publish_batch(messages_to_publish)
                
                # Small delay to prevent busy waiting
                await asyncio.sleep(0.01)  # 10ms
                
            except Exception as e:
                logger.error(f"Error in batch publisher: {e}")
                await asyncio.sleep(1)  # Longer delay on error
    
    def queue_message(self, message: BaseMessage):
        """
        Queue message for batch publishing
        
        Args:
            message: Message to queue
        """
        with self.lock:
            # Insert message based on priority
            if message.priority == Priority.CRITICAL:
                # Critical messages bypass queue and publish immediately
                asyncio.create_task(self.publish_message(message))
            else:
                self.pending_messages.append(message)
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get publisher statistics
        
        Returns:
            Dictionary with performance statistics
        """
        current_time = time.time()
        uptime = current_time - self.start_time
        
        with self.lock:
            stats = {
                "is_connected": self.is_connected,
                "port": self.port,
                "uptime_seconds": uptime,
                "total_messages": self.message_count,
                "total_bytes": self.bytes_sent,
                "messages_per_second": self.message_count / uptime if uptime > 0 else 0,
                "bytes_per_second": self.bytes_sent / uptime if uptime > 0 else 0,
                "pending_messages": len(self.pending_messages),
                "topic_filters": len(self.topic_filters),
                "high_water_mark": self.high_water_mark,
                "batch_size": self.batch_size,
                "batch_timeout": self.batch_timeout
            }
        
        return stats
    
    def log_statistics(self):
        """Log current statistics"""
        stats = self.get_statistics()
        
        logger.info(
            f"📊 Publisher Stats: "
            f"{stats['total_messages']:,} msgs, "
            f"{stats['messages_per_second']:.1f} msg/sec, "
            f"{stats['bytes_per_second']/1024:.1f} KB/sec, "
            f"Pending: {stats['pending_messages']}"
        )
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on publisher
        
        Returns:
            Health check results
        """
        health = {
            "status": "healthy" if self.is_connected else "unhealthy",
            "timestamp": datetime.utcnow().isoformat(),
            "component": "zmq_publisher",
            "port": self.port,
            "is_connected": self.is_connected,
            "message_count": self.message_count,
            "pending_messages": len(self.pending_messages)
        }
        
        # Check for potential issues
        issues = []
        
        if not self.is_connected:
            issues.append("Not connected to ZMQ socket")
        
        if len(self.pending_messages) > self.batch_size * 10:
            issues.append(f"High pending message count: {len(self.pending_messages)}")
        
        current_time = time.time()
        if current_time - self.last_batch_time > self.batch_timeout * 10:
            issues.append("Batch publishing appears stalled")
        
        if issues:
            health["status"] = "warning"
            health["issues"] = issues
        
        return health
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.disconnect()
    
    async def __aenter__(self):
        """Async context manager entry"""
        await self.connect()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        self.disconnect()


# Convenience functions for common publishing patterns
async def publish_market_data(
    publisher: ZMQPublisher,
    symbol: str,
    exchange: str,
    data: Dict[str, Any],
    source: str = "market_data"
):
    """Publish market data message"""
    from .message_types import create_market_data_message
    
    message = create_market_data_message(symbol, exchange, data, source)
    await publisher.publish_message(message)


async def publish_signal(
    publisher: ZMQPublisher,
    strategy_name: str,
    symbol: str,
    signal_type: str,
    confidence: float,
    source: str = "strategy_engine",
    **kwargs
):
    """Publish trading signal message"""
    from .message_types import create_signal_message
    
    message = create_signal_message(
        strategy_name, symbol, signal_type, confidence, source, **kwargs
    )
    await publisher.publish_message(message)


async def publish_error(
    publisher: ZMQPublisher,
    component: str,
    error_type: str,
    error_message: str,
    source: str = "system",
    **kwargs
):
    """Publish error message"""
    from .message_types import create_error_message
    
    message = create_error_message(
        component, error_type, error_message, source, **kwargs
    )
    await publisher.publish_message(message)
