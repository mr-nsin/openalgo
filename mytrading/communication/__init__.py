"""
Communication Module
===================

High-performance messaging system for the MyTrading system using ZeroMQ
for inter-component communication.

Components:
- ZMQPublisher: High-performance message publishing
- ZMQSubscriber: Message subscription and handling
- MessageRouter: Route messages between components
- MessageTypes: Standard message format definitions
"""

from .zmq_publisher import ZMQPublisher
from .zmq_subscriber import ZMQSubscriber
from .message_router import MessageRouter
from .message_types import MessageType, MarketDataMessage, SignalMessage, TradeMessage

__all__ = [
    "ZMQPublisher",
    "ZMQSubscriber", 
    "MessageRouter",
    "MessageType",
    "MarketDataMessage",
    "SignalMessage",
    "TradeMessage"
]
