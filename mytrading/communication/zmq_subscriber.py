"""
ZeroMQ Subscriber
=================

High-performance ZeroMQ subscriber for receiving messages from publishers.
"""

import asyncio
import zmq
import zmq.asyncio
from typing import Optional, List, Any
from ..utils.logging_config import get_logger

logger = get_logger(__name__)


class ZMQSubscriber:
    """
    Asynchronous ZeroMQ subscriber
    
    Provides high-performance message receiving with topic filtering
    """
    
    def __init__(self, host: str = "localhost", port: int = 5555):
        self.host = host
        self.port = port
        self.context: Optional[zmq.asyncio.Context] = None
        self.socket: Optional[zmq.asyncio.Socket] = None
        self.is_connected = False
        
        logger.info(f"📡 ZMQSubscriber initialized - {host}:{port}")
    
    async def connect(self):
        """Connect to the publisher"""
        try:
            self.context = zmq.asyncio.Context()
            self.socket = self.context.socket(zmq.SUB)
            
            # Connect to publisher
            address = f"tcp://{self.host}:{self.port}"
            self.socket.connect(address)
            
            self.is_connected = True
            logger.success(f"✅ Connected to ZMQ publisher at {address}")
            
        except Exception as e:
            logger.error(f"❌ Failed to connect to ZMQ publisher: {e}")
            raise
    
    async def disconnect(self):
        """Disconnect from the publisher"""
        if self.socket:
            self.socket.close()
        if self.context:
            self.context.term()
        
        self.is_connected = False
        logger.info("📡 ZMQ subscriber disconnected")
    
    async def subscribe(self, topic: str = ""):
        """Subscribe to a topic"""
        if not self.socket:
            raise RuntimeError("Not connected to publisher")
        
        self.socket.setsockopt_string(zmq.SUBSCRIBE, topic)
        logger.info(f"📡 Subscribed to topic: '{topic}'")
    
    async def unsubscribe(self, topic: str = ""):
        """Unsubscribe from a topic"""
        if not self.socket:
            raise RuntimeError("Not connected to publisher")
        
        self.socket.setsockopt_string(zmq.UNSUBSCRIBE, topic)
        logger.info(f"📡 Unsubscribed from topic: '{topic}'")
    
    async def receive_message(self, timeout: Optional[int] = None) -> Optional[bytes]:
        """Receive a message"""
        if not self.socket:
            return None
        
        try:
            if timeout:
                # Use poller for timeout
                poller = zmq.asyncio.Poller()
                poller.register(self.socket, zmq.POLLIN)
                
                events = await poller.poll(timeout)
                if not events:
                    return None  # Timeout
            
            # Receive message
            message = await self.socket.recv()
            return message
            
        except zmq.Again:
            return None
        except Exception as e:
            logger.error(f"Error receiving message: {e}")
            return None
    
    async def receive_multipart(self, timeout: Optional[int] = None) -> Optional[List[bytes]]:
        """Receive a multipart message"""
        if not self.socket:
            return None
        
        try:
            if timeout:
                poller = zmq.asyncio.Poller()
                poller.register(self.socket, zmq.POLLIN)
                
                events = await poller.poll(timeout)
                if not events:
                    return None
            
            # Receive multipart message
            message_parts = await self.socket.recv_multipart()
            return message_parts
            
        except zmq.Again:
            return None
        except Exception as e:
            logger.error(f"Error receiving multipart message: {e}")
            return None
