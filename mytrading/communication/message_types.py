"""
Message Type Definitions
========================

Standard message formats and types for inter-component communication
in the MyTrading system.
"""

import json
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum


class MessageType(str, Enum):
    """Standard message types for the trading system"""
    
    # Market data messages
    MARKET_DATA = "market_data"
    TICK_DATA = "tick_data"
    DEPTH_DATA = "depth_data"
    HISTORICAL_DATA = "historical_data"
    
    # Strategy and signal messages
    STRATEGY_SIGNAL = "strategy_signal"
    TRADE_SIGNAL = "trade_signal"
    RISK_SIGNAL = "risk_signal"
    
    # Trade execution messages
    ORDER_REQUEST = "order_request"
    ORDER_UPDATE = "order_update"
    TRADE_EXECUTION = "trade_execution"
    POSITION_UPDATE = "position_update"
    
    # System messages
    SYSTEM_STATUS = "system_status"
    HEALTH_CHECK = "health_check"
    PERFORMANCE_METRICS = "performance_metrics"
    ERROR_MESSAGE = "error_message"
    
    # Control messages
    START_TRADING = "start_trading"
    STOP_TRADING = "stop_trading"
    PAUSE_TRADING = "pause_trading"
    SHUTDOWN = "shutdown"


class Priority(int, Enum):
    """Message priority levels"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class BaseMessage:
    """Base message class with common fields"""
    message_type: MessageType
    timestamp: datetime
    source: str
    priority: Priority = Priority.NORMAL
    correlation_id: Optional[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        
        # Ensure timestamp is datetime object
        if isinstance(self.timestamp, str):
            self.timestamp = datetime.fromisoformat(self.timestamp)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary"""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        data['message_type'] = self.message_type.value
        data['priority'] = self.priority.value
        return data
    
    def to_json(self) -> str:
        """Convert message to JSON string"""
        return json.dumps(self.to_dict(), default=str)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BaseMessage':
        """Create message from dictionary"""
        # Convert string enums back to enum objects
        if 'message_type' in data:
            data['message_type'] = MessageType(data['message_type'])
        if 'priority' in data:
            data['priority'] = Priority(data['priority'])
        if 'timestamp' in data and isinstance(data['timestamp'], str):
            data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        
        return cls(**data)
    
    @classmethod
    def from_json(cls, json_str: str) -> 'BaseMessage':
        """Create message from JSON string"""
        data = json.loads(json_str)
        return cls.from_dict(data)


@dataclass
class MarketDataMessage(BaseMessage):
    """Market data message"""
    symbol: str
    exchange: str
    data: Dict[str, Any]
    
    # Market data specific fields
    ltp: Optional[float] = None
    volume: Optional[int] = None
    open_interest: Optional[int] = None
    bid: Optional[float] = None
    ask: Optional[float] = None
    bid_quantity: Optional[int] = None
    ask_quantity: Optional[int] = None
    
    # OHLC data
    open: Optional[float] = None
    high: Optional[float] = None
    low: Optional[float] = None
    close: Optional[float] = None
    
    # Depth data
    bid_levels: Optional[List[Dict[str, float]]] = None
    ask_levels: Optional[List[Dict[str, float]]] = None
    
    def __post_init__(self):
        super().__post_init__()
        self.message_type = MessageType.MARKET_DATA
        
        # Extract common fields from data if not provided
        if self.data:
            self.ltp = self.ltp or self.data.get('ltp')
            self.volume = self.volume or self.data.get('volume')
            self.open_interest = self.open_interest or self.data.get('oi')
            self.bid = self.bid or self.data.get('bid')
            self.ask = self.ask or self.data.get('ask')
            self.open = self.open or self.data.get('open')
            self.high = self.high or self.data.get('high')
            self.low = self.low or self.data.get('low')
            self.close = self.close or self.data.get('close')


@dataclass
class TickDataMessage(BaseMessage):
    """Tick data message for high-frequency updates"""
    symbol: str
    exchange: str
    price: float
    quantity: int
    timestamp_exchange: Optional[datetime] = None
    
    def __post_init__(self):
        super().__post_init__()
        self.message_type = MessageType.TICK_DATA


@dataclass
class DepthDataMessage(BaseMessage):
    """Market depth/order book message"""
    symbol: str
    exchange: str
    bid_levels: List[Dict[str, Union[float, int]]]  # [{"price": 100.0, "quantity": 50}, ...]
    ask_levels: List[Dict[str, Union[float, int]]]  # [{"price": 101.0, "quantity": 25}, ...]
    
    def __post_init__(self):
        super().__post_init__()
        self.message_type = MessageType.DEPTH_DATA


@dataclass
class SignalMessage(BaseMessage):
    """Trading signal message"""
    strategy_name: str
    symbol: str
    signal_type: str  # BUY, SELL, HOLD, EXIT
    confidence: float  # 0.0 to 1.0
    
    # Signal parameters
    entry_price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    quantity: Optional[int] = None
    
    # Strategy context
    timeframe: Optional[str] = None
    indicators: Optional[Dict[str, float]] = None
    reasoning: Optional[str] = None
    
    def __post_init__(self):
        super().__post_init__()
        self.message_type = MessageType.STRATEGY_SIGNAL


@dataclass
class TradeMessage(BaseMessage):
    """Trade execution message"""
    order_id: str
    symbol: str
    action: str  # BUY, SELL
    quantity: int
    order_type: str  # MARKET, LIMIT, STOP_LOSS, etc.
    
    # Order parameters
    price: Optional[float] = None
    stop_price: Optional[float] = None
    
    # Execution details
    filled_quantity: Optional[int] = None
    average_price: Optional[float] = None
    status: Optional[str] = None  # PENDING, FILLED, CANCELLED, REJECTED
    
    # Strategy context
    strategy_name: Optional[str] = None
    signal_id: Optional[str] = None
    
    def __post_init__(self):
        super().__post_init__()
        self.message_type = MessageType.TRADE_EXECUTION


@dataclass
class OrderMessage(BaseMessage):
    """Order management message"""
    order_id: str
    symbol: str
    action: str  # BUY, SELL, CANCEL, MODIFY
    quantity: int
    order_type: str
    
    # Order parameters
    price: Optional[float] = None
    stop_price: Optional[float] = None
    validity: Optional[str] = None  # DAY, IOC, GTD
    
    # Order status
    status: Optional[str] = None
    filled_quantity: Optional[int] = None
    remaining_quantity: Optional[int] = None
    average_price: Optional[float] = None
    
    # Timestamps
    order_time: Optional[datetime] = None
    update_time: Optional[datetime] = None
    
    def __post_init__(self):
        super().__post_init__()
        self.message_type = MessageType.ORDER_UPDATE


@dataclass
class PositionMessage(BaseMessage):
    """Position update message"""
    symbol: str
    quantity: int  # Positive for long, negative for short
    average_price: float
    current_price: float
    
    # P&L information
    unrealized_pnl: float
    realized_pnl: float
    
    # Position details
    market_value: float
    day_change: Optional[float] = None
    day_change_percent: Optional[float] = None
    
    def __post_init__(self):
        super().__post_init__()
        self.message_type = MessageType.POSITION_UPDATE


@dataclass
class SystemStatusMessage(BaseMessage):
    """System status message"""
    component: str
    status: str  # RUNNING, STOPPED, ERROR, WARNING
    details: Optional[str] = None
    
    # Performance metrics
    cpu_usage: Optional[float] = None
    memory_usage: Optional[float] = None
    uptime: Optional[float] = None
    
    def __post_init__(self):
        super().__post_init__()
        self.message_type = MessageType.SYSTEM_STATUS


@dataclass
class PerformanceMessage(BaseMessage):
    """Performance metrics message"""
    component: str
    operation: str
    duration_ms: float
    success: bool
    
    # Additional metrics
    throughput: Optional[float] = None
    error_rate: Optional[float] = None
    latency_p95: Optional[float] = None
    latency_p99: Optional[float] = None
    
    def __post_init__(self):
        super().__post_init__()
        self.message_type = MessageType.PERFORMANCE_METRICS


@dataclass
class ErrorMessage(BaseMessage):
    """Error message"""
    component: str
    error_type: str
    error_message: str
    
    # Error details
    exception_type: Optional[str] = None
    stack_trace: Optional[str] = None
    context: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        super().__post_init__()
        self.message_type = MessageType.ERROR_MESSAGE
        self.priority = Priority.HIGH


@dataclass
class ControlMessage(BaseMessage):
    """System control message"""
    command: str  # START, STOP, PAUSE, SHUTDOWN, RESTART
    target_component: Optional[str] = None  # Specific component or None for all
    parameters: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        super().__post_init__()
        # Set message type based on command
        command_map = {
            "START": MessageType.START_TRADING,
            "STOP": MessageType.STOP_TRADING,
            "PAUSE": MessageType.PAUSE_TRADING,
            "SHUTDOWN": MessageType.SHUTDOWN
        }
        self.message_type = command_map.get(self.command, MessageType.SYSTEM_STATUS)
        self.priority = Priority.HIGH


# Message factory functions
def create_market_data_message(
    symbol: str,
    exchange: str,
    data: Dict[str, Any],
    source: str = "market_data"
) -> MarketDataMessage:
    """Create a market data message"""
    return MarketDataMessage(
        message_type=MessageType.MARKET_DATA,
        timestamp=datetime.utcnow(),
        source=source,
        symbol=symbol,
        exchange=exchange,
        data=data
    )


def create_signal_message(
    strategy_name: str,
    symbol: str,
    signal_type: str,
    confidence: float,
    source: str = "strategy_engine",
    **kwargs
) -> SignalMessage:
    """Create a trading signal message"""
    return SignalMessage(
        message_type=MessageType.STRATEGY_SIGNAL,
        timestamp=datetime.utcnow(),
        source=source,
        strategy_name=strategy_name,
        symbol=symbol,
        signal_type=signal_type,
        confidence=confidence,
        **kwargs
    )


def create_trade_message(
    order_id: str,
    symbol: str,
    action: str,
    quantity: int,
    order_type: str,
    source: str = "trade_manager",
    **kwargs
) -> TradeMessage:
    """Create a trade execution message"""
    return TradeMessage(
        message_type=MessageType.TRADE_EXECUTION,
        timestamp=datetime.utcnow(),
        source=source,
        order_id=order_id,
        symbol=symbol,
        action=action,
        quantity=quantity,
        order_type=order_type,
        **kwargs
    )


def create_error_message(
    component: str,
    error_type: str,
    error_message: str,
    source: str = "system",
    **kwargs
) -> ErrorMessage:
    """Create an error message"""
    return ErrorMessage(
        message_type=MessageType.ERROR_MESSAGE,
        timestamp=datetime.utcnow(),
        source=source,
        component=component,
        error_type=error_type,
        error_message=error_message,
        priority=Priority.HIGH,
        **kwargs
    )


def create_system_status_message(
    component: str,
    status: str,
    source: str = "system",
    **kwargs
) -> SystemStatusMessage:
    """Create a system status message"""
    return SystemStatusMessage(
        message_type=MessageType.SYSTEM_STATUS,
        timestamp=datetime.utcnow(),
        source=source,
        component=component,
        status=status,
        **kwargs
    )


# Message parsing utilities
def parse_message(message_data: Union[str, bytes, Dict]) -> BaseMessage:
    """
    Parse message from various formats
    
    Args:
        message_data: Message data (JSON string, bytes, or dict)
        
    Returns:
        Parsed message object
        
    Raises:
        ValueError: If message format is invalid
    """
    if isinstance(message_data, bytes):
        message_data = message_data.decode('utf-8')
    
    if isinstance(message_data, str):
        try:
            message_data = json.loads(message_data)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON message: {e}")
    
    if not isinstance(message_data, dict):
        raise ValueError("Message must be a dictionary")
    
    # Get message type
    message_type_str = message_data.get('message_type')
    if not message_type_str:
        raise ValueError("Message missing 'message_type' field")
    
    try:
        message_type = MessageType(message_type_str)
    except ValueError:
        raise ValueError(f"Unknown message type: {message_type_str}")
    
    # Create appropriate message object based on type
    message_classes = {
        MessageType.MARKET_DATA: MarketDataMessage,
        MessageType.TICK_DATA: TickDataMessage,
        MessageType.DEPTH_DATA: DepthDataMessage,
        MessageType.STRATEGY_SIGNAL: SignalMessage,
        MessageType.TRADE_EXECUTION: TradeMessage,
        MessageType.ORDER_UPDATE: OrderMessage,
        MessageType.POSITION_UPDATE: PositionMessage,
        MessageType.SYSTEM_STATUS: SystemStatusMessage,
        MessageType.PERFORMANCE_METRICS: PerformanceMessage,
        MessageType.ERROR_MESSAGE: ErrorMessage,
        MessageType.START_TRADING: ControlMessage,
        MessageType.STOP_TRADING: ControlMessage,
        MessageType.PAUSE_TRADING: ControlMessage,
        MessageType.SHUTDOWN: ControlMessage,
    }
    
    message_class = message_classes.get(message_type, BaseMessage)
    
    try:
        return message_class.from_dict(message_data)
    except Exception as e:
        raise ValueError(f"Error creating {message_class.__name__}: {e}")


def get_message_topic(message: BaseMessage) -> str:
    """
    Get ZMQ topic for message routing
    
    Args:
        message: Message object
        
    Returns:
        Topic string for ZMQ publishing
    """
    base_topic = message.message_type.value
    
    # Add symbol-specific routing for market data and signals
    if hasattr(message, 'symbol') and message.symbol:
        return f"{base_topic}.{message.symbol}"
    
    # Add component-specific routing for system messages
    if hasattr(message, 'component') and message.component:
        return f"{base_topic}.{message.component}"
    
    return base_topic
