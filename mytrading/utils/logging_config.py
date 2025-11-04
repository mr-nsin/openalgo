"""
Logging Configuration
====================

Advanced logging setup for the MyTrading system with structured logging,
performance tracking, and multiple output formats.
"""

import sys
import os
from pathlib import Path
from typing import Optional, Dict, Any
from loguru import logger
import json
from datetime import datetime


class StructuredFormatter:
    """Custom formatter for structured logging"""
    
    def __init__(self, include_extra: bool = True):
        self.include_extra = include_extra
    
    def format(self, record: Dict[str, Any]) -> str:
        """Format log record as structured JSON"""
        # Base log data
        log_data = {
            "timestamp": record["time"].isoformat(),
            "level": record["level"].name,
            "logger": record["name"],
            "message": record["message"],
            "module": record["module"],
            "function": record["function"],
            "line": record["line"]
        }
        
        # Add extra fields if available
        if self.include_extra and "extra" in record:
            extra = record["extra"]
            if extra:
                log_data["extra"] = extra
        
        # Add exception info if present
        if record.get("exception"):
            log_data["exception"] = {
                "type": record["exception"].type.__name__,
                "value": str(record["exception"].value),
                "traceback": record["exception"].traceback
            }
        
        return json.dumps(log_data, default=str)


class TradingLogger:
    """Enhanced logger for trading system with context management"""
    
    def __init__(self, name: str):
        self.name = name
        self.context: Dict[str, Any] = {}
    
    def set_context(self, **kwargs):
        """Set logging context for this logger"""
        self.context.update(kwargs)
    
    def clear_context(self):
        """Clear logging context"""
        self.context.clear()
    
    def _log_with_context(self, level: str, message: str, **kwargs):
        """Log message with context"""
        extra = {**self.context, **kwargs}
        getattr(logger.bind(**extra), level)(message)
    
    def debug(self, message: str, **kwargs):
        """Log debug message"""
        self._log_with_context("debug", message, **kwargs)
    
    def info(self, message: str, **kwargs):
        """Log info message"""
        self._log_with_context("info", message, **kwargs)
    
    def warning(self, message: str, **kwargs):
        """Log warning message"""
        self._log_with_context("warning", message, **kwargs)
    
    def error(self, message: str, **kwargs):
        """Log error message"""
        self._log_with_context("error", message, **kwargs)
    
    def critical(self, message: str, **kwargs):
        """Log critical message"""
        self._log_with_context("critical", message, **kwargs)
    
    def trade(self, message: str, **kwargs):
        """Log trading-specific message"""
        self._log_with_context("info", f"[TRADE] {message}", trade_event=True, **kwargs)
    
    def performance(self, message: str, **kwargs):
        """Log performance-specific message"""
        self._log_with_context("info", f"[PERF] {message}", performance_event=True, **kwargs)
    
    def market_data(self, message: str, **kwargs):
        """Log market data message"""
        self._log_with_context("debug", f"[MARKET] {message}", market_data_event=True, **kwargs)
    
    def strategy(self, message: str, **kwargs):
        """Log strategy-specific message"""
        self._log_with_context("info", f"[STRATEGY] {message}", strategy_event=True, **kwargs)


def setup_logging(
    level: str = "INFO",
    log_directory: str = "logs",
    console_enabled: bool = True,
    file_enabled: bool = True,
    structured_logging: bool = False,
    max_file_size: str = "100 MB",
    retention_days: int = 30
) -> None:
    """
    Setup comprehensive logging for the trading system
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_directory: Directory for log files
        console_enabled: Enable console logging
        file_enabled: Enable file logging
        structured_logging: Enable structured JSON logging
        max_file_size: Maximum size per log file
        retention_days: Number of days to retain logs
    """
    
    # Remove default logger
    logger.remove()
    
    # Create log directory
    log_dir = Path(log_directory)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Console logging
    if console_enabled:
        console_format = (
            "<green>{time:HH:mm:ss.SSS}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>"
        )
        
        if structured_logging:
            logger.add(
                sys.stdout,
                format=StructuredFormatter().format,
                level=level,
                colorize=False,
                serialize=True
            )
        else:
            logger.add(
                sys.stdout,
                format=console_format,
                level=level,
                colorize=True
            )
    
    # File logging
    if file_enabled:
        # Main log file
        main_log_file = log_dir / "trading_{time:YYYY-MM-DD}.log"
        
        if structured_logging:
            file_format = StructuredFormatter().format
        else:
            file_format = (
                "{time:YYYY-MM-DD HH:mm:ss.SSS} | "
                "{level: <8} | "
                "{name}:{function}:{line} | "
                "{message}"
            )
        
        logger.add(
            str(main_log_file),
            format=file_format,
            level=level,
            rotation=max_file_size,
            retention=f"{retention_days} days",
            compression="zip",
            serialize=structured_logging
        )
        
        # Separate log files for different components
        component_logs = {
            "trades": "trades_{time:YYYY-MM-DD}.log",
            "performance": "performance_{time:YYYY-MM-DD}.log", 
            "errors": "errors_{time:YYYY-MM-DD}.log",
            "market_data": "market_data_{time:YYYY-MM-DD}.log"
        }
        
        for component, filename in component_logs.items():
            log_file = log_dir / filename
            
            # Filter function for component-specific logs
            def make_filter(comp):
                def filter_func(record):
                    extra = record.get("extra", {})
                    if comp == "trades":
                        return extra.get("trade_event", False)
                    elif comp == "performance":
                        return extra.get("performance_event", False)
                    elif comp == "errors":
                        return record["level"].name in ["ERROR", "CRITICAL"]
                    elif comp == "market_data":
                        return extra.get("market_data_event", False)
                    return False
                return filter_func
            
            logger.add(
                str(log_file),
                format=file_format,
                level="DEBUG",
                filter=make_filter(component),
                rotation=max_file_size,
                retention=f"{retention_days} days",
                compression="zip",
                serialize=structured_logging
            )
    
    # Add process info to all logs
    logger.configure(
        extra={
            "process_id": os.getpid(),
            "system": "MyTrading",
            "version": "1.0.0"
        }
    )
    
    logger.info(f"Logging initialized - Level: {level}, Directory: {log_directory}")


def get_logger(name: str) -> TradingLogger:
    """
    Get a trading logger instance with the specified name
    
    Args:
        name: Logger name (usually module name)
        
    Returns:
        TradingLogger instance
    """
    return TradingLogger(name)


def log_system_info():
    """Log system information at startup"""
    import platform
    import psutil
    
    system_info = {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "cpu_count": psutil.cpu_count(),
        "memory_total": psutil.virtual_memory().total,
        "disk_space": psutil.disk_usage('/').total if os.name != 'nt' else psutil.disk_usage('C:').total
    }
    
    logger.info("System Information", **system_info)


def log_performance_metrics(
    component: str,
    operation: str,
    duration: float,
    success: bool = True,
    **kwargs
):
    """
    Log performance metrics in a structured format
    
    Args:
        component: Component name (e.g., 'websocket', 'strategy')
        operation: Operation name (e.g., 'data_processing', 'signal_generation')
        duration: Operation duration in seconds
        success: Whether operation was successful
        **kwargs: Additional metrics
    """
    metrics = {
        "component": component,
        "operation": operation,
        "duration_ms": round(duration * 1000, 2),
        "success": success,
        "timestamp": datetime.utcnow().isoformat(),
        **kwargs
    }
    
    logger.bind(performance_event=True).info(
        f"Performance: {component}.{operation} took {duration*1000:.2f}ms",
        **metrics
    )


def log_trade_event(
    event_type: str,
    symbol: str,
    action: str,
    quantity: Optional[float] = None,
    price: Optional[float] = None,
    **kwargs
):
    """
    Log trading events in a structured format
    
    Args:
        event_type: Type of trade event (order, fill, cancel, etc.)
        symbol: Trading symbol
        action: Action taken (buy, sell, cancel, etc.)
        quantity: Order/fill quantity
        price: Order/fill price
        **kwargs: Additional trade data
    """
    trade_data = {
        "event_type": event_type,
        "symbol": symbol,
        "action": action,
        "timestamp": datetime.utcnow().isoformat(),
        **kwargs
    }
    
    if quantity is not None:
        trade_data["quantity"] = quantity
    if price is not None:
        trade_data["price"] = price
    
    logger.bind(trade_event=True).info(
        f"Trade Event: {event_type} - {symbol} {action}",
        **trade_data
    )


def log_market_data_event(
    symbol: str,
    data_type: str,
    price: Optional[float] = None,
    volume: Optional[int] = None,
    **kwargs
):
    """
    Log market data events
    
    Args:
        symbol: Trading symbol
        data_type: Type of market data (tick, quote, depth, etc.)
        price: Current price
        volume: Current volume
        **kwargs: Additional market data
    """
    market_data = {
        "symbol": symbol,
        "data_type": data_type,
        "timestamp": datetime.utcnow().isoformat(),
        **kwargs
    }
    
    if price is not None:
        market_data["price"] = price
    if volume is not None:
        market_data["volume"] = volume
    
    logger.bind(market_data_event=True).debug(
        f"Market Data: {symbol} {data_type}",
        **market_data
    )


# Context managers for logging
class LoggingContext:
    """Context manager for adding logging context"""
    
    def __init__(self, **context):
        self.context = context
        self.token = None
    
    def __enter__(self):
        self.token = logger.contextualize(**self.context)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.token:
            self.token.__exit__(exc_type, exc_val, exc_tb)


class PerformanceLoggingContext:
    """Context manager for performance logging"""
    
    def __init__(self, component: str, operation: str, **kwargs):
        self.component = component
        self.operation = operation
        self.kwargs = kwargs
        self.start_time = None
    
    def __enter__(self):
        self.start_time = datetime.utcnow()
        logger.debug(f"Starting {self.component}.{self.operation}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time:
            duration = (datetime.utcnow() - self.start_time).total_seconds()
            success = exc_type is None
            
            log_performance_metrics(
                self.component,
                self.operation,
                duration,
                success,
                **self.kwargs
            )


# Convenience functions
def with_logging_context(**context):
    """Decorator for adding logging context to functions"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            with LoggingContext(**context):
                return func(*args, **kwargs)
        return wrapper
    return decorator


def with_performance_logging(component: str, operation: str = None):
    """Decorator for performance logging"""
    def decorator(func):
        op_name = operation or func.__name__
        
        def wrapper(*args, **kwargs):
            with PerformanceLoggingContext(component, op_name):
                return func(*args, **kwargs)
        return wrapper
    return decorator
