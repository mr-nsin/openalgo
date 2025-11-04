"""
Logging Configuration
=====================

Centralized logging configuration using loguru for the MyTrading system.
"""

import os
import sys
from pathlib import Path
from typing import Optional, Dict, Any
from loguru import logger


class LoggingConfig:
    """Centralized logging configuration"""
    
    def __init__(self):
        self.is_configured = False
        self.log_level = os.getenv('LOG_LEVEL', 'INFO')
        self.log_dir = Path(os.getenv('LOG_DIR', 'logs'))
        self.enable_file_logging = os.getenv('ENABLE_FILE_LOGGING', 'true').lower() == 'true'
        self.enable_console_logging = os.getenv('ENABLE_CONSOLE_LOGGING', 'true').lower() == 'true'
        self.log_rotation = os.getenv('LOG_ROTATION', '100 MB')
        self.log_retention = os.getenv('LOG_RETENTION', '30 days')
        
    def setup_logging(self, component_name: str = "mytrading") -> None:
        """Setup logging configuration"""
        if self.is_configured:
            return
        
        # Remove default logger
        logger.remove()
        
        # Create log directory
        self.log_dir.mkdir(exist_ok=True)
        
        # Console logging
        if self.enable_console_logging:
            logger.add(
                sys.stdout,
                level=self.log_level,
                format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
                       "<level>{level: <8}</level> | "
                       "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
                       "<level>{message}</level>",
                colorize=True
            )
        
        # File logging
        if self.enable_file_logging:
            # Main log file
            logger.add(
                self.log_dir / f"{component_name}.log",
                level=self.log_level,
                format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} | {message}",
                rotation=self.log_rotation,
                retention=self.log_retention,
                compression="zip"
            )
            
            # Error log file
            logger.add(
                self.log_dir / f"{component_name}_errors.log",
                level="ERROR",
                format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} | {message}",
                rotation=self.log_rotation,
                retention=self.log_retention,
                compression="zip"
            )
            
            # Performance log file (for performance monitoring)
            logger.add(
                self.log_dir / f"{component_name}_performance.log",
                level="DEBUG",
                format="{time:YYYY-MM-DD HH:mm:ss.SSS} | PERF | {name}:{function}:{line} | {message}",
                rotation=self.log_rotation,
                retention=self.log_retention,
                compression="zip",
                filter=lambda record: "PERF" in record["message"]
            )
        
        # Add custom log levels
        logger.level("SUCCESS", no=25, color="<green><bold>")
        logger.level("TRADE", no=35, color="<blue><bold>")
        logger.level("SIGNAL", no=33, color="<yellow><bold>")
        
        self.is_configured = True
        logger.info(f"🔧 Logging configured for {component_name}")
        logger.info(f"📁 Log directory: {self.log_dir.absolute()}")
        logger.info(f"📊 Log level: {self.log_level}")


# Global logging configuration instance
_logging_config = LoggingConfig()


def setup_logging(component_name: str = "mytrading") -> None:
    """Setup logging for the application"""
    _logging_config.setup_logging(component_name)


def get_logger(name: str) -> Any:
    """Get a logger instance"""
    if not _logging_config.is_configured:
        setup_logging()
    return logger.bind(name=name)


def log_trade(symbol: str, action: str, quantity: int, price: float, **kwargs) -> None:
    """Log a trade event"""
    trade_logger = get_logger("trade")
    trade_logger.log(
        "TRADE",
        f"TRADE: {action} {quantity} {symbol} @ {price:.2f} | {kwargs}"
    )


def log_signal(strategy: str, symbol: str, signal_type: str, confidence: float, **kwargs) -> None:
    """Log a signal event"""
    signal_logger = get_logger("signal")
    signal_logger.log(
        "SIGNAL",
        f"SIGNAL: {strategy} | {symbol} {signal_type} (confidence: {confidence:.1%}) | {kwargs}"
    )


def log_performance(component: str, operation: str, duration: float, **kwargs) -> None:
    """Log a performance event"""
    perf_logger = get_logger("performance")
    perf_logger.debug(
        f"PERF: {component}.{operation} took {duration*1000:.2f}ms | {kwargs}"
    )


# Custom log format functions
def format_currency(amount: float) -> str:
    """Format currency for logging"""
    return f"${amount:,.2f}"


def format_percentage(value: float) -> str:
    """Format percentage for logging"""
    return f"{value:.2%}"


def format_duration(seconds: float) -> str:
    """Format duration for logging"""
    if seconds < 1:
        return f"{seconds*1000:.1f}ms"
    elif seconds < 60:
        return f"{seconds:.1f}s"
    else:
        return f"{seconds/60:.1f}m"


# Context managers for structured logging
class LogContext:
    """Context manager for structured logging"""
    
    def __init__(self, logger_name: str, operation: str, **context):
        self.logger = get_logger(logger_name)
        self.operation = operation
        self.context = context
        self.start_time = None
    
    def __enter__(self):
        import time
        self.start_time = time.time()
        self.logger.debug(f"Starting {self.operation}", **self.context)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        import time
        duration = time.time() - self.start_time
        
        if exc_type is None:
            self.logger.debug(f"Completed {self.operation} in {format_duration(duration)}", **self.context)
        else:
            self.logger.error(f"Failed {self.operation} after {format_duration(duration)}: {exc_val}", **self.context)


def log_context(logger_name: str, operation: str, **context):
    """Create a logging context"""
    return LogContext(logger_name, operation, **context)


# Logging decorators
def log_function_call(logger_name: Optional[str] = None):
    """Decorator to log function calls"""
    def decorator(func):
        import functools
        import time
        
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            func_logger = get_logger(logger_name or func.__module__)
            start_time = time.time()
            
            func_logger.debug(f"Calling {func.__name__}")
            try:
                result = await func(*args, **kwargs)
                duration = time.time() - start_time
                func_logger.debug(f"Completed {func.__name__} in {format_duration(duration)}")
                return result
            except Exception as e:
                duration = time.time() - start_time
                func_logger.error(f"Failed {func.__name__} after {format_duration(duration)}: {e}")
                raise
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            func_logger = get_logger(logger_name or func.__module__)
            start_time = time.time()
            
            func_logger.debug(f"Calling {func.__name__}")
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                func_logger.debug(f"Completed {func.__name__} in {format_duration(duration)}")
                return result
            except Exception as e:
                duration = time.time() - start_time
                func_logger.error(f"Failed {func.__name__} after {format_duration(duration)}: {e}")
                raise
        
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


# Error handling utilities
def log_exception(logger_name: str, operation: str, exception: Exception, **context):
    """Log an exception with context"""
    error_logger = get_logger(logger_name)
    error_logger.error(
        f"Exception in {operation}: {type(exception).__name__}: {exception}",
        **context
    )


def safe_log_dict(data: Dict[str, Any], max_length: int = 1000) -> str:
    """Safely log a dictionary, truncating if too long"""
    try:
        import json
        json_str = json.dumps(data, default=str)
        if len(json_str) > max_length:
            return json_str[:max_length] + "... (truncated)"
        return json_str
    except Exception:
        return str(data)[:max_length]


# Health check logging
def log_health_check(component: str, status: str, metrics: Dict[str, Any]):
    """Log health check results"""
    health_logger = get_logger("health")
    
    if status == "healthy":
        health_logger.debug(f"Health check: {component} is {status}")
    elif status == "warning":
        health_logger.warning(f"Health check: {component} is {status} - {safe_log_dict(metrics)}")
    else:
        health_logger.error(f"Health check: {component} is {status} - {safe_log_dict(metrics)}")


# Startup and shutdown logging
def log_startup(component: str, version: str, config: Dict[str, Any]):
    """Log application startup"""
    startup_logger = get_logger("startup")
    startup_logger.success(f"🚀 {component} v{version} starting up")
    startup_logger.info(f"Configuration: {safe_log_dict(config)}")


def log_shutdown(component: str, reason: str = "Normal shutdown"):
    """Log application shutdown"""
    shutdown_logger = get_logger("shutdown")
    shutdown_logger.success(f"🛑 {component} shutting down: {reason}")


# Performance logging utilities
class PerformanceLogger:
    """Performance logging utility"""
    
    def __init__(self, logger_name: str = "performance"):
        self.logger = get_logger(logger_name)
        self.timers = {}
    
    def start_timer(self, name: str):
        """Start a performance timer"""
        import time
        self.timers[name] = time.time()
    
    def end_timer(self, name: str, **context):
        """End a performance timer and log the result"""
        import time
        if name in self.timers:
            duration = time.time() - self.timers[name]
            self.logger.debug(f"Timer {name}: {format_duration(duration)}", **context)
            del self.timers[name]
            return duration
        return None
    
    def log_metric(self, metric_name: str, value: float, unit: str = "", **context):
        """Log a performance metric"""
        self.logger.debug(f"Metric {metric_name}: {value}{unit}", **context)


# Global performance logger
performance_logger = PerformanceLogger()


# Export commonly used functions
__all__ = [
    'setup_logging',
    'get_logger',
    'log_trade',
    'log_signal',
    'log_performance',
    'log_context',
    'log_function_call',
    'log_exception',
    'log_health_check',
    'log_startup',
    'log_shutdown',
    'format_currency',
    'format_percentage',
    'format_duration',
    'performance_logger'
]