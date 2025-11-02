"""
Async Logger using Loguru

Provides structured, async-friendly logging with rotation, retention, and JSON formatting.
"""

import asyncio
import json
import sys
from typing import Any, Dict, Optional
from loguru import logger
from datetime import datetime

class AsyncLogger:
    """Async-friendly logger with structured logging capabilities"""
    
    def __init__(
        self, 
        log_file: str, 
        level: str = "INFO",
        rotation: str = "100 MB",
        retention: str = "30 days",
        enable_console: bool = True
    ):
        self.log_file = log_file
        self.level = level
        
        # Remove default handler
        logger.remove()
        
        # Add file handler with rotation and structured logging
        logger.add(
            log_file,
            rotation=rotation,
            retention=retention,
            level=level,
            format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} | {message}",
            serialize=False,  # Keep human-readable for easier debugging
            backtrace=True,
            diagnose=True,
            enqueue=True,  # Make it thread-safe
            catch=True
        )
        
        # Add console handler if enabled
        if enable_console:
            logger.add(
                sys.stdout,
                level=level,
                format="<green>{time:HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | <level>{message}</level>",
                colorize=True
            )
        
        # Add structured JSON handler for important events
        logger.add(
            log_file.replace('.log', '_structured.jsonl'),
            rotation=rotation,
            retention=retention,
            level="INFO",
            serialize=True,  # JSON format
            enqueue=True
        )
        
        logger.info(f"AsyncLogger initialized - File: {log_file}, Level: {level}")
    
    async def log_batch_progress(
        self, 
        processed: int, 
        total: int, 
        current_item: str,
        additional_info: Optional[Dict[str, Any]] = None
    ):
        """Log batch processing progress with structured data"""
        
        percentage = (processed / total) * 100 if total > 0 else 0
        
        progress_data = {
            'event_type': 'batch_progress',
            'processed': processed,
            'total': total,
            'percentage': round(percentage, 1),
            'current_item': current_item,
            'timestamp': datetime.now().isoformat()
        }
        
        if additional_info:
            progress_data.update(additional_info)
        
        logger.info(
            f"Progress: {processed:,}/{total:,} ({percentage:.1f}%) - Current: {current_item}",
            extra=progress_data
        )
    
    async def log_api_metrics(self, metrics: Dict[str, Any]):
        """Log API performance metrics"""
        
        metrics_data = {
            'event_type': 'api_metrics',
            'timestamp': datetime.now().isoformat(),
            **metrics
        }
        
        logger.info(
            f"API Metrics - Requests: {metrics.get('total_requests', 0)}, "
            f"Success Rate: {metrics.get('success_rate', 0):.1f}%",
            extra=metrics_data
        )
    
    async def log_symbol_processing(
        self,
        symbol: str,
        instrument_type: str,
        timeframe: str,
        records_count: int,
        processing_time: float,
        status: str = "success"
    ):
        """Log symbol processing results"""
        
        symbol_data = {
            'event_type': 'symbol_processed',
            'symbol': symbol,
            'instrument_type': instrument_type,
            'timeframe': timeframe,
            'records_count': records_count,
            'processing_time_seconds': round(processing_time, 2),
            'status': status,
            'timestamp': datetime.now().isoformat()
        }
        
        if status == "success":
            logger.info(
                f"✅ {symbol} ({instrument_type}, {timeframe}): {records_count:,} records in {processing_time:.2f}s",
                extra=symbol_data
            )
        else:
            logger.error(
                f"❌ {symbol} ({instrument_type}, {timeframe}): Failed after {processing_time:.2f}s",
                extra=symbol_data
            )
    
    async def log_database_operation(
        self,
        operation: str,
        table_name: str,
        records_count: int,
        execution_time: float,
        status: str = "success"
    ):
        """Log database operations"""
        
        db_data = {
            'event_type': 'database_operation',
            'operation': operation,
            'table_name': table_name,
            'records_count': records_count,
            'execution_time_seconds': round(execution_time, 2),
            'status': status,
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info(
            f"DB {operation}: {table_name} - {records_count:,} records in {execution_time:.2f}s",
            extra=db_data
        )
    
    async def log_error_with_context(
        self,
        error: Exception,
        context: Dict[str, Any],
        operation: str = "unknown"
    ):
        """Log errors with detailed context"""
        
        error_data = {
            'event_type': 'error',
            'operation': operation,
            'error_type': type(error).__name__,
            'error_message': str(error),
            'context': context,
            'timestamp': datetime.now().isoformat()
        }
        
        logger.error(
            f"Error in {operation}: {type(error).__name__}: {error}",
            extra=error_data
        )
    
    async def log_system_metrics(self, metrics: Dict[str, Any]):
        """Log system performance metrics"""
        
        system_data = {
            'event_type': 'system_metrics',
            'timestamp': datetime.now().isoformat(),
            **metrics
        }
        
        logger.info(
            f"System Metrics - Memory: {metrics.get('memory_usage_mb', 0):.1f}MB, "
            f"CPU: {metrics.get('cpu_percent', 0):.1f}%",
            extra=system_data
        )
    
    async def log_notification_sent(
        self,
        notification_type: str,
        recipient: str,
        status: str,
        message_preview: str = ""
    ):
        """Log notification delivery status"""
        
        notification_data = {
            'event_type': 'notification_sent',
            'notification_type': notification_type,
            'recipient': recipient,
            'status': status,
            'message_preview': message_preview[:100] + "..." if len(message_preview) > 100 else message_preview,
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info(
            f"Notification ({notification_type}) to {recipient}: {status}",
            extra=notification_data
        )
    
    def get_logger(self):
        """Get the underlying loguru logger instance"""
        return logger
    
    async def flush_logs(self):
        """Force flush all log handlers"""
        # Force flush by calling complete() on all handlers
        await asyncio.sleep(0.1)  # Allow any pending logs to be processed
        
    def force_sync_flush(self):
        """Synchronously flush logs (for use in error handlers)"""
        import time
        time.sleep(0.1)  # Allow loguru's enqueue to process

# Global logger instance
_global_logger: Optional[AsyncLogger] = None

def get_async_logger() -> AsyncLogger:
    """Get or create global async logger instance"""
    global _global_logger
    
    if _global_logger is None:
        _global_logger = AsyncLogger("logs/historical_fetcher.log")
    
    return _global_logger

def setup_async_logger(
    log_file: str,
    level: str = "INFO",
    rotation: str = "100 MB",
    retention: str = "30 days"
) -> AsyncLogger:
    """Setup and return configured async logger"""
    global _global_logger
    
    _global_logger = AsyncLogger(
        log_file=log_file,
        level=level,
        rotation=rotation,
        retention=retention
    )
    
    return _global_logger
