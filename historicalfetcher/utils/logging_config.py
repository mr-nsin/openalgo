"""
Centralized Logging Configuration for OpenAlgo Historical Data Fetcher

This module provides a unified logging setup using loguru for all components
of the historical data fetcher with async support and structured logging.
"""

import sys
import os
from pathlib import Path
from loguru import logger
from typing import Optional, Dict, Any
from datetime import datetime

class HistoricalFetcherLogger:
    """Centralized logger configuration for the historical data fetcher"""
    
    _instance: Optional['HistoricalFetcherLogger'] = None
    _initialized: bool = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            self._setup_default_logger()
            HistoricalFetcherLogger._initialized = True
    
    def _setup_default_logger(self):
        """Setup default logger configuration"""
        # Remove default handler
        logger.remove()
        
        # Create logs directory if it doesn't exist
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        # Console handler with colors
        logger.add(
            sys.stdout,
            level="INFO",
            format="<green>{time:HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | <level>{message}</level>",
            colorize=True,
            enqueue=True
        )
        
        # File handler for general logs
        logger.add(
            log_dir / "historical_fetcher.log",
            rotation="100 MB",
            retention="30 days",
            level="DEBUG",
            format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} | {message}",
            enqueue=True,
            backtrace=True,
            diagnose=True,
            catch=True
        )
        
        # Structured JSON handler for important events
        logger.add(
            log_dir / "historical_fetcher_structured.jsonl",
            rotation="100 MB",
            retention="30 days",
            level="INFO",
            serialize=True,
            enqueue=True
        )
        
        # Error-only handler
        logger.add(
            log_dir / "historical_fetcher_errors.log",
            rotation="50 MB",
            retention="60 days",
            level="ERROR",
            format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} | {message}",
            enqueue=True,
            backtrace=True,
            diagnose=True
        )
        
        logger.info("Historical Fetcher Logger initialized")
    
    def configure_from_settings(self, settings):
        """Configure logger from OpenAlgoSettings"""
        # Remove existing handlers
        logger.remove()
        
        # Create logs directory
        log_dir = Path(settings.log_file_path).parent
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Console handler
        logger.add(
            sys.stdout,
            level=settings.log_level,
            format="<green>{time:HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | <level>{message}</level>",
            colorize=True,
            enqueue=True
        )
        
        # Main log file
        logger.add(
            settings.log_file_path,
            rotation=settings.log_rotation,
            retention=settings.log_retention,
            level=settings.log_level,
            format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} | {message}",
            enqueue=True,
            backtrace=True,
            diagnose=True,
            catch=True
        )
        
        # Structured logs
        structured_path = settings.log_file_path.replace('.log', '_structured.jsonl')
        logger.add(
            structured_path,
            rotation=settings.log_rotation,
            retention=settings.log_retention,
            level="INFO",
            serialize=True,
            enqueue=True
        )
        
        # Error logs
        error_path = settings.log_file_path.replace('.log', '_errors.log')
        logger.add(
            error_path,
            rotation="50 MB",
            retention="60 days",
            level="ERROR",
            format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} | {message}",
            enqueue=True,
            backtrace=True,
            diagnose=True
        )
        
        logger.info(f"Logger reconfigured - Level: {settings.log_level}, File: {settings.log_file_path}")
    
    @staticmethod
    def get_logger():
        """Get the configured logger instance"""
        return logger
    
    @staticmethod
    def log_structured(event_type: str, data: Dict[str, Any], message: str = ""):
        """Log structured data with event type"""
        structured_data = {
            'event_type': event_type,
            'timestamp': datetime.now().isoformat(),
            **data
        }
        
        logger.info(message or f"{event_type} event", extra=structured_data)
    
    @staticmethod
    def log_symbol_processing(
        symbol: str,
        instrument_type: str,
        timeframe: str,
        records_count: int,
        processing_time: float,
        status: str = "success"
    ):
        """Log symbol processing with structured data"""
        data = {
            'symbol': symbol,
            'instrument_type': instrument_type,
            'timeframe': timeframe,
            'records_count': records_count,
            'processing_time_seconds': round(processing_time, 2),
            'status': status
        }
        
        message = f"{'✅' if status == 'success' else '❌'} {symbol} ({instrument_type}, {timeframe}): {records_count:,} records in {processing_time:.2f}s"
        
        if status == "success":
            logger.info(message, extra={'event_type': 'symbol_processed', **data})
        else:
            logger.error(message, extra={'event_type': 'symbol_failed', **data})
    
    @staticmethod
    def log_batch_progress(processed: int, total: int, current_item: str = ""):
        """Log batch processing progress"""
        percentage = (processed / total) * 100 if total > 0 else 0
        
        data = {
            'processed': processed,
            'total': total,
            'percentage': round(percentage, 1),
            'current_item': current_item
        }
        
        message = f"Progress: {processed:,}/{total:,} ({percentage:.1f}%)"
        if current_item:
            message += f" - Current: {current_item}"
        
        logger.info(message, extra={'event_type': 'batch_progress', **data})
    
    @staticmethod
    def log_api_metrics(metrics: Dict[str, Any]):
        """Log API performance metrics"""
        message = f"API Metrics - Requests: {metrics.get('total_requests', 0)}, Success Rate: {metrics.get('success_rate', 0):.1f}%"
        logger.info(message, extra={'event_type': 'api_metrics', **metrics})
    
    @staticmethod
    def log_database_operation(
        operation: str,
        table_name: str,
        records_count: int,
        execution_time: float,
        status: str = "success"
    ):
        """Log database operations"""
        data = {
            'operation': operation,
            'table_name': table_name,
            'records_count': records_count,
            'execution_time_seconds': round(execution_time, 2),
            'status': status
        }
        
        message = f"DB {operation}: {table_name} - {records_count:,} records in {execution_time:.2f}s"
        logger.info(message, extra={'event_type': 'database_operation', **data})

# Global logger instance
_logger_instance = None

def setup_logging(settings=None):
    """Setup logging for the historical data fetcher"""
    global _logger_instance
    
    if _logger_instance is None:
        _logger_instance = HistoricalFetcherLogger()
    
    if settings:
        _logger_instance.configure_from_settings(settings)
    
    return _logger_instance.get_logger()

def get_logger():
    """Get the configured logger instance"""
    global _logger_instance
    
    if _logger_instance is None:
        _logger_instance = HistoricalFetcherLogger()
    
    return _logger_instance.get_logger()

# Convenience functions for structured logging
def log_symbol_processing(*args, **kwargs):
    """Convenience function for symbol processing logs"""
    return HistoricalFetcherLogger.log_symbol_processing(*args, **kwargs)

def log_batch_progress(*args, **kwargs):
    """Convenience function for batch progress logs"""
    return HistoricalFetcherLogger.log_batch_progress(*args, **kwargs)

def log_api_metrics(*args, **kwargs):
    """Convenience function for API metrics logs"""
    return HistoricalFetcherLogger.log_api_metrics(*args, **kwargs)

def log_database_operation(*args, **kwargs):
    """Convenience function for database operation logs"""
    return HistoricalFetcherLogger.log_database_operation(*args, **kwargs)

def log_structured(*args, **kwargs):
    """Convenience function for structured logging"""
    return HistoricalFetcherLogger.log_structured(*args, **kwargs)



