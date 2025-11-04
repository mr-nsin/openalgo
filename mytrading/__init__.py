"""
MyTrading - Advanced Real-time Trading System
============================================

A high-performance, multi-threaded trading system built on OpenAlgo infrastructure
with real-time WebSocket data feeds, historical data integration, and automated
strategy execution.

Features:
- Real-time market data processing via WebSocket
- Historical data integration and backtesting
- Multi-timeframe strategy execution
- Advanced risk management
- High-performance messaging with ZeroMQ
- Comprehensive logging and monitoring

Author: Trading System Developer
Version: 1.0.0
"""

__version__ = "1.0.0"
__author__ = "Trading System Developer"

# Core imports for easy access
from .core.orchestrator import TradingOrchestrator
from .config.settings import TradingSettings
from .utils.logging_config import setup_logging

__all__ = [
    "TradingOrchestrator",
    "TradingSettings", 
    "setup_logging"
]
