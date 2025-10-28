"""
Historical Data Fetcher for OpenAlgo
====================================

A comprehensive, modular system for fetching historical market data from Zerodha API
and storing it in QuestDB with instrument-specific table structures.

Features:
- Multi-instrument support (Equity, Futures, Options, Indices)
- Multiple timeframe support (1m, 5m, 15m, 1h, Daily)
- Async processing with rate limiting
- QuestDB integration with optimized schemas
- Comprehensive notification system (Telegram & Email)
- Market-aware scheduling
- Robust error handling and retry logic
"""

__version__ = "1.0.0"
__author__ = "OpenAlgo Team"
