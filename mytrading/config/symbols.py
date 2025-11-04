"""
Symbol Configuration
====================

Defines trading symbols, exchanges, timeframes, and instrument types.
"""

import os
from dataclasses import dataclass, field
from typing import List, Dict, Set, Optional
from enum import Enum


class Exchange(Enum):
    """Supported exchanges"""
    NSE = "NSE"
    BSE = "BSE"
    MCX = "MCX"
    NCDEX = "NCDEX"


class InstrumentType(Enum):
    """Instrument types"""
    EQUITY = "EQ"
    FUTURES = "FUT"
    OPTIONS = "OPT"
    CURRENCY = "CUR"
    COMMODITY = "COM"


class TimeFrame(Enum):
    """Supported timeframes"""
    TICK = "TICK"
    SECOND_1 = "1s"
    SECOND_5 = "5s"
    SECOND_15 = "15s"
    SECOND_30 = "30s"
    MINUTE_1 = "1m"
    MINUTE_3 = "3m"
    MINUTE_5 = "5m"
    MINUTE_15 = "15m"
    MINUTE_30 = "30m"
    HOUR_1 = "1h"
    HOUR_4 = "4h"
    DAY_1 = "1d"
    WEEK_1 = "1w"
    MONTH_1 = "1M"


@dataclass
class SymbolInfo:
    """Information about a trading symbol"""
    symbol: str
    exchange: Exchange
    instrument_type: InstrumentType
    lot_size: int = 1
    tick_size: float = 0.01
    enabled: bool = True
    
    # Option-specific fields
    strike_price: Optional[float] = None
    option_type: Optional[str] = None  # "CE" or "PE"
    expiry_date: Optional[str] = None
    
    # Additional metadata
    description: Optional[str] = None
    sector: Optional[str] = None


@dataclass
class SymbolConfig:
    """Configuration for all trading symbols"""
    symbols: Dict[str, SymbolInfo] = field(default_factory=dict)
    enabled_exchanges: Set[Exchange] = field(default_factory=set)
    enabled_timeframes: Set[TimeFrame] = field(default_factory=set)
    enabled_instrument_types: Set[InstrumentType] = field(default_factory=set)
    
    @classmethod
    def create_default_config(cls) -> 'SymbolConfig':
        """Create default symbol configuration from environment variables"""
        config = cls()
        
        # Load from environment variables
        config._load_from_environment()
        
        return config
    
    def _load_from_environment(self):
        """Load configuration from environment variables"""
        # Load enabled symbols
        enabled_symbols_str = os.getenv('ENABLED_SYMBOLS', 'NIFTY,BANKNIFTY,SENSEX')
        enabled_symbols = [s.strip() for s in enabled_symbols_str.split(',') if s.strip()]
        
        # Load enabled exchanges
        enabled_exchanges_str = os.getenv('ENABLED_EXCHANGES', 'NSE,BSE')
        for exchange_str in enabled_exchanges_str.split(','):
            try:
                exchange = Exchange(exchange_str.strip())
                self.enabled_exchanges.add(exchange)
            except ValueError:
                continue
        
        # Load enabled timeframes
        enabled_timeframes_str = os.getenv('ENABLED_TIMEFRAMES', '1m,5m,15m,1h,1d')
        for timeframe_str in enabled_timeframes_str.split(','):
            try:
                timeframe = TimeFrame(timeframe_str.strip())
                self.enabled_timeframes.add(timeframe)
            except ValueError:
                continue
        
        # Load enabled instrument types
        enabled_instruments_str = os.getenv('ENABLED_INSTRUMENT_TYPES', 'EQ,OPT,FUT')
        for instrument_str in enabled_instruments_str.split(','):
            try:
                instrument = InstrumentType(instrument_str.strip())
                self.enabled_instrument_types.add(instrument)
            except ValueError:
                continue
        
        # Create symbol info for each enabled symbol
        for symbol in enabled_symbols:
            self._create_symbol_info(symbol)
    
    def _create_symbol_info(self, symbol: str):
        """Create symbol info for a given symbol"""
        # Determine instrument type and exchange based on symbol
        if symbol in ['NIFTY', 'BANKNIFTY', 'SENSEX', 'FINNIFTY']:
            # Index symbols - create both equity and options
            
            # Equity/Index
            self.symbols[symbol] = SymbolInfo(
                symbol=symbol,
                exchange=Exchange.NSE,
                instrument_type=InstrumentType.EQUITY,
                lot_size=1,
                description=f"{symbol} Index"
            )
            
            # Options (ATM and nearby strikes)
            if InstrumentType.OPTIONS in self.enabled_instrument_types:
                self._create_option_symbols(symbol)
        
        else:
            # Regular equity symbols
            self.symbols[symbol] = SymbolInfo(
                symbol=symbol,
                exchange=Exchange.NSE,
                instrument_type=InstrumentType.EQUITY,
                lot_size=1,
                description=f"{symbol} Equity"
            )
    
    def _create_option_symbols(self, underlying: str):
        """Create option symbols for an underlying"""
        # This is a placeholder - in a full implementation, you would:
        # 1. Fetch current price of underlying
        # 2. Calculate ATM strike
        # 3. Create CE/PE symbols for ATM and nearby strikes
        # 4. Handle different expiry dates
        
        # For now, create placeholder option symbols
        base_strikes = self._get_base_strikes(underlying)
        
        for strike in base_strikes:
            for option_type in ['CE', 'PE']:
                option_symbol = f"{underlying}{strike}{option_type}"
                self.symbols[option_symbol] = SymbolInfo(
                    symbol=option_symbol,
                    exchange=Exchange.NSE,
                    instrument_type=InstrumentType.OPTIONS,
                    lot_size=self._get_lot_size(underlying),
                    strike_price=float(strike),
                    option_type=option_type,
                    description=f"{underlying} {strike} {option_type}"
                )
    
    def _get_base_strikes(self, underlying: str) -> List[str]:
        """Get base strikes for an underlying (placeholder)"""
        # This should be dynamic based on current market price
        if underlying == 'NIFTY':
            return ['24000', '24050', '24100', '24150', '24200']
        elif underlying == 'BANKNIFTY':
            return ['51000', '51100', '51200', '51300', '51400']
        elif underlying == 'SENSEX':
            return ['79000', '79100', '79200', '79300', '79400']
        else:
            return ['100', '110', '120', '130', '140']
    
    def _get_lot_size(self, underlying: str) -> int:
        """Get lot size for an underlying"""
        lot_sizes = {
            'NIFTY': 50,
            'BANKNIFTY': 15,
            'SENSEX': 10,
            'FINNIFTY': 40
        }
        return lot_sizes.get(underlying, 1)
    
    def get_symbols_by_exchange(self, exchange: Exchange) -> List[SymbolInfo]:
        """Get all symbols for a specific exchange"""
        return [info for info in self.symbols.values() if info.exchange == exchange]
    
    def get_symbols_by_instrument_type(self, instrument_type: InstrumentType) -> List[SymbolInfo]:
        """Get all symbols for a specific instrument type"""
        return [info for info in self.symbols.values() if info.instrument_type == instrument_type]
    
    def get_option_symbols(self, underlying: str) -> List[SymbolInfo]:
        """Get all option symbols for an underlying"""
        return [
            info for info in self.symbols.values()
            if info.instrument_type == InstrumentType.OPTIONS and underlying in info.symbol
        ]
    
    def is_symbol_enabled(self, symbol: str) -> bool:
        """Check if a symbol is enabled"""
        return symbol in self.symbols and self.symbols[symbol].enabled
    
    def get_enabled_symbols(self) -> List[str]:
        """Get list of all enabled symbols"""
        return [symbol for symbol, info in self.symbols.items() if info.enabled]