"""
Symbol Configuration
===================

Configuration for trading symbols, timeframes, and market data subscriptions.
Supports multiple indices (NIFTY, BANKNIFTY, SENSEX) with their options chains.
"""

import yaml
from pathlib import Path
from typing import Dict, List, Optional, Set
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta


class TimeFrame(str, Enum):
    """Supported timeframes for market data and strategies"""
    TICK = "tick"           # Tick-by-tick data
    SECOND_1 = "1s"         # 1 second
    SECOND_5 = "5s"         # 5 seconds
    SECOND_15 = "15s"       # 15 seconds
    SECOND_30 = "30s"       # 30 seconds
    MINUTE_1 = "1m"         # 1 minute
    MINUTE_2 = "2m"         # 2 minutes
    MINUTE_3 = "3m"         # 3 minutes
    MINUTE_5 = "5m"         # 5 minutes
    MINUTE_10 = "10m"       # 10 minutes
    MINUTE_15 = "15m"       # 15 minutes
    MINUTE_30 = "30m"       # 30 minutes
    HOUR_1 = "1h"           # 1 hour
    HOUR_2 = "2h"           # 2 hours
    HOUR_4 = "4h"           # 4 hours
    DAY_1 = "1d"            # 1 day
    WEEK_1 = "1w"           # 1 week
    MONTH_1 = "1M"          # 1 month


class InstrumentType(str, Enum):
    """Types of financial instruments"""
    INDEX = "INDEX"         # Index (NIFTY, BANKNIFTY, etc.)
    EQUITY = "EQ"           # Equity stocks
    FUTURES = "FUT"         # Futures contracts
    CALL_OPTION = "CE"      # Call options
    PUT_OPTION = "PE"       # Put options
    CURRENCY = "CUR"        # Currency pairs
    COMMODITY = "COM"       # Commodities


class Exchange(str, Enum):
    """Supported exchanges"""
    NSE = "NSE"             # National Stock Exchange
    BSE = "BSE"             # Bombay Stock Exchange  
    NFO = "NFO"             # NSE Futures & Options
    BFO = "BFO"             # BSE Futures & Options
    NSE_INDEX = "NSE_INDEX" # NSE Indices
    BSE_INDEX = "BSE_INDEX" # BSE Indices


@dataclass
class SymbolInfo:
    """Information about a trading symbol"""
    symbol: str
    exchange: Exchange
    instrument_type: InstrumentType
    timeframes: List[TimeFrame] = field(default_factory=list)
    
    # Options-specific fields
    underlying: Optional[str] = None
    strike: Optional[float] = None
    expiry: Optional[str] = None
    
    # Additional metadata
    lot_size: int = 1
    tick_size: float = 0.05
    enabled: bool = True
    
    # Strategy assignment
    strategies: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        """Post-initialization validation"""
        if isinstance(self.exchange, str):
            self.exchange = Exchange(self.exchange)
        if isinstance(self.instrument_type, str):
            self.instrument_type = InstrumentType(self.instrument_type)
        
        # Convert string timeframes to TimeFrame enums
        converted_timeframes = []
        for tf in self.timeframes:
            if isinstance(tf, str):
                converted_timeframes.append(TimeFrame(tf))
            else:
                converted_timeframes.append(tf)
        self.timeframes = converted_timeframes
    
    @property
    def full_symbol(self) -> str:
        """Get full symbol identifier"""
        return f"{self.exchange.value}:{self.symbol}"
    
    def is_option(self) -> bool:
        """Check if this is an options contract"""
        return self.instrument_type in [InstrumentType.CALL_OPTION, InstrumentType.PUT_OPTION]
    
    def is_index(self) -> bool:
        """Check if this is an index"""
        return self.instrument_type == InstrumentType.INDEX


@dataclass
class OptionsChainConfig:
    """Configuration for options chain monitoring"""
    underlying: str
    exchange: Exchange
    expiry_dates: List[str] = field(default_factory=list)
    
    # Strike selection
    atm_range: int = 20          # Number of strikes above and below ATM
    strike_step: int = 50        # Strike price step (e.g., 50 for NIFTY)
    
    # Instrument types to include
    include_calls: bool = True
    include_puts: bool = True
    
    # Timeframes for options data
    timeframes: List[TimeFrame] = field(default_factory=lambda: [TimeFrame.MINUTE_1, TimeFrame.MINUTE_5])
    
    # Auto-discovery settings
    auto_discover_strikes: bool = True
    auto_discover_expiries: bool = True
    
    def __post_init__(self):
        """Post-initialization validation"""
        if isinstance(self.exchange, str):
            self.exchange = Exchange(self.exchange)
        
        # Convert string timeframes to TimeFrame enums
        converted_timeframes = []
        for tf in self.timeframes:
            if isinstance(tf, str):
                converted_timeframes.append(TimeFrame(tf))
            else:
                converted_timeframes.append(tf)
        self.timeframes = converted_timeframes


@dataclass
class MarketDataConfig:
    """Configuration for market data subscriptions"""
    # Real-time data settings
    enable_tick_data: bool = True
    enable_depth_data: bool = True
    depth_levels: int = 5
    
    # Historical data settings
    enable_historical_data: bool = True
    historical_days: int = 30
    
    # Data quality settings
    enable_data_validation: bool = True
    max_price_change_percent: float = 10.0  # Maximum allowed price change
    
    # Performance settings
    buffer_size: int = 10000
    batch_size: int = 100
    update_frequency: float = 0.1  # seconds


class SymbolConfig:
    """Main symbol configuration manager"""
    
    def __init__(self):
        self.symbols: Dict[str, SymbolInfo] = {}
        self.options_chains: Dict[str, OptionsChainConfig] = {}
        self.market_data_config = MarketDataConfig()
        
        # Predefined symbol groups
        self.symbol_groups: Dict[str, List[str]] = {
            "indices": [],
            "nifty_options": [],
            "banknifty_options": [],
            "sensex_options": [],
            "equities": [],
            "futures": []
        }
    
    def add_symbol(self, symbol_info: SymbolInfo):
        """Add a symbol to the configuration"""
        self.symbols[symbol_info.full_symbol] = symbol_info
        
        # Add to appropriate group
        if symbol_info.is_index():
            self.symbol_groups["indices"].append(symbol_info.full_symbol)
        elif symbol_info.is_option():
            if symbol_info.underlying == "NIFTY":
                self.symbol_groups["nifty_options"].append(symbol_info.full_symbol)
            elif symbol_info.underlying == "BANKNIFTY":
                self.symbol_groups["banknifty_options"].append(symbol_info.full_symbol)
            elif symbol_info.underlying == "SENSEX":
                self.symbol_groups["sensex_options"].append(symbol_info.full_symbol)
        elif symbol_info.instrument_type == InstrumentType.EQUITY:
            self.symbol_groups["equities"].append(symbol_info.full_symbol)
        elif symbol_info.instrument_type == InstrumentType.FUTURES:
            self.symbol_groups["futures"].append(symbol_info.full_symbol)
    
    def add_options_chain(self, chain_config: OptionsChainConfig):
        """Add an options chain configuration"""
        self.options_chains[chain_config.underlying] = chain_config
    
    def get_symbols_by_timeframe(self, timeframe: TimeFrame) -> List[SymbolInfo]:
        """Get all symbols that support a specific timeframe"""
        return [
            symbol for symbol in self.symbols.values()
            if timeframe in symbol.timeframes and symbol.enabled
        ]
    
    def get_symbols_by_exchange(self, exchange: Exchange) -> List[SymbolInfo]:
        """Get all symbols from a specific exchange"""
        return [
            symbol for symbol in self.symbols.values()
            if symbol.exchange == exchange and symbol.enabled
        ]
    
    def get_symbols_by_type(self, instrument_type: InstrumentType) -> List[SymbolInfo]:
        """Get all symbols of a specific instrument type"""
        return [
            symbol for symbol in self.symbols.values()
            if symbol.instrument_type == instrument_type and symbol.enabled
        ]
    
    def get_symbols_by_strategy(self, strategy_name: str) -> List[SymbolInfo]:
        """Get all symbols assigned to a specific strategy"""
        return [
            symbol for symbol in self.symbols.values()
            if strategy_name in symbol.strategies and symbol.enabled
        ]
    
    def get_options_for_underlying(self, underlying: str) -> List[SymbolInfo]:
        """Get all options for a specific underlying"""
        return [
            symbol for symbol in self.symbols.values()
            if symbol.underlying == underlying and symbol.is_option() and symbol.enabled
        ]
    
    def get_all_timeframes(self) -> Set[TimeFrame]:
        """Get all unique timeframes across all symbols"""
        timeframes = set()
        for symbol in self.symbols.values():
            if symbol.enabled:
                timeframes.update(symbol.timeframes)
        return timeframes
    
    def get_websocket_subscriptions(self) -> List[Dict[str, str]]:
        """Get list of symbols for WebSocket subscription"""
        subscriptions = []
        for symbol in self.symbols.values():
            if symbol.enabled:
                subscriptions.append({
                    "exchange": symbol.exchange.value,
                    "symbol": symbol.symbol
                })
        return subscriptions
    
    @classmethod
    def create_default_config(cls) -> 'SymbolConfig':
        """Create a default configuration with common symbols"""
        config = cls()
        
        # Add major indices
        indices = [
            SymbolInfo("NIFTY", Exchange.NSE_INDEX, InstrumentType.INDEX, 
                      [TimeFrame.TICK, TimeFrame.MINUTE_1, TimeFrame.MINUTE_5, TimeFrame.HOUR_1, TimeFrame.DAY_1]),
            SymbolInfo("BANKNIFTY", Exchange.NSE_INDEX, InstrumentType.INDEX,
                      [TimeFrame.TICK, TimeFrame.MINUTE_1, TimeFrame.MINUTE_5, TimeFrame.HOUR_1, TimeFrame.DAY_1]),
            SymbolInfo("SENSEX", Exchange.BSE_INDEX, InstrumentType.INDEX,
                      [TimeFrame.TICK, TimeFrame.MINUTE_1, TimeFrame.MINUTE_5, TimeFrame.HOUR_1, TimeFrame.DAY_1]),
        ]
        
        for index in indices:
            config.add_symbol(index)
        
        # Add options chain configurations
        nifty_options = OptionsChainConfig(
            underlying="NIFTY",
            exchange=Exchange.NFO,
            atm_range=20,
            strike_step=50,
            timeframes=[TimeFrame.MINUTE_1, TimeFrame.MINUTE_5]
        )
        
        banknifty_options = OptionsChainConfig(
            underlying="BANKNIFTY", 
            exchange=Exchange.NFO,
            atm_range=15,
            strike_step=100,
            timeframes=[TimeFrame.MINUTE_1, TimeFrame.MINUTE_5]
        )
        
        config.add_options_chain(nifty_options)
        config.add_options_chain(banknifty_options)
        
        return config
    
    @classmethod
    def from_file(cls, config_path: str) -> 'SymbolConfig':
        """Load symbol configuration from YAML file"""
        config = cls()
        
        config_file = Path(config_path)
        if not config_file.exists():
            raise FileNotFoundError(f"Symbol configuration file not found: {config_path}")
        
        with open(config_file, 'r', encoding='utf-8') as f:
            config_data = yaml.safe_load(f)
        
        # Load symbols
        if 'symbols' in config_data:
            for symbol_data in config_data['symbols']:
                symbol_info = SymbolInfo(**symbol_data)
                config.add_symbol(symbol_info)
        
        # Load options chains
        if 'options_chains' in config_data:
            for chain_data in config_data['options_chains']:
                chain_config = OptionsChainConfig(**chain_data)
                config.add_options_chain(chain_config)
        
        # Load market data config
        if 'market_data' in config_data:
            market_data = config_data['market_data']
            for key, value in market_data.items():
                if hasattr(config.market_data_config, key):
                    setattr(config.market_data_config, key, value)
        
        return config
    
    def to_dict(self) -> Dict:
        """Convert configuration to dictionary"""
        return {
            'symbols': [
                {
                    'symbol': symbol.symbol,
                    'exchange': symbol.exchange.value,
                    'instrument_type': symbol.instrument_type.value,
                    'timeframes': [tf.value for tf in symbol.timeframes],
                    'underlying': symbol.underlying,
                    'strike': symbol.strike,
                    'expiry': symbol.expiry,
                    'lot_size': symbol.lot_size,
                    'tick_size': symbol.tick_size,
                    'enabled': symbol.enabled,
                    'strategies': symbol.strategies,
                }
                for symbol in self.symbols.values()
            ],
            'options_chains': [
                {
                    'underlying': chain.underlying,
                    'exchange': chain.exchange.value,
                    'expiry_dates': chain.expiry_dates,
                    'atm_range': chain.atm_range,
                    'strike_step': chain.strike_step,
                    'include_calls': chain.include_calls,
                    'include_puts': chain.include_puts,
                    'timeframes': [tf.value for tf in chain.timeframes],
                    'auto_discover_strikes': chain.auto_discover_strikes,
                    'auto_discover_expiries': chain.auto_discover_expiries,
                }
                for chain in self.options_chains.values()
            ],
            'market_data': {
                'enable_tick_data': self.market_data_config.enable_tick_data,
                'enable_depth_data': self.market_data_config.enable_depth_data,
                'depth_levels': self.market_data_config.depth_levels,
                'enable_historical_data': self.market_data_config.enable_historical_data,
                'historical_days': self.market_data_config.historical_days,
                'enable_data_validation': self.market_data_config.enable_data_validation,
                'max_price_change_percent': self.market_data_config.max_price_change_percent,
                'buffer_size': self.market_data_config.buffer_size,
                'batch_size': self.market_data_config.batch_size,
                'update_frequency': self.market_data_config.update_frequency,
            }
        }
    
    def save_to_file(self, config_path: str):
        """Save configuration to YAML file"""
        config_file = Path(config_path)
        config_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(config_file, 'w', encoding='utf-8') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, indent=2)
    
    def validate(self) -> List[str]:
        """Validate symbol configuration"""
        errors = []
        
        if not self.symbols:
            errors.append("No symbols configured")
        
        # Validate each symbol
        for full_symbol, symbol in self.symbols.items():
            if not symbol.symbol:
                errors.append(f"Symbol name is empty for {full_symbol}")
            
            if not symbol.timeframes:
                errors.append(f"No timeframes configured for {full_symbol}")
            
            if symbol.is_option():
                if not symbol.underlying:
                    errors.append(f"Underlying not specified for option {full_symbol}")
                if symbol.strike is None:
                    errors.append(f"Strike price not specified for option {full_symbol}")
        
        # Validate options chains
        for underlying, chain in self.options_chains.items():
            if chain.atm_range <= 0:
                errors.append(f"ATM range must be positive for {underlying}")
            if chain.strike_step <= 0:
                errors.append(f"Strike step must be positive for {underlying}")
        
        return errors
