"""
Strategy Configuration
======================

Defines trading strategy configurations and parameters.
"""

import os
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from enum import Enum


class StrategyType(Enum):
    """Strategy types"""
    TECHNICAL = "TECHNICAL"
    FUNDAMENTAL = "FUNDAMENTAL"
    QUANTITATIVE = "QUANTITATIVE"
    ARBITRAGE = "ARBITRAGE"
    MOMENTUM = "MOMENTUM"
    MEAN_REVERSION = "MEAN_REVERSION"


class SignalType(Enum):
    """Signal types"""
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"
    EXIT = "EXIT"


@dataclass
class IndicatorConfig:
    """Configuration for technical indicators"""
    name: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    enabled: bool = True


@dataclass
class StrategyConfig:
    """Configuration for a trading strategy"""
    name: str
    strategy_type: StrategyType
    description: str
    enabled: bool = True
    
    # Symbol and timeframe configuration
    symbols: List[str] = field(default_factory=list)
    timeframes: List[str] = field(default_factory=list)
    
    # Strategy parameters
    parameters: Dict[str, Any] = field(default_factory=dict)
    
    # Technical indicators
    indicators: List[IndicatorConfig] = field(default_factory=list)
    
    # Risk management
    max_position_size: float = 10000.0
    stop_loss_pct: float = 0.02  # 2%
    take_profit_pct: float = 0.04  # 4%
    
    # Signal generation
    min_confidence: float = 0.6
    signal_cooldown: int = 300  # seconds
    
    # Execution settings
    order_type: str = "MARKET"
    execution_delay: float = 0.0


class StrategyManager:
    """Manages strategy configurations"""
    
    def __init__(self):
        self.strategies: Dict[str, StrategyConfig] = {}
    
    @classmethod
    def create_default_strategies(cls) -> 'StrategyManager':
        """Create default strategy configurations"""
        manager = cls()
        
        # Load strategies from environment or create defaults
        manager._create_default_strategies()
        manager._load_from_environment()
        
        return manager
    
    def _create_default_strategies(self):
        """Create default strategy configurations"""
        
        # SMA Crossover Strategy
        sma_crossover = StrategyConfig(
            name="SMA_Crossover",
            strategy_type=StrategyType.TECHNICAL,
            description="Simple Moving Average Crossover Strategy",
            symbols=["NIFTY", "BANKNIFTY"],
            timeframes=["5m", "15m"],
            parameters={
                "fast_period": 10,
                "slow_period": 20,
                "volume_threshold": 1000
            },
            indicators=[
                IndicatorConfig("SMA", {"period": 10}),
                IndicatorConfig("SMA", {"period": 20}),
                IndicatorConfig("Volume", {})
            ],
            max_position_size=50000.0,
            stop_loss_pct=0.015,
            take_profit_pct=0.03,
            min_confidence=0.7
        )
        self.strategies["SMA_Crossover"] = sma_crossover
        
        # RSI Mean Reversion Strategy
        rsi_mean_reversion = StrategyConfig(
            name="RSI_MeanReversion",
            strategy_type=StrategyType.MEAN_REVERSION,
            description="RSI-based Mean Reversion Strategy",
            symbols=["NIFTY", "BANKNIFTY", "SENSEX"],
            timeframes=["1m", "5m"],
            parameters={
                "rsi_period": 14,
                "oversold_level": 30,
                "overbought_level": 70,
                "rsi_smoothing": 3
            },
            indicators=[
                IndicatorConfig("RSI", {"period": 14}),
                IndicatorConfig("EMA", {"period": 21}),
                IndicatorConfig("VWAP", {})
            ],
            max_position_size=30000.0,
            stop_loss_pct=0.01,
            take_profit_pct=0.02,
            min_confidence=0.65
        )
        self.strategies["RSI_MeanReversion"] = rsi_mean_reversion
        
        # Bollinger Bands Strategy
        bollinger_bands = StrategyConfig(
            name="BollingerBands_Breakout",
            strategy_type=StrategyType.MOMENTUM,
            description="Bollinger Bands Breakout Strategy",
            symbols=["NIFTY", "BANKNIFTY"],
            timeframes=["15m", "30m"],
            parameters={
                "bb_period": 20,
                "bb_std_dev": 2.0,
                "volume_multiplier": 1.5,
                "breakout_threshold": 0.001
            },
            indicators=[
                IndicatorConfig("BollingerBands", {"period": 20, "std_dev": 2.0}),
                IndicatorConfig("Volume", {}),
                IndicatorConfig("ATR", {"period": 14})
            ],
            max_position_size=40000.0,
            stop_loss_pct=0.02,
            take_profit_pct=0.05,
            min_confidence=0.75
        )
        self.strategies["BollingerBands_Breakout"] = bollinger_bands
        
        # MACD Strategy
        macd_strategy = StrategyConfig(
            name="MACD_Momentum",
            strategy_type=StrategyType.MOMENTUM,
            description="MACD Momentum Strategy",
            symbols=["NIFTY", "BANKNIFTY", "FINNIFTY"],
            timeframes=["5m", "15m", "1h"],
            parameters={
                "fast_ema": 12,
                "slow_ema": 26,
                "signal_ema": 9,
                "histogram_threshold": 0.5
            },
            indicators=[
                IndicatorConfig("MACD", {"fast": 12, "slow": 26, "signal": 9}),
                IndicatorConfig("EMA", {"period": 50}),
                IndicatorConfig("Volume", {})
            ],
            max_position_size=35000.0,
            stop_loss_pct=0.018,
            take_profit_pct=0.035,
            min_confidence=0.68
        )
        self.strategies["MACD_Momentum"] = macd_strategy
        
        # Options Straddle Strategy
        options_straddle = StrategyConfig(
            name="Options_Straddle",
            strategy_type=StrategyType.QUANTITATIVE,
            description="Options Straddle Strategy for High Volatility",
            symbols=["NIFTY*", "BANKNIFTY*"],  # * indicates options
            timeframes=["1m", "5m"],
            parameters={
                "iv_threshold": 20.0,
                "time_to_expiry_min": 7,
                "time_to_expiry_max": 30,
                "delta_range": [0.4, 0.6]
            },
            indicators=[
                IndicatorConfig("ImpliedVolatility", {}),
                IndicatorConfig("Delta", {}),
                IndicatorConfig("Gamma", {}),
                IndicatorConfig("Theta", {})
            ],
            max_position_size=25000.0,
            stop_loss_pct=0.25,
            take_profit_pct=0.50,
            min_confidence=0.8
        )
        self.strategies["Options_Straddle"] = options_straddle
        
        # Scalping Strategy
        scalping_strategy = StrategyConfig(
            name="Scalping_Quick",
            strategy_type=StrategyType.MOMENTUM,
            description="Quick Scalping Strategy for Intraday",
            symbols=["NIFTY", "BANKNIFTY"],
            timeframes=["1m", "3m"],
            parameters={
                "price_change_threshold": 0.001,
                "volume_spike_multiplier": 2.0,
                "max_hold_time": 300,  # 5 minutes
                "profit_target": 0.005
            },
            indicators=[
                IndicatorConfig("EMA", {"period": 5}),
                IndicatorConfig("EMA", {"period": 13}),
                IndicatorConfig("Volume", {}),
                IndicatorConfig("VWAP", {})
            ],
            max_position_size=20000.0,
            stop_loss_pct=0.005,
            take_profit_pct=0.01,
            min_confidence=0.85,
            signal_cooldown=60  # 1 minute cooldown
        )
        self.strategies["Scalping_Quick"] = scalping_strategy
    
    def _load_from_environment(self):
        """Load strategy configurations from environment variables"""
        # Load enabled strategies
        enabled_strategies_str = os.getenv('ENABLED_STRATEGIES', 'SMA_Crossover,RSI_MeanReversion')
        enabled_strategies = [s.strip() for s in enabled_strategies_str.split(',') if s.strip()]
        
        # Disable strategies not in enabled list
        for strategy_name in self.strategies:
            if strategy_name not in enabled_strategies:
                self.strategies[strategy_name].enabled = False
        
        # Load strategy-specific parameters from environment
        for strategy_name, strategy in self.strategies.items():
            self._load_strategy_parameters(strategy_name, strategy)
    
    def _load_strategy_parameters(self, strategy_name: str, strategy: StrategyConfig):
        """Load strategy-specific parameters from environment"""
        prefix = f"STRATEGY_{strategy_name.upper()}_"
        
        # Load max position size
        max_position_key = f"{prefix}MAX_POSITION_SIZE"
        if os.getenv(max_position_key):
            try:
                strategy.max_position_size = float(os.getenv(max_position_key))
            except ValueError:
                pass
        
        # Load stop loss percentage
        stop_loss_key = f"{prefix}STOP_LOSS_PCT"
        if os.getenv(stop_loss_key):
            try:
                strategy.stop_loss_pct = float(os.getenv(stop_loss_key))
            except ValueError:
                pass
        
        # Load take profit percentage
        take_profit_key = f"{prefix}TAKE_PROFIT_PCT"
        if os.getenv(take_profit_key):
            try:
                strategy.take_profit_pct = float(os.getenv(take_profit_key))
            except ValueError:
                pass
        
        # Load minimum confidence
        min_confidence_key = f"{prefix}MIN_CONFIDENCE"
        if os.getenv(min_confidence_key):
            try:
                strategy.min_confidence = float(os.getenv(min_confidence_key))
            except ValueError:
                pass
        
        # Load symbols
        symbols_key = f"{prefix}SYMBOLS"
        if os.getenv(symbols_key):
            symbols_str = os.getenv(symbols_key)
            strategy.symbols = [s.strip() for s in symbols_str.split(',') if s.strip()]
        
        # Load timeframes
        timeframes_key = f"{prefix}TIMEFRAMES"
        if os.getenv(timeframes_key):
            timeframes_str = os.getenv(timeframes_key)
            strategy.timeframes = [t.strip() for t in timeframes_str.split(',') if t.strip()]
    
    def get_strategy(self, name: str) -> Optional[StrategyConfig]:
        """Get strategy configuration by name"""
        return self.strategies.get(name)
    
    def get_enabled_strategies(self) -> List[StrategyConfig]:
        """Get all enabled strategies"""
        return [strategy for strategy in self.strategies.values() if strategy.enabled]
    
    def get_strategies_by_type(self, strategy_type: StrategyType) -> List[StrategyConfig]:
        """Get strategies by type"""
        return [
            strategy for strategy in self.strategies.values()
            if strategy.strategy_type == strategy_type and strategy.enabled
        ]
    
    def get_strategies_for_symbol(self, symbol: str) -> List[StrategyConfig]:
        """Get strategies that trade a specific symbol"""
        matching_strategies = []
        
        for strategy in self.strategies.values():
            if not strategy.enabled:
                continue
            
            # Check if symbol matches any pattern in strategy symbols
            for pattern in strategy.symbols:
                if pattern.endswith('*'):
                    # Wildcard pattern (e.g., "NIFTY*" matches "NIFTY24000CE")
                    if symbol.startswith(pattern[:-1]):
                        matching_strategies.append(strategy)
                        break
                elif pattern == symbol:
                    # Exact match
                    matching_strategies.append(strategy)
                    break
        
        return matching_strategies
    
    def add_strategy(self, strategy: StrategyConfig):
        """Add a new strategy configuration"""
        self.strategies[strategy.name] = strategy
    
    def remove_strategy(self, name: str):
        """Remove a strategy configuration"""
        if name in self.strategies:
            del self.strategies[name]
    
    def enable_strategy(self, name: str):
        """Enable a strategy"""
        if name in self.strategies:
            self.strategies[name].enabled = True
    
    def disable_strategy(self, name: str):
        """Disable a strategy"""
        if name in self.strategies:
            self.strategies[name].enabled = False
    
    def get_all_strategies(self) -> Dict[str, StrategyConfig]:
        """Get all strategy configurations"""
        return dict(self.strategies)
    
    def validate_strategies(self) -> List[str]:
        """Validate all strategy configurations"""
        errors = []
        
        for name, strategy in self.strategies.items():
            # Check required fields
            if not strategy.name:
                errors.append(f"Strategy {name}: Missing name")
            
            if not strategy.symbols:
                errors.append(f"Strategy {name}: No symbols configured")
            
            if not strategy.timeframes:
                errors.append(f"Strategy {name}: No timeframes configured")
            
            # Check parameter ranges
            if strategy.min_confidence < 0 or strategy.min_confidence > 1:
                errors.append(f"Strategy {name}: Invalid min_confidence {strategy.min_confidence}")
            
            if strategy.stop_loss_pct < 0 or strategy.stop_loss_pct > 1:
                errors.append(f"Strategy {name}: Invalid stop_loss_pct {strategy.stop_loss_pct}")
            
            if strategy.take_profit_pct < 0:
                errors.append(f"Strategy {name}: Invalid take_profit_pct {strategy.take_profit_pct}")
            
            if strategy.max_position_size <= 0:
                errors.append(f"Strategy {name}: Invalid max_position_size {strategy.max_position_size}")
        
        return errors