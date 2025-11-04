"""
Strategy Configuration
=====================

Configuration for trading strategies, including technical indicators,
risk parameters, and execution rules.
"""

import yaml
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum

from .symbols import TimeFrame


class StrategyType(str, Enum):
    """Types of trading strategies"""
    TECHNICAL = "technical"         # Technical analysis based
    MOMENTUM = "momentum"           # Momentum strategies
    MEAN_REVERSION = "mean_reversion"  # Mean reversion strategies
    ARBITRAGE = "arbitrage"         # Arbitrage opportunities
    OPTIONS = "options"             # Options strategies
    SCALPING = "scalping"           # High-frequency scalping
    SWING = "swing"                 # Swing trading
    CUSTOM = "custom"               # Custom user-defined strategies


class SignalType(str, Enum):
    """Types of trading signals"""
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"
    EXIT = "exit"
    STOP_LOSS = "stop_loss"
    TAKE_PROFIT = "take_profit"


class OrderType(str, Enum):
    """Order types for strategy execution"""
    MARKET = "market"
    LIMIT = "limit"
    STOP_LOSS = "stop_loss"
    STOP_LIMIT = "stop_limit"
    BRACKET = "bracket"


@dataclass
class IndicatorConfig:
    """Configuration for technical indicators"""
    name: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    enabled: bool = True
    
    # Common indicator parameters
    period: Optional[int] = None
    source: str = "close"  # open, high, low, close, volume
    
    def __post_init__(self):
        """Set default parameters based on indicator name"""
        if self.name.upper() == "SMA" and "period" not in self.parameters:
            self.parameters["period"] = self.period or 20
        elif self.name.upper() == "EMA" and "period" not in self.parameters:
            self.parameters["period"] = self.period or 20
        elif self.name.upper() == "RSI" and "period" not in self.parameters:
            self.parameters["period"] = self.period or 14
        elif self.name.upper() == "MACD":
            if "fast_period" not in self.parameters:
                self.parameters["fast_period"] = 12
            if "slow_period" not in self.parameters:
                self.parameters["slow_period"] = 26
            if "signal_period" not in self.parameters:
                self.parameters["signal_period"] = 9
        elif self.name.upper() == "BOLLINGER_BANDS":
            if "period" not in self.parameters:
                self.parameters["period"] = self.period or 20
            if "std_dev" not in self.parameters:
                self.parameters["std_dev"] = 2


@dataclass
class RiskParameters:
    """Risk management parameters for strategies"""
    # Position sizing
    max_position_size: float = 10000.0  # Maximum position size in currency
    position_size_percent: float = 0.02  # Position size as % of capital
    
    # Stop loss settings
    stop_loss_percent: float = 0.02      # Stop loss as % of entry price
    trailing_stop_enabled: bool = False
    trailing_stop_percent: float = 0.01  # Trailing stop distance
    
    # Take profit settings
    take_profit_percent: float = 0.04    # Take profit as % of entry price
    partial_profit_enabled: bool = False
    partial_profit_levels: List[float] = field(default_factory=lambda: [0.02, 0.03])
    
    # Risk limits
    max_daily_trades: int = 10
    max_concurrent_positions: int = 3
    max_drawdown_percent: float = 0.05   # Maximum strategy drawdown
    
    # Time-based rules
    trading_start_time: str = "09:15"    # Market open time
    trading_end_time: str = "15:30"      # Market close time
    avoid_first_minutes: int = 5         # Avoid trading in first N minutes
    avoid_last_minutes: int = 15         # Avoid trading in last N minutes


@dataclass
class ExecutionParameters:
    """Order execution parameters"""
    default_order_type: OrderType = OrderType.MARKET
    limit_order_offset: float = 0.001    # Offset for limit orders (as % of price)
    order_timeout: int = 300             # Order timeout in seconds
    
    # Slippage and fees
    expected_slippage: float = 0.0005    # Expected slippage (0.05%)
    transaction_cost: float = 0.0003     # Transaction cost (0.03%)
    
    # Order management
    enable_order_modification: bool = True
    max_order_modifications: int = 3
    modification_timeout: int = 60       # Seconds to wait before modification


@dataclass
class StrategyConfig:
    """Complete strategy configuration"""
    name: str
    strategy_type: StrategyType
    description: str = ""
    
    # Strategy parameters
    timeframes: List[TimeFrame] = field(default_factory=list)
    symbols: List[str] = field(default_factory=list)  # Symbol patterns or specific symbols
    
    # Technical indicators
    indicators: List[IndicatorConfig] = field(default_factory=list)
    
    # Strategy-specific parameters
    parameters: Dict[str, Any] = field(default_factory=dict)
    
    # Risk and execution
    risk_parameters: RiskParameters = field(default_factory=RiskParameters)
    execution_parameters: ExecutionParameters = field(default_factory=ExecutionParameters)
    
    # Control flags
    enabled: bool = True
    paper_trading: bool = True
    
    # Performance tracking
    track_performance: bool = True
    benchmark_symbol: Optional[str] = None
    
    def __post_init__(self):
        """Post-initialization processing"""
        if isinstance(self.strategy_type, str):
            self.strategy_type = StrategyType(self.strategy_type)
        
        # Convert string timeframes to TimeFrame enums
        converted_timeframes = []
        for tf in self.timeframes:
            if isinstance(tf, str):
                converted_timeframes.append(TimeFrame(tf))
            else:
                converted_timeframes.append(tf)
        self.timeframes = converted_timeframes
    
    def add_indicator(self, name: str, parameters: Dict[str, Any] = None, **kwargs):
        """Add a technical indicator to the strategy"""
        if parameters is None:
            parameters = {}
        
        # Merge kwargs into parameters
        parameters.update(kwargs)
        
        indicator = IndicatorConfig(name=name, parameters=parameters)
        self.indicators.append(indicator)
    
    def get_indicator(self, name: str) -> Optional[IndicatorConfig]:
        """Get indicator configuration by name"""
        for indicator in self.indicators:
            if indicator.name.upper() == name.upper():
                return indicator
        return None
    
    def remove_indicator(self, name: str) -> bool:
        """Remove indicator by name"""
        for i, indicator in enumerate(self.indicators):
            if indicator.name.upper() == name.upper():
                del self.indicators[i]
                return True
        return False


class StrategyManager:
    """Manager for multiple strategy configurations"""
    
    def __init__(self):
        self.strategies: Dict[str, StrategyConfig] = {}
        self.strategy_groups: Dict[str, List[str]] = {}
    
    def add_strategy(self, strategy: StrategyConfig):
        """Add a strategy configuration"""
        self.strategies[strategy.name] = strategy
    
    def get_strategy(self, name: str) -> Optional[StrategyConfig]:
        """Get strategy by name"""
        return self.strategies.get(name)
    
    def remove_strategy(self, name: str) -> bool:
        """Remove strategy by name"""
        if name in self.strategies:
            del self.strategies[name]
            return True
        return False
    
    def get_strategies_by_type(self, strategy_type: StrategyType) -> List[StrategyConfig]:
        """Get all strategies of a specific type"""
        return [
            strategy for strategy in self.strategies.values()
            if strategy.strategy_type == strategy_type and strategy.enabled
        ]
    
    def get_strategies_by_timeframe(self, timeframe: TimeFrame) -> List[StrategyConfig]:
        """Get all strategies that use a specific timeframe"""
        return [
            strategy for strategy in self.strategies.values()
            if timeframe in strategy.timeframes and strategy.enabled
        ]
    
    def get_strategies_for_symbol(self, symbol: str) -> List[StrategyConfig]:
        """Get all strategies that trade a specific symbol"""
        matching_strategies = []
        for strategy in self.strategies.values():
            if not strategy.enabled:
                continue
            
            # Check if symbol matches any pattern in strategy.symbols
            for pattern in strategy.symbols:
                if pattern == symbol or pattern == "*" or symbol.startswith(pattern.replace("*", "")):
                    matching_strategies.append(strategy)
                    break
        
        return matching_strategies
    
    def get_enabled_strategies(self) -> List[StrategyConfig]:
        """Get all enabled strategies"""
        return [strategy for strategy in self.strategies.values() if strategy.enabled]
    
    @classmethod
    def create_default_strategies(cls) -> 'StrategyManager':
        """Create default strategy configurations"""
        manager = cls()
        
        # 1. Simple Moving Average Crossover Strategy
        sma_strategy = StrategyConfig(
            name="SMA_Crossover",
            strategy_type=StrategyType.TECHNICAL,
            description="Simple Moving Average crossover strategy",
            timeframes=[TimeFrame.MINUTE_5, TimeFrame.MINUTE_15],
            symbols=["NIFTY", "BANKNIFTY"]
        )
        sma_strategy.add_indicator("SMA", period=20)
        sma_strategy.add_indicator("SMA", period=50)
        sma_strategy.parameters = {
            "fast_period": 20,
            "slow_period": 50,
            "min_crossover_strength": 0.5
        }
        manager.add_strategy(sma_strategy)
        
        # 2. RSI Mean Reversion Strategy
        rsi_strategy = StrategyConfig(
            name="RSI_MeanReversion",
            strategy_type=StrategyType.MEAN_REVERSION,
            description="RSI-based mean reversion strategy",
            timeframes=[TimeFrame.MINUTE_1, TimeFrame.MINUTE_5],
            symbols=["NIFTY*", "BANKNIFTY*"]  # All NIFTY and BANKNIFTY instruments
        )
        rsi_strategy.add_indicator("RSI", period=14)
        rsi_strategy.add_indicator("EMA", period=20)
        rsi_strategy.parameters = {
            "rsi_oversold": 30,
            "rsi_overbought": 70,
            "rsi_extreme_oversold": 20,
            "rsi_extreme_overbought": 80
        }
        rsi_strategy.risk_parameters.stop_loss_percent = 0.015  # 1.5% stop loss
        rsi_strategy.risk_parameters.take_profit_percent = 0.03  # 3% take profit
        manager.add_strategy(rsi_strategy)
        
        # 3. MACD Momentum Strategy
        macd_strategy = StrategyConfig(
            name="MACD_Momentum",
            strategy_type=StrategyType.MOMENTUM,
            description="MACD-based momentum strategy",
            timeframes=[TimeFrame.MINUTE_15, TimeFrame.HOUR_1],
            symbols=["NSE_INDEX:NIFTY", "NSE_INDEX:BANKNIFTY"]
        )
        macd_strategy.add_indicator("MACD", fast_period=12, slow_period=26, signal_period=9)
        macd_strategy.add_indicator("EMA", period=50)
        macd_strategy.parameters = {
            "macd_signal_threshold": 0.1,
            "trend_confirmation_required": True,
            "volume_confirmation": True
        }
        manager.add_strategy(macd_strategy)
        
        # 4. Options Straddle Strategy
        options_strategy = StrategyConfig(
            name="ATM_Straddle",
            strategy_type=StrategyType.OPTIONS,
            description="ATM straddle strategy for high volatility",
            timeframes=[TimeFrame.MINUTE_5],
            symbols=["NFO:NIFTY*CE", "NFO:NIFTY*PE"]
        )
        options_strategy.add_indicator("ATR", period=14)
        options_strategy.add_indicator("IV", period=20)  # Implied Volatility
        options_strategy.parameters = {
            "min_iv_percentile": 70,    # Enter when IV is above 70th percentile
            "max_dte": 7,               # Maximum days to expiry
            "profit_target": 0.25,      # 25% profit target
            "loss_limit": 0.50          # 50% loss limit
        }
        options_strategy.risk_parameters.max_concurrent_positions = 2
        manager.add_strategy(options_strategy)
        
        # 5. Scalping Strategy
        scalping_strategy = StrategyConfig(
            name="Quick_Scalp",
            strategy_type=StrategyType.SCALPING,
            description="High-frequency scalping strategy",
            timeframes=[TimeFrame.MINUTE_1],
            symbols=["NIFTY", "BANKNIFTY"]
        )
        scalping_strategy.add_indicator("EMA", period=9)
        scalping_strategy.add_indicator("EMA", period=21)
        scalping_strategy.add_indicator("RSI", period=7)
        scalping_strategy.parameters = {
            "min_price_movement": 0.001,  # Minimum 0.1% price movement
            "max_hold_time": 300,         # Maximum 5 minutes hold time
            "volume_spike_threshold": 1.5  # 1.5x average volume
        }
        scalping_strategy.risk_parameters.stop_loss_percent = 0.005  # 0.5% stop loss
        scalping_strategy.risk_parameters.take_profit_percent = 0.01  # 1% take profit
        scalping_strategy.risk_parameters.max_daily_trades = 50
        scalping_strategy.execution_parameters.default_order_type = OrderType.LIMIT
        manager.add_strategy(scalping_strategy)
        
        return manager
    
    @classmethod
    def from_file(cls, config_path: str) -> 'StrategyManager':
        """Load strategy configurations from YAML file"""
        manager = cls()
        
        config_file = Path(config_path)
        if not config_file.exists():
            raise FileNotFoundError(f"Strategy configuration file not found: {config_path}")
        
        with open(config_file, 'r', encoding='utf-8') as f:
            config_data = yaml.safe_load(f)
        
        # Load strategies
        if 'strategies' in config_data:
            for strategy_data in config_data['strategies']:
                # Handle indicators separately
                indicators_data = strategy_data.pop('indicators', [])
                
                # Handle risk_parameters separately
                risk_data = strategy_data.pop('risk_parameters', {})
                
                # Handle execution_parameters separately
                execution_data = strategy_data.pop('execution_parameters', {})
                
                # Create strategy
                strategy = StrategyConfig(**strategy_data)
                
                # Add indicators
                for indicator_data in indicators_data:
                    indicator = IndicatorConfig(**indicator_data)
                    strategy.indicators.append(indicator)
                
                # Update risk parameters
                for key, value in risk_data.items():
                    if hasattr(strategy.risk_parameters, key):
                        setattr(strategy.risk_parameters, key, value)
                
                # Update execution parameters
                for key, value in execution_data.items():
                    if hasattr(strategy.execution_parameters, key):
                        if key == 'default_order_type':
                            strategy.execution_parameters.default_order_type = OrderType(value)
                        else:
                            setattr(strategy.execution_parameters, key, value)
                
                manager.add_strategy(strategy)
        
        # Load strategy groups
        if 'strategy_groups' in config_data:
            manager.strategy_groups = config_data['strategy_groups']
        
        return manager
    
    def to_dict(self) -> Dict:
        """Convert strategy manager to dictionary"""
        return {
            'strategies': [
                {
                    'name': strategy.name,
                    'strategy_type': strategy.strategy_type.value,
                    'description': strategy.description,
                    'timeframes': [tf.value for tf in strategy.timeframes],
                    'symbols': strategy.symbols,
                    'indicators': [
                        {
                            'name': indicator.name,
                            'parameters': indicator.parameters,
                            'enabled': indicator.enabled,
                            'period': indicator.period,
                            'source': indicator.source,
                        }
                        for indicator in strategy.indicators
                    ],
                    'parameters': strategy.parameters,
                    'risk_parameters': {
                        'max_position_size': strategy.risk_parameters.max_position_size,
                        'position_size_percent': strategy.risk_parameters.position_size_percent,
                        'stop_loss_percent': strategy.risk_parameters.stop_loss_percent,
                        'trailing_stop_enabled': strategy.risk_parameters.trailing_stop_enabled,
                        'trailing_stop_percent': strategy.risk_parameters.trailing_stop_percent,
                        'take_profit_percent': strategy.risk_parameters.take_profit_percent,
                        'partial_profit_enabled': strategy.risk_parameters.partial_profit_enabled,
                        'partial_profit_levels': strategy.risk_parameters.partial_profit_levels,
                        'max_daily_trades': strategy.risk_parameters.max_daily_trades,
                        'max_concurrent_positions': strategy.risk_parameters.max_concurrent_positions,
                        'max_drawdown_percent': strategy.risk_parameters.max_drawdown_percent,
                        'trading_start_time': strategy.risk_parameters.trading_start_time,
                        'trading_end_time': strategy.risk_parameters.trading_end_time,
                        'avoid_first_minutes': strategy.risk_parameters.avoid_first_minutes,
                        'avoid_last_minutes': strategy.risk_parameters.avoid_last_minutes,
                    },
                    'execution_parameters': {
                        'default_order_type': strategy.execution_parameters.default_order_type.value,
                        'limit_order_offset': strategy.execution_parameters.limit_order_offset,
                        'order_timeout': strategy.execution_parameters.order_timeout,
                        'expected_slippage': strategy.execution_parameters.expected_slippage,
                        'transaction_cost': strategy.execution_parameters.transaction_cost,
                        'enable_order_modification': strategy.execution_parameters.enable_order_modification,
                        'max_order_modifications': strategy.execution_parameters.max_order_modifications,
                        'modification_timeout': strategy.execution_parameters.modification_timeout,
                    },
                    'enabled': strategy.enabled,
                    'paper_trading': strategy.paper_trading,
                    'track_performance': strategy.track_performance,
                    'benchmark_symbol': strategy.benchmark_symbol,
                }
                for strategy in self.strategies.values()
            ],
            'strategy_groups': self.strategy_groups
        }
    
    def save_to_file(self, config_path: str):
        """Save strategy configurations to YAML file"""
        config_file = Path(config_path)
        config_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(config_file, 'w', encoding='utf-8') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, indent=2)
    
    def validate(self) -> List[str]:
        """Validate all strategy configurations"""
        errors = []
        
        if not self.strategies:
            errors.append("No strategies configured")
        
        for name, strategy in self.strategies.items():
            # Validate basic fields
            if not strategy.name:
                errors.append(f"Strategy name is empty")
            
            if not strategy.timeframes:
                errors.append(f"No timeframes configured for strategy {name}")
            
            if not strategy.symbols:
                errors.append(f"No symbols configured for strategy {name}")
            
            # Validate risk parameters
            risk = strategy.risk_parameters
            if risk.max_position_size <= 0:
                errors.append(f"Invalid max_position_size for strategy {name}")
            
            if not (0 < risk.position_size_percent <= 1):
                errors.append(f"Invalid position_size_percent for strategy {name}")
            
            if not (0 < risk.stop_loss_percent < 1):
                errors.append(f"Invalid stop_loss_percent for strategy {name}")
            
            if not (0 < risk.take_profit_percent < 1):
                errors.append(f"Invalid take_profit_percent for strategy {name}")
            
            # Validate indicators
            for indicator in strategy.indicators:
                if not indicator.name:
                    errors.append(f"Indicator name is empty in strategy {name}")
        
        return errors
