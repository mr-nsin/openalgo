"""
Trading System Settings
=======================

Main configuration class for the MyTrading system with all system-wide settings,
API configurations, performance parameters, and operational modes.

Uses environment variables for configuration (similar to historicalfetcher).
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    # Try to load from mytrading directory first, then from OpenAlgo root
    env_paths = [
        os.path.join(os.path.dirname(__file__), '..', '.env'),
        os.path.join(os.path.dirname(__file__), '..', '..', '.env')
    ]
    
    for env_path in env_paths:
        if os.path.exists(env_path):
            load_dotenv(dotenv_path=env_path, override=False)
            print(f"Loaded environment variables from: {env_path}")
            break
    else:
        print("No .env file found. Using system environment variables only.")
        
except ImportError:
    print("python-dotenv not installed. Using system environment variables only.")


class TradingMode(str, Enum):
    """Trading system operational modes"""
    LIVE = "live"           # Live trading with real money
    PAPER = "paper"         # Paper trading (simulation)
    BACKTEST = "backtest"   # Historical backtesting
    DRY_RUN = "dry_run"     # Dry run mode (no orders)


class LogLevel(str, Enum):
    """Logging levels"""
    DEBUG = "DEBUG"
    INFO = "INFO" 
    WARNING = "WARNING"
    ERROR = "ERROR"


@dataclass
class OpenAlgoConfig:
    """OpenAlgo API configuration"""
    api_key: str = field(default_factory=lambda: os.getenv('OPENALGO_API_KEY', ''))
    api_host: str = field(default_factory=lambda: os.getenv('OPENALGO_API_HOST', 'http://127.0.0.1:5000'))
    websocket_host: str = field(default_factory=lambda: os.getenv('OPENALGO_WS_HOST', '127.0.0.1'))
    websocket_port: int = field(default_factory=lambda: int(os.getenv('OPENALGO_WS_PORT', '8765')))
    timeout: int = field(default_factory=lambda: int(os.getenv('OPENALGO_TIMEOUT', '30')))
    max_retries: int = field(default_factory=lambda: int(os.getenv('OPENALGO_MAX_RETRIES', '3')))
    retry_delay: float = field(default_factory=lambda: float(os.getenv('OPENALGO_RETRY_DELAY', '1.0')))


@dataclass
class MessagingConfig:
    """ZeroMQ and messaging configuration"""
    zmq_base_port: int = field(default_factory=lambda: int(os.getenv('ZMQ_BASE_PORT', '5555')))
    zmq_high_water_mark: int = field(default_factory=lambda: int(os.getenv('ZMQ_HIGH_WATER_MARK', '10000')))
    message_timeout: int = field(default_factory=lambda: int(os.getenv('ZMQ_MESSAGE_TIMEOUT', '5000')))  # milliseconds
    
    # Different ports for different message types
    market_data_port: int = field(default_factory=lambda: int(os.getenv('ZMQ_MARKET_DATA_PORT', '0')))
    strategy_signals_port: int = field(default_factory=lambda: int(os.getenv('ZMQ_STRATEGY_SIGNALS_PORT', '0'))) 
    trade_commands_port: int = field(default_factory=lambda: int(os.getenv('ZMQ_TRADE_COMMANDS_PORT', '0')))
    system_status_port: int = field(default_factory=lambda: int(os.getenv('ZMQ_SYSTEM_STATUS_PORT', '0')))
    
    def __post_init__(self):
        """Set default ports based on base port"""
        if self.market_data_port == 0:  # Default value
            self.market_data_port = self.zmq_base_port
            self.strategy_signals_port = self.zmq_base_port + 1
            self.trade_commands_port = self.zmq_base_port + 2
            self.system_status_port = self.zmq_base_port + 3


@dataclass
class PerformanceConfig:
    """Performance and optimization settings"""
    max_concurrent_requests: int = field(default_factory=lambda: int(os.getenv('MAX_CONCURRENT_REQUESTS', '10')))
    websocket_batch_size: int = field(default_factory=lambda: int(os.getenv('WEBSOCKET_BATCH_SIZE', '50')))
    websocket_batch_delay: float = field(default_factory=lambda: float(os.getenv('WEBSOCKET_BATCH_DELAY', '0.05')))  # seconds
    max_websocket_subscriptions: int = field(default_factory=lambda: int(os.getenv('MAX_WEBSOCKET_SUBSCRIPTIONS', '3000')))
    
    # Data processing
    market_data_buffer_size: int = field(default_factory=lambda: int(os.getenv('MARKET_DATA_BUFFER_SIZE', '10000')))
    strategy_calculation_workers: int = field(default_factory=lambda: int(os.getenv('STRATEGY_CALCULATION_WORKERS', '4')))
    
    # Memory management
    enable_memory_optimization: bool = field(default_factory=lambda: os.getenv('ENABLE_MEMORY_OPTIMIZATION', 'true').lower() == 'true')
    gc_threshold: int = field(default_factory=lambda: int(os.getenv('GC_THRESHOLD', '1000000')))  # bytes
    
    # CPU optimization
    enable_cpu_affinity: bool = field(default_factory=lambda: os.getenv('ENABLE_CPU_AFFINITY', 'false').lower() == 'true')
    cpu_cores: List[int] = field(default_factory=lambda: [int(x.strip()) for x in os.getenv('CPU_CORES', '').split(',') if x.strip()])


@dataclass
class RiskManagementConfig:
    """Risk management parameters"""
    max_position_size: float = field(default_factory=lambda: float(os.getenv('MAX_POSITION_SIZE', '100000.0')))  # Maximum position size in currency
    max_daily_loss: float = field(default_factory=lambda: float(os.getenv('MAX_DAILY_LOSS', '10000.0')))      # Maximum daily loss
    max_drawdown: float = field(default_factory=lambda: float(os.getenv('MAX_DRAWDOWN', '0.05')))           # Maximum drawdown (5%)
    
    # Position limits
    max_positions_per_symbol: int = field(default_factory=lambda: int(os.getenv('MAX_POSITIONS_PER_SYMBOL', '5')))
    max_total_positions: int = field(default_factory=lambda: int(os.getenv('MAX_TOTAL_POSITIONS', '50')))
    
    # Order limits
    max_order_size: float = field(default_factory=lambda: float(os.getenv('MAX_ORDER_SIZE', '10000.0')))
    min_order_size: float = field(default_factory=lambda: float(os.getenv('MIN_ORDER_SIZE', '100.0')))
    
    # Stop loss settings
    default_stop_loss: float = field(default_factory=lambda: float(os.getenv('DEFAULT_STOP_LOSS', '0.02')))      # 2% stop loss
    trailing_stop_enabled: bool = field(default_factory=lambda: os.getenv('TRAILING_STOP_ENABLED', 'true').lower() == 'true')
    trailing_stop_distance: float = field(default_factory=lambda: float(os.getenv('TRAILING_STOP_DISTANCE', '0.01')))  # 1% trailing distance


@dataclass
class DatabaseConfig:
    """Database configuration"""
    enabled: bool = field(default_factory=lambda: os.getenv('DATABASE_ENABLED', 'true').lower() == 'true')
    database_type: str = field(default_factory=lambda: os.getenv('DATABASE_TYPE', 'sqlite'))  # sqlite, postgresql, questdb
    
    # SQLite settings
    sqlite_path: str = field(default_factory=lambda: os.getenv('SQLITE_PATH', 'database/trading.db'))
    
    # PostgreSQL settings (if used)
    postgres_host: str = field(default_factory=lambda: os.getenv('POSTGRES_HOST', 'localhost'))
    postgres_port: int = field(default_factory=lambda: int(os.getenv('POSTGRES_PORT', '5432')))
    postgres_database: str = field(default_factory=lambda: os.getenv('POSTGRES_DATABASE', 'trading'))
    postgres_username: str = field(default_factory=lambda: os.getenv('POSTGRES_USERNAME', ''))
    postgres_password: str = field(default_factory=lambda: os.getenv('POSTGRES_PASSWORD', ''))
    
    # Connection pool settings
    pool_size: int = field(default_factory=lambda: int(os.getenv('DB_POOL_SIZE', '10')))
    max_overflow: int = field(default_factory=lambda: int(os.getenv('DB_MAX_OVERFLOW', '20')))
    pool_timeout: int = field(default_factory=lambda: int(os.getenv('DB_POOL_TIMEOUT', '30')))


@dataclass
class LoggingConfig:
    """Logging configuration"""
    level: LogLevel = field(default_factory=lambda: LogLevel(os.getenv('LOG_LEVEL', 'INFO')))
    console_enabled: bool = field(default_factory=lambda: os.getenv('CONSOLE_LOGGING_ENABLED', 'true').lower() == 'true')
    file_enabled: bool = field(default_factory=lambda: os.getenv('FILE_LOGGING_ENABLED', 'true').lower() == 'true')
    
    # File logging
    log_directory: str = field(default_factory=lambda: os.getenv('LOG_DIRECTORY', 'logs'))
    log_filename: str = field(default_factory=lambda: os.getenv('LOG_FILENAME', 'trading_{time:YYYY-MM-DD}.log'))
    max_file_size: str = field(default_factory=lambda: os.getenv('LOG_MAX_FILE_SIZE', '100 MB'))
    retention_days: int = field(default_factory=lambda: int(os.getenv('LOG_RETENTION_DAYS', '30')))
    
    # Performance logging
    performance_logging: bool = field(default_factory=lambda: os.getenv('PERFORMANCE_LOGGING', 'true').lower() == 'true')
    trade_logging: bool = field(default_factory=lambda: os.getenv('TRADE_LOGGING', 'true').lower() == 'true')
    error_logging: bool = field(default_factory=lambda: os.getenv('ERROR_LOGGING', 'true').lower() == 'true')
    
    # Log formatting
    console_format: str = field(default_factory=lambda: os.getenv('LOG_CONSOLE_FORMAT', '<green>{time:HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan> - <level>{message}</level>'))
    file_format: str = field(default_factory=lambda: os.getenv('LOG_FILE_FORMAT', '{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}'))


class TradingSettings:
    """Main trading system configuration"""
    
    def __init__(self):
        # Core settings from environment variables
        trading_mode_str = os.getenv('TRADING_MODE', 'paper')
        self.trading_mode: TradingMode = TradingMode(trading_mode_str) if trading_mode_str in [m.value for m in TradingMode] else TradingMode.PAPER
        self.system_name: str = os.getenv('SYSTEM_NAME', 'MyTrading')
        self.version: str = os.getenv('SYSTEM_VERSION', '1.0.0')
        
        # Component configurations
        self.openalgo = OpenAlgoConfig()
        self.messaging = MessagingConfig()
        self.performance = PerformanceConfig()
        self.risk_management = RiskManagementConfig()
        self.database = DatabaseConfig()
        self.logging = LoggingConfig()
        
        # System settings from environment variables
        self.enable_websocket: bool = os.getenv('ENABLE_WEBSOCKET', 'true').lower() == 'true'
        self.enable_historical_data: bool = os.getenv('ENABLE_HISTORICAL_DATA', 'true').lower() == 'true'
        self.enable_strategy_engine: bool = os.getenv('ENABLE_STRATEGY_ENGINE', 'true').lower() == 'true'
        self.enable_trade_execution: bool = os.getenv('ENABLE_TRADE_EXECUTION', 'false').lower() == 'true'  # Default to false for safety
        
        # Monitoring and alerts from environment variables
        self.enable_performance_monitoring: bool = os.getenv('ENABLE_PERFORMANCE_MONITORING', 'true').lower() == 'true'
        self.enable_health_checks: bool = os.getenv('ENABLE_HEALTH_CHECKS', 'true').lower() == 'true'
        self.health_check_interval: int = int(os.getenv('HEALTH_CHECK_INTERVAL', '60'))  # seconds
    
    
    @classmethod
    def from_file(cls, config_path: str) -> 'TradingSettings':
        """Load settings from YAML configuration file"""
        settings = cls()
        
        config_file = Path(config_path)
        if not config_file.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_file, 'r', encoding='utf-8') as f:
            config_data = yaml.safe_load(f)
        
        # Update settings from config file
        settings._update_from_dict(config_data)
        
        return settings
    
    def _update_from_dict(self, config_data: Dict[str, Any]):
        """Update settings from dictionary"""
        if not config_data:
            return
        
        # Update trading mode
        if 'trading_mode' in config_data:
            mode = config_data['trading_mode']
            if mode in [m.value for m in TradingMode]:
                self.trading_mode = TradingMode(mode)
        
        # Update OpenAlgo config
        if 'openalgo' in config_data:
            openalgo_config = config_data['openalgo']
            for key, value in openalgo_config.items():
                if hasattr(self.openalgo, key):
                    setattr(self.openalgo, key, value)
        
        # Update messaging config
        if 'messaging' in config_data:
            messaging_config = config_data['messaging']
            for key, value in messaging_config.items():
                if hasattr(self.messaging, key):
                    setattr(self.messaging, key, value)
        
        # Update performance config
        if 'performance' in config_data:
            perf_config = config_data['performance']
            for key, value in perf_config.items():
                if hasattr(self.performance, key):
                    setattr(self.performance, key, value)
        
        # Update risk management config
        if 'risk_management' in config_data:
            risk_config = config_data['risk_management']
            for key, value in risk_config.items():
                if hasattr(self.risk_management, key):
                    setattr(self.risk_management, key, value)
        
        # Update database config
        if 'database' in config_data:
            db_config = config_data['database']
            for key, value in db_config.items():
                if hasattr(self.database, key):
                    setattr(self.database, key, value)
        
        # Update logging config
        if 'logging' in config_data:
            log_config = config_data['logging']
            for key, value in log_config.items():
                if key == 'level' and value in [l.value for l in LogLevel]:
                    self.logging.level = LogLevel(value)
                elif hasattr(self.logging, key):
                    setattr(self.logging, key, value)
        
        # Update system settings
        system_settings = [
            'enable_websocket', 'enable_historical_data', 'enable_strategy_engine',
            'enable_trade_execution', 'enable_performance_monitoring', 'enable_health_checks',
            'health_check_interval'
        ]
        
        for setting in system_settings:
            if setting in config_data:
                setattr(self, setting, config_data[setting])
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert settings to dictionary"""
        return {
            'trading_mode': self.trading_mode.value,
            'system_name': self.system_name,
            'version': self.version,
            'openalgo': {
                'api_key': self.openalgo.api_key,
                'api_host': self.openalgo.api_host,
                'websocket_host': self.openalgo.websocket_host,
                'websocket_port': self.openalgo.websocket_port,
                'timeout': self.openalgo.timeout,
                'max_retries': self.openalgo.max_retries,
                'retry_delay': self.openalgo.retry_delay,
            },
            'messaging': {
                'zmq_base_port': self.messaging.zmq_base_port,
                'zmq_high_water_mark': self.messaging.zmq_high_water_mark,
                'message_timeout': self.messaging.message_timeout,
                'market_data_port': self.messaging.market_data_port,
                'strategy_signals_port': self.messaging.strategy_signals_port,
                'trade_commands_port': self.messaging.trade_commands_port,
                'system_status_port': self.messaging.system_status_port,
            },
            'performance': {
                'max_concurrent_requests': self.performance.max_concurrent_requests,
                'websocket_batch_size': self.performance.websocket_batch_size,
                'websocket_batch_delay': self.performance.websocket_batch_delay,
                'max_websocket_subscriptions': self.performance.max_websocket_subscriptions,
                'market_data_buffer_size': self.performance.market_data_buffer_size,
                'strategy_calculation_workers': self.performance.strategy_calculation_workers,
                'enable_memory_optimization': self.performance.enable_memory_optimization,
                'gc_threshold': self.performance.gc_threshold,
                'enable_cpu_affinity': self.performance.enable_cpu_affinity,
                'cpu_cores': self.performance.cpu_cores,
            },
            'risk_management': {
                'max_position_size': self.risk_management.max_position_size,
                'max_daily_loss': self.risk_management.max_daily_loss,
                'max_drawdown': self.risk_management.max_drawdown,
                'max_positions_per_symbol': self.risk_management.max_positions_per_symbol,
                'max_total_positions': self.risk_management.max_total_positions,
                'max_order_size': self.risk_management.max_order_size,
                'min_order_size': self.risk_management.min_order_size,
                'default_stop_loss': self.risk_management.default_stop_loss,
                'trailing_stop_enabled': self.risk_management.trailing_stop_enabled,
                'trailing_stop_distance': self.risk_management.trailing_stop_distance,
            },
            'database': {
                'enabled': self.database.enabled,
                'database_type': self.database.database_type,
                'sqlite_path': self.database.sqlite_path,
                'postgres_host': self.database.postgres_host,
                'postgres_port': self.database.postgres_port,
                'postgres_database': self.database.postgres_database,
                'postgres_username': self.database.postgres_username,
                'postgres_password': self.database.postgres_password,
                'pool_size': self.database.pool_size,
                'max_overflow': self.database.max_overflow,
                'pool_timeout': self.database.pool_timeout,
            },
            'logging': {
                'level': self.logging.level.value,
                'console_enabled': self.logging.console_enabled,
                'file_enabled': self.logging.file_enabled,
                'log_directory': self.logging.log_directory,
                'log_filename': self.logging.log_filename,
                'max_file_size': self.logging.max_file_size,
                'retention_days': self.logging.retention_days,
                'performance_logging': self.logging.performance_logging,
                'trade_logging': self.logging.trade_logging,
                'error_logging': self.logging.error_logging,
                'console_format': self.logging.console_format,
                'file_format': self.logging.file_format,
            },
            'enable_websocket': self.enable_websocket,
            'enable_historical_data': self.enable_historical_data,
            'enable_strategy_engine': self.enable_strategy_engine,
            'enable_trade_execution': self.enable_trade_execution,
            'enable_performance_monitoring': self.enable_performance_monitoring,
            'enable_health_checks': self.enable_health_checks,
            'health_check_interval': self.health_check_interval,
        }
    
    def save_to_file(self, config_path: str):
        """Save settings to YAML configuration file"""
        config_file = Path(config_path)
        config_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(config_file, 'w', encoding='utf-8') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, indent=2)
    
    def validate(self) -> List[str]:
        """Validate configuration and return list of errors"""
        errors = []
        
        # Validate OpenAlgo settings
        if not self.openalgo.api_key:
            errors.append("OpenAlgo API key is required")
        
        if not self.openalgo.api_host:
            errors.append("OpenAlgo API host is required")
        
        # Validate ports
        if not (1024 <= self.messaging.zmq_base_port <= 65535):
            errors.append("ZMQ base port must be between 1024 and 65535")
        
        if not (1024 <= self.openalgo.websocket_port <= 65535):
            errors.append("WebSocket port must be between 1024 and 65535")
        
        # Validate risk management
        if self.risk_management.max_daily_loss <= 0:
            errors.append("Maximum daily loss must be positive")
        
        if self.risk_management.max_position_size <= 0:
            errors.append("Maximum position size must be positive")
        
        if not (0 < self.risk_management.max_drawdown < 1):
            errors.append("Maximum drawdown must be between 0 and 1")
        
        # Validate performance settings
        if self.performance.max_concurrent_requests <= 0:
            errors.append("Maximum concurrent requests must be positive")
        
        if self.performance.websocket_batch_size <= 0:
            errors.append("WebSocket batch size must be positive")
        
        return errors
