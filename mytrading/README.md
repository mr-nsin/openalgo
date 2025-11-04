# MyTrading - Advanced Real-time Trading System

A high-performance, multi-component trading system built on OpenAlgo infrastructure with real-time WebSocket data feeds, historical data integration, and automated strategy execution.

## 🚀 Features

### Core Capabilities
- **Real-time Market Data**: WebSocket-based data feeds with microsecond precision
- **Historical Data Integration**: Seamless fusion of real-time and historical data
- **Multi-Strategy Engine**: Support for technical, momentum, mean-reversion, and options strategies
- **Advanced Risk Management**: Position sizing, stop-loss, and drawdown controls
- **High-Performance Messaging**: ZeroMQ-based inter-component communication
- **Comprehensive Monitoring**: Performance metrics, health checks, and alerting

### Supported Instruments
- **Indices**: NIFTY, BANKNIFTY, SENSEX
- **Options**: Full options chain support with Greeks calculation
- **Equities**: Individual stocks and ETFs
- **Futures**: Futures contracts with roll-over management

### Trading Modes
- **Live Trading**: Real money trading with full risk controls
- **Paper Trading**: Simulation mode for strategy testing
- **Backtesting**: Historical strategy validation
- **Dry Run**: System testing without order placement

## 📁 Project Structure

```
mytrading/
├── config/                 # Configuration classes
│   ├── settings.py         # Main system configuration
│   ├── symbols.py          # Symbol and market data config
│   └── strategies.py       # Strategy definitions
├── core/                   # Core system components
│   ├── orchestrator.py     # Main system orchestrator
│   ├── data_manager.py     # Data layer management
│   ├── strategy_engine.py  # Strategy execution engine
│   └── trade_manager.py    # Trade execution and management
├── data/                   # Data layer
│   ├── websocket_feed.py   # Real-time WebSocket data
│   ├── historical_fetcher.py # Historical data integration
│   └── data_fusion.py      # Data combination logic
├── strategies/             # Trading strategies
│   ├── base_strategy.py    # Base strategy interface
│   ├── technical_indicators.py # Technical analysis
│   └── options_strategies.py # Options-specific strategies
├── communication/          # Messaging system
│   ├── zmq_publisher.py    # ZeroMQ message publishing
│   ├── zmq_subscriber.py   # ZeroMQ message subscription
│   └── message_types.py    # Message format definitions
├── utils/                  # Utility modules
│   ├── logging_config.py   # Advanced logging setup
│   ├── performance_monitor.py # Performance tracking
│   └── helpers.py          # Common utilities
└── main.py                 # Main entry point
```

## 🛠️ Installation

### Prerequisites
- Python 3.9 or higher
- OpenAlgo system running locally
- ZeroMQ library

### Setup Steps

1. **Clone or create the project directory**:
   ```bash
   mkdir mytrading
   cd mytrading
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure the system**:
   ```bash
   # Copy and edit environment configuration
   cp env_template.txt .env
   # Edit .env file with your settings
   nano .env
   ```

4. **Set your API key in .env file**:
   ```bash
   # Edit .env file and set:
   OPENALGO_API_KEY=your_api_key_here
   OPENALGO_API_HOST=http://127.0.0.1:5000
   TRADING_MODE=paper  # paper, live, backtest, dry_run
   ```

## 🚀 Quick Start

### Basic Usage

1. **Start the trading system**:
   ```bash
   python main.py
   ```

2. **With debug logging**:
   ```bash
   python main.py --log-level DEBUG
   ```

3. **Paper trading mode with debug logging**:
   ```bash
   python main.py --dry-run --log-level DEBUG
   ```

### Configuration Examples

#### Basic NIFTY Trading Setup (.env file)
```bash
# Trading Configuration
TRADING_MODE=paper
OPENALGO_API_KEY=your_api_key_here

# Enabled Strategies
ENABLED_STRATEGIES=SMA_Crossover,RSI_MeanReversion

# Strategy Parameters
SMA_FAST_PERIOD=20
SMA_SLOW_PERIOD=50
RSI_PERIOD=14
RSI_OVERSOLD=30
RSI_OVERBOUGHT=70

# Enabled Instruments
ENABLED_TIMEFRAMES=1m,5m,15m,1h
ENABLED_INSTRUMENT_TYPES=INDEX,CE,PE
ENABLED_EXCHANGES=NSE_INDEX,NFO

# Risk Management
MAX_POSITION_SIZE=50000.0
DEFAULT_STOP_LOSS=0.02
TRAILING_STOP_ENABLED=true
```

## 📊 System Architecture

### Component Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    MAIN ORCHESTRATOR                        │
│                   (Async Event Loop)                        │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────┬─────────────────┬─────────────────┬─────────────────┐
│   DATA LAYER    │  STRATEGY LAYER │  SIGNAL LAYER   │   TRADE LAYER   │
│   (Async Task)  │  (Async Task)   │  (Async Task)   │  (Async Task)   │
└─────────────────┴─────────────────┴─────────────────┴─────────────────┘
         │                 │                 │                 │
         ▼                 ▼                 ▼                 ▼
    ZMQ Publisher    ZMQ Subscriber     ZMQ Subscriber    ZMQ Subscriber
                     ZMQ Publisher      ZMQ Publisher     ZMQ Publisher
```

### Data Flow

1. **Market Data**: WebSocket → Data Manager → ZMQ Publisher
2. **Historical Data**: Database/API → Data Fusion → ZMQ Publisher
3. **Strategy Signals**: Strategy Engine → Signal Generator → ZMQ Publisher
4. **Trade Execution**: Trade Manager → OpenAlgo API → Position Updates

### Message Types

- **Market Data**: Real-time price, volume, and depth data
- **Strategy Signals**: Buy/sell signals with confidence levels
- **Trade Messages**: Order placement and execution updates
- **System Status**: Health checks and performance metrics

## 📈 Performance Features

### High-Frequency Capabilities
- **Sub-millisecond latency**: Optimized message passing
- **10Hz display refresh**: Real-time option chain updates
- **Batch processing**: Efficient handling of multiple symbols
- **Memory optimization**: Minimal garbage collection impact

### Monitoring and Alerting
- **Real-time metrics**: CPU, memory, and network usage
- **Performance tracking**: Message throughput and latency
- **Health checks**: Component status monitoring
- **Alert system**: Configurable thresholds and notifications

## 🛡️ Risk Management

### Built-in Controls
- **Position limits**: Maximum position size and count
- **Stop-loss orders**: Automatic loss limitation
- **Drawdown protection**: Maximum portfolio drawdown
- **Time-based rules**: Trading hours and session limits

### Configuration Example (.env file)
```bash
# Risk Management Configuration
MAX_POSITION_SIZE=100000.0
MAX_DAILY_LOSS=10000.0
MAX_DRAWDOWN=0.05
DEFAULT_STOP_LOSS=0.02
TRAILING_STOP_ENABLED=true
MAX_POSITIONS_PER_SYMBOL=5
MAX_TOTAL_POSITIONS=50
```

## 📝 Logging and Monitoring

### Log Levels and Files
- **Console**: Real-time system status
- **Main Log**: Complete system activity
- **Trade Log**: All trading activities
- **Performance Log**: System performance metrics
- **Error Log**: Errors and exceptions

### Performance Monitoring
```python
# Example: Monitor strategy performance
from mytrading.utils.performance_monitor import get_performance_monitor

monitor = get_performance_monitor()
stats = monitor.get_summary_report()
print(f"Average strategy execution: {stats['strategy_engine.execute']['mean']:.2f}ms")
```

## 🧪 Testing

### Running Tests
```bash
# Run all tests
pytest

# Run specific test category
pytest tests/test_strategies.py

# Run with coverage
pytest --cov=mytrading tests/
```

### Paper Trading
The system includes comprehensive paper trading capabilities for strategy validation:

```bash
# Set TRADING_MODE=paper in .env file, then run:
python main.py

# Or use dry-run flag for simulation mode:
python main.py --dry-run
```

## 🔧 Development

### Adding New Strategies
1. Create strategy class inheriting from `BaseStrategy`
2. Implement required methods: `initialize()`, `on_data()`, `generate_signals()`
3. Add strategy to `ENABLED_STRATEGIES` in `.env` file
4. Add strategy-specific parameters to `.env` file
5. Register strategy in the strategy engine

### Custom Indicators
```python
from mytrading.strategies.base_strategy import BaseStrategy

class CustomStrategy(BaseStrategy):
    def initialize(self):
        self.add_indicator("SMA", period=20)
        self.add_indicator("RSI", period=14)
    
    def on_data(self, data):
        sma = self.get_indicator_value("SMA")
        rsi = self.get_indicator_value("RSI")
        
        if rsi < 30 and data.close > sma:
            return self.create_signal("BUY", confidence=0.8)
```

## 📚 API Reference

### Core Classes
- `TradingOrchestrator`: Main system coordinator
- `DataManager`: Market data handling
- `StrategyEngine`: Strategy execution
- `TradeManager`: Order management
- `ZMQPublisher/Subscriber`: Message passing

### Configuration Classes
- `TradingSettings`: System configuration
- `SymbolConfig`: Symbol and market data setup
- `StrategyConfig`: Strategy parameters

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## ⚠️ Disclaimer

This software is for educational and research purposes. Trading involves substantial risk of loss. The authors are not responsible for any financial losses incurred through the use of this software.

## 📞 Support

For questions and support:
- Create an issue in the repository
- Check the documentation in the `docs/` directory
- Review the `env_template.txt` file for configuration options
- Copy `env_template.txt` to `.env` and customize for your setup

---

**Happy Trading! 🚀📈**
