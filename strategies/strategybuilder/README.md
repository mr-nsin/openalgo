# Strategy Builder

A comprehensive framework for backtesting trading strategies using OpenAlgo historical data.

## Features

- **Multiple Data Sources**: Fetch data from QuestDB (fast) or OpenAlgo API (always available)
- **Modular Strategy Design**: Easy to create and test new strategies
- **Comprehensive Backtesting**: Full backtesting engine with detailed statistics
- **Configuration via .env**: All settings managed through environment variables
- **OpenAlgo Integration**: Seamlessly integrates with OpenAlgo's data infrastructure

## Structure

```
strategybuilder/
├── config/
│   ├── __init__.py
│   └── strategy_settings.py      # Configuration from .env
├── services/
│   ├── __init__.py
│   └── data_service.py            # Data fetching service
├── strategies/
│   ├── __init__.py
│   └── moving_average_crossover.py # Example strategy
├── backtesting/
│   ├── __init__.py
│   └── backtest_engine.py          # Backtesting framework
├── main.py                         # Main entry point
├── requirements.txt                # Python dependencies
├── env_template.txt                # Environment variable template
└── README.md                       # This file
```

## Installation

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure Environment**:
   ```bash
   cp env_template.txt .env
   # Edit .env with your OpenAlgo API key and other settings
   ```

3. **Ensure OpenAlgo is Running**:
   - Make sure your OpenAlgo instance is running
   - Generate an API key from OpenAlgo web interface
   - Add the API key to `.env` file

## Usage

### Configuration-Based Usage (Recommended)

All configuration is done through the `.env` file. No command-line arguments needed!

1. **Copy the environment template**:
   ```bash
   cp env_template.txt .env
   ```

2. **Edit `.env` file** with your settings:
   ```bash
   # Strategy Configuration
   STRATEGY_NAME=mvg_avg_crossover_original
   
   # Backtest Configuration
   STRATEGY_SYMBOL=RELIANCE
   STRATEGY_EXCHANGE=NSE
   STRATEGY_INTERVAL=D
   STRATEGY_START_DATE=2023-01-01
   STRATEGY_END_DATE=2024-01-01
   STRATEGY_INITIAL_CAPITAL=100000
   
   # Strategy Parameters
   STRATEGY_SHORT_SMA_PERIOD=50
   STRATEGY_MID_EMA_PERIOD=100
   STRATEGY_LONG_SMA_PERIOD=200
   STRATEGY_SWING_LOOKBACK=5
   STRATEGY_VOLUME_FILTER=true
   STRATEGY_MACD_FILTER=true
   
   # OpenAlgo API (Required)
   OPENALGO_API_KEY=your_api_key_here
   ```

3. **Run the backtest**:
   ```bash
   python main.py
   ```

That's it! All configuration is read from `.env` file.

## Architecture

The strategy builder is designed with a modular architecture:

```
main.py
├── fetch_historical_data()    # Data fetching (separated)
├── run_strategy_backtest()     # Strategy execution (separated)
└── main()                      # Orchestrates the flow

strategies/
├── strategy_registry.py        # Strategy registry
└── mvg_avg_crossover_original.py  # Original strategy implementation
```

### Key Features

1. **Separated Data Fetching**: Data fetching is completely separate from strategy logic
2. **Strategy Registry**: Easy to add new strategies by registering them
3. **Configuration-Driven**: All settings in `.env` file, no command-line arguments
4. **Original Logic Preserved**: The original `mvg_avg_co_bt_stgy.py` logic is embedded exactly as-is

## Strategy: Moving Average Crossover (Original)

The included strategy is the exact implementation from `mvg_avg_co_bt_stgy.py`:

### Entry Rules
- **Long**: SMA50 crosses above SMA200, price above EMA100, volume increasing, MACD bullish
- **Short**: SMA50 crosses below SMA200, price below EMA100, volume increasing, MACD bearish

### Exit Rules
- Stop loss based on swing high/low (trailing stop)
- Opposite signal

### Parameters (configured in .env)
- `STRATEGY_SHORT_SMA_PERIOD`: Short SMA period (default: 50)
- `STRATEGY_MID_EMA_PERIOD`: Mid EMA period (default: 100)
- `STRATEGY_LONG_SMA_PERIOD`: Long SMA period (default: 200)
- `STRATEGY_SWING_LOOKBACK`: Swing lookback period for stop loss (default: 5)
- `STRATEGY_VOLUME_FILTER`: Enable volume filter (default: true)
- `STRATEGY_MACD_FILTER`: Enable MACD filter (default: true)

## Creating New Strategies

1. **Create a new strategy class** in `strategies/` directory:

```python
# strategies/my_strategy.py
import pandas as pd
from typing import Dict, Any, Optional

class MyStrategy:
    def __init__(self, params: Optional[Dict[str, Any]] = None):
        self.name = "My Strategy"
        self.params = params or {}
        # Initialize your parameters
        self.param1 = self.params.get('param1', 10)
    
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate your indicators"""
        df = df.copy()
        # Your indicator logic here
        return df
    
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate trading signals (1=Buy, -1=Sell, 0=Hold)"""
        df = df.copy()
        df['Signal'] = 0
        # Your signal logic here
        return df
    
    def run_backtest(self, df: pd.DataFrame, initial_capital: float) -> Dict[str, Any]:
        """Run backtest and return results"""
        # Your backtest logic here
        return {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'win_rate': 0.0,
            'total_pnl': 0.0,
            'final_capital': initial_capital
        }
    
    def prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare data by calculating indicators and generating signals"""
        df = self.calculate_indicators(df)
        df = self.generate_signals(df)
        return df
```

2. **Register the strategy** in `strategies/strategy_registry.py`:

```python
from .my_strategy import MyStrategy

STRATEGIES: Dict[str, Type] = {
    'mvg_avg_crossover_original': MovingAverageCrossoverOriginal,
    'my_strategy': MyStrategy,  # Add your strategy
}
```

3. **Add configuration** to `config/strategy_settings.py` and `env_template.txt` if needed

4. **Use it** by setting `STRATEGY_NAME=my_strategy` in your `.env` file

## Backtest Results

The backtest engine provides comprehensive statistics:

- **Total Trades**: Number of trades executed
- **Winning/Losing Trades**: Count of profitable and losing trades
- **Win Rate**: Percentage of winning trades
- **Total P&L**: Total profit/loss
- **Total Return**: Percentage return on initial capital
- **Max Drawdown**: Maximum drawdown percentage
- **Average Win/Loss**: Average profit per winning/losing trade
- **Profit Factor**: Ratio of gross profit to gross loss

## Data Sources

### QuestDB (Recommended)
- **Pros**: Fast, efficient for large datasets
- **Cons**: Requires historicalfetcher to populate data
- **Use When**: You have QuestDB set up and populated

### OpenAlgo API (Fallback)
- **Pros**: Always available, no setup required
- **Cons**: Slower, rate-limited
- **Use When**: QuestDB is not available or for small tests

The strategy builder automatically falls back to OpenAlgo API if QuestDB is unavailable.

## Configuration

All configuration is done through environment variables in `.env` file:

- `OPENALGO_API_KEY`: Your OpenAlgo API key (required)
- `OPENALGO_API_HOST`: OpenAlgo API host (default: http://127.0.0.1:5000)
- `STRATEGY_DATA_SOURCE`: Data source (`questdb` or `openalgo_api`)
- `STRATEGY_QUESTDB_HOST`: QuestDB host (if using QuestDB)
- `STRATEGY_QUESTDB_PORT`: QuestDB port (default: 8812)
- `STRATEGY_LOG_LEVEL`: Logging level (default: INFO)

See `env_template.txt` for all available options.

## Examples

### Example 1: Daily Backtest
Edit `.env`:
```bash
STRATEGY_SYMBOL=RELIANCE
STRATEGY_EXCHANGE=NSE
STRATEGY_INTERVAL=D
STRATEGY_START_DATE=2023-01-01
STRATEGY_END_DATE=2024-01-01
```

Run:
```bash
python main.py
```

### Example 2: Intraday Backtest (5-minute)
Edit `.env`:
```bash
STRATEGY_SYMBOL=RELIANCE
STRATEGY_EXCHANGE=NSE
STRATEGY_INTERVAL=5m
STRATEGY_START_DATE=2024-01-01
STRATEGY_END_DATE=2024-01-31
```

Run:
```bash
python main.py
```

### Example 3: Custom Parameters
Edit `.env`:
```bash
STRATEGY_SYMBOL=TCS
STRATEGY_EXCHANGE=NSE
STRATEGY_INTERVAL=D
STRATEGY_START_DATE=2023-01-01
STRATEGY_END_DATE=2024-01-01
STRATEGY_SHORT_SMA_PERIOD=20
STRATEGY_LONG_SMA_PERIOD=100
STRATEGY_SWING_LOOKBACK=10
```

Run:
```bash
python main.py
```

## Troubleshooting

### "No data found" Error
- Check if the symbol exists in the exchange
- Verify the date range is valid
- Ensure OpenAlgo API is accessible
- If using QuestDB, check if historicalfetcher has populated data

### "Connection refused" Error
- Verify QuestDB is running (if using QuestDB)
- Check QuestDB port (should be 8812 for PostgreSQL wire protocol)
- Verify OpenAlgo API is running

### Import Errors
- Ensure all dependencies are installed: `pip install -r requirements.txt`
- Check that OpenAlgo is properly installed in the parent directory

## License

Part of the OpenAlgo project.

