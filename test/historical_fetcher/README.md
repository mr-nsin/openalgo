# Historical Data Fetcher for OpenAlgo

A comprehensive, production-ready system for fetching historical market data from Zerodha API and storing it in QuestDB with instrument-specific optimizations.

## 🎯 Features

- **Multi-Instrument Support**: Equity, Futures, Options (CE/PE), and Indices
- **Multiple Timeframes**: 1m, 3m, 5m, 15m, 30m, 1h, Daily
- **Async Processing**: High-performance concurrent data fetching
- **QuestDB Integration**: Optimized time-series database storage
- **Smart Rate Limiting**: Respects Zerodha API limits
- **Comprehensive Notifications**: Telegram and Email alerts
- **Market-Aware Scheduling**: Runs at optimal times based on market calendar
- **Robust Error Handling**: Retry logic and graceful failure recovery
- **Performance Monitoring**: System metrics and processing statistics
- **Modular Architecture**: Easy to extend and maintain

## 📊 Database Schema

The system creates optimized QuestDB tables for different instrument types:

- `equity_historical_data` - Equity stocks (NSE, BSE)
- `futures_historical_data` - Futures contracts with OI data
- `options_historical_data` - Options with Greeks support
- `index_historical_data` - Market indices
- `fetch_status` - Processing status tracking
- `fetch_summary` - Daily execution summaries

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- OpenAlgo installed and running
- QuestDB (will be installed automatically)
- Zerodha Kite Connect API credentials

### Installation

1. **Clone and navigate to the directory:**
   ```bash
   cd test/historical_fetcher
   ```

2. **Run the installation script:**
   ```bash
   chmod +x install.sh
   ./install.sh
   ```

3. **Configure your settings:**
   ```bash
   cp env_template.txt .env
   # Edit .env with your actual credentials
   nano .env
   ```

4. **Test the setup:**
   ```bash
   ./run.sh
   ```

### Manual Installation

If you prefer manual installation:

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install and start QuestDB
# See: https://questdb.io/get-questdb/

# Configure environment
cp env_template.txt .env
# Edit .env file with your settings
```

## ⚙️ Configuration

### Environment Variables

Key configuration options in `.env`:

```bash
# Zerodha API
HIST_FETCHER_ZERODHA_API_KEY=your_api_key
HIST_FETCHER_ZERODHA_ACCESS_TOKEN=your_access_token

# Data Configuration
HIST_FETCHER_ENABLED_TIMEFRAMES=1m,5m,15m,1h,D
HIST_FETCHER_ENABLED_INSTRUMENT_TYPES=EQ,FUT,CE,PE,INDEX
HIST_FETCHER_HISTORICAL_DAYS_LIMIT=365

# Performance
HIST_FETCHER_BATCH_SIZE=50
HIST_FETCHER_MAX_CONCURRENT_REQUESTS=5
HIST_FETCHER_API_REQUESTS_PER_SECOND=3

# Notifications
HIST_FETCHER_TELEGRAM_BOT_TOKEN=your_bot_token
HIST_FETCHER_TELEGRAM_CHAT_IDS=["your_chat_id"]
```

### Notification Setup

#### Telegram
1. Create a bot via [@BotFather](https://t.me/botfather)
2. Get your chat ID by messaging [@userinfobot](https://t.me/userinfobot)
3. Add credentials to `.env`

#### Email
1. Use Gmail App Password (not regular password)
2. Enable 2FA and generate App Password
3. Add SMTP settings to `.env`

## 🏃 Usage

### Manual Execution

```bash
# Single run
./run.sh

# With custom settings
HIST_FETCHER_BATCH_SIZE=100 ./run.sh
```

### Scheduled Execution

```bash
# Start scheduler service
./run_scheduler.sh

# The scheduler will run:
# - Daily at 6:30 PM IST (after market close)
# - Weekend full sync on Saturday 10:00 PM
# - Monthly cleanup on first Sunday
```

### Python API

```python
from main import HistoricalDataFetcher

# Create and run fetcher
fetcher = HistoricalDataFetcher()
await fetcher.run()
```

## 📈 Data Access

### QuestDB Console
Access the web console at: http://localhost:9000

### Sample Queries

```sql
-- Get latest equity data
SELECT * FROM equity_historical_data 
WHERE symbol = 'RELIANCE' AND timeframe = 'D'
ORDER BY timestamp DESC LIMIT 10;

-- Options data with strike analysis
SELECT underlying_symbol, strike_price, option_type, 
       AVG(close) as avg_price, SUM(volume) as total_volume
FROM options_historical_data 
WHERE underlying_symbol = 'NIFTY' 
  AND timeframe = '1h'
  AND timestamp > dateadd('d', -7, now())
GROUP BY underlying_symbol, strike_price, option_type;

-- Futures OI analysis
SELECT contract_symbol, 
       FIRST(oi) as opening_oi,
       LAST(oi) as closing_oi,
       LAST(oi) - FIRST(oi) as oi_change
FROM futures_historical_data 
WHERE underlying_symbol = 'BANKNIFTY'
  AND timeframe = 'D'
  AND timestamp > dateadd('d', -30, now())
SAMPLE BY 1d;
```

## 🔧 Architecture

### Components

```
historical_fetcher/
├── config/           # Configuration management
├── fetchers/         # Data fetching logic
├── database/         # QuestDB client and models
├── notifications/    # Telegram and email alerts
├── utils/           # Utilities (logging, rate limiting)
├── scheduler/       # Market-aware job scheduling
└── main.py          # Main orchestrator
```

### Data Flow

1. **Symbol Discovery**: Query OpenAlgo's symtoken table
2. **Instrument Classification**: Group by type (EQ/FUT/CE/PE/INDEX)
3. **Batch Processing**: Process symbols concurrently with rate limiting
4. **Data Fetching**: Retrieve historical data from Zerodha API
5. **Storage**: Store in appropriate QuestDB tables
6. **Monitoring**: Track progress and send notifications

## 📊 Monitoring

### Logs
```bash
# View real-time logs
tail -f logs/historical_fetcher.log

# View structured logs
tail -f logs/historical_fetcher_structured.jsonl
```

### Performance Metrics
- Processing rate (symbols/second)
- API success rate
- Memory and CPU usage
- Database insertion performance

### Notifications
- Success/failure alerts
- Progress updates (every 5 minutes)
- Performance warnings
- Error details with context

## 🛠️ Troubleshooting

### Common Issues

1. **QuestDB Connection Failed**
   ```bash
   # Check if QuestDB is running
   ps aux | grep questdb
   
   # Start QuestDB
   questdb start -d data/questdb
   ```

2. **Zerodha API Errors**
   - Verify API credentials in `.env`
   - Check rate limiting settings
   - Ensure access token is valid

3. **No Symbols Found**
   - Ensure OpenAlgo is running
   - Download master contract in OpenAlgo
   - Check DATABASE_URL in `.env`

4. **Memory Issues**
   - Reduce BATCH_SIZE
   - Lower MAX_CONCURRENT_REQUESTS
   - Increase MEMORY_LIMIT_MB

### Debug Mode

```bash
# Enable debug logging
HIST_FETCHER_LOG_LEVEL=DEBUG ./run.sh

# Test specific components
python -c "
from config.settings import Settings
from database.questdb_client import QuestDBClient
import asyncio

async def test():
    settings = Settings()
    client = QuestDBClient(settings)
    await client.connect()
    print('QuestDB connection successful')

asyncio.run(test())
"
```

## 🔄 Maintenance

### Regular Tasks

1. **Monitor Disk Space**: QuestDB data grows over time
2. **Update Market Calendar**: Add new holidays annually
3. **Rotate Logs**: Automatic with loguru configuration
4. **Performance Tuning**: Adjust batch sizes based on usage

### Backup Strategy

```bash
# Backup QuestDB data
cp -r data/questdb data/questdb_backup_$(date +%Y%m%d)

# Backup configuration
cp .env .env.backup
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is part of OpenAlgo and follows the same license terms.

## 🆘 Support

- **Documentation**: Check this README and inline code comments
- **Issues**: Report bugs via GitHub issues
- **Community**: Join OpenAlgo community discussions
- **Logs**: Always check logs first for error details

## 📚 Additional Resources

- [QuestDB Documentation](https://questdb.io/docs/)
- [Zerodha Kite Connect API](https://kite.trade/docs/connect/v3/)
- [OpenAlgo Documentation](https://docs.openalgo.in/)
- [APScheduler Documentation](https://apscheduler.readthedocs.io/)

---

**Note**: This system is designed for educational and research purposes. Ensure compliance with your broker's terms of service and applicable regulations when using in production.
