# OpenAlgo Historical Data Fetcher - Setup Guide

This guide walks you through setting up the historical data fetcher to work with OpenAlgo's API layer.

## 🔑 Understanding the Authentication

### How Authentication Works

The historical data fetcher uses **OpenAlgo's API layer** for authentication, not direct broker APIs:

```
┌─────────────────┐    ┌─────────────┐    ┌─────────────┐    ┌──────────────┐
│ Historical      │───▶│ OpenAlgo    │───▶│ Broker      │───▶│ Historical   │
│ Fetcher         │    │ API Layer   │    │ API         │    │ Data         │
└─────────────────┘    └─────────────┘    └─────────────┘    └──────────────┘
```

### Why This Approach?

1. **Unified Authentication**: One OpenAlgo API key works across all supported brokers
2. **Security**: Broker credentials are stored securely in OpenAlgo
3. **Consistency**: Same authentication method as other OpenAlgo services
4. **Error Handling**: Centralized error handling and rate limiting

## 📋 Prerequisites

Before setting up the historical fetcher, ensure you have:

### 1. OpenAlgo Instance Running
- OpenAlgo should be installed and running
- Your broker should be configured and authenticated in OpenAlgo
- You should be able to login to OpenAlgo web interface

### 2. Broker Authentication in OpenAlgo
Make sure your broker is properly configured in OpenAlgo:
- Login to OpenAlgo web interface
- Navigate to broker configuration
- Ensure your broker credentials are saved and working
- Test that you can fetch live quotes or place test orders

### 3. OpenAlgo API Key
Generate an API key from OpenAlgo:
- Login to OpenAlgo web interface
- Go to Settings → API Keys
- Click "Generate New Key"
- Copy the generated API key (you'll need this for the fetcher)

## 🚀 Step-by-Step Setup

### Step 1: Navigate to Historical Fetcher Directory
```bash
cd test/historical_fetcher
```

### Step 2: Create Environment File
```bash
# Copy the template
cp env_template.txt .env

# Edit the environment file
nano .env  # or use your preferred editor
```

### Step 3: Configure Environment Variables

Edit the `.env` file with your settings:

```bash
# ============================================================================
# REQUIRED SETTINGS
# ============================================================================

# Your OpenAlgo API Key (from OpenAlgo web interface)
OPENALGO_API_KEY=your_actual_openalgo_api_key_here

# OpenAlgo instance URL (change if running on different host/port)
OPENALGO_API_HOST=http://127.0.0.1:5000

# Database URL (should match your OpenAlgo DATABASE_URL)
DATABASE_URL=sqlite:///db/openalgo.db

# ============================================================================
# OPTIONAL SETTINGS (can use defaults)
# ============================================================================

# What data to fetch
HIST_FETCHER_ENABLED_TIMEFRAMES=1m,5m,15m,1h,D
HIST_FETCHER_ENABLED_INSTRUMENT_TYPES=EQ,FUT,CE,PE,INDEX
HIST_FETCHER_HISTORICAL_DAYS_LIMIT=365

# Performance settings
HIST_FETCHER_BATCH_SIZE=50
HIST_FETCHER_MAX_CONCURRENT_REQUESTS=5
HIST_FETCHER_API_REQUESTS_PER_SECOND=3
```

### Step 4: Install Dependencies
```bash
# Install Python dependencies
pip install -r requirements.txt

# If you get a Pydantic error, install pydantic-settings separately:
pip install pydantic-settings

# Install and start QuestDB (for storing historical data)
# See QuestDB installation guide: https://questdb.io/get-questdb/
```

### Step 5: Test the Setup
```bash
# First, test the Pydantic compatibility fix
python test_pydantic_fix.py

# If the test passes, run the main historical fetcher
python openalgo_main.py
```

## 🔧 Verification Steps

### 1. Test OpenAlgo API Connection
Before running the fetcher, verify your OpenAlgo API key works:

```bash
# Test API key (replace with your actual key)
curl -X GET "http://127.0.0.1:5000/api/v1/quotes?symbol=RELIANCE&exchange=NSE&apikey=your_openalgo_api_key"
```

Expected response: JSON with quote data (not an error)

### 2. Check Database Connection
Verify the fetcher can connect to OpenAlgo's database:

```python
# Quick test script
from config.openalgo_settings import OpenAlgoSettings
from database.auth_db import get_auth_token_broker

settings = OpenAlgoSettings()
auth_token, broker_name = get_auth_token_broker(settings.openalgo_api_key)

if auth_token:
    print(f"✅ Authentication successful! Broker: {broker_name}")
else:
    print("❌ Authentication failed! Check your OPENALGO_API_KEY")
```

### 3. Test Symbol Retrieval
Verify the fetcher can access OpenAlgo's symbol database:

```python
from fetchers import OpenAlgoSymbolManager
from config.openalgo_settings import OpenAlgoSettings
import asyncio

async def test_symbols():
    settings = OpenAlgoSettings()
    symbol_manager = OpenAlgoSymbolManager(settings)
    symbols = await symbol_manager.get_all_active_symbols()
    
    total_symbols = sum(len(symbols) for symbols in symbols.values())
    print(f"✅ Found {total_symbols} symbols in OpenAlgo database")
    
    for inst_type, symbol_list in symbols.items():
        if symbol_list:
            print(f"  {inst_type}: {len(symbol_list)} symbols")

asyncio.run(test_symbols())
```

## ❌ Common Issues and Solutions

### Issue 1: "Invalid openalgo apikey"
**Cause**: Wrong or expired OpenAlgo API key
**Solution**: 
- Generate a new API key from OpenAlgo web interface
- Update OPENALGO_API_KEY in .env file
- Ensure no extra spaces or characters in the key

### Issue 2: "No symbols found"
**Cause**: OpenAlgo database doesn't have symbol data
**Solution**:
- Login to OpenAlgo web interface
- Download/update master contract data
- Ensure DATABASE_URL points to correct OpenAlgo database

### Issue 3: "Connection refused to OpenAlgo"
**Cause**: OpenAlgo instance is not running
**Solution**:
- Start your OpenAlgo instance
- Verify OPENALGO_API_HOST is correct
- Check if OpenAlgo is accessible at the specified URL

### Issue 4: "Broker authentication failed"
**Cause**: Broker credentials not configured in OpenAlgo
**Solution**:
- Login to OpenAlgo web interface
- Configure your broker credentials
- Test broker connection in OpenAlgo first

### Issue 5: "QuestDB connection failed"
**Cause**: QuestDB is not running or misconfigured
**Solution**:
- Install and start QuestDB
- Check HIST_FETCHER_QUESTDB_HOST and HIST_FETCHER_QUESTDB_PORT
- Ensure QuestDB is accessible

### Issue 6: "BaseSettings has been moved to pydantic-settings"
**Cause**: Using Pydantic v2+ without pydantic-settings package
**Solution**:
```bash
# Install pydantic-settings
pip install pydantic-settings

# Or upgrade all dependencies
pip install -r requirements.txt --upgrade
```

## 🔍 Debugging Tips

### Enable Debug Logging
```bash
# Set debug level in .env
HIST_FETCHER_LOG_LEVEL=DEBUG

# Run with debug output
python openalgo_main.py
```

### Check Log Files
```bash
# View real-time logs
tail -f logs/historical_fetcher.log

# View structured logs
tail -f logs/historical_fetcher_structured.jsonl
```

### Test Individual Components
```python
# Test OpenAlgo API connection
from openalgo import api
client = api(api_key="your_key", host="http://127.0.0.1:5000")
print(client.quotes("RELIANCE", "NSE"))

# Test database connection
from database.auth_db import get_auth_token_broker
result = get_auth_token_broker("your_openalgo_api_key")
print(f"Auth result: {result}")
```

## 📞 Getting Help

If you encounter issues:

1. **Check Logs**: Always check the log files first
2. **Verify Prerequisites**: Ensure OpenAlgo is running and broker is configured
3. **Test Components**: Use the verification steps above
4. **Community Support**: Join OpenAlgo community discussions
5. **GitHub Issues**: Report bugs with log details

## 🎯 Next Steps

Once setup is complete:

1. **Run Initial Fetch**: Start with a small date range to test
2. **Monitor Performance**: Check logs and system resources
3. **Schedule Regular Runs**: Set up cron jobs for automated fetching
4. **Explore Data**: Use QuestDB console to query historical data

Remember: The historical fetcher is designed to work seamlessly with OpenAlgo's existing infrastructure. You don't need to manage broker credentials directly - OpenAlgo handles all of that for you!
