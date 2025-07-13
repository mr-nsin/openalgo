# Broker Configuration Management System

## Overview

The Broker Configuration Management System provides a centralized, dynamic way to manage broker-specific configurations in OpenAlgo. This system allows brokers to define their API endpoints, WebSocket settings, authentication parameters, and feature capabilities in JSON configuration files.

## Architecture

### Core Components

1. **BrokerConfigManager** (`utils/broker_config.py`)
   - Singleton pattern for consistent access
   - Thread-safe configuration loading
   - Automatic broker detection from calling context
   - Configuration caching for performance

2. **Broker Configuration Files**
   - Location: `broker/{broker_name}/config/{broker_name}_config.json`
   - Comprehensive JSON schema covering all broker aspects
   - Example: `broker/angel/config/angel_config.json`

3. **Integration Points**
   - Flask app initialization (`app.py`)
   - Broker API modules (auth, data, orders, etc.)
   - WebSocket adapters
   - Plugin loader system

## Configuration Schema

Each broker configuration file follows this structure:

```json
{
  "broker_name": "angel",
  "display_name": "Angel One",
  "description": "Angel One (formerly Angel Broking) API integration",
  "api_config": {
    "base_url": "https://apiconnect.angelbroking.com",
    "headers": {
      "Content-Type": "application/json",
      "Accept": "application/json"
    },
    "endpoints": {
      "login": "/rest/auth/angelbroking/user/v1/loginByPassword",
      "logout": "/rest/secure/angelbroking/user/v1/logout",
      "profile": "/rest/secure/angelbroking/user/v1/getProfile"
    },
    "timeout": 30,
    "retry_count": 3,
    "backoff_factor": 0.3
  },
  "websocket_config": {
    "url": "wss://smartapisocket.angelone.in/smart-stream",
    "heartbeat_interval": 30,
    "reconnect_delay": 5,
    "max_reconnect_attempts": 10
  },
  "features": {
    "supports_market_data": true,
    "supports_websocket": true,
    "supports_options": true
  },
  "exchanges": {
    "supported": ["NSE", "BSE", "NFO", "BFO", "CDS", "MCX"]
  },
  "timeframes": {
    "supported": ["1m", "3m", "5m", "15m", "30m", "1h", "1D"]
  }
}
```

## Usage

### In Broker API Files

```python
from utils.broker_config import BrokerConfigManager

# Get configuration manager
manager = BrokerConfigManager()

# Get broker-specific configuration (auto-detected from calling context)
base_url = manager.get_base_url()
headers = manager.get_headers()
endpoint = manager.get_endpoint('login')

# Or specify broker explicitly
base_url = manager.get_base_url('angel')
```

### Convenience Functions

```python
from utils.broker_config import (
    get_current_broker_config,
    get_current_broker_headers,
    get_current_broker_base_url,
    get_current_broker_endpoint
)

# These functions automatically detect the broker from calling context
config = get_current_broker_config()
headers = get_current_broker_headers()
base_url = get_current_broker_base_url()
endpoint = get_current_broker_endpoint('login')
```

## Key Features

### 1. Automatic Broker Detection
The system can automatically detect which broker is being used based on the calling file path:
- Looks for pattern: `broker/{broker_name}/api/*.py`
- Eliminates need to hardcode broker names in most cases

### 2. Configuration Caching
- Configurations are cached after first load
- Thread-safe caching mechanism
- Can be cleared/reloaded when needed

### 3. Fallback Mechanisms
- Graceful handling of missing configuration files
- Default configuration generation
- Error handling with logging

### 4. Dynamic Loading
- Configurations loaded on-demand
- No need to restart application for config changes
- Supports hot-reloading of configurations

## Migration from Hardcoded Values

### Before (Hardcoded)
```python
BASE_URL = "https://apiconnect.angelbroking.com"
HEADERS = {
    "Content-Type": "application/json",
    "Accept": "application/json"
}
```

### After (Configuration-based)
```python
from utils.broker_config import BrokerConfigManager

manager = BrokerConfigManager()
base_url = manager.get_base_url()
headers = manager.get_headers()
```

## Benefits

1. **Centralized Management**: All broker settings in one place
2. **Easy Maintenance**: Update configurations without code changes
3. **Dynamic Loading**: No application restarts needed
4. **Consistency**: Standardized configuration schema across brokers
5. **Extensibility**: Easy to add new brokers and features
6. **Type Safety**: Structured configuration with validation
7. **Performance**: Caching reduces repeated file operations

## File Structure

```
openalgo/
├── utils/
│   └── broker_config.py          # BrokerConfigManager
├── broker/
│   └── angel/
│       ├── config/
│       │   └── angel_config.json  # Angel broker configuration
│       ├── api/
│       │   ├── auth_api.py       # Updated to use config
│       │   ├── data.py           # Updated to use config
│       │   ├── funds.py          # Updated to use config
│       │   └── order_api.py      # Updated to use config
│       └── streaming/
│           └── angel_adapter.py  # Updated to use config
└── app.py                        # Initializes BrokerConfigManager
```

## Integration Status

### Completed
- ✅ BrokerConfigManager implementation
- ✅ Angel broker configuration file
- ✅ Angel API files updated
- ✅ WebSocket adapter updated
- ✅ Flask app integration
- ✅ Plugin loader enhancement

### Testing
- ✅ Configuration loading works
- ✅ Singleton pattern verified
- ✅ Thread-safe access confirmed
- ✅ Flask app integration tested
- ✅ Angel broker integration validated

## Future Enhancements

1. **Configuration Validation**: JSON schema validation
2. **Hot Reloading**: File system watchers for config changes
3. **Multi-Environment**: Support for dev/staging/prod configs
4. **Encryption**: Sensitive configuration encryption
5. **Web UI**: Configuration management interface
6. **Version Control**: Configuration versioning and rollback

## Error Handling

The system includes comprehensive error handling:

- Missing configuration files → Default configuration generated
- Invalid JSON → Error logged, fallback to defaults
- Network timeouts → Retry mechanism with exponential backoff
- Threading conflicts → Thread-safe locking mechanisms

## Performance Considerations

- Configurations cached in memory after first load
- Lazy loading - only loaded when needed
- Thread-safe singleton pattern
- Minimal file system operations

This system provides a robust, scalable foundation for managing broker configurations in OpenAlgo while maintaining backward compatibility and ease of use.
