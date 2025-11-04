"""
Helper Utilities
================

Common utility functions and helpers for the MyTrading system.
"""

import re
import json
import hashlib
from typing import Any, Dict, List, Optional, Union, Tuple
from datetime import datetime, timedelta, timezone
from decimal import Decimal, ROUND_HALF_UP
import asyncio
from pathlib import Path


def format_currency(
    amount: Union[float, Decimal, int],
    currency: str = "₹",
    decimal_places: int = 2
) -> str:
    """
    Format amount as currency string
    
    Args:
        amount: Amount to format
        currency: Currency symbol
        decimal_places: Number of decimal places
        
    Returns:
        Formatted currency string
    """
    if amount is None:
        return f"{currency}0.00"
    
    # Convert to Decimal for precise formatting
    if not isinstance(amount, Decimal):
        amount = Decimal(str(amount))
    
    # Round to specified decimal places
    quantizer = Decimal('0.' + '0' * decimal_places)
    rounded_amount = amount.quantize(quantizer, rounding=ROUND_HALF_UP)
    
    # Format with commas for thousands separator
    formatted = f"{rounded_amount:,.{decimal_places}f}"
    
    return f"{currency}{formatted}"


def format_percentage(
    value: Union[float, Decimal, int],
    decimal_places: int = 2,
    include_sign: bool = True
) -> str:
    """
    Format value as percentage string
    
    Args:
        value: Value to format (0.05 = 5%)
        decimal_places: Number of decimal places
        include_sign: Include + sign for positive values
        
    Returns:
        Formatted percentage string
    """
    if value is None:
        return "0.00%"
    
    # Convert to percentage
    percentage = float(value) * 100
    
    # Format with specified decimal places
    formatted = f"{percentage:.{decimal_places}f}%"
    
    # Add + sign for positive values if requested
    if include_sign and percentage > 0:
        formatted = f"+{formatted}"
    
    return formatted


def format_number(
    value: Union[float, int],
    decimal_places: int = 2,
    use_thousands_separator: bool = True
) -> str:
    """
    Format number with proper decimal places and separators
    
    Args:
        value: Number to format
        decimal_places: Number of decimal places
        use_thousands_separator: Use comma as thousands separator
        
    Returns:
        Formatted number string
    """
    if value is None:
        return "0"
    
    if use_thousands_separator:
        return f"{value:,.{decimal_places}f}"
    else:
        return f"{value:.{decimal_places}f}"


def format_volume(volume: int) -> str:
    """
    Format volume with appropriate suffixes (K, M, B)
    
    Args:
        volume: Volume to format
        
    Returns:
        Formatted volume string
    """
    if volume is None or volume == 0:
        return "0"
    
    if volume >= 1_000_000_000:
        return f"{volume / 1_000_000_000:.1f}B"
    elif volume >= 1_000_000:
        return f"{volume / 1_000_000:.1f}M"
    elif volume >= 1_000:
        return f"{volume / 1_000:.1f}K"
    else:
        return str(volume)


def validate_symbol(symbol: str) -> bool:
    """
    Validate trading symbol format
    
    Args:
        symbol: Symbol to validate
        
    Returns:
        True if valid symbol format
    """
    if not symbol or not isinstance(symbol, str):
        return False
    
    # Basic symbol validation (alphanumeric + some special chars)
    pattern = r'^[A-Z0-9_\-&\.]+$'
    return bool(re.match(pattern, symbol.upper()))


def validate_exchange_symbol(exchange_symbol: str) -> bool:
    """
    Validate exchange:symbol format
    
    Args:
        exchange_symbol: Exchange:symbol string to validate
        
    Returns:
        True if valid format
    """
    if not exchange_symbol or ':' not in exchange_symbol:
        return False
    
    parts = exchange_symbol.split(':')
    if len(parts) != 2:
        return False
    
    exchange, symbol = parts
    return validate_symbol(exchange) and validate_symbol(symbol)


def parse_exchange_symbol(exchange_symbol: str) -> Tuple[str, str]:
    """
    Parse exchange:symbol string into components
    
    Args:
        exchange_symbol: Exchange:symbol string
        
    Returns:
        Tuple of (exchange, symbol)
        
    Raises:
        ValueError: If format is invalid
    """
    if not validate_exchange_symbol(exchange_symbol):
        raise ValueError(f"Invalid exchange:symbol format: {exchange_symbol}")
    
    exchange, symbol = exchange_symbol.split(':', 1)
    return exchange.upper(), symbol.upper()


def calculate_percentage_change(old_value: float, new_value: float) -> float:
    """
    Calculate percentage change between two values
    
    Args:
        old_value: Original value
        new_value: New value
        
    Returns:
        Percentage change (0.05 = 5% increase)
    """
    if old_value == 0:
        return 0.0 if new_value == 0 else float('inf')
    
    return (new_value - old_value) / old_value


def calculate_compound_return(returns: List[float]) -> float:
    """
    Calculate compound return from list of individual returns
    
    Args:
        returns: List of individual returns (0.05 = 5%)
        
    Returns:
        Compound return
    """
    if not returns:
        return 0.0
    
    compound = 1.0
    for ret in returns:
        compound *= (1.0 + ret)
    
    return compound - 1.0


def calculate_sharpe_ratio(
    returns: List[float],
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252
) -> float:
    """
    Calculate Sharpe ratio from returns
    
    Args:
        returns: List of returns
        risk_free_rate: Risk-free rate (annual)
        periods_per_year: Number of periods per year (252 for daily)
        
    Returns:
        Sharpe ratio
    """
    if not returns or len(returns) < 2:
        return 0.0
    
    import statistics
    
    # Calculate excess returns
    excess_returns = [r - (risk_free_rate / periods_per_year) for r in returns]
    
    # Calculate mean and standard deviation
    mean_excess_return = statistics.mean(excess_returns)
    std_excess_return = statistics.stdev(excess_returns)
    
    if std_excess_return == 0:
        return 0.0
    
    # Annualize Sharpe ratio
    return (mean_excess_return / std_excess_return) * (periods_per_year ** 0.5)


def calculate_max_drawdown(values: List[float]) -> Tuple[float, int, int]:
    """
    Calculate maximum drawdown from a series of values
    
    Args:
        values: List of portfolio values
        
    Returns:
        Tuple of (max_drawdown, start_index, end_index)
    """
    if not values or len(values) < 2:
        return 0.0, 0, 0
    
    peak = values[0]
    peak_index = 0
    max_drawdown = 0.0
    max_dd_start = 0
    max_dd_end = 0
    
    for i, value in enumerate(values):
        if value > peak:
            peak = value
            peak_index = i
        
        drawdown = (peak - value) / peak if peak > 0 else 0.0
        
        if drawdown > max_drawdown:
            max_drawdown = drawdown
            max_dd_start = peak_index
            max_dd_end = i
    
    return max_drawdown, max_dd_start, max_dd_end


def round_to_tick_size(price: float, tick_size: float) -> float:
    """
    Round price to nearest tick size
    
    Args:
        price: Price to round
        tick_size: Minimum tick size
        
    Returns:
        Rounded price
    """
    if tick_size <= 0:
        return price
    
    return round(price / tick_size) * tick_size


def calculate_position_size(
    account_value: float,
    risk_percent: float,
    entry_price: float,
    stop_loss_price: float
) -> int:
    """
    Calculate position size based on risk management
    
    Args:
        account_value: Total account value
        risk_percent: Risk percentage (0.02 = 2%)
        entry_price: Entry price
        stop_loss_price: Stop loss price
        
    Returns:
        Position size (number of shares/contracts)
    """
    if entry_price <= 0 or stop_loss_price <= 0:
        return 0
    
    risk_amount = account_value * risk_percent
    price_risk = abs(entry_price - stop_loss_price)
    
    if price_risk <= 0:
        return 0
    
    position_size = risk_amount / price_risk
    return int(position_size)


def generate_order_id(prefix: str = "ORD") -> str:
    """
    Generate unique order ID
    
    Args:
        prefix: Prefix for order ID
        
    Returns:
        Unique order ID
    """
    timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S%f")
    return f"{prefix}_{timestamp}"


def generate_hash(data: Union[str, Dict, List]) -> str:
    """
    Generate SHA-256 hash of data
    
    Args:
        data: Data to hash
        
    Returns:
        Hexadecimal hash string
    """
    if isinstance(data, (dict, list)):
        data_str = json.dumps(data, sort_keys=True)
    else:
        data_str = str(data)
    
    return hashlib.sha256(data_str.encode()).hexdigest()


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """
    Safely divide two numbers, returning default if denominator is zero
    
    Args:
        numerator: Numerator
        denominator: Denominator
        default: Default value if division by zero
        
    Returns:
        Division result or default
    """
    if denominator == 0:
        return default
    return numerator / denominator


def clamp(value: float, min_value: float, max_value: float) -> float:
    """
    Clamp value between min and max
    
    Args:
        value: Value to clamp
        min_value: Minimum value
        max_value: Maximum value
        
    Returns:
        Clamped value
    """
    return max(min_value, min(value, max_value))


def is_market_hours(
    current_time: Optional[datetime] = None,
    market_open: str = "09:15",
    market_close: str = "15:30",
    timezone_name: str = "Asia/Kolkata"
) -> bool:
    """
    Check if current time is within market hours
    
    Args:
        current_time: Time to check (defaults to now)
        market_open: Market open time (HH:MM format)
        market_close: Market close time (HH:MM format)
        timezone_name: Timezone name
        
    Returns:
        True if within market hours
    """
    if current_time is None:
        current_time = datetime.now()
    
    # Parse market hours
    open_hour, open_minute = map(int, market_open.split(':'))
    close_hour, close_minute = map(int, market_close.split(':'))
    
    # Create time objects for comparison
    market_open_time = current_time.replace(
        hour=open_hour, minute=open_minute, second=0, microsecond=0
    )
    market_close_time = current_time.replace(
        hour=close_hour, minute=close_minute, second=0, microsecond=0
    )
    
    # Check if current time is within market hours
    return market_open_time <= current_time <= market_close_time


def is_trading_day(date: Optional[datetime] = None) -> bool:
    """
    Check if given date is a trading day (Monday-Friday)
    
    Args:
        date: Date to check (defaults to today)
        
    Returns:
        True if trading day
    """
    if date is None:
        date = datetime.now()
    
    # Monday = 0, Sunday = 6
    return date.weekday() < 5


def get_next_trading_day(date: Optional[datetime] = None) -> datetime:
    """
    Get next trading day from given date
    
    Args:
        date: Starting date (defaults to today)
        
    Returns:
        Next trading day
    """
    if date is None:
        date = datetime.now()
    
    next_day = date + timedelta(days=1)
    
    # Skip weekends
    while next_day.weekday() >= 5:  # Saturday = 5, Sunday = 6
        next_day += timedelta(days=1)
    
    return next_day


def retry_async(max_retries: int = 3, delay: float = 1.0, backoff: float = 2.0):
    """
    Decorator for retrying async functions
    
    Args:
        max_retries: Maximum number of retries
        delay: Initial delay between retries
        backoff: Backoff multiplier for delay
    """
    def decorator(func):
        async def wrapper(*args, **kwargs):
            current_delay = delay
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    
                    if attempt == max_retries:
                        break
                    
                    await asyncio.sleep(current_delay)
                    current_delay *= backoff
            
            # Re-raise the last exception
            raise last_exception
        
        return wrapper
    return decorator


def load_json_file(file_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load JSON data from file
    
    Args:
        file_path: Path to JSON file
        
    Returns:
        Loaded JSON data
        
    Raises:
        FileNotFoundError: If file doesn't exist
        json.JSONDecodeError: If file contains invalid JSON
    """
    path = Path(file_path)
    
    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {file_path}")
    
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_json_file(data: Dict[str, Any], file_path: Union[str, Path], indent: int = 2):
    """
    Save data to JSON file
    
    Args:
        data: Data to save
        file_path: Path to save file
        indent: JSON indentation
    """
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=indent, default=str)


def deep_merge_dicts(dict1: Dict, dict2: Dict) -> Dict:
    """
    Deep merge two dictionaries
    
    Args:
        dict1: First dictionary
        dict2: Second dictionary (takes precedence)
        
    Returns:
        Merged dictionary
    """
    result = dict1.copy()
    
    for key, value in dict2.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge_dicts(result[key], value)
        else:
            result[key] = value
    
    return result


def flatten_dict(d: Dict, parent_key: str = '', sep: str = '.') -> Dict:
    """
    Flatten nested dictionary
    
    Args:
        d: Dictionary to flatten
        parent_key: Parent key prefix
        sep: Separator for nested keys
        
    Returns:
        Flattened dictionary
    """
    items = []
    
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    
    return dict(items)


def get_memory_usage() -> Dict[str, float]:
    """
    Get current memory usage information
    
    Returns:
        Dictionary with memory usage stats in MB
    """
    import psutil
    import os
    
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    
    return {
        "rss": memory_info.rss / 1024 / 1024,  # Resident Set Size
        "vms": memory_info.vms / 1024 / 1024,  # Virtual Memory Size
        "percent": process.memory_percent(),    # Percentage of total memory
        "available": psutil.virtual_memory().available / 1024 / 1024
    }
