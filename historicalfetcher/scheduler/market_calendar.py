"""
Market Calendar for Indian Stock Exchanges

Handles market holidays, trading days, and market hours for NSE, BSE, and other exchanges.
"""

import sys
import os
from datetime import date, time, datetime, timedelta
from typing import List, Dict, Optional, Set
import calendar

# Add the OpenAlgo root directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from historicalfetcher.utils.async_logger import get_async_logger

_async_logger = get_async_logger()
logger = _async_logger.get_logger()

# Logger is imported from loguru above

class MarketCalendar:
    """Market calendar for Indian stock exchanges"""
    
    def __init__(self):
        """Initialize market calendar with holidays and timings"""
        
        # Market holidays for 2024-2025 (Indian stock exchanges)
        # This should be updated annually or fetched from an API
        self.holidays = {
            # 2024 holidays
            date(2024, 1, 26): "Republic Day",
            date(2024, 3, 8): "Holi",
            date(2024, 3, 29): "Good Friday",
            date(2024, 4, 11): "Id-Ul-Fitr (Ramzan Id)",
            date(2024, 4, 17): "Ram Navami",
            date(2024, 5, 1): "Maharashtra Day",
            date(2024, 6, 17): "Bakri Id",
            date(2024, 8, 15): "Independence Day",
            date(2024, 8, 26): "Janmashtami",
            date(2024, 10, 2): "Gandhi Jayanti",
            date(2024, 10, 31): "Diwali-Laxmi Pujan",
            date(2024, 11, 1): "Diwali-Balipratipada",
            date(2024, 11, 15): "Guru Nanak Jayanti",
            
            # 2025 holidays (preliminary - should be updated)
            date(2025, 1, 26): "Republic Day",
            date(2025, 3, 14): "Holi",
            date(2025, 4, 18): "Good Friday",
            date(2025, 4, 30): "Id-Ul-Fitr (Ramzan Id)",
            date(2025, 5, 1): "Maharashtra Day",
            date(2025, 8, 15): "Independence Day",
            date(2025, 10, 2): "Gandhi Jayanti",
        }
        
        # Market timings for different exchanges
        self.market_timings = {
            'NSE': {
                'pre_open_start': time(9, 0),
                'pre_open_end': time(9, 15),
                'normal_start': time(9, 15),
                'normal_end': time(15, 30),
                'post_close_start': time(15, 40),
                'post_close_end': time(16, 0)
            },
            'BSE': {
                'pre_open_start': time(9, 0),
                'pre_open_end': time(9, 15),
                'normal_start': time(9, 15),
                'normal_end': time(15, 30),
                'post_close_start': time(15, 40),
                'post_close_end': time(16, 0)
            },
            'NFO': {  # NSE F&O
                'normal_start': time(9, 15),
                'normal_end': time(15, 30)
            },
            'BFO': {  # BSE F&O
                'normal_start': time(9, 15),
                'normal_end': time(15, 30)
            },
            'MCX': {  # Multi Commodity Exchange
                'normal_start': time(9, 0),
                'normal_end': time(23, 30)
            },
            'CDS': {  # Currency Derivatives
                'normal_start': time(9, 0),
                'normal_end': time(17, 0)
            }
        }
        
        # Default market timings
        self.default_timings = {
            'normal_start': time(9, 15),
            'normal_end': time(15, 30)
        }
    
    def is_trading_day(self, check_date: date) -> bool:
        """
        Check if given date is a trading day
        
        Args:
            check_date: Date to check
            
        Returns:
            True if it's a trading day, False otherwise
        """
        
        # Check if it's a weekend (Saturday = 5, Sunday = 6)
        if check_date.weekday() >= 5:
            return False
        
        # Check if it's a declared holiday
        if check_date in self.holidays:
            logger.debug(f"{check_date} is a holiday: {self.holidays[check_date]}")
            return False
        
        return True
    
    def is_market_open(self, exchange: str = 'NSE', check_time: Optional[datetime] = None) -> bool:
        """
        Check if market is currently open for given exchange
        
        Args:
            exchange: Exchange code (NSE, BSE, etc.)
            check_time: Time to check (default: current time)
            
        Returns:
            True if market is open, False otherwise
        """
        
        if check_time is None:
            check_time = datetime.now()
        
        check_date = check_time.date()
        current_time = check_time.time()
        
        # Check if it's a trading day
        if not self.is_trading_day(check_date):
            return False
        
        # Get market timings for exchange
        timings = self.market_timings.get(exchange, self.default_timings)
        
        # Check if current time is within normal trading hours
        start_time = timings.get('normal_start', time(9, 15))
        end_time = timings.get('normal_end', time(15, 30))
        
        return start_time <= current_time <= end_time
    
    def is_pre_market_open(self, exchange: str = 'NSE', check_time: Optional[datetime] = None) -> bool:
        """Check if pre-market session is open"""
        
        if check_time is None:
            check_time = datetime.now()
        
        check_date = check_time.date()
        current_time = check_time.time()
        
        if not self.is_trading_day(check_date):
            return False
        
        timings = self.market_timings.get(exchange, {})
        
        if 'pre_open_start' not in timings or 'pre_open_end' not in timings:
            return False
        
        return timings['pre_open_start'] <= current_time <= timings['pre_open_end']
    
    def is_post_market_open(self, exchange: str = 'NSE', check_time: Optional[datetime] = None) -> bool:
        """Check if post-market session is open"""
        
        if check_time is None:
            check_time = datetime.now()
        
        check_date = check_time.date()
        current_time = check_time.time()
        
        if not self.is_trading_day(check_date):
            return False
        
        timings = self.market_timings.get(exchange, {})
        
        if 'post_close_start' not in timings or 'post_close_end' not in timings:
            return False
        
        return timings['post_close_start'] <= current_time <= timings['post_close_end']
    
    def get_next_trading_day(self, from_date: Optional[date] = None) -> date:
        """
        Get next trading day from given date
        
        Args:
            from_date: Starting date (default: today)
            
        Returns:
            Next trading day
        """
        
        if from_date is None:
            from_date = date.today()
        
        next_date = from_date + timedelta(days=1)
        
        # Keep incrementing until we find a trading day
        while not self.is_trading_day(next_date):
            next_date += timedelta(days=1)
            
            # Safety check to avoid infinite loop
            if (next_date - from_date).days > 30:
                logger.warning(f"Could not find trading day within 30 days of {from_date}")
                break
        
        return next_date
    
    def get_previous_trading_day(self, from_date: Optional[date] = None) -> date:
        """
        Get previous trading day from given date
        
        Args:
            from_date: Starting date (default: today)
            
        Returns:
            Previous trading day
        """
        
        if from_date is None:
            from_date = date.today()
        
        prev_date = from_date - timedelta(days=1)
        
        # Keep decrementing until we find a trading day
        while not self.is_trading_day(prev_date):
            prev_date -= timedelta(days=1)
            
            # Safety check to avoid infinite loop
            if (from_date - prev_date).days > 30:
                logger.warning(f"Could not find trading day within 30 days before {from_date}")
                break
        
        return prev_date
    
    def get_trading_days_in_month(self, year: int, month: int) -> List[date]:
        """Get all trading days in a given month"""
        
        trading_days = []
        
        # Get all days in the month
        _, last_day = calendar.monthrange(year, month)
        
        for day in range(1, last_day + 1):
            check_date = date(year, month, day)
            if self.is_trading_day(check_date):
                trading_days.append(check_date)
        
        return trading_days
    
    def get_trading_days_in_range(self, start_date: date, end_date: date) -> List[date]:
        """Get all trading days in a date range"""
        
        trading_days = []
        current_date = start_date
        
        while current_date <= end_date:
            if self.is_trading_day(current_date):
                trading_days.append(current_date)
            current_date += timedelta(days=1)
        
        return trading_days
    
    def get_market_status(self, exchange: str = 'NSE', check_time: Optional[datetime] = None) -> Dict[str, any]:
        """
        Get comprehensive market status
        
        Args:
            exchange: Exchange code
            check_time: Time to check (default: current time)
            
        Returns:
            Dictionary with market status information
        """
        
        if check_time is None:
            check_time = datetime.now()
        
        check_date = check_time.date()
        
        status = {
            'date': check_date,
            'exchange': exchange,
            'is_trading_day': self.is_trading_day(check_date),
            'is_market_open': False,
            'is_pre_market_open': False,
            'is_post_market_open': False,
            'market_phase': 'closed',
            'next_trading_day': None,
            'previous_trading_day': None,
            'holiday_reason': None
        }
        
        if not status['is_trading_day']:
            if check_date in self.holidays:
                status['holiday_reason'] = self.holidays[check_date]
            elif check_date.weekday() >= 5:
                status['holiday_reason'] = 'Weekend'
        else:
            status['is_market_open'] = self.is_market_open(exchange, check_time)
            status['is_pre_market_open'] = self.is_pre_market_open(exchange, check_time)
            status['is_post_market_open'] = self.is_post_market_open(exchange, check_time)
            
            # Determine market phase
            if status['is_pre_market_open']:
                status['market_phase'] = 'pre_market'
            elif status['is_market_open']:
                status['market_phase'] = 'normal_trading'
            elif status['is_post_market_open']:
                status['market_phase'] = 'post_market'
            else:
                status['market_phase'] = 'closed'
        
        # Get next and previous trading days
        status['next_trading_day'] = self.get_next_trading_day(check_date)
        status['previous_trading_day'] = self.get_previous_trading_day(check_date)
        
        return status
    
    def get_market_timings(self, exchange: str = 'NSE') -> Dict[str, time]:
        """Get market timings for an exchange"""
        return self.market_timings.get(exchange, self.default_timings).copy()
    
    def add_holiday(self, holiday_date: date, reason: str):
        """Add a new holiday to the calendar"""
        self.holidays[holiday_date] = reason
        logger.info(f"Added holiday: {holiday_date} - {reason}")
    
    def remove_holiday(self, holiday_date: date):
        """Remove a holiday from the calendar"""
        if holiday_date in self.holidays:
            reason = self.holidays.pop(holiday_date)
            logger.info(f"Removed holiday: {holiday_date} - {reason}")
    
    def get_holidays_in_range(self, start_date: date, end_date: date) -> Dict[date, str]:
        """Get all holidays in a date range"""
        
        holidays_in_range = {}
        
        for holiday_date, reason in self.holidays.items():
            if start_date <= holiday_date <= end_date:
                holidays_in_range[holiday_date] = reason
        
        return holidays_in_range
    
    def is_settlement_day(self, check_date: date) -> bool:
        """
        Check if given date is a settlement day (T+2 from a trading day)
        
        Args:
            check_date: Date to check
            
        Returns:
            True if it's a settlement day
        """
        
        # Find the trading day that would settle on check_date (T+2)
        trade_date = check_date - timedelta(days=2)
        
        # Adjust for weekends and holidays
        while not self.is_trading_day(trade_date):
            trade_date -= timedelta(days=1)
        
        # Check if the settlement date (T+2) falls on check_date
        settlement_date = trade_date + timedelta(days=2)
        
        # Adjust settlement date for weekends and holidays
        while not self.is_trading_day(settlement_date):
            settlement_date += timedelta(days=1)
        
        return settlement_date == check_date
    
    def get_optimal_fetch_time(self, exchange: str = 'NSE') -> time:
        """
        Get optimal time for fetching historical data (after market close)
        
        Args:
            exchange: Exchange code
            
        Returns:
            Optimal fetch time
        """
        
        timings = self.market_timings.get(exchange, self.default_timings)
        market_close = timings.get('normal_end', time(15, 30))
        
        # Add buffer time after market close
        close_datetime = datetime.combine(date.today(), market_close)
        optimal_datetime = close_datetime + timedelta(hours=3)  # 3 hours after close
        
        return optimal_datetime.time()
    
    def should_run_historical_fetch(self, check_time: Optional[datetime] = None) -> bool:
        """
        Determine if historical data fetch should run at given time
        
        Args:
            check_time: Time to check (default: current time)
            
        Returns:
            True if fetch should run
        """
        
        if check_time is None:
            check_time = datetime.now()
        
        check_date = check_time.date()
        current_time = check_time.time()
        
        # Don't run on non-trading days
        if not self.is_trading_day(check_date):
            return False
        
        # Don't run during market hours
        if self.is_market_open('NSE', check_time):
            return False
        
        # Run after market close (after 6:30 PM)
        optimal_time = time(18, 30)  # 6:30 PM
        
        return current_time >= optimal_time
