"""
Retry Handler with exponential backoff

Provides robust retry logic for handling transient failures in API calls
and database operations.
"""

import asyncio
import random
import time
from typing import Any, Callable, Optional, Type, Union, List
from dataclasses import dataclass
from enum import Enum

class RetryStrategy(Enum):
    """Retry strategy types"""
    EXPONENTIAL_BACKOFF = "exponential_backoff"
    LINEAR_BACKOFF = "linear_backoff"
    FIXED_DELAY = "fixed_delay"
    FIBONACCI_BACKOFF = "fibonacci_backoff"

@dataclass
class RetryStats:
    """Statistics for retry operations"""
    total_attempts: int = 0
    successful_attempts: int = 0
    failed_attempts: int = 0
    total_retry_time: float = 0.0
    max_retries_reached: int = 0

class RetryHandler:
    """
    Handles retry logic with various backoff strategies
    """
    
    def __init__(
        self,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        backoff_factor: float = 2.0,
        jitter: bool = True,
        strategy: RetryStrategy = RetryStrategy.EXPONENTIAL_BACKOFF,
        retryable_exceptions: Optional[List[Type[Exception]]] = None
    ):
        """
        Initialize retry handler
        
        Args:
            max_retries: Maximum number of retry attempts
            base_delay: Base delay between retries in seconds
            max_delay: Maximum delay between retries in seconds
            backoff_factor: Multiplier for exponential backoff
            jitter: Add random jitter to delays
            strategy: Retry strategy to use
            retryable_exceptions: List of exceptions that should trigger retries
        """
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.backoff_factor = backoff_factor
        self.jitter = jitter
        self.strategy = strategy
        
        # Default retryable exceptions
        self.retryable_exceptions = retryable_exceptions or [
            ConnectionError,
            TimeoutError,
            OSError,
            asyncio.TimeoutError
        ]
        
        # Statistics
        self.stats = RetryStats()
    
    async def execute(
        self,
        func: Callable,
        *args,
        **kwargs
    ) -> Any:
        """
        Execute function with retry logic
        
        Args:
            func: Function to execute
            *args: Positional arguments for function
            **kwargs: Keyword arguments for function
            
        Returns:
            Result of successful function execution
            
        Raises:
            Last exception if all retries failed
        """
        last_exception = None
        start_time = time.monotonic()
        
        for attempt in range(self.max_retries + 1):
            self.stats.total_attempts += 1
            
            try:
                # Execute the function
                if asyncio.iscoroutinefunction(func):
                    result = await func(*args, **kwargs)
                else:
                    result = func(*args, **kwargs)
                
                # Success!
                self.stats.successful_attempts += 1
                self.stats.total_retry_time += time.monotonic() - start_time
                
                return result
                
            except Exception as e:
                last_exception = e
                self.stats.failed_attempts += 1
                
                # Check if this exception is retryable
                if not self._is_retryable_exception(e):
                    raise e
                
                # Check if we've exhausted retries
                if attempt >= self.max_retries:
                    self.stats.max_retries_reached += 1
                    break
                
                # Calculate delay for next attempt
                delay = self._calculate_delay(attempt)
                
                # Wait before retry
                await asyncio.sleep(delay)
        
        # All retries failed
        self.stats.total_retry_time += time.monotonic() - start_time
        raise last_exception
    
    def _is_retryable_exception(self, exception: Exception) -> bool:
        """Check if exception should trigger a retry"""
        return any(
            isinstance(exception, exc_type) 
            for exc_type in self.retryable_exceptions
        )
    
    def _calculate_delay(self, attempt: int) -> float:
        """Calculate delay for given attempt number"""
        if self.strategy == RetryStrategy.EXPONENTIAL_BACKOFF:
            delay = self.base_delay * (self.backoff_factor ** attempt)
        elif self.strategy == RetryStrategy.LINEAR_BACKOFF:
            delay = self.base_delay * (attempt + 1)
        elif self.strategy == RetryStrategy.FIXED_DELAY:
            delay = self.base_delay
        elif self.strategy == RetryStrategy.FIBONACCI_BACKOFF:
            delay = self.base_delay * self._fibonacci(attempt + 1)
        else:
            delay = self.base_delay
        
        # Apply maximum delay limit
        delay = min(delay, self.max_delay)
        
        # Add jitter if enabled
        if self.jitter:
            jitter_amount = delay * 0.1  # 10% jitter
            delay += random.uniform(-jitter_amount, jitter_amount)
            delay = max(0, delay)  # Ensure non-negative
        
        return delay
    
    def _fibonacci(self, n: int) -> int:
        """Calculate nth Fibonacci number"""
        if n <= 1:
            return n
        
        a, b = 0, 1
        for _ in range(2, n + 1):
            a, b = b, a + b
        
        return b
    
    def get_stats(self) -> RetryStats:
        """Get retry statistics"""
        return self.stats
    
    def reset_stats(self):
        """Reset statistics"""
        self.stats = RetryStats()

class ConditionalRetryHandler(RetryHandler):
    """
    Retry handler with conditional retry logic based on exception details
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.retry_conditions = []
    
    def add_retry_condition(
        self,
        exception_type: Type[Exception],
        condition_func: Callable[[Exception], bool]
    ):
        """
        Add conditional retry logic
        
        Args:
            exception_type: Type of exception to check
            condition_func: Function that returns True if retry should happen
        """
        self.retry_conditions.append((exception_type, condition_func))
    
    def _is_retryable_exception(self, exception: Exception) -> bool:
        """Check if exception should trigger a retry with conditions"""
        # First check base retryable exceptions
        if super()._is_retryable_exception(exception):
            return True
        
        # Check conditional retry logic
        for exc_type, condition_func in self.retry_conditions:
            if isinstance(exception, exc_type):
                try:
                    return condition_func(exception)
                except Exception:
                    # If condition check fails, don't retry
                    return False
        
        return False

class CircuitBreakerRetryHandler(RetryHandler):
    """
    Retry handler with circuit breaker pattern
    
    Stops retrying if failure rate is too high within a time window.
    """
    
    def __init__(
        self,
        *args,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        
        # Circuit breaker state
        self.failure_count = 0
        self.last_failure_time = 0
        self.circuit_open = False
    
    async def execute(self, func: Callable, *args, **kwargs) -> Any:
        """Execute with circuit breaker logic"""
        
        # Check if circuit is open
        if self.circuit_open:
            if time.monotonic() - self.last_failure_time > self.recovery_timeout:
                # Try to close circuit
                self.circuit_open = False
                self.failure_count = 0
            else:
                # Circuit is still open
                raise Exception("Circuit breaker is open - too many recent failures")
        
        try:
            result = await super().execute(func, *args, **kwargs)
            
            # Success - reset failure count
            self.failure_count = 0
            return result
            
        except Exception as e:
            # Failure - increment counter
            self.failure_count += 1
            self.last_failure_time = time.monotonic()
            
            # Check if we should open circuit
            if self.failure_count >= self.failure_threshold:
                self.circuit_open = True
            
            raise e

# Decorator for easy retry functionality
def retry(
    max_retries: int = 3,
    base_delay: float = 1.0,
    backoff_factor: float = 2.0,
    retryable_exceptions: Optional[List[Type[Exception]]] = None
):
    """
    Decorator for adding retry logic to functions
    
    Usage:
        @retry(max_retries=3, base_delay=1.0)
        async def my_function():
            # Function that might fail
            pass
    """
    def decorator(func):
        retry_handler = RetryHandler(
            max_retries=max_retries,
            base_delay=base_delay,
            backoff_factor=backoff_factor,
            retryable_exceptions=retryable_exceptions
        )
        
        async def wrapper(*args, **kwargs):
            return await retry_handler.execute(func, *args, **kwargs)
        
        return wrapper
    
    return decorator

# Utility functions for common retry scenarios
def create_api_retry_handler() -> RetryHandler:
    """Create retry handler optimized for API calls"""
    return RetryHandler(
        max_retries=3,
        base_delay=1.0,
        max_delay=30.0,
        backoff_factor=2.0,
        jitter=True,
        strategy=RetryStrategy.EXPONENTIAL_BACKOFF,
        retryable_exceptions=[
            ConnectionError,
            TimeoutError,
            OSError,
            asyncio.TimeoutError,
            # Add API-specific exceptions here
        ]
    )

def create_database_retry_handler() -> RetryHandler:
    """Create retry handler optimized for database operations"""
    return RetryHandler(
        max_retries=5,
        base_delay=0.5,
        max_delay=10.0,
        backoff_factor=1.5,
        jitter=True,
        strategy=RetryStrategy.EXPONENTIAL_BACKOFF,
        retryable_exceptions=[
            ConnectionError,
            TimeoutError,
            OSError,
            # Add database-specific exceptions here
        ]
    )
