"""
Rate Limiter for API requests

Implements token bucket algorithm for smooth rate limiting with burst support.
"""

import asyncio
import time
from typing import Optional
from dataclasses import dataclass

@dataclass
class RateLimitStats:
    """Statistics for rate limiter"""
    total_requests: int = 0
    requests_allowed: int = 0
    requests_throttled: int = 0
    total_wait_time: float = 0.0
    average_wait_time: float = 0.0

class RateLimiter:
    """
    Token bucket rate limiter for async operations
    
    Allows burst requests up to bucket capacity while maintaining
    average rate over time.
    """
    
    def __init__(
        self,
        requests_per_second: float,
        burst_capacity: Optional[int] = None,
        initial_tokens: Optional[int] = None
    ):
        """
        Initialize rate limiter
        
        Args:
            requests_per_second: Average requests per second allowed
            burst_capacity: Maximum burst requests (default: 2x rate)
            initial_tokens: Initial tokens in bucket (default: full capacity)
        """
        self.rate = requests_per_second
        self.burst_capacity = burst_capacity or max(int(requests_per_second * 2), 1)
        
        # Token bucket state
        self.tokens = initial_tokens if initial_tokens is not None else self.burst_capacity
        self.last_refill = time.monotonic()
        
        # Synchronization
        self._lock = asyncio.Lock()
        
        # Statistics
        self.stats = RateLimitStats()
    
    async def __aenter__(self):
        """Async context manager entry"""
        await self.acquire()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        pass
    
    async def acquire(self, tokens: int = 1) -> float:
        """
        Acquire tokens from the bucket
        
        Args:
            tokens: Number of tokens to acquire
            
        Returns:
            Time waited in seconds
        """
        async with self._lock:
            self.stats.total_requests += 1
            
            # Refill tokens based on elapsed time
            now = time.monotonic()
            elapsed = now - self.last_refill
            
            # Add tokens based on rate and elapsed time
            tokens_to_add = elapsed * self.rate
            self.tokens = min(self.burst_capacity, self.tokens + tokens_to_add)
            self.last_refill = now
            
            # Check if we have enough tokens
            if self.tokens >= tokens:
                # We have enough tokens, consume them
                self.tokens -= tokens
                self.stats.requests_allowed += 1
                return 0.0
            else:
                # Not enough tokens, calculate wait time
                tokens_needed = tokens - self.tokens
                wait_time = tokens_needed / self.rate
                
                # Wait for tokens to be available
                await asyncio.sleep(wait_time)
                
                # Update state after waiting
                self.tokens = 0  # All tokens consumed
                self.last_refill = time.monotonic()
                
                # Update statistics
                self.stats.requests_throttled += 1
                self.stats.total_wait_time += wait_time
                self.stats.average_wait_time = (
                    self.stats.total_wait_time / self.stats.requests_throttled
                )
                
                return wait_time
    
    def get_current_tokens(self) -> float:
        """Get current number of tokens in bucket"""
        now = time.monotonic()
        elapsed = now - self.last_refill
        tokens_to_add = elapsed * self.rate
        return min(self.burst_capacity, self.tokens + tokens_to_add)
    
    def get_stats(self) -> RateLimitStats:
        """Get rate limiter statistics"""
        return self.stats
    
    def reset_stats(self):
        """Reset statistics"""
        self.stats = RateLimitStats()

class AdaptiveRateLimiter(RateLimiter):
    """
    Adaptive rate limiter that adjusts rate based on success/failure patterns
    """
    
    def __init__(
        self,
        initial_rate: float,
        min_rate: float = 0.1,
        max_rate: float = 10.0,
        adaptation_factor: float = 0.1
    ):
        """
        Initialize adaptive rate limiter
        
        Args:
            initial_rate: Starting requests per second
            min_rate: Minimum allowed rate
            max_rate: Maximum allowed rate
            adaptation_factor: How quickly to adapt (0.0 to 1.0)
        """
        super().__init__(initial_rate)
        
        self.min_rate = min_rate
        self.max_rate = max_rate
        self.adaptation_factor = adaptation_factor
        
        # Adaptation state
        self.success_count = 0
        self.failure_count = 0
        self.last_adaptation = time.monotonic()
        self.adaptation_window = 60.0  # Adapt every 60 seconds
    
    async def report_success(self):
        """Report successful request"""
        self.success_count += 1
        await self._maybe_adapt()
    
    async def report_failure(self):
        """Report failed request (e.g., rate limit error)"""
        self.failure_count += 1
        await self._maybe_adapt()
    
    async def _maybe_adapt(self):
        """Adapt rate based on success/failure ratio"""
        now = time.monotonic()
        
        if now - self.last_adaptation < self.adaptation_window:
            return
        
        total_requests = self.success_count + self.failure_count
        
        if total_requests == 0:
            return
        
        success_rate = self.success_count / total_requests
        
        # Adjust rate based on success rate
        if success_rate > 0.95:
            # High success rate, increase rate
            new_rate = self.rate * (1 + self.adaptation_factor)
        elif success_rate < 0.8:
            # Low success rate, decrease rate
            new_rate = self.rate * (1 - self.adaptation_factor)
        else:
            # Acceptable success rate, no change
            new_rate = self.rate
        
        # Apply bounds
        new_rate = max(self.min_rate, min(self.max_rate, new_rate))
        
        if abs(new_rate - self.rate) > 0.01:  # Only update if significant change
            async with self._lock:
                self.rate = new_rate
                # Reset counters
                self.success_count = 0
                self.failure_count = 0
                self.last_adaptation = now

class MultiLevelRateLimiter:
    """
    Multi-level rate limiter for different types of operations
    """
    
    def __init__(self, limits: dict):
        """
        Initialize multi-level rate limiter
        
        Args:
            limits: Dict mapping operation types to rate limits
                   e.g., {'api': 3.0, 'db': 10.0, 'notification': 1.0}
        """
        self.limiters = {
            operation: RateLimiter(rate) 
            for operation, rate in limits.items()
        }
    
    async def acquire(self, operation: str, tokens: int = 1) -> float:
        """Acquire tokens for specific operation type"""
        if operation not in self.limiters:
            raise ValueError(f"Unknown operation type: {operation}")
        
        return await self.limiters[operation].acquire(tokens)
    
    def get_limiter(self, operation: str) -> RateLimiter:
        """Get rate limiter for specific operation"""
        if operation not in self.limiters:
            raise ValueError(f"Unknown operation type: {operation}")
        
        return self.limiters[operation]
    
    def get_all_stats(self) -> dict:
        """Get statistics for all rate limiters"""
        return {
            operation: limiter.get_stats()
            for operation, limiter in self.limiters.items()
        }
