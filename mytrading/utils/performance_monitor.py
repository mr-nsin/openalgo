"""
Performance Monitor
===================

Performance monitoring and profiling utilities for the MyTrading system.
"""

import time
import asyncio
import functools
import threading
from typing import Dict, List, Any, Optional, Callable
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from .logging_config import get_logger, log_performance

logger = get_logger(__name__)


@dataclass
class PerformanceMetric:
    """Performance metric data"""
    name: str
    component: str
    operation: str
    duration: float
    timestamp: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ComponentStats:
    """Statistics for a component"""
    total_calls: int = 0
    total_duration: float = 0.0
    min_duration: float = float('inf')
    max_duration: float = 0.0
    avg_duration: float = 0.0
    last_call_time: float = 0.0
    error_count: int = 0
    
    def update(self, duration: float, timestamp: float, is_error: bool = False):
        """Update statistics with new measurement"""
        self.total_calls += 1
        self.total_duration += duration
        self.min_duration = min(self.min_duration, duration)
        self.max_duration = max(self.max_duration, duration)
        self.avg_duration = self.total_duration / self.total_calls
        self.last_call_time = timestamp
        
        if is_error:
            self.error_count += 1


class PerformanceMonitor:
    """
    Performance monitoring system
    
    Tracks execution times, throughput, and system metrics
    """
    
    def __init__(self, max_history: int = 10000):
        self.max_history = max_history
        self.is_monitoring = False
        self.start_time = time.time()
        
        # Metrics storage
        self.metrics: deque = deque(maxlen=max_history)
        self.component_stats: Dict[str, Dict[str, ComponentStats]] = defaultdict(lambda: defaultdict(ComponentStats))
        
        # Real-time monitoring
        self.monitoring_task: Optional[asyncio.Task] = None
        self.monitoring_interval = 5.0  # seconds
        
        # Thread safety
        self.lock = threading.RLock()
        
        logger.info("📊 PerformanceMonitor initialized")
    
    def record_metric(self, component: str, operation: str, duration: float, 
                     metadata: Optional[Dict[str, Any]] = None, is_error: bool = False):
        """Record a performance metric"""
        timestamp = time.time()
        
        with self.lock:
            # Create metric
            metric = PerformanceMetric(
                name=f"{component}.{operation}",
                component=component,
                operation=operation,
                duration=duration,
                timestamp=timestamp,
                metadata=metadata or {}
            )
            
            # Store metric
            self.metrics.append(metric)
            
            # Update component stats
            stats = self.component_stats[component][operation]
            stats.update(duration, timestamp, is_error)
            
            # Log performance if enabled
            if duration > 1.0:  # Log slow operations (>1 second)
                logger.warning(f"Slow operation: {component}.{operation} took {duration*1000:.1f}ms")
            
            log_performance(component, operation, duration, **metadata or {})
    
    def get_component_stats(self, component: str) -> Dict[str, ComponentStats]:
        """Get statistics for a component"""
        with self.lock:
            return dict(self.component_stats[component])
    
    def get_operation_stats(self, component: str, operation: str) -> Optional[ComponentStats]:
        """Get statistics for a specific operation"""
        with self.lock:
            return self.component_stats[component].get(operation)
    
    def get_recent_metrics(self, component: Optional[str] = None, 
                          operation: Optional[str] = None, 
                          duration_seconds: int = 300) -> List[PerformanceMetric]:
        """Get recent metrics within specified duration"""
        cutoff_time = time.time() - duration_seconds
        
        with self.lock:
            recent_metrics = []
            for metric in reversed(self.metrics):
                if metric.timestamp < cutoff_time:
                    break
                
                if component and metric.component != component:
                    continue
                
                if operation and metric.operation != operation:
                    continue
                
                recent_metrics.append(metric)
            
            return list(reversed(recent_metrics))
    
    def get_slowest_operations(self, limit: int = 10) -> List[tuple]:
        """Get slowest operations"""
        with self.lock:
            slow_ops = []
            
            for component, operations in self.component_stats.items():
                for operation, stats in operations.items():
                    slow_ops.append((
                        f"{component}.{operation}",
                        stats.max_duration,
                        stats.avg_duration,
                        stats.total_calls
                    ))
            
            # Sort by max duration
            slow_ops.sort(key=lambda x: x[1], reverse=True)
            return slow_ops[:limit]
    
    def get_most_called_operations(self, limit: int = 10) -> List[tuple]:
        """Get most frequently called operations"""
        with self.lock:
            frequent_ops = []
            
            for component, operations in self.component_stats.items():
                for operation, stats in operations.items():
                    frequent_ops.append((
                        f"{component}.{operation}",
                        stats.total_calls,
                        stats.avg_duration,
                        stats.total_duration
                    ))
            
            # Sort by total calls
            frequent_ops.sort(key=lambda x: x[1], reverse=True)
            return frequent_ops[:limit]
    
    def get_error_rates(self) -> Dict[str, float]:
        """Get error rates by component"""
        with self.lock:
            error_rates = {}
            
            for component, operations in self.component_stats.items():
                total_calls = sum(stats.total_calls for stats in operations.values())
                total_errors = sum(stats.error_count for stats in operations.values())
                
                if total_calls > 0:
                    error_rates[component] = total_errors / total_calls
                else:
                    error_rates[component] = 0.0
            
            return error_rates
    
    def get_throughput_stats(self, duration_seconds: int = 300) -> Dict[str, float]:
        """Get throughput statistics (operations per second)"""
        recent_metrics = self.get_recent_metrics(duration_seconds=duration_seconds)
        
        if not recent_metrics:
            return {}
        
        # Count operations by component
        component_counts = defaultdict(int)
        for metric in recent_metrics:
            component_counts[metric.component] += 1
        
        # Calculate throughput
        throughput = {}
        for component, count in component_counts.items():
            throughput[component] = count / duration_seconds
        
        return throughput
    
    def get_summary_report(self) -> Dict[str, Any]:
        """Get comprehensive performance summary"""
        current_time = time.time()
        uptime = current_time - self.start_time
        
        with self.lock:
            total_operations = sum(
                sum(stats.total_calls for stats in operations.values())
                for operations in self.component_stats.values()
            )
            
            return {
                "uptime_seconds": uptime,
                "total_operations": total_operations,
                "operations_per_second": total_operations / uptime if uptime > 0 else 0,
                "components_monitored": len(self.component_stats),
                "metrics_stored": len(self.metrics),
                "slowest_operations": self.get_slowest_operations(5),
                "most_called_operations": self.get_most_called_operations(5),
                "error_rates": self.get_error_rates(),
                "throughput_5min": self.get_throughput_stats(300),
                "memory_usage": self._get_memory_usage()
            }
    
    def _get_memory_usage(self) -> Dict[str, Any]:
        """Get memory usage statistics"""
        try:
            import psutil
            process = psutil.Process()
            memory_info = process.memory_info()
            
            return {
                "rss_mb": memory_info.rss / 1024 / 1024,
                "vms_mb": memory_info.vms / 1024 / 1024,
                "percent": process.memory_percent()
            }
        except ImportError:
            return {"error": "psutil not available"}
    
    async def start_monitoring(self):
        """Start real-time monitoring"""
        if self.is_monitoring:
            return
        
        self.is_monitoring = True
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        logger.info("📊 Performance monitoring started")
    
    async def stop_monitoring(self):
        """Stop real-time monitoring"""
        self.is_monitoring = False
        
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
        
        logger.info("📊 Performance monitoring stopped")
    
    async def _monitoring_loop(self):
        """Real-time monitoring loop"""
        try:
            while self.is_monitoring:
                # Log summary statistics
                summary = self.get_summary_report()
                
                logger.debug(
                    f"📊 Performance: {summary['total_operations']} ops, "
                    f"{summary['operations_per_second']:.1f} ops/sec, "
                    f"{summary['components_monitored']} components"
                )
                
                # Check for performance issues
                await self._check_performance_issues()
                
                await asyncio.sleep(self.monitoring_interval)
                
        except asyncio.CancelledError:
            logger.debug("Performance monitoring loop cancelled")
    
    async def _check_performance_issues(self):
        """Check for performance issues and alert"""
        # Check for slow operations
        slowest = self.get_slowest_operations(3)
        for op_name, max_duration, avg_duration, calls in slowest:
            if max_duration > 5.0:  # 5 seconds
                logger.warning(f"⚠️  Slow operation detected: {op_name} max={max_duration*1000:.0f}ms")
        
        # Check error rates
        error_rates = self.get_error_rates()
        for component, error_rate in error_rates.items():
            if error_rate > 0.1:  # 10% error rate
                logger.warning(f"⚠️  High error rate in {component}: {error_rate:.1%}")
    
    def reset_stats(self):
        """Reset all statistics"""
        with self.lock:
            self.metrics.clear()
            self.component_stats.clear()
            self.start_time = time.time()
        
        logger.info("📊 Performance statistics reset")


# Global performance monitor instance
_performance_monitor = PerformanceMonitor()


def get_performance_monitor() -> PerformanceMonitor:
    """Get the global performance monitor instance"""
    return _performance_monitor


def performance_timer(component: str, operation: str, metadata: Optional[Dict[str, Any]] = None):
    """Decorator to time function execution"""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = time.time()
            is_error = False
            
            try:
                result = await func(*args, **kwargs)
                return result
            except Exception as e:
                is_error = True
                raise
            finally:
                duration = time.time() - start_time
                _performance_monitor.record_metric(
                    component, operation, duration, metadata, is_error
                )
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            start_time = time.time()
            is_error = False
            
            try:
                result = func(*args, **kwargs)
                return result
            except Exception as e:
                is_error = True
                raise
            finally:
                duration = time.time() - start_time
                _performance_monitor.record_metric(
                    component, operation, duration, metadata, is_error
                )
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


class PerformanceContext:
    """Context manager for timing operations"""
    
    def __init__(self, component: str, operation: str, metadata: Optional[Dict[str, Any]] = None):
        self.component = component
        self.operation = operation
        self.metadata = metadata or {}
        self.start_time = None
        self.is_error = False
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = time.time() - self.start_time
        self.is_error = exc_type is not None
        
        _performance_monitor.record_metric(
            self.component, self.operation, duration, self.metadata, self.is_error
        )


def performance_context(component: str, operation: str, metadata: Optional[Dict[str, Any]] = None):
    """Create a performance timing context"""
    return PerformanceContext(component, operation, metadata)


# Throughput monitoring
class ThroughputMonitor:
    """Monitor throughput for specific operations"""
    
    def __init__(self, window_size: int = 60):
        self.window_size = window_size
        self.timestamps: deque = deque(maxlen=window_size)
        self.lock = threading.Lock()
    
    def record_event(self):
        """Record an event occurrence"""
        with self.lock:
            self.timestamps.append(time.time())
    
    def get_rate(self) -> float:
        """Get current rate (events per second)"""
        with self.lock:
            if len(self.timestamps) < 2:
                return 0.0
            
            current_time = time.time()
            # Remove old timestamps
            while self.timestamps and current_time - self.timestamps[0] > self.window_size:
                self.timestamps.popleft()
            
            if len(self.timestamps) < 2:
                return 0.0
            
            time_span = self.timestamps[-1] - self.timestamps[0]
            return len(self.timestamps) / time_span if time_span > 0 else 0.0


# Memory profiling utilities
def profile_memory(func: Callable) -> Callable:
    """Decorator to profile memory usage"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            import tracemalloc
            import psutil
            
            # Start tracing
            tracemalloc.start()
            process = psutil.Process()
            memory_before = process.memory_info().rss
            
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                # Get memory usage
                memory_after = process.memory_info().rss
                current, peak = tracemalloc.get_traced_memory()
                tracemalloc.stop()
                
                memory_diff = memory_after - memory_before
                
                logger.debug(
                    f"Memory profile for {func.__name__}: "
                    f"RSS diff: {memory_diff/1024/1024:.1f}MB, "
                    f"Peak traced: {peak/1024/1024:.1f}MB"
                )
        
        except ImportError:
            # Fallback if profiling libraries not available
            return func(*args, **kwargs)
    
    return wrapper


# CPU profiling utilities
def profile_cpu(func: Callable) -> Callable:
    """Decorator to profile CPU usage"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            import cProfile
            import io
            import pstats
            
            profiler = cProfile.Profile()
            profiler.enable()
            
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                profiler.disable()
                
                # Get stats
                stats_stream = io.StringIO()
                stats = pstats.Stats(profiler, stream=stats_stream)
                stats.sort_stats('cumulative').print_stats(10)
                
                logger.debug(f"CPU profile for {func.__name__}:\n{stats_stream.getvalue()}")
        
        except ImportError:
            # Fallback if profiling not available
            return func(*args, **kwargs)
    
    return wrapper


# Export commonly used functions
__all__ = [
    'PerformanceMonitor',
    'PerformanceMetric',
    'ComponentStats',
    'get_performance_monitor',
    'performance_timer',
    'performance_context',
    'ThroughputMonitor',
    'profile_memory',
    'profile_cpu'
]