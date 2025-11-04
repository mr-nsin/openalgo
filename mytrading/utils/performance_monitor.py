"""
Performance Monitor
==================

Comprehensive performance monitoring and metrics collection for the
MyTrading system with real-time statistics and alerting.
"""

import time
import asyncio
import threading
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from collections import defaultdict, deque
from datetime import datetime, timedelta
import statistics
import psutil
import gc
from contextlib import contextmanager

from .logging_config import get_logger, log_performance_metrics

logger = get_logger(__name__)


@dataclass
class PerformanceMetric:
    """Individual performance metric"""
    name: str
    value: float
    unit: str
    timestamp: datetime
    component: str
    operation: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PerformanceStats:
    """Statistical summary of performance metrics"""
    count: int = 0
    total: float = 0.0
    min_value: float = float('inf')
    max_value: float = 0.0
    mean: float = 0.0
    median: float = 0.0
    std_dev: float = 0.0
    percentile_95: float = 0.0
    percentile_99: float = 0.0
    
    def update(self, values: List[float]):
        """Update statistics with new values"""
        if not values:
            return
        
        self.count = len(values)
        self.total = sum(values)
        self.min_value = min(values)
        self.max_value = max(values)
        self.mean = statistics.mean(values)
        
        if len(values) > 1:
            self.median = statistics.median(values)
            self.std_dev = statistics.stdev(values)
            
            # Calculate percentiles
            sorted_values = sorted(values)
            self.percentile_95 = sorted_values[int(0.95 * len(sorted_values))]
            self.percentile_99 = sorted_values[int(0.99 * len(sorted_values))]
        else:
            self.median = values[0]
            self.std_dev = 0.0
            self.percentile_95 = values[0]
            self.percentile_99 = values[0]


class Timer:
    """High-precision timer for performance measurement"""
    
    def __init__(self, name: str = "operation"):
        self.name = name
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.duration: Optional[float] = None
    
    def start(self):
        """Start the timer"""
        self.start_time = time.perf_counter()
        return self
    
    def stop(self):
        """Stop the timer and calculate duration"""
        if self.start_time is None:
            raise ValueError("Timer not started")
        
        self.end_time = time.perf_counter()
        self.duration = self.end_time - self.start_time
        return self.duration
    
    def __enter__(self):
        """Context manager entry"""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.stop()
    
    @property
    def elapsed(self) -> float:
        """Get elapsed time (even if timer is still running)"""
        if self.start_time is None:
            return 0.0
        
        end_time = self.end_time or time.perf_counter()
        return end_time - self.start_time
    
    @property
    def elapsed_ms(self) -> float:
        """Get elapsed time in milliseconds"""
        return self.elapsed * 1000


class SystemMonitor:
    """System resource monitoring"""
    
    def __init__(self):
        self.process = psutil.Process()
        self.start_time = time.time()
    
    def get_cpu_usage(self) -> float:
        """Get current CPU usage percentage"""
        return self.process.cpu_percent()
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get memory usage information"""
        memory_info = self.process.memory_info()
        memory_percent = self.process.memory_percent()
        
        return {
            "rss": memory_info.rss / 1024 / 1024,  # MB
            "vms": memory_info.vms / 1024 / 1024,  # MB
            "percent": memory_percent,
            "available": psutil.virtual_memory().available / 1024 / 1024  # MB
        }
    
    def get_disk_io(self) -> Dict[str, int]:
        """Get disk I/O statistics"""
        try:
            io_counters = self.process.io_counters()
            return {
                "read_bytes": io_counters.read_bytes,
                "write_bytes": io_counters.write_bytes,
                "read_count": io_counters.read_count,
                "write_count": io_counters.write_count
            }
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return {"read_bytes": 0, "write_bytes": 0, "read_count": 0, "write_count": 0}
    
    def get_network_io(self) -> Dict[str, int]:
        """Get network I/O statistics"""
        try:
            net_io = psutil.net_io_counters()
            return {
                "bytes_sent": net_io.bytes_sent,
                "bytes_recv": net_io.bytes_recv,
                "packets_sent": net_io.packets_sent,
                "packets_recv": net_io.packets_recv
            }
        except AttributeError:
            return {"bytes_sent": 0, "bytes_recv": 0, "packets_sent": 0, "packets_recv": 0}
    
    def get_thread_count(self) -> int:
        """Get current thread count"""
        return threading.active_count()
    
    def get_gc_stats(self) -> Dict[str, Any]:
        """Get garbage collection statistics"""
        gc_stats = gc.get_stats()
        return {
            "collections": sum(stat["collections"] for stat in gc_stats),
            "collected": sum(stat["collected"] for stat in gc_stats),
            "uncollectable": sum(stat["uncollectable"] for stat in gc_stats)
        }


class PerformanceMonitor:
    """Comprehensive performance monitoring system"""
    
    def __init__(self, max_history: int = 10000):
        self.max_history = max_history
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=max_history))
        self.stats_cache: Dict[str, PerformanceStats] = {}
        self.cache_expiry: Dict[str, datetime] = {}
        self.cache_duration = timedelta(seconds=30)
        
        # System monitoring
        self.system_monitor = SystemMonitor()
        
        # Performance thresholds and alerts
        self.thresholds: Dict[str, Dict[str, float]] = {}
        self.alert_callbacks: List[Callable] = []
        
        # Monitoring state
        self.monitoring_active = False
        self.monitoring_interval = 1.0  # seconds
        self.monitoring_task: Optional[asyncio.Task] = None
        
        # Thread safety
        self.lock = threading.RLock()
        
        logger.info("Performance monitor initialized")
    
    def record_metric(
        self,
        component: str,
        operation: str,
        value: float,
        unit: str = "ms",
        **metadata
    ):
        """Record a performance metric"""
        metric = PerformanceMetric(
            name=f"{component}.{operation}",
            value=value,
            unit=unit,
            timestamp=datetime.utcnow(),
            component=component,
            operation=operation,
            metadata=metadata
        )
        
        with self.lock:
            self.metrics[metric.name].append(metric)
            
            # Invalidate cache for this metric
            if metric.name in self.cache_expiry:
                del self.cache_expiry[metric.name]
        
        # Check thresholds
        self._check_thresholds(metric)
        
        # Log performance metric
        log_performance_metrics(
            component, operation, value / 1000 if unit == "ms" else value,
            **metadata
        )
    
    def record_duration(
        self,
        component: str,
        operation: str,
        duration: float,
        **metadata
    ):
        """Record a duration metric in seconds"""
        self.record_metric(
            component, operation, duration * 1000, "ms", **metadata
        )
    
    def get_stats(self, metric_name: str) -> Optional[PerformanceStats]:
        """Get statistical summary for a metric"""
        with self.lock:
            # Check cache
            if (metric_name in self.cache_expiry and 
                datetime.utcnow() < self.cache_expiry[metric_name]):
                return self.stats_cache.get(metric_name)
            
            # Calculate fresh stats
            if metric_name not in self.metrics:
                return None
            
            values = [m.value for m in self.metrics[metric_name]]
            if not values:
                return None
            
            stats = PerformanceStats()
            stats.update(values)
            
            # Cache results
            self.stats_cache[metric_name] = stats
            self.cache_expiry[metric_name] = datetime.utcnow() + self.cache_duration
            
            return stats
    
    def get_recent_metrics(
        self,
        metric_name: str,
        duration: timedelta = timedelta(minutes=5)
    ) -> List[PerformanceMetric]:
        """Get recent metrics within specified duration"""
        cutoff_time = datetime.utcnow() - duration
        
        with self.lock:
            if metric_name not in self.metrics:
                return []
            
            return [
                m for m in self.metrics[metric_name]
                if m.timestamp >= cutoff_time
            ]
    
    def get_all_metrics(self) -> Dict[str, List[PerformanceMetric]]:
        """Get all recorded metrics"""
        with self.lock:
            return {
                name: list(metrics)
                for name, metrics in self.metrics.items()
            }
    
    def get_system_metrics(self) -> Dict[str, Any]:
        """Get current system performance metrics"""
        return {
            "cpu_usage": self.system_monitor.get_cpu_usage(),
            "memory_usage": self.system_monitor.get_memory_usage(),
            "disk_io": self.system_monitor.get_disk_io(),
            "network_io": self.system_monitor.get_network_io(),
            "thread_count": self.system_monitor.get_thread_count(),
            "gc_stats": self.system_monitor.get_gc_stats(),
            "timestamp": datetime.utcnow().isoformat()
        }
    
    def set_threshold(
        self,
        metric_name: str,
        warning_threshold: float,
        critical_threshold: float
    ):
        """Set performance thresholds for alerting"""
        self.thresholds[metric_name] = {
            "warning": warning_threshold,
            "critical": critical_threshold
        }
    
    def add_alert_callback(self, callback: Callable[[str, PerformanceMetric], None]):
        """Add callback for performance alerts"""
        self.alert_callbacks.append(callback)
    
    def _check_thresholds(self, metric: PerformanceMetric):
        """Check if metric exceeds thresholds"""
        if metric.name not in self.thresholds:
            return
        
        thresholds = self.thresholds[metric.name]
        
        if metric.value >= thresholds["critical"]:
            self._trigger_alert("critical", metric)
        elif metric.value >= thresholds["warning"]:
            self._trigger_alert("warning", metric)
    
    def _trigger_alert(self, level: str, metric: PerformanceMetric):
        """Trigger performance alert"""
        alert_message = (
            f"Performance {level}: {metric.name} = {metric.value}{metric.unit} "
            f"at {metric.timestamp}"
        )
        
        if level == "critical":
            logger.critical(alert_message, **metric.metadata)
        else:
            logger.warning(alert_message, **metric.metadata)
        
        # Call alert callbacks
        for callback in self.alert_callbacks:
            try:
                callback(level, metric)
            except Exception as e:
                logger.error(f"Error in alert callback: {e}")
    
    async def start_monitoring(self, interval: float = 1.0):
        """Start continuous system monitoring"""
        self.monitoring_interval = interval
        self.monitoring_active = True
        
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        logger.info(f"Started performance monitoring (interval: {interval}s)")
    
    async def stop_monitoring(self):
        """Stop continuous monitoring"""
        self.monitoring_active = False
        
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Stopped performance monitoring")
    
    async def _monitoring_loop(self):
        """Continuous monitoring loop"""
        try:
            while self.monitoring_active:
                # Record system metrics
                system_metrics = self.get_system_metrics()
                
                self.record_metric("system", "cpu_usage", system_metrics["cpu_usage"], "%")
                self.record_metric("system", "memory_usage", system_metrics["memory_usage"]["percent"], "%")
                self.record_metric("system", "thread_count", system_metrics["thread_count"], "count")
                
                await asyncio.sleep(self.monitoring_interval)
                
        except asyncio.CancelledError:
            logger.debug("Monitoring loop cancelled")
        except Exception as e:
            logger.error(f"Error in monitoring loop: {e}")
    
    def clear_metrics(self, metric_name: Optional[str] = None):
        """Clear metrics (all or specific metric)"""
        with self.lock:
            if metric_name:
                if metric_name in self.metrics:
                    self.metrics[metric_name].clear()
                if metric_name in self.stats_cache:
                    del self.stats_cache[metric_name]
                if metric_name in self.cache_expiry:
                    del self.cache_expiry[metric_name]
            else:
                self.metrics.clear()
                self.stats_cache.clear()
                self.cache_expiry.clear()
    
    def get_summary_report(self) -> Dict[str, Any]:
        """Get comprehensive performance summary"""
        with self.lock:
            report = {
                "timestamp": datetime.utcnow().isoformat(),
                "system_metrics": self.get_system_metrics(),
                "performance_stats": {},
                "total_metrics": sum(len(metrics) for metrics in self.metrics.values()),
                "active_metrics": len(self.metrics)
            }
            
            # Add stats for each metric
            for metric_name in self.metrics.keys():
                stats = self.get_stats(metric_name)
                if stats:
                    report["performance_stats"][metric_name] = {
                        "count": stats.count,
                        "mean": round(stats.mean, 2),
                        "min": round(stats.min_value, 2),
                        "max": round(stats.max_value, 2),
                        "p95": round(stats.percentile_95, 2),
                        "p99": round(stats.percentile_99, 2)
                    }
            
            return report


# Global performance monitor instance
_global_monitor: Optional[PerformanceMonitor] = None


def get_performance_monitor() -> PerformanceMonitor:
    """Get global performance monitor instance"""
    global _global_monitor
    if _global_monitor is None:
        _global_monitor = PerformanceMonitor()
    return _global_monitor


@contextmanager
def measure_performance(component: str, operation: str, **metadata):
    """Context manager for measuring performance"""
    monitor = get_performance_monitor()
    timer = Timer(f"{component}.{operation}")
    
    try:
        timer.start()
        yield timer
    finally:
        duration = timer.stop()
        monitor.record_duration(component, operation, duration, **metadata)


def performance_timer(component: str, operation: str = None):
    """Decorator for automatic performance measurement"""
    def decorator(func):
        op_name = operation or func.__name__
        
        if asyncio.iscoroutinefunction(func):
            async def async_wrapper(*args, **kwargs):
                with measure_performance(component, op_name):
                    return await func(*args, **kwargs)
            return async_wrapper
        else:
            def sync_wrapper(*args, **kwargs):
                with measure_performance(component, op_name):
                    return func(*args, **kwargs)
            return sync_wrapper
    
    return decorator
