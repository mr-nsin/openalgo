"""
Performance Monitor

Tracks system performance metrics including memory usage, CPU usage,
and processing statistics.
"""

import asyncio
import psutil
import time
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from datetime import datetime, timedelta

@dataclass
class PerformanceMetrics:
    """Performance metrics snapshot"""
    timestamp: datetime
    memory_usage_mb: float
    memory_percent: float
    cpu_percent: float
    disk_usage_percent: float
    network_bytes_sent: int
    network_bytes_recv: int
    
    # Process-specific metrics
    process_memory_mb: float
    process_cpu_percent: float
    process_threads: int
    
    # Custom application metrics
    custom_metrics: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ProcessingStats:
    """Statistics for processing operations"""
    total_items: int = 0
    processed_items: int = 0
    successful_items: int = 0
    failed_items: int = 0
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate percentage"""
        if self.processed_items == 0:
            return 0.0
        return (self.successful_items / self.processed_items) * 100
    
    @property
    def processing_time(self) -> Optional[timedelta]:
        """Calculate total processing time"""
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return None
    
    @property
    def items_per_second(self) -> float:
        """Calculate processing rate"""
        if not self.processing_time:
            return 0.0
        
        total_seconds = self.processing_time.total_seconds()
        if total_seconds == 0:
            return 0.0
        
        return self.processed_items / total_seconds

class PerformanceMonitor:
    """
    Monitors system and application performance metrics
    """
    
    def __init__(
        self,
        collection_interval: float = 60.0,
        history_size: int = 100,
        enable_auto_collection: bool = True
    ):
        """
        Initialize performance monitor
        
        Args:
            collection_interval: Seconds between metric collections
            history_size: Number of historical metrics to keep
            enable_auto_collection: Whether to automatically collect metrics
        """
        self.collection_interval = collection_interval
        self.history_size = history_size
        self.enable_auto_collection = enable_auto_collection
        
        # Metrics storage
        self.metrics_history: List[PerformanceMetrics] = []
        self.processing_stats = ProcessingStats()
        
        # System info
        self.process = psutil.Process()
        self.system_boot_time = datetime.fromtimestamp(psutil.boot_time())
        
        # Collection task
        self._collection_task: Optional[asyncio.Task] = None
        self._stop_collection = False
        
        # Custom metrics
        self._custom_counters: Dict[str, int] = {}
        self._custom_gauges: Dict[str, float] = {}
        self._custom_timers: Dict[str, List[float]] = {}
    
    async def start_monitoring(self):
        """Start automatic metric collection"""
        if self.enable_auto_collection and not self._collection_task:
            self._stop_collection = False
            self._collection_task = asyncio.create_task(self._collection_loop())
    
    async def stop_monitoring(self):
        """Stop automatic metric collection"""
        self._stop_collection = True
        if self._collection_task:
            self._collection_task.cancel()
            try:
                await self._collection_task
            except asyncio.CancelledError:
                pass
            self._collection_task = None
    
    async def _collection_loop(self):
        """Main collection loop"""
        while not self._stop_collection:
            try:
                await self.collect_metrics()
                await asyncio.sleep(self.collection_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                # Log error but continue collection
                print(f"Error collecting metrics: {e}")
                await asyncio.sleep(self.collection_interval)
    
    async def collect_metrics(self) -> PerformanceMetrics:
        """Collect current performance metrics"""
        
        # System metrics
        memory = psutil.virtual_memory()
        cpu_percent = psutil.cpu_percent(interval=None)
        disk = psutil.disk_usage('/')
        network = psutil.net_io_counters()
        
        # Process metrics
        process_memory = self.process.memory_info()
        process_cpu = self.process.cpu_percent()
        
        # Create metrics snapshot
        metrics = PerformanceMetrics(
            timestamp=datetime.now(),
            memory_usage_mb=memory.used / (1024 * 1024),
            memory_percent=memory.percent,
            cpu_percent=cpu_percent,
            disk_usage_percent=disk.percent,
            network_bytes_sent=network.bytes_sent,
            network_bytes_recv=network.bytes_recv,
            process_memory_mb=process_memory.rss / (1024 * 1024),
            process_cpu_percent=process_cpu,
            process_threads=self.process.num_threads(),
            custom_metrics={
                'counters': self._custom_counters.copy(),
                'gauges': self._custom_gauges.copy(),
                'timers': {k: sum(v) / len(v) if v else 0 for k, v in self._custom_timers.items()}
            }
        )
        
        # Add to history
        self.metrics_history.append(metrics)
        
        # Maintain history size
        if len(self.metrics_history) > self.history_size:
            self.metrics_history = self.metrics_history[-self.history_size:]
        
        return metrics
    
    def start_processing(self, total_items: int):
        """Start tracking processing statistics"""
        self.processing_stats = ProcessingStats(
            total_items=total_items,
            start_time=datetime.now()
        )
    
    def update_processing_progress(
        self,
        processed: int,
        successful: int,
        failed: int
    ):
        """Update processing progress"""
        self.processing_stats.processed_items = processed
        self.processing_stats.successful_items = successful
        self.processing_stats.failed_items = failed
    
    def finish_processing(self):
        """Mark processing as finished"""
        self.processing_stats.end_time = datetime.now()
    
    def increment_counter(self, name: str, value: int = 1):
        """Increment a custom counter"""
        self._custom_counters[name] = self._custom_counters.get(name, 0) + value
    
    def set_gauge(self, name: str, value: float):
        """Set a custom gauge value"""
        self._custom_gauges[name] = value
    
    def record_timer(self, name: str, duration: float):
        """Record a timing measurement"""
        if name not in self._custom_timers:
            self._custom_timers[name] = []
        
        self._custom_timers[name].append(duration)
        
        # Keep only recent measurements
        if len(self._custom_timers[name]) > 100:
            self._custom_timers[name] = self._custom_timers[name][-100:]
    
    def get_current_metrics(self) -> Optional[PerformanceMetrics]:
        """Get the most recent metrics"""
        return self.metrics_history[-1] if self.metrics_history else None
    
    def get_metrics_summary(self, minutes: int = 10) -> Dict[str, Any]:
        """Get summary of metrics over specified time period"""
        
        if not self.metrics_history:
            return {}
        
        # Filter metrics within time window
        cutoff_time = datetime.now() - timedelta(minutes=minutes)
        recent_metrics = [
            m for m in self.metrics_history 
            if m.timestamp >= cutoff_time
        ]
        
        if not recent_metrics:
            return {}
        
        # Calculate averages
        avg_memory = sum(m.memory_usage_mb for m in recent_metrics) / len(recent_metrics)
        avg_cpu = sum(m.cpu_percent for m in recent_metrics) / len(recent_metrics)
        avg_process_memory = sum(m.process_memory_mb for m in recent_metrics) / len(recent_metrics)
        avg_process_cpu = sum(m.process_cpu_percent for m in recent_metrics) / len(recent_metrics)
        
        # Get min/max values
        max_memory = max(m.memory_usage_mb for m in recent_metrics)
        max_cpu = max(m.cpu_percent for m in recent_metrics)
        
        return {
            'time_window_minutes': minutes,
            'sample_count': len(recent_metrics),
            'avg_memory_mb': round(avg_memory, 2),
            'max_memory_mb': round(max_memory, 2),
            'avg_cpu_percent': round(avg_cpu, 2),
            'max_cpu_percent': round(max_cpu, 2),
            'avg_process_memory_mb': round(avg_process_memory, 2),
            'avg_process_cpu_percent': round(avg_process_cpu, 2),
            'processing_stats': {
                'total_items': self.processing_stats.total_items,
                'processed_items': self.processing_stats.processed_items,
                'success_rate': round(self.processing_stats.success_rate, 2),
                'items_per_second': round(self.processing_stats.items_per_second, 2)
            }
        }
    
    def check_resource_limits(
        self,
        max_memory_mb: Optional[float] = None,
        max_cpu_percent: Optional[float] = None
    ) -> Dict[str, Any]:
        """Check if resource usage exceeds limits"""
        
        current = self.get_current_metrics()
        if not current:
            return {'status': 'no_data'}
        
        warnings = []
        
        if max_memory_mb and current.process_memory_mb > max_memory_mb:
            warnings.append(f"Memory usage ({current.process_memory_mb:.1f}MB) exceeds limit ({max_memory_mb}MB)")
        
        if max_cpu_percent and current.process_cpu_percent > max_cpu_percent:
            warnings.append(f"CPU usage ({current.process_cpu_percent:.1f}%) exceeds limit ({max_cpu_percent}%)")
        
        return {
            'status': 'warning' if warnings else 'ok',
            'warnings': warnings,
            'current_memory_mb': current.process_memory_mb,
            'current_cpu_percent': current.process_cpu_percent
        }
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get system information"""
        
        return {
            'cpu_count': psutil.cpu_count(),
            'cpu_count_logical': psutil.cpu_count(logical=True),
            'memory_total_gb': psutil.virtual_memory().total / (1024**3),
            'disk_total_gb': psutil.disk_usage('/').total / (1024**3),
            'system_boot_time': self.system_boot_time.isoformat(),
            'python_process_id': self.process.pid,
            'python_process_started': datetime.fromtimestamp(self.process.create_time()).isoformat()
        }

# Context manager for timing operations
class Timer:
    """Context manager for timing operations"""
    
    def __init__(self, monitor: PerformanceMonitor, timer_name: str):
        self.monitor = monitor
        self.timer_name = timer_name
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.monotonic()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time:
            duration = time.monotonic() - self.start_time
            self.monitor.record_timer(self.timer_name, duration)

# Async context manager for timing async operations
class AsyncTimer:
    """Async context manager for timing operations"""
    
    def __init__(self, monitor: PerformanceMonitor, timer_name: str):
        self.monitor = monitor
        self.timer_name = timer_name
        self.start_time = None
    
    async def __aenter__(self):
        self.start_time = time.monotonic()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.start_time:
            duration = time.monotonic() - self.start_time
            self.monitor.record_timer(self.timer_name, duration)
