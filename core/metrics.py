"""
Metrics and monitoring for AI-Nexus
====================================

Provides Prometheus metrics and performance monitoring.
"""

from prometheus_client import Counter, Gauge, Histogram, Summary, start_http_server
from typing import Dict, Optional
import time
import psutil
from functools import wraps
from core.logger import get_logger

logger = get_logger(__name__)


# Define metrics
TASKS_PROCESSED = Counter(
    'ainexus_tasks_processed_total',
    'Total number of tasks processed',
    ['task_type', 'status']
)

TASKS_IN_FLIGHT = Gauge(
    'ainexus_tasks_in_flight',
    'Number of tasks currently being processed',
    ['task_type']
)

TASK_DURATION = Histogram(
    'ainexus_task_duration_seconds',
    'Task processing duration',
    ['task_type'],
    buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 30.0, 60.0]
)

NETWORK_LATENCY = Histogram(
    'ainexus_network_latency_seconds',
    'Network communication latency',
    ['operation'],
    buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]
)

NODE_HEALTH = Gauge(
    'ainexus_node_health',
    'Node health status (0=down, 1=up)',
    ['node_id']
)

CPU_USAGE = Gauge(
    'ainexus_cpu_usage_percent',
    'CPU usage percentage'
)

MEMORY_USAGE = Gauge(
    'ainexus_memory_usage_bytes',
    'Memory usage in bytes'
)

PRIVACY_BUDGET = Gauge(
    'ainexus_privacy_budget_remaining',
    'Remaining differential privacy budget'
)

MODEL_ACCURACY = Gauge(
    'ainexus_model_accuracy',
    'Model accuracy metric',
    ['model_id']
)

BLOCKCHAIN_HEIGHT = Gauge(
    'ainexus_blockchain_height',
    'Current blockchain height'
)

TOKENS_DISTRIBUTED = Counter(
    'ainexus_tokens_distributed_total',
    'Total tokens distributed as rewards'
)


class MetricsCollector:
    """Centralized metrics collection"""
    
    def __init__(self, port: int = 9090):
        self.port = port
        self.started = False
        
    def start(self):
        """Start Prometheus metrics server"""
        if not self.started:
            start_http_server(self.port)
            self.started = True
            logger.info(f"Metrics server started on port {self.port}")
    
    def record_task(self, task_type: str, duration: float, status: str = "success"):
        """Record task execution metrics"""
        TASKS_PROCESSED.labels(task_type=task_type, status=status).inc()
        TASK_DURATION.labels(task_type=task_type).observe(duration)
    
    def record_network_latency(self, operation: str, latency: float):
        """Record network operation latency"""
        NETWORK_LATENCY.labels(operation=operation).observe(latency)
    
    def update_node_health(self, node_id: int, is_healthy: bool):
        """Update node health status"""
        NODE_HEALTH.labels(node_id=str(node_id)).set(1 if is_healthy else 0)
    
    def update_system_metrics(self):
        """Update system resource metrics"""
        CPU_USAGE.set(psutil.cpu_percent())
        MEMORY_USAGE.set(psutil.virtual_memory().used)
    
    def update_privacy_budget(self, remaining: float):
        """Update privacy budget"""
        PRIVACY_BUDGET.set(remaining)
    
    def update_model_accuracy(self, model_id: str, accuracy: float):
        """Update model accuracy"""
        MODEL_ACCURACY.labels(model_id=model_id).set(accuracy)
    
    def update_blockchain_height(self, height: int):
        """Update blockchain height"""
        BLOCKCHAIN_HEIGHT.set(height)
    
    def record_token_distribution(self, amount: float):
        """Record token distribution"""
        TOKENS_DISTRIBUTED.inc(amount)


def measure_time(task_type: str = "unknown"):
    """Decorator to measure function execution time"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            TASKS_IN_FLIGHT.labels(task_type=task_type).inc()
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                TASKS_PROCESSED.labels(task_type=task_type, status="success").inc()
                TASK_DURATION.labels(task_type=task_type).observe(duration)
                return result
            except Exception as e:
                duration = time.time() - start_time
                TASKS_PROCESSED.labels(task_type=task_type, status="error").inc()
                TASK_DURATION.labels(task_type=task_type).observe(duration)
                raise
            finally:
                TASKS_IN_FLIGHT.labels(task_type=task_type).dec()
        return wrapper
    return decorator


def track_latency(operation: str = "unknown"):
    """Decorator to track network operation latency"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                latency = time.time() - start_time
                NETWORK_LATENCY.labels(operation=operation).observe(latency)
                return result
            except Exception:
                latency = time.time() - start_time
                NETWORK_LATENCY.labels(operation=operation).observe(latency)
                raise
        return wrapper
    return decorator


# Global metrics collector
_metrics_collector: Optional[MetricsCollector] = None


def get_metrics_collector(port: int = 9090) -> MetricsCollector:
    """Get global metrics collector instance"""
    global _metrics_collector
    if _metrics_collector is None:
        _metrics_collector = MetricsCollector(port)
    return _metrics_collector
