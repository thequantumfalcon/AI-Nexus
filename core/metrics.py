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

# ============================================================================
# Enterprise LLM Metrics (Week 5 Addition)
# ============================================================================

# HTTP Request Metrics
HTTP_REQUESTS_TOTAL = Counter(
    'ainexus_http_requests_total',
    'Total HTTP requests',
    ['method', 'endpoint', 'status']
)

HTTP_REQUEST_DURATION = Histogram(
    'ainexus_http_request_duration_seconds',
    'HTTP request latency',
    ['method', 'endpoint'],
    buckets=[0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0]
)

HTTP_REQUESTS_IN_PROGRESS = Gauge(
    'ainexus_http_requests_in_progress',
    'HTTP requests currently in progress',
    ['method', 'endpoint']
)

# LLM Inference Metrics
LLM_INFERENCE_TOTAL = Counter(
    'ainexus_llm_inference_total',
    'Total LLM inference requests',
    ['model', 'operation', 'status']
)

LLM_INFERENCE_DURATION = Histogram(
    'ainexus_llm_inference_duration_seconds',
    'LLM inference latency',
    ['model', 'operation'],
    buckets=[0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0]
)

LLM_TOKENS_PER_SECOND = Summary(
    'ainexus_llm_tokens_per_second',
    'LLM token generation rate',
    ['model']
)

LLM_INPUT_TOKENS_TOTAL = Counter(
    'ainexus_llm_input_tokens_total',
    'Total LLM input tokens',
    ['model', 'user']
)

LLM_OUTPUT_TOKENS_TOTAL = Counter(
    'ainexus_llm_output_tokens_total',
    'Total LLM output tokens',
    ['model', 'user']
)

# GPU Metrics
GPU_UTILIZATION_PERCENT = Gauge(
    'ainexus_gpu_utilization_percent',
    'GPU utilization percentage',
    ['gpu_id']
)

GPU_MEMORY_USED_BYTES = Gauge(
    'ainexus_gpu_memory_used_bytes',
    'GPU memory used in bytes',
    ['gpu_id']
)

GPU_TEMPERATURE_CELSIUS = Gauge(
    'ainexus_gpu_temperature_celsius',
    'GPU temperature in Celsius',
    ['gpu_id']
)

# Vector Database Metrics
VECTOR_SEARCH_TOTAL = Counter(
    'ainexus_vector_search_total',
    'Total vector searches',
    ['collection', 'status']
)

VECTOR_SEARCH_DURATION = Histogram(
    'ainexus_vector_search_duration_seconds',
    'Vector search latency',
    ['collection'],
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0]
)

# RAG Metrics
RAG_QUERY_TOTAL = Counter(
    'ainexus_rag_query_total',
    'Total RAG queries',
    ['status']
)

RAG_QUERY_DURATION = Histogram(
    'ainexus_rag_query_duration_seconds',
    'RAG query end-to-end latency',
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0]
)

# Cost Metrics
COST_TOTAL_DOLLARS = Counter(
    'ainexus_cost_total_dollars',
    'Total cost in dollars',
    ['model', 'user']
)

# Error Metrics
ERRORS_TOTAL = Counter(
    'ainexus_errors_total',
    'Total errors',
    ['error_type', 'endpoint']
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
    
    # ============================================================================
    # Enterprise LLM Metrics Methods (Week 5 Addition)
    # ============================================================================
    
    def record_http_request(self, method: str, endpoint: str, status: str, duration: float):
        """Record HTTP request metrics"""
        HTTP_REQUESTS_TOTAL.labels(method=method, endpoint=endpoint, status=status).inc()
        HTTP_REQUEST_DURATION.labels(method=method, endpoint=endpoint).observe(duration)
    
    def record_llm_inference(
        self,
        model: str,
        operation: str,
        status: str,
        duration: float,
        input_tokens: int = 0,
        output_tokens: int = 0,
        user: str = "unknown"
    ):
        """Record LLM inference metrics"""
        LLM_INFERENCE_TOTAL.labels(model=model, operation=operation, status=status).inc()
        LLM_INFERENCE_DURATION.labels(model=model, operation=operation).observe(duration)
        
        if input_tokens > 0:
            LLM_INPUT_TOKENS_TOTAL.labels(model=model, user=user).inc(input_tokens)
        
        if output_tokens > 0:
            LLM_OUTPUT_TOKENS_TOTAL.labels(model=model, user=user).inc(output_tokens)
            if duration > 0:
                tokens_per_sec = output_tokens / duration
                LLM_TOKENS_PER_SECOND.labels(model=model).observe(tokens_per_sec)
    
    def update_gpu_metrics(self, gpu_id: str, utilization: float, memory_used: int, temperature: float):
        """Update GPU metrics"""
        GPU_UTILIZATION_PERCENT.labels(gpu_id=gpu_id).set(utilization)
        GPU_MEMORY_USED_BYTES.labels(gpu_id=gpu_id).set(memory_used)
        GPU_TEMPERATURE_CELSIUS.labels(gpu_id=gpu_id).set(temperature)
    
    def record_vector_search(self, collection: str, status: str, duration: float):
        """Record vector search metrics"""
        VECTOR_SEARCH_TOTAL.labels(collection=collection, status=status).inc()
        VECTOR_SEARCH_DURATION.labels(collection=collection).observe(duration)
    
    def record_rag_query(self, status: str, duration: float):
        """Record RAG query metrics"""
        RAG_QUERY_TOTAL.labels(status=status).inc()
        RAG_QUERY_DURATION.observe(duration)
    
    def record_cost(self, model: str, user: str, cost: float):
        """Record cost metrics"""
        COST_TOTAL_DOLLARS.labels(model=model, user=user).inc(cost)
    
    def record_error(self, error_type: str, endpoint: str):
        """Record error metrics"""
        ERRORS_TOTAL.labels(error_type=error_type, endpoint=endpoint).inc()


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
