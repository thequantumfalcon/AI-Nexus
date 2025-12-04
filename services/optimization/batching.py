"""
Batch request optimization for LLM inference.

Features:
- Dynamic batching with timeout
- Request grouping by model
- Automatic batch size optimization
- Concurrent request handling
- Priority queuing
"""

import asyncio
import time
from typing import List, Optional, Any, Callable, TypeVar, Generic
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
import threading


# ============================================================================
# Configuration
# ============================================================================

class Priority(int, Enum):
    """Request priority levels."""
    LOW = 0
    NORMAL = 1
    HIGH = 2
    CRITICAL = 3


@dataclass
class BatchConfig:
    """Batch processing configuration."""
    # Batch size
    max_batch_size: int = 32
    min_batch_size: int = 1
    
    # Timeout
    max_wait_ms: int = 100  # Max wait for batch to fill
    
    # Concurrency
    max_concurrent_batches: int = 4
    
    # Auto-tuning
    enable_auto_tuning: bool = True
    target_latency_ms: int = 1000


# ============================================================================
# Request/Response
# ============================================================================

T = TypeVar('T')
R = TypeVar('R')


@dataclass
class BatchRequest(Generic[T]):
    """Batch request."""
    id: str
    input: T
    priority: Priority = Priority.NORMAL
    timestamp: float = field(default_factory=time.time)
    future: asyncio.Future = field(default_factory=asyncio.Future)


@dataclass
class BatchResult(Generic[R]):
    """Batch result."""
    request_id: str
    output: Optional[R] = None
    error: Optional[Exception] = None
    latency_ms: float = 0


# ============================================================================
# Batch Processor
# ============================================================================

class BatchProcessor(Generic[T, R]):
    """
    Dynamic batch processor.
    
    Features:
    - Automatic batching with timeout
    - Priority queueing
    - Concurrent batch processing
    - Auto-tuning of batch size
    """
    
    def __init__(
        self,
        process_func: Callable[[List[T]], List[R]],
        config: BatchConfig = None,
        model_id: str = "default"
    ):
        """
        Initialize batch processor.
        
        Args:
            process_func: Function to process batch of inputs
            config: Batch configuration
            model_id: Model identifier
        """
        self.process_func = process_func
        self.config = config or BatchConfig()
        self.model_id = model_id
        
        # Request queue
        self.pending_requests: List[BatchRequest[T]] = []
        self.lock = asyncio.Lock()
        
        # Processing
        self.processing = False
        self.background_task: Optional[asyncio.Task] = None
        
        # Stats
        self.total_requests = 0
        self.total_batches = 0
        self.total_latency_ms = 0
        self.avg_batch_size = 0
        
        # Auto-tuning
        self.current_batch_size = self.config.max_batch_size
        self.recent_latencies: List[float] = []
    
    async def start(self):
        """Start background batch processing."""
        if self.processing:
            return
        
        self.processing = True
        self.background_task = asyncio.create_task(self._process_loop())
    
    async def stop(self):
        """Stop background batch processing."""
        self.processing = False
        if self.background_task:
            await self.background_task
    
    async def submit(
        self,
        input_data: T,
        priority: Priority = Priority.NORMAL,
        request_id: Optional[str] = None
    ) -> R:
        """
        Submit request for batch processing.
        
        Args:
            input_data: Input data
            priority: Request priority
            request_id: Optional request ID
        
        Returns:
            Processed result
        """
        # Create request
        req_id = request_id or f"req_{time.time()}_{id(input_data)}"
        request = BatchRequest(
            id=req_id,
            input=input_data,
            priority=priority,
        )
        
        # Add to queue
        async with self.lock:
            self.pending_requests.append(request)
            self.total_requests += 1
            
            # Sort by priority
            self.pending_requests.sort(key=lambda r: r.priority, reverse=True)
        
        # Wait for result
        result = await request.future
        
        if isinstance(result, Exception):
            raise result
        
        return result
    
    async def _process_loop(self):
        """Background loop to process batches."""
        while self.processing:
            try:
                # Wait for requests or timeout
                await asyncio.sleep(self.config.max_wait_ms / 1000)
                
                # Get batch
                batch = await self._get_next_batch()
                
                if batch:
                    # Process batch
                    await self._process_batch(batch)
            
            except Exception as e:
                print(f"Error in batch processing loop: {e}")
    
    async def _get_next_batch(self) -> List[BatchRequest[T]]:
        """Get next batch of requests to process."""
        async with self.lock:
            if not self.pending_requests:
                return []
            
            # Determine batch size
            batch_size = min(
                len(self.pending_requests),
                self.current_batch_size
            )
            
            # Check min batch size and timeout
            if batch_size < self.config.min_batch_size:
                # Check if oldest request has been waiting too long
                oldest = self.pending_requests[0]
                wait_time_ms = (time.time() - oldest.timestamp) * 1000
                
                if wait_time_ms < self.config.max_wait_ms:
                    # Wait for more requests
                    return []
            
            # Extract batch
            batch = self.pending_requests[:batch_size]
            self.pending_requests = self.pending_requests[batch_size:]
            
            return batch
    
    async def _process_batch(self, batch: List[BatchRequest[T]]):
        """Process a batch of requests."""
        if not batch:
            return
        
        start_time = time.time()
        
        try:
            # Extract inputs
            inputs = [req.input for req in batch]
            
            # Process batch
            if asyncio.iscoroutinefunction(self.process_func):
                outputs = await self.process_func(inputs)
            else:
                # Run in executor for sync functions
                loop = asyncio.get_event_loop()
                outputs = await loop.run_in_executor(None, self.process_func, inputs)
            
            # Calculate latency
            latency_ms = (time.time() - start_time) * 1000
            
            # Update stats
            self.total_batches += 1
            self.total_latency_ms += latency_ms
            self.avg_batch_size = (
                (self.avg_batch_size * (self.total_batches - 1) + len(batch))
                / self.total_batches
            )
            
            # Auto-tune batch size
            if self.config.enable_auto_tuning:
                self._tune_batch_size(latency_ms, len(batch))
            
            # Set results
            for req, output in zip(batch, outputs):
                if not req.future.done():
                    req.future.set_result(output)
        
        except Exception as e:
            # Set error for all requests
            for req in batch:
                if not req.future.done():
                    req.future.set_exception(e)
    
    def _tune_batch_size(self, latency_ms: float, batch_size: int):
        """Auto-tune batch size based on latency."""
        self.recent_latencies.append(latency_ms)
        
        # Keep last 10 latencies
        if len(self.recent_latencies) > 10:
            self.recent_latencies.pop(0)
        
        # Calculate average latency
        avg_latency = sum(self.recent_latencies) / len(self.recent_latencies)
        
        # Adjust batch size
        if avg_latency > self.config.target_latency_ms:
            # Too slow, decrease batch size
            self.current_batch_size = max(
                self.config.min_batch_size,
                int(self.current_batch_size * 0.8)
            )
        elif avg_latency < self.config.target_latency_ms * 0.7:
            # Fast enough, increase batch size
            self.current_batch_size = min(
                self.config.max_batch_size,
                int(self.current_batch_size * 1.2)
            )
    
    def get_stats(self) -> dict:
        """Get batch processing statistics."""
        avg_latency_ms = (
            self.total_latency_ms / self.total_batches
            if self.total_batches > 0
            else 0
        )
        
        return {
            "total_requests": self.total_requests,
            "total_batches": self.total_batches,
            "avg_batch_size": self.avg_batch_size,
            "avg_latency_ms": avg_latency_ms,
            "current_batch_size": self.current_batch_size,
            "pending_requests": len(self.pending_requests),
        }


# ============================================================================
# Batch Manager
# ============================================================================

class BatchManager:
    """
    Manage multiple batch processors.
    
    Features:
    - Per-model batch processors
    - Automatic processor creation
    - Global statistics
    """
    
    def __init__(self):
        self.processors: dict[str, BatchProcessor] = {}
        self.lock = threading.Lock()
    
    def get_processor(
        self,
        model_id: str,
        process_func: Callable[[List[T]], List[R]],
        config: Optional[BatchConfig] = None
    ) -> BatchProcessor:
        """Get or create batch processor for model."""
        if model_id not in self.processors:
            with self.lock:
                if model_id not in self.processors:
                    processor = BatchProcessor(
                        process_func=process_func,
                        config=config,
                        model_id=model_id
                    )
                    self.processors[model_id] = processor
        
        return self.processors[model_id]
    
    async def start_all(self):
        """Start all batch processors."""
        for processor in self.processors.values():
            await processor.start()
    
    async def stop_all(self):
        """Stop all batch processors."""
        for processor in self.processors.values():
            await processor.stop()
    
    def get_global_stats(self) -> dict:
        """Get statistics from all processors."""
        return {
            model_id: processor.get_stats()
            for model_id, processor in self.processors.items()
        }


# ============================================================================
# Global Instance
# ============================================================================

_batch_manager: Optional[BatchManager] = None


def get_batch_manager() -> BatchManager:
    """Get global batch manager instance."""
    global _batch_manager
    if _batch_manager is None:
        _batch_manager = BatchManager()
    return _batch_manager


# ============================================================================
# Helper Functions
# ============================================================================

async def batch_process(
    inputs: List[T],
    process_func: Callable[[List[T]], List[R]],
    model_id: str = "default",
    priority: Priority = Priority.NORMAL,
    config: Optional[BatchConfig] = None
) -> List[R]:
    """
    Process inputs with automatic batching.
    
    Args:
        inputs: List of inputs to process
        process_func: Function to process batch
        model_id: Model identifier
        priority: Request priority
        config: Batch configuration
    
    Returns:
        List of results
    """
    manager = get_batch_manager()
    processor = manager.get_processor(model_id, process_func, config)
    
    # Ensure processor is running
    await processor.start()
    
    # Submit all inputs
    tasks = [
        processor.submit(input_data, priority)
        for input_data in inputs
    ]
    
    # Wait for all results
    results = await asyncio.gather(*tasks)
    
    return results
