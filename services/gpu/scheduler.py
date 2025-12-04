"""
GPU Scheduler for Intelligent GPU Allocation
Manages GPU resources across training jobs and inference workloads
"""

import torch
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import logging
from datetime import datetime
import psutil

logger = logging.getLogger(__name__)


class JobPriority(Enum):
    """Job priority levels."""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


class JobType(Enum):
    """Types of GPU workloads."""
    TRAINING = "training"
    INFERENCE = "inference"
    PREPROCESSING = "preprocessing"
    EVALUATION = "evaluation"


@dataclass
class GPUResource:
    """GPU resource information."""
    device_id: int
    name: str
    total_memory: int  # bytes
    used_memory: int   # bytes
    utilization: float  # 0-100
    temperature: float  # Celsius
    power_usage: float  # Watts
    compute_capability: Tuple[int, int]
    is_available: bool = True
    
    @property
    def free_memory(self) -> int:
        """Available memory in bytes."""
        return self.total_memory - self.used_memory
    
    @property
    def memory_utilization(self) -> float:
        """Memory utilization percentage."""
        return (self.used_memory / self.total_memory) * 100 if self.total_memory > 0 else 0.0


@dataclass
class JobRequest:
    """GPU job request."""
    job_id: str
    job_type: JobType
    priority: JobPriority
    required_gpus: int
    min_memory_per_gpu: int  # bytes
    preferred_gpu_ids: Optional[List[int]] = None
    created_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()


class GPUScheduler:
    """
    Intelligent GPU scheduler for multi-job management.
    
    Features:
    - Priority-based scheduling
    - Memory-aware allocation
    - Load balancing across GPUs
    - Dynamic reallocation
    - Job queuing
    """
    
    def __init__(self, min_free_memory_gb: float = 1.0):
        """
        Initialize GPU scheduler.
        
        Args:
            min_free_memory_gb: Minimum free memory to keep on each GPU (GB)
        """
        self.min_free_memory = int(min_free_memory_gb * 1024**3)  # Convert to bytes
        self.gpu_resources: List[GPUResource] = []
        self.active_jobs: Dict[str, List[int]] = {}  # job_id -> [gpu_ids]
        self.job_queue: List[JobRequest] = []
        
        self._refresh_gpu_info()
        logger.info(f"GPU Scheduler initialized with {len(self.gpu_resources)} GPUs")
    
    def _refresh_gpu_info(self) -> None:
        """Refresh GPU resource information."""
        if not torch.cuda.is_available():
            logger.warning("CUDA not available - no GPUs to schedule")
            return
        
        self.gpu_resources = []
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            
            # Get memory info
            torch.cuda.set_device(i)
            total_memory = torch.cuda.get_device_properties(i).total_memory
            used_memory = torch.cuda.memory_allocated(i)
            
            # Get utilization (simplified - would use nvidia-ml-py in production)
            try:
                utilization = torch.cuda.utilization(i) if hasattr(torch.cuda, 'utilization') else 0.0
            except:
                utilization = 0.0
            
            gpu = GPUResource(
                device_id=i,
                name=props.name,
                total_memory=total_memory,
                used_memory=used_memory,
                utilization=utilization,
                temperature=0.0,  # Would use nvidia-ml-py
                power_usage=0.0,   # Would use nvidia-ml-py
                compute_capability=(props.major, props.minor),
                is_available=True
            )
            
            self.gpu_resources.append(gpu)
            logger.debug(
                f"GPU {i}: {gpu.name}, "
                f"Memory: {gpu.used_memory / 1024**3:.2f}/{gpu.total_memory / 1024**3:.2f} GB, "
                f"Util: {gpu.utilization:.1f}%"
            )
    
    def allocate_gpus(self, request: JobRequest) -> Optional[List[int]]:
        """
        Allocate GPUs for a job request.
        
        Args:
            request: Job request with resource requirements
            
        Returns:
            List of allocated GPU IDs, or None if allocation failed
        """
        self._refresh_gpu_info()
        
        # Check if request can be satisfied
        available_gpus = [
            gpu for gpu in self.gpu_resources
            if gpu.is_available and gpu.free_memory >= request.min_memory_per_gpu
        ]
        
        if len(available_gpus) < request.required_gpus:
            logger.warning(
                f"Not enough GPUs available for job {request.job_id}: "
                f"need {request.required_gpus}, have {len(available_gpus)}"
            )
            # Add to queue
            self.job_queue.append(request)
            self.job_queue.sort(key=lambda x: (x.priority.value, x.created_at), reverse=True)
            return None
        
        # Allocate based on strategy
        allocated = self._select_gpus(available_gpus, request)
        
        if allocated:
            # Mark GPUs as allocated
            for gpu_id in allocated:
                self.gpu_resources[gpu_id].is_available = False
            
            self.active_jobs[request.job_id] = allocated
            logger.info(
                f"Allocated GPUs {allocated} to job {request.job_id} "
                f"(type={request.job_type.value}, priority={request.priority.name})"
            )
        
        return allocated
    
    def _select_gpus(
        self,
        available_gpus: List[GPUResource],
        request: JobRequest
    ) -> List[int]:
        """
        Select best GPUs for the request.
        
        Strategy:
        1. Prefer user-specified GPUs if available
        2. For training: prefer GPUs with lower utilization
        3. For inference: prefer GPUs with more free memory
        4. Balance load across GPUs
        """
        # If user specified GPUs, try to use them
        if request.preferred_gpu_ids:
            preferred_available = [
                gpu for gpu in available_gpus
                if gpu.device_id in request.preferred_gpu_ids
            ]
            if len(preferred_available) >= request.required_gpus:
                available_gpus = preferred_available
        
        # Sort by selection criteria
        if request.job_type == JobType.TRAINING:
            # For training: prefer lower utilization (more compute available)
            available_gpus.sort(key=lambda x: (x.utilization, -x.free_memory))
        else:
            # For inference: prefer more free memory
            available_gpus.sort(key=lambda x: (-x.free_memory, x.utilization))
        
        # Select top N GPUs
        selected = [gpu.device_id for gpu in available_gpus[:request.required_gpus]]
        return selected
    
    def release_gpus(self, job_id: str) -> None:
        """
        Release GPUs allocated to a job.
        
        Args:
            job_id: Job identifier
        """
        if job_id not in self.active_jobs:
            logger.warning(f"Job {job_id} not found in active jobs")
            return
        
        gpu_ids = self.active_jobs[job_id]
        
        # Mark GPUs as available
        for gpu_id in gpu_ids:
            if gpu_id < len(self.gpu_resources):
                self.gpu_resources[gpu_id].is_available = True
        
        del self.active_jobs[job_id]
        logger.info(f"Released GPUs {gpu_ids} from job {job_id}")
        
        # Try to schedule queued jobs
        self._process_queue()
    
    def _process_queue(self) -> None:
        """Process queued jobs and try to allocate resources."""
        if not self.job_queue:
            return
        
        logger.info(f"Processing queue with {len(self.job_queue)} jobs")
        
        # Try to allocate for each queued job
        scheduled = []
        for i, request in enumerate(self.job_queue):
            allocated = self.allocate_gpus(request)
            if allocated is not None:
                scheduled.append(i)
        
        # Remove scheduled jobs from queue
        for i in reversed(scheduled):
            self.job_queue.pop(i)
    
    def get_gpu_status(self) -> Dict[str, any]:
        """
        Get current GPU status.
        
        Returns:
            Dictionary with GPU status information
        """
        self._refresh_gpu_info()
        
        return {
            "total_gpus": len(self.gpu_resources),
            "available_gpus": sum(1 for gpu in self.gpu_resources if gpu.is_available),
            "active_jobs": len(self.active_jobs),
            "queued_jobs": len(self.job_queue),
            "gpus": [
                {
                    "id": gpu.device_id,
                    "name": gpu.name,
                    "memory_used_gb": gpu.used_memory / 1024**3,
                    "memory_total_gb": gpu.total_memory / 1024**3,
                    "memory_util": gpu.memory_utilization,
                    "compute_util": gpu.utilization,
                    "available": gpu.is_available,
                    "compute_capability": f"{gpu.compute_capability[0]}.{gpu.compute_capability[1]}"
                }
                for gpu in self.gpu_resources
            ],
            "jobs": {
                job_id: gpu_ids
                for job_id, gpu_ids in self.active_jobs.items()
            }
        }
    
    def get_optimal_batch_size(
        self,
        model: torch.nn.Module,
        input_shape: Tuple[int, ...],
        gpu_id: int = 0,
        safety_factor: float = 0.8
    ) -> int:
        """
        Calculate optimal batch size for a model on a GPU.
        
        Args:
            model: PyTorch model
            input_shape: Input tensor shape (without batch dimension)
            gpu_id: GPU device ID
            safety_factor: Memory safety factor (0-1)
            
        Returns:
            Recommended batch size
        """
        if gpu_id >= len(self.gpu_resources):
            raise ValueError(f"Invalid GPU ID: {gpu_id}")
        
        gpu = self.gpu_resources[gpu_id]
        available_memory = gpu.free_memory * safety_factor
        
        # Estimate memory per sample
        # This is a simplified estimation
        sample = torch.randn(1, *input_shape).to(f"cuda:{gpu_id}")
        
        # Forward pass to estimate activation memory
        model = model.to(f"cuda:{gpu_id}")
        model.eval()
        
        with torch.no_grad():
            _ = model(sample)
        
        # Get memory used
        memory_per_sample = torch.cuda.memory_allocated(gpu_id)
        
        # Calculate batch size
        batch_size = int(available_memory / memory_per_sample)
        batch_size = max(1, batch_size)  # At least 1
        
        logger.info(
            f"Optimal batch size for GPU {gpu_id}: {batch_size} "
            f"(memory per sample: {memory_per_sample / 1024**2:.2f} MB)"
        )
        
        return batch_size


# Global scheduler instance
_scheduler = None


def get_scheduler() -> GPUScheduler:
    """Get global GPU scheduler instance."""
    global _scheduler
    if _scheduler is None:
        _scheduler = GPUScheduler()
    return _scheduler
