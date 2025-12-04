"""
Distributed Training Module for Multi-GPU/Multi-Node Training
Supports PyTorch DistributedDataParallel (DDP) with NCCL backend
"""

import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from typing import Optional, Callable, Dict, Any
import logging
from contextlib import contextmanager

logger = logging.getLogger(__name__)


class DistributedTrainer:
    """
    Manages distributed training across multiple GPUs and nodes.
    
    Features:
    - Automatic rank/world_size detection
    - NCCL backend for GPU communication
    - Gradient synchronization
    - Distributed checkpointing
    """
    
    def __init__(
        self,
        backend: str = "nccl",
        init_method: Optional[str] = None,
        timeout_minutes: int = 30
    ):
        """
        Initialize distributed training environment.
        
        Args:
            backend: Communication backend ('nccl', 'gloo', 'mpi')
            init_method: Initialization method URL (auto-detected if None)
            timeout_minutes: Timeout for initialization
        """
        self.backend = backend
        self.init_method = init_method or self._get_init_method()
        self.timeout = torch.distributed.timedelta(minutes=timeout_minutes)
        
        # Distributed state
        self.rank: int = -1
        self.world_size: int = 1
        self.local_rank: int = -1
        self.is_initialized: bool = False
        
    def _get_init_method(self) -> str:
        """Auto-detect initialization method from environment."""
        # Kubernetes environment
        if os.environ.get("MASTER_ADDR") and os.environ.get("MASTER_PORT"):
            master_addr = os.environ["MASTER_ADDR"]
            master_port = os.environ["MASTER_PORT"]
            return f"tcp://{master_addr}:{master_port}"
        
        # SLURM environment
        if os.environ.get("SLURM_JOB_ID"):
            node_list = os.environ.get("SLURM_NODELIST", "localhost")
            return f"tcp://{node_list}:29500"
        
        # Default to localhost
        return "tcp://localhost:29500"
    
    def initialize(self) -> None:
        """Initialize distributed process group."""
        if self.is_initialized:
            logger.warning("Distributed training already initialized")
            return
        
        # Get rank and world_size from environment
        self.rank = int(os.environ.get("RANK", 0))
        self.world_size = int(os.environ.get("WORLD_SIZE", 1))
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        
        if self.world_size == 1:
            logger.info("Single process mode - skipping distributed init")
            self.is_initialized = True
            return
        
        # Initialize process group
        logger.info(
            f"Initializing distributed training: "
            f"rank={self.rank}, world_size={self.world_size}, "
            f"backend={self.backend}, init_method={self.init_method}"
        )
        
        dist.init_process_group(
            backend=self.backend,
            init_method=self.init_method,
            rank=self.rank,
            world_size=self.world_size,
            timeout=self.timeout
        )
        
        # Set device
        if torch.cuda.is_available():
            torch.cuda.set_device(self.local_rank)
            logger.info(f"Rank {self.rank} using GPU {self.local_rank}")
        
        self.is_initialized = True
        logger.info(f"Distributed training initialized on rank {self.rank}")
    
    def cleanup(self) -> None:
        """Cleanup distributed resources."""
        if self.is_initialized and self.world_size > 1:
            dist.destroy_process_group()
            self.is_initialized = False
            logger.info(f"Distributed training cleaned up on rank {self.rank}")
    
    def wrap_model(
        self,
        model: torch.nn.Module,
        device_ids: Optional[list] = None,
        find_unused_parameters: bool = False
    ) -> torch.nn.Module:
        """
        Wrap model for distributed training.
        
        Args:
            model: PyTorch model to wrap
            device_ids: GPU device IDs (auto-detected if None)
            find_unused_parameters: Whether to find unused parameters
            
        Returns:
            DDP-wrapped model
        """
        if not self.is_initialized:
            self.initialize()
        
        if self.world_size == 1:
            # Single process - just move to GPU
            if torch.cuda.is_available():
                return model.cuda(self.local_rank)
            return model
        
        # Multi-process - wrap with DDP
        if torch.cuda.is_available():
            device_ids = device_ids or [self.local_rank]
            model = model.cuda(self.local_rank)
        
        ddp_model = DDP(
            model,
            device_ids=device_ids if torch.cuda.is_available() else None,
            find_unused_parameters=find_unused_parameters
        )
        
        logger.info(f"Model wrapped with DDP on rank {self.rank}")
        return ddp_model
    
    def barrier(self) -> None:
        """Synchronize all processes."""
        if self.is_initialized and self.world_size > 1:
            dist.barrier()
    
    def is_main_process(self) -> bool:
        """Check if this is the main process (rank 0)."""
        return self.rank <= 0
    
    def reduce_metrics(
        self,
        metrics: Dict[str, float],
        reduction: str = "mean"
    ) -> Dict[str, float]:
        """
        Reduce metrics across all processes.
        
        Args:
            metrics: Dictionary of metrics to reduce
            reduction: Reduction operation ('mean', 'sum', 'max', 'min')
            
        Returns:
            Reduced metrics dictionary
        """
        if not self.is_initialized or self.world_size == 1:
            return metrics
        
        reduced = {}
        for key, value in metrics.items():
            tensor = torch.tensor(value, device=f"cuda:{self.local_rank}")
            
            if reduction == "mean":
                dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
                tensor /= self.world_size
            elif reduction == "sum":
                dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
            elif reduction == "max":
                dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
            elif reduction == "min":
                dist.all_reduce(tensor, op=dist.ReduceOp.MIN)
            else:
                raise ValueError(f"Unknown reduction: {reduction}")
            
            reduced[key] = tensor.item()
        
        return reduced
    
    def save_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        filepath: str,
        **kwargs
    ) -> None:
        """
        Save checkpoint (only on main process).
        
        Args:
            model: Model to save
            optimizer: Optimizer to save
            epoch: Current epoch
            filepath: Path to save checkpoint
            **kwargs: Additional state to save
        """
        if not self.is_main_process():
            return
        
        # Unwrap DDP model if needed
        model_state = model.module.state_dict() if isinstance(model, DDP) else model.state_dict()
        
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model_state,
            "optimizer_state_dict": optimizer.state_dict(),
            "rank": self.rank,
            "world_size": self.world_size,
            **kwargs
        }
        
        torch.save(checkpoint, filepath)
        logger.info(f"Checkpoint saved to {filepath}")
    
    def load_checkpoint(
        self,
        filepath: str,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None
    ) -> Dict[str, Any]:
        """
        Load checkpoint.
        
        Args:
            filepath: Path to checkpoint
            model: Model to load state into
            optimizer: Optional optimizer to load state into
            
        Returns:
            Checkpoint dictionary
        """
        # Map location based on device
        map_location = f"cuda:{self.local_rank}" if torch.cuda.is_available() else "cpu"
        checkpoint = torch.load(filepath, map_location=map_location)
        
        # Load model state
        if isinstance(model, DDP):
            model.module.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint["model_state_dict"])
        
        # Load optimizer state
        if optimizer is not None and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
        logger.info(f"Checkpoint loaded from {filepath}")
        return checkpoint
    
    @contextmanager
    def distributed_context(self):
        """Context manager for distributed training."""
        try:
            self.initialize()
            yield self
        finally:
            self.cleanup()


def get_distributed_sampler(
    dataset: torch.utils.data.Dataset,
    shuffle: bool = True,
    seed: int = 0
) -> torch.utils.data.Sampler:
    """
    Create distributed sampler for dataset.
    
    Args:
        dataset: PyTorch dataset
        shuffle: Whether to shuffle data
        seed: Random seed
        
    Returns:
        DistributedSampler or SequentialSampler
    """
    if dist.is_initialized():
        return torch.utils.data.DistributedSampler(
            dataset,
            shuffle=shuffle,
            seed=seed
        )
    else:
        if shuffle:
            return torch.utils.data.RandomSampler(dataset)
        return torch.utils.data.SequentialSampler(dataset)


def setup_distributed_logger(rank: int) -> None:
    """
    Setup logging for distributed training.
    Only main process logs to console.
    
    Args:
        rank: Process rank
    """
    if rank == 0:
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    else:
        logging.basicConfig(level=logging.WARNING)
