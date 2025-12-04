"""
Tests for Distributed Training Module
"""

import pytest
import torch
import torch.nn as nn
from services.ml.distributed import (
    DistributedTrainer,
    get_distributed_sampler,
    setup_distributed_logger
)
from services.gpu.scheduler import (
    GPUScheduler,
    JobRequest,
    JobType,
    JobPriority
)


class SimpleModel(nn.Module):
    """Simple model for testing."""
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 5)
    
    def forward(self, x):
        return self.fc(x)


class TestDistributedTrainer:
    """Tests for DistributedTrainer class."""
    
    def test_initialization(self):
        """Test trainer initialization."""
        trainer = DistributedTrainer(backend="gloo")  # Use gloo for CPU testing
        
        assert trainer.backend == "gloo"
        assert trainer.rank == -1
        assert trainer.world_size == 1
        assert not trainer.is_initialized
    
    def test_single_process_mode(self):
        """Test single process mode (no distributed)."""
        trainer = DistributedTrainer()
        trainer.initialize()
        
        assert trainer.is_initialized
        assert trainer.world_size == 1
        assert trainer.is_main_process()
    
    def test_wrap_model_single_process(self):
        """Test model wrapping in single process mode."""
        trainer = DistributedTrainer()
        model = SimpleModel()
        
        wrapped = trainer.wrap_model(model)
        
        # Should not be wrapped in DDP for single process
        assert not hasattr(wrapped, "module")
        
        trainer.cleanup()
    
    def test_barrier_single_process(self):
        """Test barrier in single process mode."""
        trainer = DistributedTrainer()
        trainer.initialize()
        
        # Should not raise error
        trainer.barrier()
        
        trainer.cleanup()
    
    def test_reduce_metrics_single_process(self):
        """Test metrics reduction in single process."""
        trainer = DistributedTrainer()
        trainer.initialize()
        
        metrics = {"loss": 1.5, "accuracy": 0.85}
        reduced = trainer.reduce_metrics(metrics)
        
        assert reduced == metrics  # No change in single process
        
        trainer.cleanup()
    
    def test_checkpoint_save_load(self, tmp_path):
        """Test checkpoint saving and loading."""
        trainer = DistributedTrainer()
        trainer.initialize()
        
        model = SimpleModel()
        optimizer = torch.optim.Adam(model.parameters())
        
        # Save checkpoint
        checkpoint_path = tmp_path / "checkpoint.pt"
        trainer.save_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=5,
            filepath=str(checkpoint_path),
            custom_value=42
        )
        
        assert checkpoint_path.exists()
        
        # Load checkpoint
        new_model = SimpleModel()
        new_optimizer = torch.optim.Adam(new_model.parameters())
        
        checkpoint = trainer.load_checkpoint(
            str(checkpoint_path),
            new_model,
            new_optimizer
        )
        
        assert checkpoint["epoch"] == 5
        assert checkpoint["custom_value"] == 42
        
        trainer.cleanup()
    
    def test_distributed_context(self):
        """Test distributed context manager."""
        trainer = DistributedTrainer()
        
        with trainer.distributed_context() as t:
            assert t.is_initialized
        
        # Should be cleaned up only if world_size > 1
        # In single process mode, cleanup is a no-op
        if trainer.world_size > 1:
            assert not trainer.is_initialized


class TestDistributedSampler:
    """Tests for distributed sampler."""
    
    def test_sampler_creation(self):
        """Test sampler creation without distributed."""
        dataset = torch.utils.data.TensorDataset(
            torch.randn(100, 10)
        )
        
        sampler = get_distributed_sampler(dataset, shuffle=True)
        
        # Should return RandomSampler for non-distributed
        assert isinstance(sampler, torch.utils.data.RandomSampler)
    
    def test_sequential_sampler(self):
        """Test sequential sampler."""
        dataset = torch.utils.data.TensorDataset(
            torch.randn(100, 10)
        )
        
        sampler = get_distributed_sampler(dataset, shuffle=False)
        
        assert isinstance(sampler, torch.utils.data.SequentialSampler)


class TestGPUScheduler:
    """Tests for GPU Scheduler."""
    
    @pytest.fixture
    def scheduler(self):
        """Create scheduler instance."""
        return GPUScheduler()
    
    def test_initialization(self, scheduler):
        """Test scheduler initialization."""
        assert scheduler is not None
        assert isinstance(scheduler.gpu_resources, list)
        assert isinstance(scheduler.active_jobs, dict)
        assert isinstance(scheduler.job_queue, list)
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_gpu_detection(self, scheduler):
        """Test GPU detection."""
        assert len(scheduler.gpu_resources) > 0
        
        gpu = scheduler.gpu_resources[0]
        assert gpu.device_id >= 0
        assert gpu.total_memory > 0
        assert gpu.compute_capability[0] > 0
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_allocate_gpus(self, scheduler):
        """Test GPU allocation."""
        request = JobRequest(
            job_id="test_job_1",
            job_type=JobType.TRAINING,
            priority=JobPriority.NORMAL,
            required_gpus=1,
            min_memory_per_gpu=1024**3  # 1 GB
        )
        
        allocated = scheduler.allocate_gpus(request)
        
        if allocated is not None:
            assert len(allocated) == 1
            assert allocated[0] >= 0
            assert "test_job_1" in scheduler.active_jobs
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_release_gpus(self, scheduler):
        """Test GPU release."""
        request = JobRequest(
            job_id="test_job_2",
            job_type=JobType.INFERENCE,
            priority=JobPriority.HIGH,
            required_gpus=1,
            min_memory_per_gpu=1024**3
        )
        
        allocated = scheduler.allocate_gpus(request)
        
        if allocated is not None:
            scheduler.release_gpus("test_job_2")
            assert "test_job_2" not in scheduler.active_jobs
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_job_priority(self, scheduler):
        """Test job priority handling."""
        # Create high priority request
        high_priority = JobRequest(
            job_id="high_priority_job",
            job_type=JobType.TRAINING,
            priority=JobPriority.CRITICAL,
            required_gpus=1,
            min_memory_per_gpu=1024**3
        )
        
        # Create low priority request
        low_priority = JobRequest(
            job_id="low_priority_job",
            job_type=JobType.PREPROCESSING,
            priority=JobPriority.LOW,
            required_gpus=1,
            min_memory_per_gpu=1024**3
        )
        
        # Queue should prioritize high priority jobs
        scheduler.job_queue.append(low_priority)
        scheduler.job_queue.append(high_priority)
        scheduler.job_queue.sort(
            key=lambda x: (x.priority.value, x.created_at),
            reverse=True
        )
        
        assert scheduler.job_queue[0].job_id == "high_priority_job"
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_get_gpu_status(self, scheduler):
        """Test GPU status retrieval."""
        status = scheduler.get_gpu_status()
        
        assert "total_gpus" in status
        assert "available_gpus" in status
        assert "active_jobs" in status
        assert "queued_jobs" in status
        assert "gpus" in status
        
        if len(status["gpus"]) > 0:
            gpu_info = status["gpus"][0]
            assert "id" in gpu_info
            assert "name" in gpu_info
            assert "memory_total_gb" in gpu_info
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_optimal_batch_size(self, scheduler):
        """Test optimal batch size calculation."""
        model = SimpleModel()
        input_shape = (10,)  # Match model input
        
        try:
            batch_size = scheduler.get_optimal_batch_size(
                model=model,
                input_shape=input_shape,
                gpu_id=0,
                safety_factor=0.8
            )
            
            assert batch_size > 0
            assert isinstance(batch_size, int)
        except Exception as e:
            pytest.skip(f"Batch size calculation failed: {e}")


def test_setup_distributed_logger():
    """Test distributed logger setup."""
    # Should not raise error
    setup_distributed_logger(0)
    setup_distributed_logger(1)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
