"""
GPU Manager for AI-Nexus
Handles GPU detection, memory management, and monitoring
"""

import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import time

logger = logging.getLogger(__name__)


@dataclass
class GPUInfo:
    """GPU device information"""
    id: int
    name: str
    compute_capability: Tuple[int, int]
    total_memory: int  # bytes
    free_memory: int  # bytes
    used_memory: int  # bytes
    temperature: Optional[float]  # Celsius
    power_usage: Optional[float]  # Watts
    utilization: Optional[float]  # Percentage
    driver_version: str
    cuda_version: str


class GPUManager:
    """
    Manages GPU resources and provides monitoring
    Supports NVIDIA GPUs with CUDA 13.0+
    """
    
    def __init__(self, device_id: int = 0, memory_fraction: float = 0.9):
        self.device_id = device_id
        self.memory_fraction = memory_fraction
        self._initialize_gpu()
        
    def _initialize_gpu(self):
        """Initialize GPU and set memory limits"""
        try:
            import torch
            
            if not torch.cuda.is_available():
                raise RuntimeError(
                    "CUDA not available. Install PyTorch with CUDA support:\n"
                    "pip install torch --index-url https://download.pytorch.org/whl/cu121"
                )
            
            self.device = torch.device(f'cuda:{self.device_id}')
            torch.cuda.set_device(self.device)
            
            # Set memory fraction to prevent OOM
            if hasattr(torch.cuda, 'set_per_process_memory_fraction'):
                torch.cuda.set_per_process_memory_fraction(
                    self.memory_fraction,
                    self.device_id
                )
            
            # Enable TF32 for RTX 5080 (faster matmul)
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            
            # Enable cudnn autotuner for optimized convolutions
            torch.backends.cudnn.benchmark = True
            
            info = self.get_gpu_info()
            logger.info(f"GPU initialized: {info.name}")
            logger.info(f"Memory: {info.free_memory / 1e9:.2f}GB free / {info.total_memory / 1e9:.2f}GB total")
            logger.info(f"Compute capability: {info.compute_capability}")
            
        except ImportError:
            raise RuntimeError(
                "PyTorch not installed. Install with:\n"
                "pip install torch --index-url https://download.pytorch.org/whl/cu121"
            )
    
    def get_gpu_info(self, device_id: Optional[int] = None) -> GPUInfo:
        """Get detailed GPU information"""
        import torch
        
        dev_id = device_id if device_id is not None else self.device_id
        
        # Get basic info
        props = torch.cuda.get_device_properties(dev_id)
        memory_allocated = torch.cuda.memory_allocated(dev_id)
        memory_reserved = torch.cuda.memory_reserved(dev_id)
        
        info = GPUInfo(
            id=dev_id,
            name=props.name,
            compute_capability=(props.major, props.minor),
            total_memory=props.total_memory,
            free_memory=props.total_memory - memory_reserved,
            used_memory=memory_allocated,
            temperature=None,
            power_usage=None,
            utilization=None,
            driver_version=torch.version.cuda or "Unknown",
            cuda_version=torch.version.cuda or "Unknown"
        )
        
        # Try to get additional stats from nvidia-ml-py
        try:
            import pynvml
            pynvml.nvmlInit()
            
            handle = pynvml.nvmlDeviceGetHandleByIndex(dev_id)
            
            # Temperature
            try:
                info.temperature = pynvml.nvmlDeviceGetTemperature(
                    handle,
                    pynvml.NVML_TEMPERATURE_GPU
                )
            except:
                pass
            
            # Power usage
            try:
                info.power_usage = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # mW to W
            except:
                pass
            
            # Utilization
            try:
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                info.utilization = util.gpu
            except:
                pass
            
            pynvml.nvmlShutdown()
            
        except ImportError:
            logger.warning(
                "pynvml not installed. Install for enhanced monitoring:\n"
                "pip install pynvml"
            )
        except Exception as e:
            logger.warning(f"Failed to get extended GPU info: {e}")
        
        return info
    
    def get_all_gpus(self) -> List[GPUInfo]:
        """Get information for all available GPUs"""
        import torch
        
        return [
            self.get_gpu_info(i)
            for i in range(torch.cuda.device_count())
        ]
    
    def allocate_memory(self, size_bytes: int) -> 'torch.Tensor':
        """Pre-allocate GPU memory"""
        import torch
        
        # Allocate tensor on GPU
        tensor = torch.empty(
            size_bytes // 4,  # float32 = 4 bytes
            dtype=torch.float32,
            device=self.device
        )
        
        logger.info(f"Allocated {size_bytes / 1e9:.2f}GB on GPU {self.device_id}")
        return tensor
    
    def clear_cache(self):
        """Clear GPU memory cache"""
        import torch
        
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        
        info = self.get_gpu_info()
        logger.info(f"GPU cache cleared. Free memory: {info.free_memory / 1e9:.2f}GB")
    
    def synchronize(self):
        """Wait for all GPU operations to complete"""
        import torch
        torch.cuda.synchronize(self.device)
    
    def monitor_memory(self) -> Dict:
        """Get current memory usage statistics"""
        import torch
        
        return {
            'allocated': torch.cuda.memory_allocated(self.device_id),
            'reserved': torch.cuda.memory_reserved(self.device_id),
            'max_allocated': torch.cuda.max_memory_allocated(self.device_id),
            'max_reserved': torch.cuda.max_memory_reserved(self.device_id)
        }
    
    def benchmark_bandwidth(self, size_mb: int = 1024) -> Dict:
        """Benchmark GPU memory bandwidth"""
        import torch
        
        size_bytes = size_mb * 1024 * 1024
        num_elements = size_bytes // 4  # float32
        
        # Create test data
        cpu_data = torch.randn(num_elements)
        gpu_data = torch.empty(num_elements, device=self.device)
        
        # Host to Device
        torch.cuda.synchronize()
        start = time.perf_counter()
        gpu_data.copy_(cpu_data)
        torch.cuda.synchronize()
        h2d_time = time.perf_counter() - start
        
        # Device to Host
        torch.cuda.synchronize()
        start = time.perf_counter()
        cpu_result = gpu_data.cpu()
        torch.cuda.synchronize()
        d2h_time = time.perf_counter() - start
        
        # Device to Device
        gpu_data2 = torch.empty_like(gpu_data)
        torch.cuda.synchronize()
        start = time.perf_counter()
        gpu_data2.copy_(gpu_data)
        torch.cuda.synchronize()
        d2d_time = time.perf_counter() - start
        
        return {
            'size_mb': size_mb,
            'host_to_device_gbps': (size_mb / 1024) / h2d_time,
            'device_to_host_gbps': (size_mb / 1024) / d2h_time,
            'device_to_device_gbps': (size_mb / 1024) / d2d_time,
            'h2d_time_ms': h2d_time * 1000,
            'd2h_time_ms': d2h_time * 1000,
            'd2d_time_ms': d2d_time * 1000
        }
    
    def set_cuda_visible_devices(self, devices: List[int]):
        """Set visible CUDA devices"""
        import os
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, devices))
        logger.info(f"CUDA_VISIBLE_DEVICES set to: {devices}")
    
    def enable_multi_gpu(self) -> bool:
        """Check if multi-GPU is available"""
        import torch
        
        num_gpus = torch.cuda.device_count()
        if num_gpus > 1:
            logger.info(f"Multi-GPU available: {num_gpus} devices")
            return True
        return False
    
    def get_optimal_batch_size(
        self,
        model_memory_mb: int,
        sample_size_mb: int,
        safety_factor: float = 0.8
    ) -> int:
        """
        Calculate optimal batch size based on available memory
        
        Args:
            model_memory_mb: Model size in MB
            sample_size_mb: Single sample size in MB
            safety_factor: Memory safety factor (0-1)
        
        Returns:
            Optimal batch size
        """
        info = self.get_gpu_info()
        available_mb = (info.free_memory / 1e6) * safety_factor
        
        # Memory needed = model + batch * sample_size
        remaining_mb = available_mb - model_memory_mb
        
        if remaining_mb <= 0:
            logger.warning("Model too large for GPU memory")
            return 1
        
        batch_size = int(remaining_mb / sample_size_mb)
        return max(1, batch_size)


def get_gpu_info(device_id: int = 0) -> GPUInfo:
    """Convenience function to get GPU info"""
    manager = GPUManager(device_id=device_id)
    return manager.get_gpu_info()


def print_gpu_info():
    """Print formatted GPU information"""
    try:
        import torch
        
        print("\n" + "="*60)
        print("GPU INFORMATION")
        print("="*60)
        
        if not torch.cuda.is_available():
            print("❌ CUDA not available")
            return
        
        manager = GPUManager()
        
        for gpu in manager.get_all_gpus():
            print(f"\nGPU {gpu.id}: {gpu.name}")
            print(f"  Compute Capability: {gpu.compute_capability[0]}.{gpu.compute_capability[1]}")
            print(f"  Total Memory: {gpu.total_memory / 1e9:.2f} GB")
            print(f"  Free Memory: {gpu.free_memory / 1e9:.2f} GB")
            print(f"  Used Memory: {gpu.used_memory / 1e9:.2f} GB")
            
            if gpu.temperature:
                print(f"  Temperature: {gpu.temperature}°C")
            if gpu.power_usage:
                print(f"  Power Usage: {gpu.power_usage:.1f}W")
            if gpu.utilization is not None:
                print(f"  Utilization: {gpu.utilization}%")
            
            print(f"  CUDA Version: {gpu.cuda_version}")
            print(f"  Driver Version: {gpu.driver_version}")
        
        # Benchmark
        print("\n" + "-"*60)
        print("MEMORY BANDWIDTH BENCHMARK")
        print("-"*60)
        
        results = manager.benchmark_bandwidth(size_mb=512)
        print(f"Test Size: {results['size_mb']} MB")
        print(f"Host → Device: {results['host_to_device_gbps']:.2f} GB/s ({results['h2d_time_ms']:.2f} ms)")
        print(f"Device → Host: {results['device_to_host_gbps']:.2f} GB/s ({results['d2h_time_ms']:.2f} ms)")
        print(f"Device → Device: {results['device_to_device_gbps']:.2f} GB/s ({results['d2d_time_ms']:.2f} ms)")
        
        print("\n" + "="*60 + "\n")
        
    except Exception as e:
        print(f"Error getting GPU info: {e}")


if __name__ == "__main__":
    print_gpu_info()
