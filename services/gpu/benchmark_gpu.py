"""
GPU Benchmarks for AI-Nexus
Comprehensive performance testing of CUDA kernels
Measures speedup vs CPU and validates correctness
"""

import numpy as np
import time
import logging
from typing import Dict, List, Tuple
from dataclasses import dataclass, asdict
import json

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Store benchmark results"""
    operation: str
    input_size: str
    cpu_time_ms: float
    gpu_time_ms: float
    speedup: float
    throughput_gflops: float
    memory_mb: float
    error: float = 0.0
    
    def to_dict(self) -> Dict:
        return asdict(self)


class GPUBenchmark:
    """
    Benchmark suite for GPU operations
    Tests matrix multiplication, convolution, attention, privacy ops
    """
    
    def __init__(self, warmup_runs: int = 3, benchmark_runs: int = 10):
        self.warmup_runs = warmup_runs
        self.benchmark_runs = benchmark_runs
        self.results: List[BenchmarkResult] = []
        
        # Check GPU availability
        self._check_gpu()
    
    def _check_gpu(self):
        """Verify GPU is available"""
        try:
            import torch
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA not available")
            
            self.device = torch.device('cuda')
            info = torch.cuda.get_device_properties(0)
            
            logger.info(f"GPU: {info.name}")
            logger.info(f"Memory: {info.total_memory / 1e9:.2f}GB")
            logger.info(f"Compute: {info.major}.{info.minor}")
            
        except ImportError:
            raise RuntimeError("PyTorch not installed. Cannot run GPU benchmarks.")
    
    def benchmark_matrix_multiply(
        self,
        sizes: List[Tuple[int, int, int]] = [(512, 512, 512), (1024, 1024, 1024), (2048, 2048, 2048)]
    ):
        """
        Benchmark matrix multiplication: C = A @ B
        
        Args:
            sizes: List of (M, K, N) dimensions
        """
        print("\n" + "="*60)
        print("MATRIX MULTIPLICATION BENCHMARK")
        print("="*60)
        
        from services.gpu.kernels import MatrixMultiplyKernel
        
        kernel = MatrixMultiplyKernel(use_tensor_cores=True)
        
        for M, K, N in sizes:
            print(f"\nSize: ({M}, {K}) @ ({K}, {N}) = ({M}, {N})")
            
            # Generate test data
            A = np.random.randn(M, K).astype(np.float32)
            B = np.random.randn(K, N).astype(np.float32)
            
            # CPU baseline
            print("  CPU: ", end="", flush=True)
            cpu_times = []
            for _ in range(self.warmup_runs):
                np.matmul(A, B)
            
            for _ in range(self.benchmark_runs):
                start = time.perf_counter()
                C_cpu = np.matmul(A, B)
                cpu_times.append(time.perf_counter() - start)
            
            cpu_time = np.median(cpu_times)
            print(f"{cpu_time*1000:.2f}ms")
            
            # GPU
            print("  GPU: ", end="", flush=True)
            gpu_times = []
            for _ in range(self.warmup_runs):
                kernel.forward(A, B, use_mixed_precision=True)
            
            for _ in range(self.benchmark_runs):
                start = time.perf_counter()
                C_gpu = kernel.forward(A, B, use_mixed_precision=True)
                gpu_times.append(time.perf_counter() - start)
            
            gpu_time = np.median(gpu_times)
            print(f"{gpu_time*1000:.2f}ms")
            
            # Calculate metrics
            speedup = cpu_time / gpu_time
            
            # FLOPS: 2*M*N*K operations
            flops = 2 * M * N * K
            throughput_gflops = (flops / gpu_time) / 1e9
            
            # Memory
            memory_mb = (M*K + K*N + M*N) * 4 / 1e6  # float32 = 4 bytes
            
            # Verify correctness
            error = np.max(np.abs(C_cpu - C_gpu))
            
            print(f"  Speedup: {speedup:.2f}x")
            print(f"  Throughput: {throughput_gflops:.2f} GFLOPS")
            print(f"  Error: {error:.2e}")
            
            # Store result
            result = BenchmarkResult(
                operation="matrix_multiply",
                input_size=f"{M}x{K} @ {K}x{N}",
                cpu_time_ms=cpu_time * 1000,
                gpu_time_ms=gpu_time * 1000,
                speedup=speedup,
                throughput_gflops=throughput_gflops,
                memory_mb=memory_mb,
                error=error
            )
            self.results.append(result)
    
    def benchmark_differential_privacy(
        self,
        sizes: List[Tuple[int, int]] = [(1000, 100), (10000, 1000), (100000, 10000)]
    ):
        """Benchmark DP noise generation"""
        print("\n" + "="*60)
        print("DIFFERENTIAL PRIVACY BENCHMARK")
        print("="*60)
        
        from services.gpu.privacy_gpu import GPUDifferentialPrivacy
        
        dp_gpu = GPUDifferentialPrivacy(epsilon=1.0, delta=1e-5)
        
        for batch_size, dim in sizes:
            print(f"\nSize: ({batch_size}, {dim})")
            
            # Generate test gradients
            gradients = np.random.randn(batch_size, dim).astype(np.float32)
            
            # CPU baseline
            print("  CPU: ", end="", flush=True)
            cpu_times = []
            
            for _ in range(self.warmup_runs):
                dp_gpu._privatize_gradients_cpu(gradients, clip_norm=1.0, noise_multiplier=1.0, batch_size=batch_size)
            
            for _ in range(self.benchmark_runs):
                start = time.perf_counter()
                result_cpu = dp_gpu._privatize_gradients_cpu(gradients, clip_norm=1.0, noise_multiplier=1.0, batch_size=batch_size)
                cpu_times.append(time.perf_counter() - start)
            
            cpu_time = np.median(cpu_times)
            print(f"{cpu_time*1000:.2f}ms")
            
            # GPU
            print("  GPU: ", end="", flush=True)
            gpu_times = []
            
            for _ in range(self.warmup_runs):
                dp_gpu.privatize_gradients(gradients, clip_norm=1.0, noise_multiplier=1.0, batch_size=batch_size)
            
            for _ in range(self.benchmark_runs):
                start = time.perf_counter()
                result_gpu = dp_gpu.privatize_gradients(gradients, clip_norm=1.0, noise_multiplier=1.0, batch_size=batch_size)
                gpu_times.append(time.perf_counter() - start)
            
            gpu_time = np.median(gpu_times)
            print(f"{gpu_time*1000:.2f}ms")
            
            speedup = cpu_time / gpu_time
            throughput_gflops = (batch_size * dim / gpu_time) / 1e9
            memory_mb = batch_size * dim * 4 / 1e6
            
            print(f"  Speedup: {speedup:.2f}x")
            print(f"  Throughput: {throughput_gflops:.2f} G samples/s")
            
            result = BenchmarkResult(
                operation="differential_privacy",
                input_size=f"{batch_size}x{dim}",
                cpu_time_ms=cpu_time * 1000,
                gpu_time_ms=gpu_time * 1000,
                speedup=speedup,
                throughput_gflops=throughput_gflops,
                memory_mb=memory_mb
            )
            self.results.append(result)
    
    def benchmark_secure_aggregation(
        self,
        configs: List[Tuple[int, int]] = [(10, 10000), (100, 100000), (1000, 1000000)]
    ):
        """Benchmark secure aggregation"""
        print("\n" + "="*60)
        print("SECURE AGGREGATION BENCHMARK")
        print("="*60)
        
        from services.gpu.privacy_gpu import GPUSecureMultiPartyComputation
        
        smpc = GPUSecureMultiPartyComputation(num_parties=100)
        
        for num_clients, model_size in configs:
            print(f"\nClients: {num_clients}, Model size: {model_size}")
            
            # Generate client updates
            client_updates = [
                np.random.randn(model_size).astype(np.float32)
                for _ in range(num_clients)
            ]
            
            # CPU baseline
            print("  CPU: ", end="", flush=True)
            cpu_times = []
            
            for _ in range(self.warmup_runs):
                np.mean(client_updates, axis=0)
            
            for _ in range(self.benchmark_runs):
                start = time.perf_counter()
                result_cpu = np.mean(client_updates, axis=0)
                cpu_times.append(time.perf_counter() - start)
            
            cpu_time = np.median(cpu_times)
            print(f"{cpu_time*1000:.2f}ms")
            
            # GPU
            print("  GPU: ", end="", flush=True)
            gpu_times = []
            
            for _ in range(self.warmup_runs):
                smpc.secure_aggregate(client_updates)
            
            for _ in range(self.benchmark_runs):
                start = time.perf_counter()
                result_gpu = smpc.secure_aggregate(client_updates)
                gpu_times.append(time.perf_counter() - start)
            
            gpu_time = np.median(gpu_times)
            print(f"{gpu_time*1000:.2f}ms")
            
            speedup = cpu_time / gpu_time
            throughput_gflops = (num_clients * model_size / gpu_time) / 1e9
            memory_mb = num_clients * model_size * 4 / 1e6
            
            print(f"  Speedup: {speedup:.2f}x")
            print(f"  Throughput: {throughput_gflops:.2f} G params/s")
            
            result = BenchmarkResult(
                operation="secure_aggregation",
                input_size=f"{num_clients} clients × {model_size} params",
                cpu_time_ms=cpu_time * 1000,
                gpu_time_ms=gpu_time * 1000,
                speedup=speedup,
                throughput_gflops=throughput_gflops,
                memory_mb=memory_mb
            )
            self.results.append(result)
    
    def benchmark_memory_bandwidth(self):
        """Benchmark GPU memory bandwidth"""
        print("\n" + "="*60)
        print("MEMORY BANDWIDTH BENCHMARK")
        print("="*60)
        
        from services.gpu.gpu_manager import GPUManager
        
        manager = GPUManager()
        
        for size_mb in [128, 512, 1024, 2048]:
            print(f"\nTransfer size: {size_mb}MB")
            
            results = manager.benchmark_bandwidth(size_mb=size_mb)
            
            print(f"  Host → Device: {results['host_to_device_gbps']:.2f} GB/s")
            print(f"  Device → Host: {results['device_to_host_gbps']:.2f} GB/s")
            print(f"  Device → Device: {results['device_to_device_gbps']:.2f} GB/s")
    
    def run_all_benchmarks(self):
        """Run complete benchmark suite"""
        print("\n" + "="*70)
        print(" "*20 + "AI-NEXUS GPU BENCHMARK SUITE")
        print("="*70)
        
        try:
            # Matrix operations
            self.benchmark_matrix_multiply()
            
            # Privacy operations
            self.benchmark_differential_privacy()
            self.benchmark_secure_aggregation()
            
            # Memory bandwidth
            self.benchmark_memory_bandwidth()
            
            # Summary
            self.print_summary()
            
            # Save results
            self.save_results()
            
        except Exception as e:
            logger.error(f"Benchmark failed: {e}", exc_info=True)
            raise
    
    def print_summary(self):
        """Print benchmark summary"""
        print("\n" + "="*70)
        print("BENCHMARK SUMMARY")
        print("="*70)
        
        if not self.results:
            print("No results to display")
            return
        
        # Group by operation
        ops = {}
        for result in self.results:
            if result.operation not in ops:
                ops[result.operation] = []
            ops[result.operation].append(result)
        
        for op_name, op_results in ops.items():
            print(f"\n{op_name.upper().replace('_', ' ')}")
            print("-" * 70)
            
            avg_speedup = np.mean([r.speedup for r in op_results])
            max_speedup = np.max([r.speedup for r in op_results])
            avg_throughput = np.mean([r.throughput_gflops for r in op_results])
            
            print(f"  Average speedup: {avg_speedup:.2f}x")
            print(f"  Max speedup: {max_speedup:.2f}x")
            print(f"  Average throughput: {avg_throughput:.2f} GFLOPS")
        
        # Overall
        all_speedups = [r.speedup for r in self.results]
        print(f"\nOVERALL AVERAGE SPEEDUP: {np.mean(all_speedups):.2f}x")
        print("="*70 + "\n")
    
    def save_results(self, filename: str = "gpu_benchmark_results.json"):
        """Save results to JSON"""
        output = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'gpu_info': self._get_gpu_info(),
            'results': [r.to_dict() for r in self.results]
        }
        
        with open(filename, 'w') as f:
            json.dump(output, f, indent=2)
        
        print(f"\nResults saved to: {filename}")
    
    def _get_gpu_info(self) -> Dict:
        """Get GPU information"""
        try:
            import torch
            props = torch.cuda.get_device_properties(0)
            
            return {
                'name': props.name,
                'compute_capability': f"{props.major}.{props.minor}",
                'total_memory_gb': props.total_memory / 1e9,
                'cuda_version': torch.version.cuda
            }
        except:
            return {}


def run_quick_benchmark():
    """Run a quick benchmark for testing"""
    print("\nRunning quick GPU benchmark...")
    
    benchmark = GPUBenchmark(warmup_runs=1, benchmark_runs=3)
    
    # Quick tests
    benchmark.benchmark_matrix_multiply(sizes=[(512, 512, 512)])
    benchmark.benchmark_differential_privacy(sizes=[(1000, 100)])
    
    benchmark.print_summary()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--quick":
        run_quick_benchmark()
    else:
        benchmark = GPUBenchmark(warmup_runs=3, benchmark_runs=10)
        benchmark.run_all_benchmarks()
