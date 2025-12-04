"""
Tests for GPU acceleration modules
"""

import pytest
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Check if GPU is available
CUDA_AVAILABLE = False
try:
    import torch
    CUDA_AVAILABLE = torch.cuda.is_available()
except ImportError:
    pass

skip_if_no_cuda = pytest.mark.skipif(
    not CUDA_AVAILABLE,
    reason="CUDA not available. Install PyTorch with CUDA: pip install torch --index-url https://download.pytorch.org/whl/cu121"
)


class TestGPUManager:
    """Test GPU manager functionality"""
    
    @skip_if_no_cuda
    def test_gpu_initialization(self):
        """Test GPU manager initialization"""
        from services.gpu.gpu_manager import GPUManager
        
        manager = GPUManager(device_id=0)
        assert manager.device_id == 0
        assert manager.device is not None
    
    @skip_if_no_cuda
    def test_get_gpu_info(self):
        """Test GPU information retrieval"""
        from services.gpu.gpu_manager import GPUManager, get_gpu_info
        
        info = get_gpu_info(device_id=0)
        
        assert info.id == 0
        assert info.name is not None
        assert info.total_memory > 0
        assert info.compute_capability[0] >= 5  # Minimum compute capability
        
        print(f"\nGPU: {info.name}")
        print(f"Memory: {info.total_memory / 1e9:.2f}GB")
        print(f"Compute: {info.compute_capability}")
    
    @skip_if_no_cuda
    def test_memory_management(self):
        """Test GPU memory allocation and clearing"""
        from services.gpu.gpu_manager import GPUManager
        
        manager = GPUManager()
        
        # Get initial memory
        initial_memory = manager.monitor_memory()
        
        # Allocate memory
        size_mb = 100
        tensor = manager.allocate_memory(size_mb * 1024 * 1024)
        
        # Check allocation
        current_memory = manager.monitor_memory()
        assert current_memory['allocated'] > initial_memory['allocated']
        
        # Clear cache
        del tensor
        manager.clear_cache()
        
        # Verify cleanup
        final_memory = manager.monitor_memory()
        assert final_memory['allocated'] <= initial_memory['allocated'] + 1e6  # Some tolerance
    
    @skip_if_no_cuda
    def test_memory_bandwidth(self):
        """Test memory bandwidth benchmark"""
        from services.gpu.gpu_manager import GPUManager
        
        manager = GPUManager()
        results = manager.benchmark_bandwidth(size_mb=128)
        
        # Check results
        assert results['size_mb'] == 128
        assert results['host_to_device_gbps'] > 0
        assert results['device_to_host_gbps'] > 0
        assert results['device_to_device_gbps'] > 0
        
        # RTX 5080 should have high bandwidth (>500 GB/s for D2D)
        print(f"\nMemory Bandwidth:")
        print(f"  H2D: {results['host_to_device_gbps']:.2f} GB/s")
        print(f"  D2H: {results['device_to_host_gbps']:.2f} GB/s")
        print(f"  D2D: {results['device_to_device_gbps']:.2f} GB/s")


class TestCUDAKernels:
    """Test CUDA kernel implementations"""
    
    @skip_if_no_cuda
    def test_matrix_multiply_kernel(self):
        """Test GPU matrix multiplication"""
        from services.gpu.kernels import MatrixMultiplyKernel
        
        kernel = MatrixMultiplyKernel(use_tensor_cores=True)
        
        # Test small matrices
        M, K, N = 128, 64, 256
        A = np.random.randn(M, K).astype(np.float32)
        B = np.random.randn(K, N).astype(np.float32)
        
        # GPU computation
        C_gpu = kernel.forward(A, B, use_mixed_precision=False)
        
        # CPU reference
        C_cpu = np.matmul(A, B)
        
        # Verify correctness
        error = np.max(np.abs(C_gpu - C_cpu))
        assert error < 1e-3, f"Matrix multiply error too large: {error}"
        
        print(f"\nMatrix Multiply ({M}x{K} @ {K}x{N}):")
        print(f"  Max error: {error:.2e}")
    
    @skip_if_no_cuda
    def test_batch_matrix_multiply(self):
        """Test batched matrix multiplication"""
        from services.gpu.kernels import MatrixMultiplyKernel
        
        kernel = MatrixMultiplyKernel()
        
        # Test batched operation
        batch, M, K, N = 16, 32, 32, 32
        A = np.random.randn(batch, M, K).astype(np.float32)
        B = np.random.randn(batch, K, N).astype(np.float32)
        
        C_gpu = kernel.batch_multiply(A, B)
        C_cpu = np.matmul(A, B)
        
        error = np.max(np.abs(C_gpu - C_cpu))
        assert error < 1e-3
        
        print(f"\nBatch Matrix Multiply ({batch}×{M}×{K} @ {batch}×{K}×{N}):")
        print(f"  Max error: {error:.2e}")
    
    @skip_if_no_cuda
    def test_convolution_kernel(self):
        """Test GPU 2D convolution"""
        from services.gpu.kernels import ConvolutionKernel
        
        kernel = ConvolutionKernel()
        
        # Test convolution
        batch, in_ch, H, W = 4, 3, 32, 32
        out_ch, kH, kW = 16, 3, 3
        
        input_data = np.random.randn(batch, in_ch, H, W).astype(np.float32)
        weight = np.random.randn(out_ch, in_ch, kH, kW).astype(np.float32)
        bias = np.random.randn(out_ch).astype(np.float32)
        
        output = kernel.forward(input_data, weight, bias, stride=1, padding=1)
        
        # Check output shape
        expected_shape = (batch, out_ch, H, W)  # Same size with padding=1
        assert output.shape == expected_shape
        
        print(f"\n2D Convolution:")
        print(f"  Input: {input_data.shape}")
        print(f"  Output: {output.shape}")
    
    @skip_if_no_cuda
    def test_attention_kernel(self):
        """Test attention computation"""
        from services.gpu.kernels import AttentionKernel
        
        kernel = AttentionKernel(use_flash_attention=False)  # Use standard attention
        
        # Test attention
        batch, seq_len, num_heads, head_dim = 2, 64, 8, 64
        
        Q = np.random.randn(batch, seq_len, num_heads, head_dim).astype(np.float32)
        K = np.random.randn(batch, seq_len, num_heads, head_dim).astype(np.float32)
        V = np.random.randn(batch, seq_len, num_heads, head_dim).astype(np.float32)
        
        output = kernel.forward(Q, K, V)
        
        # Check shape
        assert output.shape == Q.shape
        
        print(f"\nAttention:")
        print(f"  Input: {Q.shape}")
        print(f"  Output: {output.shape}")


class TestPrivacyGPU:
    """Test GPU-accelerated privacy modules"""
    
    @skip_if_no_cuda
    def test_differential_privacy_gpu(self):
        """Test GPU differential privacy"""
        from services.gpu.privacy_gpu import GPUDifferentialPrivacy
        
        dp = GPUDifferentialPrivacy(epsilon=1.0, delta=1e-5)
        
        # Test gradient privatization
        gradients = np.random.randn(32, 1000).astype(np.float32)
        
        privatized = dp.privatize_gradients(
            gradients,
            clip_norm=1.0,
            noise_multiplier=1.0,
            batch_size=32
        )
        
        # Check shape preserved
        assert privatized.shape == gradients.shape
        
        # Check noise was added
        diff = np.abs(privatized - gradients)
        assert np.mean(diff) > 0, "No noise was added"
        
        print(f"\nDifferential Privacy:")
        print(f"  Input shape: {gradients.shape}")
        print(f"  Mean noise: {np.mean(diff):.4f}")
        print(f"  Budget: ε={dp.epsilon}, δ={dp.delta}")
    
    @skip_if_no_cuda
    def test_privacy_accounting(self):
        """Test privacy loss accounting"""
        from services.gpu.privacy_gpu import GPUDifferentialPrivacy
        
        dp = GPUDifferentialPrivacy(epsilon=10.0, delta=1e-5)
        
        # Compute privacy loss for training
        eps, delta = dp.compute_privacy_loss(
            epochs=10,
            batch_size=128,
            dataset_size=10000,
            noise_multiplier=1.0
        )
        
        # Should be finite and reasonable
        assert eps > 0 and eps < 100
        assert delta == 1e-5
        
        print(f"\nPrivacy Accounting:")
        print(f"  Epochs: 10")
        print(f"  Final ε: {eps:.3f}")
        print(f"  Final δ: {delta:.2e}")
    
    @skip_if_no_cuda
    def test_privacy_kernel_laplace(self):
        """Test Laplace noise generation"""
        from services.gpu.kernels import PrivacyKernel
        
        kernel = PrivacyKernel()
        
        data = np.random.randn(1000, 100).astype(np.float32)
        noisy_data = kernel.add_laplace_noise(data, epsilon=1.0, sensitivity=1.0)
        
        # Check noise distribution
        noise = noisy_data - data
        
        # Laplace noise should have higher kurtosis than Gaussian
        assert noisy_data.shape == data.shape
        
        print(f"\nLaplace Noise:")
        print(f"  Mean: {np.mean(noise):.4f}")
        print(f"  Std: {np.std(noise):.4f}")
    
    @skip_if_no_cuda
    def test_secure_aggregation_gpu(self):
        """Test GPU secure aggregation"""
        from services.gpu.privacy_gpu import GPUSecureMultiPartyComputation
        
        smpc = GPUSecureMultiPartyComputation(num_parties=10)
        
        # Create client updates
        num_clients = 10
        model_size = 1000
        
        client_updates = [
            np.random.randn(model_size).astype(np.float32)
            for _ in range(num_clients)
        ]
        
        # Aggregate
        aggregated = smpc.secure_aggregate(client_updates)
        
        # Check result
        assert aggregated.shape == (model_size,)
        
        # Should be close to mean
        expected = np.mean(client_updates, axis=0)
        error = np.max(np.abs(aggregated - expected))
        assert error < 1e-5
        
        print(f"\nSecure Aggregation:")
        print(f"  Clients: {num_clients}")
        print(f"  Model size: {model_size}")
        print(f"  Max error: {error:.2e}")
    
    @skip_if_no_cuda
    def test_shamir_secret_sharing(self):
        """Test Shamir secret sharing on GPU"""
        from services.gpu.privacy_gpu import GPUSecureMultiPartyComputation
        
        smpc = GPUSecureMultiPartyComputation(num_parties=5)
        
        # Share a secret
        secret = np.array([42, 100, 256], dtype=np.int64)
        threshold = 3
        
        shares = smpc.shamir_share(secret, threshold=threshold)
        
        # Check we got correct number of shares
        assert len(shares) == 5
        
        # Each share should have same shape as secret
        for party_id, share in shares:
            assert share.shape == secret.shape
            assert party_id >= 1 and party_id <= 5
        
        print(f"\nShamir Secret Sharing:")
        print(f"  Secret: {secret}")
        print(f"  Parties: {len(shares)}")
        print(f"  Threshold: {threshold}")


class TestGPUIntegration:
    """Integration tests for GPU modules"""
    
    @skip_if_no_cuda
    def test_end_to_end_training(self):
        """Test GPU-accelerated training pipeline"""
        from services.gpu.kernels import MatrixMultiplyKernel
        from services.gpu.privacy_gpu import GPUDifferentialPrivacy
        
        # Setup
        batch_size = 64
        input_dim = 100
        output_dim = 10
        
        # Create simple model (linear layer)
        W = np.random.randn(input_dim, output_dim).astype(np.float32)
        b = np.random.randn(output_dim).astype(np.float32)
        
        # Forward pass on GPU
        X = np.random.randn(batch_size, input_dim).astype(np.float32)
        
        kernel = MatrixMultiplyKernel()
        logits = kernel.forward(X, W)  # [batch, output_dim]
        
        # Compute gradients (simplified)
        grad_W = np.random.randn(*W.shape).astype(np.float32)
        
        # Privatize gradients
        dp = GPUDifferentialPrivacy(epsilon=1.0, delta=1e-5)
        private_grad = dp.privatize_gradients(
            grad_W.reshape(1, -1),
            clip_norm=1.0
        )
        
        # Verify shapes
        assert logits.shape == (batch_size, output_dim)
        assert private_grad.size == grad_W.size
        
        print(f"\nEnd-to-End Training:")
        print(f"  Input: {X.shape}")
        print(f"  Logits: {logits.shape}")
        print(f"  Private gradients: {private_grad.shape}")
    
    @skip_if_no_cuda  
    def test_federated_learning_gpu(self):
        """Test GPU-accelerated federated learning"""
        from services.gpu.privacy_gpu import GPUSecureMultiPartyComputation, GPUDifferentialPrivacy
        
        # Simulate federated learning round
        num_clients = 10
        model_size = 1000
        
        # Each client computes update
        client_updates = []
        dp = GPUDifferentialPrivacy(epsilon=1.0, delta=1e-5)
        
        for _ in range(num_clients):
            # Simulate gradient
            grad = np.random.randn(model_size).astype(np.float32)
            
            # Privatize
            private_grad = dp.privatize_gradients(
                grad.reshape(1, -1),
                clip_norm=1.0
            ).flatten()
            
            client_updates.append(private_grad)
        
        # Secure aggregation
        smpc = GPUSecureMultiPartyComputation(num_parties=num_clients)
        aggregated = smpc.secure_aggregate(client_updates)
        
        # Verify result
        assert aggregated.shape == (model_size,)
        
        print(f"\nFederated Learning Round:")
        print(f"  Clients: {num_clients}")
        print(f"  Model size: {model_size}")
        print(f"  Aggregated update: {aggregated.shape}")


@pytest.mark.benchmark
class TestGPUPerformance:
    """Performance benchmarks"""
    
    @skip_if_no_cuda
    def test_matrix_multiply_speedup(self):
        """Benchmark matrix multiplication speedup"""
        import time
        from services.gpu.kernels import MatrixMultiplyKernel
        
        # Large matrices
        M, K, N = 2048, 2048, 2048
        A = np.random.randn(M, K).astype(np.float32)
        B = np.random.randn(K, N).astype(np.float32)
        
        # CPU
        start = time.perf_counter()
        C_cpu = np.matmul(A, B)
        cpu_time = time.perf_counter() - start
        
        # GPU
        kernel = MatrixMultiplyKernel()
        
        # Warmup
        for _ in range(3):
            kernel.forward(A, B)
        
        start = time.perf_counter()
        C_gpu = kernel.forward(A, B)
        gpu_time = time.perf_counter() - start
        
        speedup = cpu_time / gpu_time
        
        print(f"\nMatrix Multiply Benchmark ({M}×{K}×{N}):")
        print(f"  CPU: {cpu_time*1000:.2f}ms")
        print(f"  GPU: {gpu_time*1000:.2f}ms")
        print(f"  Speedup: {speedup:.2f}x")
        
        # Should see significant speedup on GPU
        assert speedup > 5.0, f"Expected >5x speedup, got {speedup:.2f}x"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
