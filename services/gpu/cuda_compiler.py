"""
CUDA Kernel Compiler and Runtime
Compiles and executes custom CUDA kernels for AI-Nexus
"""

import os
import subprocess
import tempfile
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class CUDACompiler:
    """
    Custom CUDA compiler for dynamic kernel compilation
    Supports CUDA 13.0+ with RTX 50-series optimizations
    """
    
    def __init__(
        self,
        cuda_path: Optional[str] = None,
        compute_capability: str = "8.9",  # RTX 5080 (Ada Lovelace)
        optimization_level: str = "O3",
        use_fast_math: bool = True,
        cache_dir: Optional[str] = None
    ):
        self.cuda_path = cuda_path or self._find_cuda_path()
        self.compute_capability = compute_capability
        self.optimization_level = optimization_level
        self.use_fast_math = use_fast_math
        self.cache_dir = Path(cache_dir or tempfile.gettempdir()) / "ai_nexus_cuda_cache"
        self.cache_dir.mkdir(exist_ok=True, parents=True)
        
        self.nvcc = self._find_nvcc()
        self._validate_cuda_environment()
        
        logger.info(f"CUDA Compiler initialized: {self.nvcc}")
        logger.info(f"Compute capability: {self.compute_capability}")
        logger.info(f"Cache directory: {self.cache_dir}")
    
    def _find_cuda_path(self) -> str:
        """Find CUDA installation path"""
        # Common CUDA installation paths
        cuda_paths = [
            os.environ.get("CUDA_PATH"),
            os.environ.get("CUDA_HOME"),
            "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0",
            "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.6",
            "/usr/local/cuda",
            "/opt/cuda"
        ]
        
        for path in cuda_paths:
            if path and os.path.exists(path):
                return path
        
        raise RuntimeError(
            "CUDA installation not found. Please set CUDA_PATH environment variable "
            "or install CUDA toolkit from https://developer.nvidia.com/cuda-downloads"
        )
    
    def _find_nvcc(self) -> str:
        """Find nvcc compiler executable"""
        if os.name == 'nt':  # Windows
            nvcc_paths = [
                os.path.join(self.cuda_path, "bin", "nvcc.exe"),
                "nvcc.exe"
            ]
        else:  # Linux/Mac
            nvcc_paths = [
                os.path.join(self.cuda_path, "bin", "nvcc"),
                "/usr/local/cuda/bin/nvcc",
                "nvcc"
            ]
        
        for nvcc in nvcc_paths:
            try:
                result = subprocess.run(
                    [nvcc, "--version"],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                if result.returncode == 0:
                    return nvcc
            except (FileNotFoundError, subprocess.TimeoutExpired):
                continue
        
        raise RuntimeError(
            f"nvcc compiler not found. Please add CUDA bin directory to PATH: {self.cuda_path}/bin"
        )
    
    def _validate_cuda_environment(self):
        """Validate CUDA environment and dependencies"""
        try:
            # Check nvcc version
            result = subprocess.run(
                [self.nvcc, "--version"],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            if "release 13" not in result.stdout and "release 12" not in result.stdout:
                logger.warning(
                    f"CUDA version may not be optimal. Expected 12.x or 13.x\n{result.stdout}"
                )
            
            logger.info(f"CUDA environment validated:\n{result.stdout}")
            
        except Exception as e:
            raise RuntimeError(f"Failed to validate CUDA environment: {e}")
    
    def compile_kernel(
        self,
        kernel_code: str,
        kernel_name: str,
        includes: Optional[List[str]] = None,
        extra_flags: Optional[List[str]] = None,
        force_recompile: bool = False
    ) -> str:
        """
        Compile CUDA kernel code to PTX or cubin
        
        Args:
            kernel_code: CUDA C++ source code
            kernel_name: Name of the kernel (for caching)
            includes: Additional include directories
            extra_flags: Extra compiler flags
            force_recompile: Force recompilation even if cached
        
        Returns:
            Path to compiled kernel (.ptx or .cubin)
        """
        # Generate cache key from kernel code
        code_hash = hashlib.sha256(kernel_code.encode()).hexdigest()[:16]
        cache_key = f"{kernel_name}_{code_hash}"
        
        # Check cache
        cached_ptx = self.cache_dir / f"{cache_key}.ptx"
        if cached_ptx.exists() and not force_recompile:
            logger.info(f"Using cached kernel: {cached_ptx}")
            return str(cached_ptx)
        
        # Write kernel source to temporary file
        with tempfile.NamedTemporaryFile(
            mode='w',
            suffix='.cu',
            delete=False,
            dir=self.cache_dir
        ) as f:
            f.write(kernel_code)
            cu_file = f.name
        
        try:
            # Build compiler command
            cmd = [
                self.nvcc,
                f"-arch=sm_{self.compute_capability.replace('.', '')}",
                f"-{self.optimization_level}",
                "--ptx",  # Generate PTX (portable)
                cu_file,
                "-o", str(cached_ptx)
            ]
            
            if self.use_fast_math:
                cmd.append("--use_fast_math")
            
            # Add GPU-specific optimizations for RTX 5080
            cmd.extend([
                "--gpu-architecture=sm_89",  # Ada Lovelace
                "--gpu-code=sm_89",
                "-allow-unsupported-compiler",  # For newer MSVC versions
                "-Xptxas=-v",  # Verbose PTX assembly
                "-Xcompiler=/openmp",  # OpenMP support on Windows
            ])
            
            if includes:
                for include in includes:
                    cmd.extend(["-I", include])
            
            if extra_flags:
                cmd.extend(extra_flags)
            
            # Compile kernel
            logger.info(f"Compiling kernel: {kernel_name}")
            logger.debug(f"Command: {' '.join(cmd)}")
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=60
            )
            
            if result.returncode != 0:
                raise RuntimeError(
                    f"Kernel compilation failed:\n"
                    f"STDOUT: {result.stdout}\n"
                    f"STDERR: {result.stderr}"
                )
            
            logger.info(f"Kernel compiled successfully: {cached_ptx}")
            logger.debug(f"Compiler output:\n{result.stdout}")
            
            return str(cached_ptx)
            
        finally:
            # Cleanup source file
            try:
                os.unlink(cu_file)
            except:
                pass
    
    def get_kernel_info(self, ptx_file: str) -> Dict:
        """Extract information from compiled PTX"""
        with open(ptx_file, 'r') as f:
            ptx_content = f.read()
        
        info = {
            'target': None,
            'address_size': None,
            'kernels': [],
            'size_bytes': os.path.getsize(ptx_file)
        }
        
        # Parse PTX header
        for line in ptx_content.split('\n'):
            if '.target' in line:
                info['target'] = line.split()[1]
            elif '.address_size' in line:
                info['address_size'] = int(line.split()[1])
            elif '.entry' in line or '.visible .entry' in line:
                kernel_name = line.split('(')[0].split()[-1]
                info['kernels'].append(kernel_name)
        
        return info


def compile_kernel(
    kernel_code: str,
    kernel_name: str,
    **kwargs
) -> str:
    """
    Convenience function to compile a kernel
    Creates a CUDACompiler instance and compiles the kernel
    """
    compiler = CUDACompiler()
    return compiler.compile_kernel(kernel_code, kernel_name, **kwargs)


# Example CUDA kernel templates
KERNEL_TEMPLATES = {
    'matrix_multiply': '''
extern "C" __global__ void matrix_multiply(
    const float* A, 
    const float* B, 
    float* C,
    int M, int N, int K
) {
    // Shared memory for tiling
    __shared__ float As[32][32];
    __shared__ float Bs[32][32];
    
    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;
    
    int row = by * 32 + ty;
    int col = bx * 32 + tx;
    
    float sum = 0.0f;
    
    // Tile the matrices
    for (int t = 0; t < (K + 31) / 32; t++) {
        // Load tiles into shared memory
        if (row < M && t * 32 + tx < K)
            As[ty][tx] = A[row * K + t * 32 + tx];
        else
            As[ty][tx] = 0.0f;
        
        if (t * 32 + ty < K && col < N)
            Bs[ty][tx] = B[(t * 32 + ty) * N + col];
        else
            Bs[ty][tx] = 0.0f;
        
        __syncthreads();
        
        // Compute partial dot product
        #pragma unroll
        for (int k = 0; k < 32; k++) {
            sum += As[ty][k] * Bs[k][tx];
        }
        
        __syncthreads();
    }
    
    // Write result
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}
''',
    
    'relu_activation': '''
extern "C" __global__ void relu_activation(
    const float* input,
    float* output,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < size) {
        output[idx] = fmaxf(0.0f, input[idx]);
    }
}
''',
    
    'attention_weights': '''
extern "C" __global__ void attention_weights(
    const float* Q,  // Query [batch, seq_len, dim]
    const float* K,  // Key [batch, seq_len, dim]
    float* scores,   // Output [batch, seq_len, seq_len]
    int batch_size,
    int seq_len,
    int dim,
    float scale
) {
    int b = blockIdx.z;
    int i = blockIdx.y * blockDim.y + threadIdx.y;  // Query position
    int j = blockIdx.x * blockDim.x + threadIdx.x;  // Key position
    
    if (b < batch_size && i < seq_len && j < seq_len) {
        float score = 0.0f;
        
        // Compute dot product Q[i] · K[j]
        int q_offset = b * seq_len * dim + i * dim;
        int k_offset = b * seq_len * dim + j * dim;
        
        #pragma unroll
        for (int d = 0; d < dim; d++) {
            score += Q[q_offset + d] * K[k_offset + d];
        }
        
        scores[b * seq_len * seq_len + i * seq_len + j] = score * scale;
    }
}
'''
}


if __name__ == "__main__":
    # Test compilation
    compiler = CUDACompiler()
    
    # Compile matrix multiply kernel
    ptx_file = compiler.compile_kernel(
        KERNEL_TEMPLATES['matrix_multiply'],
        'matrix_multiply'
    )
    
    print(f"Compiled kernel: {ptx_file}")
    print(f"Kernel info: {compiler.get_kernel_info(ptx_file)}")
