"""
AI-Nexus GPU Acceleration Module
Custom CUDA kernels for 100x performance boost
"""

from .cuda_compiler import CUDACompiler, compile_kernel
from .gpu_manager import GPUManager, get_gpu_info
from .kernels import (
    MatrixMultiplyKernel,
    ConvolutionKernel,
    AttentionKernel,
    PrivacyKernel,
    HomomorphicKernel
)

__all__ = [
    'CUDACompiler',
    'compile_kernel',
    'GPUManager',
    'get_gpu_info',
    'MatrixMultiplyKernel',
    'ConvolutionKernel',
    'AttentionKernel',
    'PrivacyKernel',
    'HomomorphicKernel'
]
