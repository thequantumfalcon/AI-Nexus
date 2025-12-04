"""
Model Optimization Module
=========================

Quantization, pruning, distillation, and acceleration techniques.
"""

import torch
import torch.nn as nn
import torch.quantization as quant
import numpy as np
from typing import Optional, Dict, Any
import copy

from core.logger import get_logger

logger = get_logger(__name__)


class ModelOptimizer:
    """
    Model optimization for faster inference and smaller models
    """
    
    def __init__(self):
        self.quantization_schemes = {
            'dynamic': self.dynamic_quantization,
            'static': self.static_quantization,
            'qat': self.quantization_aware_training
        }
    
    def dynamic_quantization(
        self,
        model: nn.Module,
        dtype: torch.dtype = torch.qint8
    ) -> nn.Module:
        """
        Dynamic quantization - quantize weights, activations stay FP32
        Best for LSTM/RNN models, 2-3x speedup
        """
        logger.info("Applying dynamic quantization")
        
        quantized_model = quant.quantize_dynamic(
            model,
            {nn.Linear, nn.LSTM, nn.GRU},
            dtype=dtype
        )
        
        logger.info("Dynamic quantization complete")
        return quantized_model
    
    def static_quantization(
        self,
        model: nn.Module,
        calibration_data: torch.Tensor,
        dtype: torch.dtype = torch.qint8
    ) -> nn.Module:
        """
        Static quantization - quantize both weights and activations
        Requires calibration data, 4x speedup possible
        """
        logger.info("Applying static quantization")
        
        # Prepare model
        model.eval()
        model.qconfig = quant.get_default_qconfig('x86')
        quant.prepare(model, inplace=True)
        
        # Calibrate
        with torch.no_grad():
            model(calibration_data)
        
        # Convert
        quantized_model = quant.convert(model, inplace=False)
        
        logger.info("Static quantization complete")
        return quantized_model
    
    def quantization_aware_training(
        self,
        model: nn.Module,
        train_loader: Any,
        epochs: int = 5
    ) -> nn.Module:
        """
        Quantization-aware training - train with quantization in mind
        Best accuracy retention, 4x speedup
        """
        logger.info("Starting quantization-aware training")
        
        model.train()
        model.qconfig = quant.get_default_qat_qconfig('x86')
        quant.prepare_qat(model, inplace=True)
        
        # Training loop (simplified)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        for epoch in range(epochs):
            for batch in train_loader:
                optimizer.zero_grad()
                # Training step would go here
                pass
        
        model.eval()
        quantized_model = quant.convert(model, inplace=False)
        
        logger.info("QAT complete")
        return quantized_model
    
    def prune_model(
        self,
        model: nn.Module,
        amount: float = 0.3,
        method: str = 'l1_unstructured'
    ) -> nn.Module:
        """
        Prune model to reduce parameters
        
        Args:
            amount: Fraction of parameters to prune (0.3 = 30% pruned)
            method: 'l1_unstructured' or 'random'
        """
        logger.info(f"Pruning model: {amount*100}% of parameters")
        
        import torch.nn.utils.prune as prune
        
        # Prune all Linear and Conv2d layers
        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                if method == 'l1_unstructured':
                    prune.l1_unstructured(module, name='weight', amount=amount)
                else:
                    prune.random_unstructured(module, name='weight', amount=amount)
                
                # Make pruning permanent
                prune.remove(module, 'weight')
        
        logger.info("Pruning complete")
        return model
    
    def knowledge_distillation(
        self,
        teacher_model: nn.Module,
        student_model: nn.Module,
        train_loader: Any,
        temperature: float = 3.0,
        alpha: float = 0.7,
        epochs: int = 10
    ) -> nn.Module:
        """
        Knowledge distillation - train smaller model to mimic larger one
        
        Args:
            teacher_model: Large pre-trained model
            student_model: Smaller model to train
            temperature: Softmax temperature for soft targets
            alpha: Weight for distillation loss vs hard label loss
        """
        logger.info("Starting knowledge distillation")
        
        teacher_model.eval()
        student_model.train()
        
        optimizer = torch.optim.Adam(student_model.parameters(), lr=0.001)
        kl_div = nn.KLDivLoss(reduction='batchmean')
        ce_loss = nn.CrossEntropyLoss()
        
        for epoch in range(epochs):
            for batch_idx, (data, target) in enumerate(train_loader):
                optimizer.zero_grad()
                
                # Teacher predictions (soft targets)
                with torch.no_grad():
                    teacher_logits = teacher_model(data)
                    soft_targets = nn.functional.softmax(teacher_logits / temperature, dim=1)
                
                # Student predictions
                student_logits = student_model(data)
                soft_predictions = nn.functional.log_softmax(student_logits / temperature, dim=1)
                
                # Distillation loss
                distill_loss = kl_div(soft_predictions, soft_targets) * (temperature ** 2)
                
                # Hard label loss
                hard_loss = ce_loss(student_logits, target)
                
                # Combined loss
                loss = alpha * distill_loss + (1 - alpha) * hard_loss
                
                loss.backward()
                optimizer.step()
        
        logger.info("Knowledge distillation complete")
        return student_model
    
    def mixed_precision_convert(self, model: nn.Module) -> nn.Module:
        """
        Convert model to use mixed precision (FP16 + FP32)
        Requires NVIDIA GPU with Tensor Cores
        """
        logger.info("Converting to mixed precision")
        
        model = model.half()  # Convert to FP16
        
        # Keep certain ops in FP32 for numerical stability
        for module in model.modules():
            if isinstance(module, (nn.BatchNorm2d, nn.LayerNorm)):
                module.float()
        
        logger.info("Mixed precision conversion complete")
        return model
    
    def fuse_modules(self, model: nn.Module) -> nn.Module:
        """
        Fuse consecutive operations (Conv+BN+ReLU) for faster inference
        """
        logger.info("Fusing modules")
        
        # Example: fuse conv+bn+relu
        fused_model = quant.fuse_modules(
            model,
            [['conv', 'bn', 'relu']],  # Adjust based on actual model structure
            inplace=False
        )
        
        logger.info("Module fusion complete")
        return fused_model
    
    def export_to_onnx(
        self,
        model: nn.Module,
        sample_input: torch.Tensor,
        output_path: str,
        opset_version: int = 14
    ):
        """
        Export model to ONNX format for deployment
        """
        logger.info(f"Exporting to ONNX: {output_path}")
        
        model.eval()
        torch.onnx.export(
            model,
            sample_input,
            output_path,
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        )
        
        logger.info("ONNX export complete")
    
    def get_model_size(self, model: nn.Module) -> Dict[str, Any]:
        """Get model statistics - handles both regular and quantized models"""
        param_count = 0
        param_size = 0
        trainable = 0
        
        # Check if quantized model
        is_quantized = any('quantized' in str(type(m)).lower() for m in model.modules())
        
        if is_quantized:
            # For quantized models, count packed parameters
            for module in model.modules():
                if hasattr(module, '_packed_params'):
                    # Quantized linear layers have packed params
                    try:
                        weight, bias = module._weight_bias()
                        param_count += weight.numel()
                        param_size += weight.numel() * weight.element_size()
                        if bias is not None:
                            param_count += bias.numel()
                            param_size += bias.numel() * bias.element_size()
                    except:
                        pass
                # Also check regular parameters
                for param in module.parameters(recurse=False):
                    param_count += param.numel()
                    param_size += param.numel() * param.element_size()
                    if param.requires_grad:
                        trainable += param.numel()
        else:
            # Regular model
            for param in model.parameters():
                param_count += param.numel()
                param_size += param.numel() * param.element_size()
                if param.requires_grad:
                    trainable += param.numel()
        
        param_size = param_size / 1024 / 1024  # Convert to MB
        
        return {
            'parameters': param_count,
            'size_mb': param_size,
            'trainable': trainable
        }
    
    def benchmark_inference(
        self,
        model: nn.Module,
        input_shape: tuple,
        iterations: int = 100,
        warmup: int = 10
    ) -> Dict[str, float]:
        """
        Benchmark inference speed
        """
        import time
        
        model.eval()
        device = next(model.parameters()).device
        dummy_input = torch.randn(input_shape).to(device)
        
        # Warmup
        with torch.no_grad():
            for _ in range(warmup):
                _ = model(dummy_input)
        
        # Benchmark
        times = []
        with torch.no_grad():
            for _ in range(iterations):
                start = time.time()
                _ = model(dummy_input)
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                times.append(time.time() - start)
        
        times = np.array(times) * 1000  # Convert to ms
        
        return {
            'mean_ms': float(np.mean(times)),
            'std_ms': float(np.std(times)),
            'min_ms': float(np.min(times)),
            'max_ms': float(np.max(times)),
            'p50_ms': float(np.percentile(times, 50)),
            'p95_ms': float(np.percentile(times, 95)),
            'p99_ms': float(np.percentile(times, 99))
        }
