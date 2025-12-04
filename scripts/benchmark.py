"""
Comprehensive Benchmarking Suite
=================================

Performance profiling and stress testing for all AI-Nexus components.
"""

import time
import numpy as np
import psutil
import torch
from typing import Dict, List, Any, Callable
from dataclasses import dataclass, asdict
import json
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import threading

from services.ml.ml_engine import PrivacyPreservingMLEngine
from services.nlp.nlp_engine import SecureNLPEngine
from services.blockchain.blockchain import AIBlockchain
from core.crypto import (
    EncryptionManager,
    DifferentialPrivacy,
    HomomorphicEncryption,
    ZeroKnowledgeProof
)
from core.logger import get_logger

logger = get_logger(__name__)


@dataclass
class BenchmarkResult:
    """Benchmark result with detailed metrics"""
    component: str
    operation: str
    mean_time_ms: float
    std_time_ms: float
    min_time_ms: float
    max_time_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    throughput_ops_sec: float
    memory_mb: float
    cpu_percent: float
    iterations: int
    success_rate: float


class Benchmarker:
    """
    Comprehensive benchmarking for all AI-Nexus components
    """
    
    def __init__(self, warmup_iterations: int = 10, benchmark_iterations: int = 100):
        self.warmup_iterations = warmup_iterations
        self.benchmark_iterations = benchmark_iterations
        self.results: List[BenchmarkResult] = []
    
    def benchmark_function(
        self,
        func: Callable,
        component: str,
        operation: str,
        *args,
        **kwargs
    ) -> BenchmarkResult:
        """
        Benchmark a single function
        """
        logger.info(f"Benchmarking {component}/{operation}")
        
        # Warmup
        for _ in range(self.warmup_iterations):
            try:
                func(*args, **kwargs)
            except Exception as e:
                logger.warning(f"Warmup failed: {e}")
        
        # Actual benchmark
        times = []
        successes = 0
        mem_usage = []
        cpu_usage = []
        
        process = psutil.Process()
        
        for _ in range(self.benchmark_iterations):
            # Measure memory and CPU before
            mem_before = process.memory_info().rss / 1024 / 1024  # MB
            cpu_before = process.cpu_percent(interval=None)
            
            start = time.perf_counter()
            try:
                func(*args, **kwargs)
                successes += 1
            except Exception as e:
                logger.warning(f"Benchmark iteration failed: {e}")
            end = time.perf_counter()
            
            # Measure after
            mem_after = process.memory_info().rss / 1024 / 1024  # MB
            cpu_after = process.cpu_percent(interval=None)
            
            times.append((end - start) * 1000)  # Convert to ms
            mem_usage.append(mem_after - mem_before)
            cpu_usage.append(cpu_after - cpu_before)
        
        # Calculate statistics
        times_arr = np.array(times)
        
        result = BenchmarkResult(
            component=component,
            operation=operation,
            mean_time_ms=float(np.mean(times_arr)),
            std_time_ms=float(np.std(times_arr)),
            min_time_ms=float(np.min(times_arr)),
            max_time_ms=float(np.max(times_arr)),
            p50_ms=float(np.percentile(times_arr, 50)),
            p95_ms=float(np.percentile(times_arr, 95)),
            p99_ms=float(np.percentile(times_arr, 99)),
            throughput_ops_sec=1000.0 / float(np.mean(times_arr)) if np.mean(times_arr) > 0 else 0,
            memory_mb=float(np.mean(mem_usage)),
            cpu_percent=float(np.mean(cpu_usage)),
            iterations=self.benchmark_iterations,
            success_rate=successes / self.benchmark_iterations
        )
        
        self.results.append(result)
        return result
    
    def benchmark_ml_engine(self) -> List[BenchmarkResult]:
        """Benchmark ML engine operations"""
        logger.info("=== Benchmarking ML Engine ===")
        
        engine = PrivacyPreservingMLEngine()
        X = np.random.randn(100, 10)
        y = np.random.randint(0, 2, 100)
        
        # Training benchmarks
        self.benchmark_function(
            engine.train_model,
            "ML", "linear_training",
            "linear", X, y, {'epochs': 5}
        )
        
        self.benchmark_function(
            engine.train_model,
            "ML", "neural_net_training",
            "neural_net", X, y, {'epochs': 3}
        )
        
        # Train a model for prediction benchmarks
        result = engine.train_model("linear", X, y, {'epochs': 5})
        model_id = result.model_id
        
        # Prediction benchmarks
        X_test = np.random.randn(10, 10)
        self.benchmark_function(
            engine.predict,
            "ML", "prediction",
            model_id, X_test
        )
        
        self.benchmark_function(
            engine.predict,
            "ML", "prediction_with_confidence",
            model_id, X_test, return_confidence=True
        )
        
        # Evaluation
        self.benchmark_function(
            engine.evaluate_model,
            "ML", "evaluation",
            model_id, X, y
        )
        
        # Model aggregation (federated learning)
        result2 = engine.train_model("linear", X, y, {'epochs': 5})
        self.benchmark_function(
            engine.aggregate_models,
            "ML", "federated_aggregation",
            [model_id, result2.model_id]
        )
        
        return self.results
    
    def benchmark_nlp_engine(self) -> List[BenchmarkResult]:
        """Benchmark NLP engine operations"""
        logger.info("=== Benchmarking NLP Engine ===")
        
        engine = SecureNLPEngine()
        
        # Sentiment analysis
        text = "This is an amazing product that I really love!"
        self.benchmark_function(
            engine.analyze_sentiment,
            "NLP", "sentiment_analysis",
            text
        )
        
        # Entity extraction
        text_ner = "John Smith works at Microsoft in Seattle, Washington."
        self.benchmark_function(
            engine.extract_entities,
            "NLP", "entity_extraction",
            text_ner
        )
        
        # Text generation
        self.benchmark_function(
            engine.generate_text,
            "NLP", "text_generation",
            "Artificial intelligence will",
            max_length=50
        )
        
        # Full text processing
        self.benchmark_function(
            engine.process_text,
            "NLP", "full_processing",
            text,
            task_type="classification",
            compliance_mode="HIPAA"
        )
        
        return self.results
    
    def benchmark_blockchain(self) -> List[BenchmarkResult]:
        """Benchmark blockchain operations"""
        logger.info("=== Benchmarking Blockchain ===")
        
        blockchain = AIBlockchain(difficulty=2)  # Lower difficulty for benchmarking
        
        # Block creation
        self.benchmark_function(
            blockchain.add_block,
            "Blockchain", "block_creation",
            {"test": "data"}, "test"
        )
        
        # Chain validation
        self.benchmark_function(
            blockchain.is_chain_valid,
            "Blockchain", "chain_validation"
        )
        
        # Token operations
        self.benchmark_function(
            blockchain.token_manager.transfer,
            "Blockchain", "token_transfer",
            "addr1", "addr2", 10
        )
        
        # Governance proposal
        self.benchmark_function(
            blockchain.governance.create_proposal,
            "Blockchain", "create_proposal",
            "test", "Test proposal", {}
        )
        
        return self.results
    
    def benchmark_crypto(self) -> List[BenchmarkResult]:
        """Benchmark cryptography operations"""
        logger.info("=== Benchmarking Cryptography ===")
        
        # Encryption
        encryption = EncryptionManager()
        data = b"sensitive data" * 100
        
        self.benchmark_function(
            encryption.encrypt,
            "Crypto", "symmetric_encryption",
            data
        )
        
        encrypted = encryption.encrypt(data)
        self.benchmark_function(
            encryption.decrypt,
            "Crypto", "symmetric_decryption",
            encrypted
        )
        
        # Asymmetric encryption
        self.benchmark_function(
            encryption.asymmetric_encrypt,
            "Crypto", "asymmetric_encryption",
            data
        )
        
        # Differential privacy
        dp = DifferentialPrivacy(epsilon=1.0, delta=1e-5)
        values = np.random.randn(1000)
        
        self.benchmark_function(
            dp.add_laplace_noise,
            "Crypto", "dp_laplace_noise",
            values, sensitivity=1.0
        )
        
        self.benchmark_function(
            dp.add_gaussian_noise,
            "Crypto", "dp_gaussian_noise",
            values, sensitivity=1.0
        )
        
        # Homomorphic encryption
        he = HomomorphicEncryption()
        value = 42
        
        self.benchmark_function(
            he.encrypt,
            "Crypto", "homomorphic_encryption",
            value
        )
        
        encrypted_val = he.encrypt(value)
        self.benchmark_function(
            he.add,
            "Crypto", "homomorphic_addition",
            encrypted_val, he.encrypt(10)
        )
        
        # Zero-knowledge proofs
        zkp = ZeroKnowledgeProof()
        
        self.benchmark_function(
            zkp.generate_commitment,
            "Crypto", "zkp_commitment",
            value
        )
        
        return self.results
    
    def benchmark_concurrent(self, max_workers: int = 4) -> List[BenchmarkResult]:
        """Benchmark concurrent operations"""
        logger.info(f"=== Benchmarking Concurrent Operations (workers={max_workers}) ===")
        
        engine = PrivacyPreservingMLEngine()
        X = np.random.randn(50, 10)
        y = np.random.randint(0, 2, 50)
        
        def train_task():
            return engine.train_model("linear", X, y, {'epochs': 3})
        
        # Thread pool
        def thread_pool_task():
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [executor.submit(train_task) for _ in range(max_workers)]
                return [f.result() for f in futures]
        
        self.benchmark_function(
            thread_pool_task,
            "Concurrency", f"thread_pool_{max_workers}",
        )
        
        return self.results
    
    def benchmark_all(self) -> List[BenchmarkResult]:
        """Run all benchmarks"""
        logger.info("=== Running Complete Benchmark Suite ===")
        
        self.benchmark_ml_engine()
        self.benchmark_nlp_engine()
        self.benchmark_blockchain()
        self.benchmark_crypto()
        self.benchmark_concurrent(max_workers=4)
        
        return self.results
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive benchmark report"""
        report = {
            'summary': {
                'total_benchmarks': len(self.results),
                'total_time_sec': sum(r.mean_time_ms for r in self.results) / 1000,
                'avg_success_rate': np.mean([r.success_rate for r in self.results])
            },
            'by_component': {},
            'detailed_results': [asdict(r) for r in self.results],
            'system_info': {
                'cpu_count': psutil.cpu_count(),
                'memory_gb': psutil.virtual_memory().total / 1024 / 1024 / 1024,
                'python_version': __import__('sys').version,
                'pytorch_version': torch.__version__,
                'cuda_available': torch.cuda.is_available()
            }
        }
        
        # Group by component
        for result in self.results:
            if result.component not in report['by_component']:
                report['by_component'][result.component] = []
            report['by_component'][result.component].append(asdict(result))
        
        return report
    
    def save_report(self, filename: str = "benchmark_report.json"):
        """Save report to file"""
        report = self.generate_report()
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2)
        logger.info(f"Benchmark report saved to {filename}")
    
    def print_summary(self):
        """Print benchmark summary"""
        print("\n" + "=" * 80)
        print("BENCHMARK SUMMARY")
        print("=" * 80)
        
        for component in set(r.component for r in self.results):
            component_results = [r for r in self.results if r.component == component]
            print(f"\n{component}:")
            print("-" * 80)
            print(f"{'Operation':<30} {'Mean (ms)':<12} {'P95 (ms)':<12} {'Ops/sec':<12}")
            print("-" * 80)
            
            for result in component_results:
                print(f"{result.operation:<30} {result.mean_time_ms:<12.2f} "
                      f"{result.p95_ms:<12.2f} {result.throughput_ops_sec:<12.2f}")
        
        print("\n" + "=" * 80)


if __name__ == "__main__":
    benchmarker = Benchmarker(warmup_iterations=5, benchmark_iterations=50)
    benchmarker.benchmark_all()
    benchmarker.print_summary()
    benchmarker.save_report()
