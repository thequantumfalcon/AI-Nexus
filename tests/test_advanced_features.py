"""
Integration Tests for New Features
===================================

Test FastAPI, MLflow, quantization, federated learning, and fairness.
"""

import pytest
import numpy as np
import torch
import torch.nn as nn
from fastapi.testclient import TestClient
import tempfile
import os

from services.api.api_server import app, create_access_token
from services.ml.optimization import ModelOptimizer
from services.ml.federated import FederatedOptimizer
from services.ml.mlflow_tracker import MLflowTracker
from services.ml.fairness import FairnessAnalyzer, FairnessMetrics
from services.ml.ml_engine import SimpleNeuralNet


class TestFastAPI:
    """Test FastAPI server"""
    
    @pytest.fixture
    def client(self):
        return TestClient(app)
    
    @pytest.fixture
    def auth_token(self):
        return create_access_token({"sub": "test_user"})
    
    @pytest.fixture
    def auth_headers(self, auth_token):
        return {"Authorization": f"Bearer {auth_token}"}
    
    def test_root_endpoint(self, client):
        response = client.get("/")
        assert response.status_code == 200
        assert "AI-Nexus" in response.json()["name"]
    
    def test_health_check(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"
    
    def test_token_generation(self, client):
        response = client.post("/auth/token", json={"api_key": "demo-api-key"})
        assert response.status_code == 200
        assert "access_token" in response.json()
    
    def test_ml_training_endpoint(self, client, auth_headers):
        request_data = {
            "model_type": "linear",
            "X": [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
            "y": [0, 1, 0],
            "hyperparameters": {"epochs": 2}
        }
        response = client.post("/ml/train", json=request_data, headers=auth_headers)
        assert response.status_code == 200
        assert "model_id" in response.json()
    
    def test_nlp_sentiment_endpoint(self, client, auth_headers):
        request_data = {
            "text": "This is amazing!"
        }
        response = client.post("/nlp/sentiment", json=request_data, headers=auth_headers)
        assert response.status_code == 200
        assert "sentiment" in response.json()
    
    def test_unauthorized_access(self, client):
        response = client.post("/ml/train", json={})
        assert response.status_code == 401  # Unauthorized, not forbidden


class TestModelOptimization:
    """Test model optimization techniques"""
    
    @pytest.fixture
    def optimizer(self):
        return ModelOptimizer()
    
    @pytest.fixture
    def simple_model(self):
        return SimpleNeuralNet(input_size=10, hidden_sizes=[20], output_size=2)
    
    def test_dynamic_quantization(self, optimizer, simple_model):
        quantized = optimizer.dynamic_quantization(simple_model)
        assert quantized is not None
        
        # Check that model is quantized
        stats = optimizer.get_model_size(quantized)
        assert stats['parameters'] > 0
    
    def test_model_pruning(self, optimizer, simple_model):
        original_params = sum(p.numel() for p in simple_model.parameters())
        
        pruned = optimizer.prune_model(simple_model, amount=0.3)
        
        # Model should still work
        x = torch.randn(1, 10)
        output = pruned(x)
        assert output.shape == (1, 2)
    
    def test_model_size_calculation(self, optimizer, simple_model):
        stats = optimizer.get_model_size(simple_model)
        assert 'parameters' in stats
        assert 'size_mb' in stats
        assert stats['parameters'] > 0
    
    def test_inference_benchmark(self, optimizer, simple_model):
        benchmark_results = optimizer.benchmark_inference(
            simple_model,
            input_shape=(1, 10),
            iterations=10
        )
        
        assert 'mean_ms' in benchmark_results
        assert 'p95_ms' in benchmark_results
        assert benchmark_results['mean_ms'] > 0


class TestFederatedLearning:
    """Test advanced federated learning algorithms"""
    
    @pytest.fixture
    def fed_optimizer(self):
        return FederatedOptimizer(strategy='fedavg')
    
    @pytest.fixture
    def client_weights(self):
        # Simulate 3 clients
        return [
            {
                'layer1': torch.randn(10, 5),
                'layer2': torch.randn(5, 2)
            },
            {
                'layer1': torch.randn(10, 5),
                'layer2': torch.randn(5, 2)
            },
            {
                'layer1': torch.randn(10, 5),
                'layer2': torch.randn(5, 2)
            }
        ]
    
    def test_fedavg_aggregation(self, fed_optimizer, client_weights):
        client_samples = [100, 150, 120]
        
        aggregated = fed_optimizer.federated_averaging(client_weights, client_samples)
        
        assert 'layer1' in aggregated
        assert 'layer2' in aggregated
        assert aggregated['layer1'].shape == (10, 5)
        assert aggregated['layer2'].shape == (5, 2)
    
    def test_fedprox_aggregation(self, client_weights):
        fed_prox = FederatedOptimizer(strategy='fedprox')
        client_samples = [100, 150, 120]
        global_weights = client_weights[0]  # Use first client as initial global
        
        aggregated = fed_prox.federated_proximal(
            client_weights,
            client_samples,
            global_weights,
            mu=0.01
        )
        
        assert 'layer1' in aggregated
        assert 'layer2' in aggregated
    
    def test_fedadam_aggregation(self, client_weights):
        fed_adam = FederatedOptimizer(strategy='fedadam')
        client_samples = [100, 150, 120]
        global_weights = client_weights[0]
        m_t = {}
        v_t = {}
        
        new_weights, new_m, new_v = fed_adam.federated_adam(
            client_weights,
            client_samples,
            global_weights,
            m_t,
            v_t,
            t=1
        )
        
        assert 'layer1' in new_weights
        assert 'layer1' in new_m
        assert 'layer1' in new_v
    
    def test_client_selection(self, fed_optimizer):
        selected = fed_optimizer.client_selection(
            num_clients=100,
            selection_rate=0.1
        )
        
        assert len(selected) == 10
        assert all(0 <= idx < 100 for idx in selected)
    
    def test_secure_aggregation_smpc(self, fed_optimizer, client_weights):
        client_samples = [100, 150, 120]
        
        # This might be slow, so use smaller weights
        small_weights = [
            {'w': torch.randn(5, 3)} for _ in range(3)
        ]
        
        aggregated = fed_optimizer.secure_aggregation(
            small_weights,
            [100, 100, 100],
            use_smpc=True
        )
        
        assert 'w' in aggregated


class TestMLflowTracking:
    """Test MLflow experiment tracking"""
    
    @pytest.fixture
    def tracker(self):
        import os
        import time
        tmpdir = "mlruns_test"
        os.makedirs(tmpdir, exist_ok=True)
        
        # Use context manager for proper cleanup
        with MLflowTracker(tracking_uri=f"file:./{tmpdir}") as tracker:
            yield tracker
        
        # Give time for connections to close
        time.sleep(0.1)
        
        # Cleanup
        import shutil
        if os.path.exists(tmpdir):
            try:
                shutil.rmtree(tmpdir)
            except PermissionError:
                # On Windows, sometimes files are locked briefly
                time.sleep(0.5)
                try:
                    shutil.rmtree(tmpdir)
                except Exception as e:
                    print(f"Warning: Could not clean up {tmpdir}: {e}")
    
    def test_start_run(self, tracker):
        run_id = tracker.start_run(run_name="test_run")
        assert run_id is not None
        tracker.end_run()
    
    def test_log_params(self, tracker):
        tracker.start_run()
        tracker.log_params({"learning_rate": 0.01, "epochs": 10})
        tracker.end_run()
    
    def test_log_metrics(self, tracker):
        tracker.start_run()
        tracker.log_metrics({"loss": 0.5, "accuracy": 0.85}, step=1)
        tracker.end_run()
    
    def test_log_privacy_metrics(self, tracker):
        tracker.start_run()
        tracker.log_privacy_metrics(epsilon=1.0, delta=1e-5)
        tracker.end_run()
    
    def test_log_federated_round(self, tracker):
        tracker.start_run()
        tracker.log_federated_round(
            round_num=1,
            num_clients=10,
            aggregated_metrics={"avg_loss": 0.3, "avg_accuracy": 0.9}
        )
        tracker.end_run()


class TestFairnessAnalysis:
    """Test bias detection and fairness metrics"""
    
    @pytest.fixture
    def analyzer(self):
        return FairnessAnalyzer()
    
    @pytest.fixture
    def fair_predictions(self):
        np.random.seed(42)
        y_true = np.random.randint(0, 2, 200)
        y_pred = np.random.randint(0, 2, 200)
        protected = np.array([0] * 100 + [1] * 100)
        return y_true, y_pred, protected
    
    @pytest.fixture
    def biased_predictions(self):
        np.random.seed(42)
        # Group 0: 80% positive predictions
        # Group 1: 20% positive predictions (bias)
        y_true = np.array([1] * 80 + [0] * 20 + [1] * 20 + [0] * 80)
        y_pred = np.array([1] * 80 + [0] * 20 + [0] * 80 + [0] * 20)
        protected = np.array([0] * 100 + [1] * 100)
        return y_true, y_pred, protected
    
    def test_evaluate_fairness(self, analyzer, fair_predictions):
        y_true, y_pred, protected = fair_predictions
        
        metrics = analyzer.evaluate_fairness(y_true, y_pred, protected)
        
        assert isinstance(metrics, FairnessMetrics)
        assert 0 <= metrics.demographic_parity_difference <= 1
        assert 0 <= metrics.disparate_impact_ratio
        assert 0 <= metrics.equalized_odds_difference <= 1
    
    def test_biased_detection(self, analyzer, biased_predictions):
        y_true, y_pred, protected = biased_predictions
        
        metrics = analyzer.evaluate_fairness(y_true, y_pred, protected)
        
        # Should detect bias
        assert metrics.demographic_parity_difference > 0.1
        assert metrics.disparate_impact_ratio < 0.8
    
    def test_fairness_report(self, analyzer, biased_predictions):
        y_true, y_pred, protected = biased_predictions
        
        metrics = analyzer.evaluate_fairness(y_true, y_pred, protected)
        report = analyzer.generate_fairness_report(metrics)
        
        assert 'overall_assessment' in report
        assert 'violations' in report
        assert 'recommendations' in report
        assert report['overall_assessment'] == 'FAIL'
    
    def test_bias_mitigation_weights(self, analyzer, biased_predictions):
        y_true, _, protected = biased_predictions
        
        weights = analyzer.mitigate_bias_reweighting(y_true, protected)
        
        assert len(weights) == len(y_true)
        assert np.all(weights > 0)
        assert np.isclose(np.mean(weights), 1.0, atol=0.1)


class TestIntegration:
    """End-to-end integration tests"""
    
    def test_complete_ml_pipeline_with_tracking(self):
        """Test full ML pipeline with MLflow tracking"""
        import time
        with tempfile.TemporaryDirectory() as tmpdir:
            # Initialize tracker with context manager for proper cleanup
            with MLflowTracker(tracking_uri=f"file:{tmpdir}") as tracker:
                # Start run
                tracker.start_run(run_name="integration_test")
                
                # Log parameters
                params = {"model_type": "neural_net", "epochs": 3}
                tracker.log_params(params)
                
                # Train model (simulated)
                from services.ml.ml_engine import PrivacyPreservingMLEngine
                engine = PrivacyPreservingMLEngine()
                
                X = np.random.randn(100, 10)
                y = np.random.randint(0, 2, 100)
                
                result = engine.train_model("neural_net", X, y, {"epochs": 3})
                
                # Log metrics
                tracker.log_metrics({
                    "final_loss": result.final_loss,
                    "final_accuracy": result.final_accuracy
                })
                
                # End run
                tracker.end_run()
                
                assert result.model_id is not None
            
            # Give time for DB connections to close
            time.sleep(0.2)
    
    def test_federated_with_fairness(self):
        """Test federated learning with fairness analysis"""
        from services.ml.ml_engine import PrivacyPreservingMLEngine
        
        engine = PrivacyPreservingMLEngine()
        fed_optimizer = FederatedOptimizer(strategy='fedavg')
        analyzer = FairnessAnalyzer()
        
        # Simulate federated training
        X = np.random.randn(100, 10)
        y = np.random.randint(0, 2, 100)
        protected = np.array([0] * 50 + [1] * 50)
        
        # Train local models
        result1 = engine.train_model("linear", X[:50], y[:50], {"epochs": 5})
        result2 = engine.train_model("linear", X[50:], y[50:], {"epochs": 5})
        
        # Aggregate
        aggregated_id = engine.aggregate_models([result1.model_id, result2.model_id])
        
        # Make predictions
        pred_result = engine.predict(aggregated_id, X)
        
        # Analyze fairness
        metrics = analyzer.evaluate_fairness(y, pred_result.prediction, protected)
        
        assert metrics is not None
        assert aggregated_id is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
