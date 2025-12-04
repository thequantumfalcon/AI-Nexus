"""
Test suite for ML Service
==========================

Tests privacy-preserving machine learning.
"""

import pytest
import numpy as np
import torch

from services.ml.ml_engine import PrivacyPreservingMLEngine, SimpleNeuralNet


class TestPrivacyPreservingMLEngine:
    """Test ML engine functionality"""
    
    @pytest.fixture
    def ml_engine(self):
        """Create ML engine fixture"""
        return PrivacyPreservingMLEngine()
    
    @pytest.fixture
    def sample_data(self):
        """Generate sample training data"""
        np.random.seed(42)
        X = np.random.randn(100, 10)
        y = np.random.randint(0, 2, 100)
        return X, y
    
    def test_initialization(self, ml_engine):
        """Test ML engine initializes correctly"""
        assert ml_engine is not None
        assert ml_engine.dp is not None
        assert ml_engine.he is not None
    
    def test_train_linear_model(self, ml_engine, sample_data):
        """Test training linear model"""
        X, y = sample_data
        
        result = ml_engine.train_model(
            "linear",
            X,
            y,
            hyperparameters={'epochs': 3, 'learning_rate': 0.01}
        )
        
        assert result.model_id is not None
        assert result.final_loss >= 0
        assert 0 <= result.final_accuracy <= 1
        assert result.epochs_completed == 3
        assert len(result.metrics_history) == 3
    
    def test_train_neural_network(self, ml_engine, sample_data):
        """Test training neural network"""
        X, y = sample_data
        
        result = ml_engine.train_model(
            "neural_net",
            X,
            y,
            hyperparameters={
                'epochs': 5,
                'hidden_sizes': [64, 32],
                'batch_size': 16
            }
        )
        
        assert result.model_id is not None
        assert result.epochs_completed == 5
    
    def test_differential_privacy(self, ml_engine, sample_data):
        """Test training with differential privacy"""
        X, y = sample_data
        
        result = ml_engine.train_model(
            "neural_net",
            X,
            y,
            hyperparameters={'epochs': 3},
            use_differential_privacy=True,
            privacy_epsilon=1.0
        )
        
        assert result.model_id is not None
    
    def test_prediction(self, ml_engine, sample_data):
        """Test model prediction"""
        X, y = sample_data
        
        # Train model
        train_result = ml_engine.train_model(
            "neural_net",
            X[:80],
            y[:80],
            hyperparameters={'epochs': 3}
        )
        
        # Make predictions
        pred_result = ml_engine.predict(
            train_result.model_id,
            X[80:],
            return_confidence=True
        )
        
        assert pred_result.prediction is not None
        assert len(pred_result.prediction) == 20
        assert pred_result.confidence > 0
        assert pred_result.inference_time_ms >= 0  # Can be 0 for fast operations
    
    def test_prediction_with_explanation(self, ml_engine, sample_data):
        """Test prediction with feature importance"""
        X, y = sample_data
        
        # Train
        train_result = ml_engine.train_model(
            "neural_net",
            X[:80],
            y[:80],
            hyperparameters={'epochs': 3}
        )
        
        # Predict with explanation
        pred_result = ml_engine.predict(
            train_result.model_id,
            X[80:85],
            return_explanation=True
        )
        
        assert isinstance(pred_result.explanation, dict)
    
    def test_model_evaluation(self, ml_engine, sample_data):
        """Test model evaluation"""
        X, y = sample_data
        
        # Train
        train_result = ml_engine.train_model(
            "neural_net",
            X[:80],
            y[:80],
            hyperparameters={'epochs': 3}
        )
        
        # Evaluate
        metrics = ml_engine.evaluate_model(
            train_result.model_id,
            X[80:],
            y[80:],
            metrics_list=['accuracy', 'precision', 'recall', 'f1']
        )
        
        assert 'accuracy' in metrics
        assert 0 <= metrics['accuracy'] <= 1
    
    def test_get_model_weights(self, ml_engine, sample_data):
        """Test getting model weights"""
        X, y = sample_data
        
        # Train
        train_result = ml_engine.train_model(
            "neural_net",
            X[:50],
            y[:50],
            hyperparameters={'epochs': 2}
        )
        
        # Get weights
        weights = ml_engine.get_model_weights(train_result.model_id)
        assert weights is not None
        assert isinstance(weights, bytes)
    
    def test_model_aggregation(self, ml_engine, sample_data):
        """Test federated learning model aggregation"""
        X, y = sample_data
        
        # Train multiple models
        model_ids = []
        for i in range(3):
            result = ml_engine.train_model(
                "neural_net",
                X[i*30:(i+1)*30],
                y[i*30:(i+1)*30],
                hyperparameters={'epochs': 2, 'hidden_sizes': [32]}
            )
            model_ids.append(result.model_id)
        
        # Aggregate
        aggregated_id = ml_engine.aggregate_models(model_ids)
        assert aggregated_id is not None
        assert aggregated_id in ml_engine.models


class TestSimpleNeuralNet:
    """Test neural network architecture"""
    
    def test_network_creation(self):
        """Test creating neural network"""
        net = SimpleNeuralNet(
            input_size=10,
            hidden_sizes=[64, 32],
            output_size=2
        )
        
        assert net is not None
    
    def test_forward_pass(self):
        """Test forward pass"""
        net = SimpleNeuralNet(10, [32], 2)
        x = torch.randn(5, 10)
        output = net(x)
        
        assert output.shape == (5, 2)


@pytest.mark.integration
class TestMLIntegration:
    """Integration tests for ML service"""
    
    def test_federated_learning_workflow(self):
        """Test complete federated learning workflow"""
        engine = PrivacyPreservingMLEngine()
        
        # Simulate 3 nodes with local data
        np.random.seed(42)
        X_full = np.random.randn(150, 10)
        y_full = np.random.randint(0, 2, 150)
        
        model_ids = []
        
        # Each node trains on local data
        for i in range(3):
            start = i * 50
            end = (i + 1) * 50
            
            result = engine.train_model(
                "neural_net",
                X_full[start:end],
                y_full[start:end],
                hyperparameters={'epochs': 5, 'hidden_sizes': [32]},
                use_differential_privacy=True
            )
            model_ids.append(result.model_id)
        
        # Aggregate models
        global_model_id = engine.aggregate_models(model_ids)
        
        # Test global model
        pred_result = engine.predict(
            global_model_id,
            X_full[:10],
            return_confidence=True
        )
        
        assert pred_result.prediction is not None
        assert pred_result.confidence > 0


class TestMLPerformance:
    """Performance tests (non-benchmark)"""
    
    def test_training_speed(self):
        """Test training performance"""
        engine = PrivacyPreservingMLEngine()
        X = np.random.randn(100, 10)
        y = np.random.randint(0, 2, 100)
        
        import time
        start = time.time()
        result = engine.train_model("neural_net", X, y, hyperparameters={'epochs': 1, 'batch_size': 32})
        duration = time.time() - start
        
        assert result.model_id is not None
        assert duration < 20.0  # Should complete in under 20 seconds
    
    def test_prediction_speed(self):
        """Test prediction performance"""
        engine = PrivacyPreservingMLEngine()
        X = np.random.randn(100, 10)
        y = np.random.randint(0, 2, 100)
        
        # Train first
        result = engine.train_model("neural_net", X, y, hyperparameters={'epochs': 1})
        
        import time
        X_test = np.random.randn(10, 10)
        start = time.time()
        pred_result = engine.predict(result.model_id, X_test)
        duration = time.time() - start
        
        assert pred_result.prediction is not None
        assert duration < 2.0  # Should complete in under 2 seconds


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
