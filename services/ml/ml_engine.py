"""
ML Engine - Privacy-Preserving Machine Learning
================================================

Implements distributed ML with differential privacy, federated learning,
and homomorphic encryption.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import pickle
import time
from dataclasses import dataclass
from collections import OrderedDict

from core.crypto import DifferentialPrivacy, HomomorphicEncryption, SecureMultiPartyComputation
from core.logger import get_logger
from core.metrics import measure_time, get_metrics_collector
from core.config import get_config

logger = get_logger(__name__)
config = get_config()
metrics = get_metrics_collector()


@dataclass
class TrainResult:
    """Result from model training"""
    model_id: str
    final_loss: float
    final_accuracy: float
    training_time_ms: int
    epochs_completed: int
    metrics_history: List[Dict[str, float]]


@dataclass
class PredictResult:
    """Result from model prediction"""
    prediction: Any
    confidence: float
    explanation: Dict[str, float]
    inference_time_ms: int


class SimpleNeuralNet(nn.Module):
    """Simple neural network for demonstration"""
    
    def __init__(self, input_size: int, hidden_sizes: List[int], output_size: int):
        super().__init__()
        
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
            prev_size = hidden_size
        
        layers.append(nn.Linear(prev_size, output_size))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)


class PrivacyPreservingMLEngine:
    """
    Machine learning engine with privacy preservation
    """
    
    def __init__(self, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.models: Dict[str, nn.Module] = {}
        self.optimizers: Dict[str, optim.Optimizer] = {}
        
        # Privacy components
        self.dp = DifferentialPrivacy(
            epsilon=float(config.get('security.privacy.differential_privacy.epsilon', 1.0)),
            delta=float(config.get('security.privacy.differential_privacy.delta', 1e-5))
        )
        self.he = HomomorphicEncryption()
        self.smpc = SecureMultiPartyComputation()
        
        logger.info(f"ML Engine initialized on {device}")
    
    @measure_time(task_type="ml_train")
    def train_model(
        self,
        model_type: str,
        data: np.ndarray,
        labels: np.ndarray,
        hyperparameters: Optional[Dict[str, Any]] = None,
        use_differential_privacy: bool = True,
        privacy_epsilon: float = 1.0
    ) -> TrainResult:
        """
        Train a model with privacy preservation
        
        Args:
            model_type: Type of model (linear, neural_net, etc.)
            data: Training data
            labels: Training labels
            hyperparameters: Model hyperparameters
            use_differential_privacy: Enable differential privacy
            privacy_epsilon: Privacy budget
        
        Returns:
            TrainResult with training metrics
        """
        start_time = time.time()
        hyperparameters = hyperparameters or {}
        
        model_id = f"{model_type}_{int(time.time())}"
        
        logger.info(f"Training model {model_id} with {len(data)} samples")
        
        # Create model
        if model_type == "linear":
            model = self._create_linear_model(data.shape[1], len(np.unique(labels)))
        elif model_type == "neural_net":
            hidden_sizes = hyperparameters.get('hidden_sizes', [128, 64])
            model = SimpleNeuralNet(
                data.shape[1],
                hidden_sizes,
                len(np.unique(labels))
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        model = model.to(self.device)
        
        # Training configuration
        learning_rate = hyperparameters.get('learning_rate', 0.001)
        batch_size = hyperparameters.get('batch_size', 32)
        epochs = hyperparameters.get('epochs', 10)
        
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()
        
        # Create data loader
        X_tensor = torch.FloatTensor(data).to(self.device)
        y_tensor = torch.LongTensor(labels).to(self.device)
        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Training loop
        metrics_history = []
        
        for epoch in range(epochs):
            model.train()
            epoch_loss = 0.0
            correct = 0
            total = 0
            
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                
                loss.backward()
                
                # Apply differential privacy if enabled
                if use_differential_privacy:
                    self._apply_dp_to_gradients(model, privacy_epsilon)
                
                optimizer.step()
                
                epoch_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()
            
            accuracy = correct / total
            avg_loss = epoch_loss / len(dataloader)
            
            metrics_history.append({
                'epoch': epoch + 1,
                'loss': avg_loss,
                'accuracy': accuracy
            })
            
            logger.info(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")
        
        # Store model
        self.models[model_id] = model
        self.optimizers[model_id] = optimizer
        
        training_time = int((time.time() - start_time) * 1000)
        
        # Update metrics
        metrics.update_model_accuracy(model_id, metrics_history[-1]['accuracy'])
        
        return TrainResult(
            model_id=model_id,
            final_loss=metrics_history[-1]['loss'],
            final_accuracy=metrics_history[-1]['accuracy'],
            training_time_ms=training_time,
            epochs_completed=epochs,
            metrics_history=metrics_history
        )
    
    def _create_linear_model(self, input_size: int, output_size: int) -> nn.Module:
        """Create linear model"""
        return nn.Linear(input_size, output_size)
    
    def _apply_dp_to_gradients(self, model: nn.Module, epsilon: float):
        """Apply differential privacy to gradients"""
        clip_norm = 1.0
        
        # Clip gradients
        total_norm = 0.0
        for param in model.parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        
        clip_coef = clip_norm / (total_norm + 1e-6)
        if clip_coef < 1:
            for param in model.parameters():
                if param.grad is not None:
                    param.grad.data.mul_(clip_coef)
        
        # Add noise
        sensitivity = clip_norm
        sigma = sensitivity * np.sqrt(2 * np.log(1.25 / self.dp.delta)) / epsilon
        
        for param in model.parameters():
            if param.grad is not None:
                noise = torch.normal(0, sigma, size=param.grad.shape).to(self.device)
                param.grad.data.add_(noise)
    
    @measure_time(task_type="ml_predict")
    def predict(
        self,
        model_id: str,
        input_data: np.ndarray,
        return_confidence: bool = True,
        return_explanation: bool = False
    ) -> PredictResult:
        """
        Make predictions with trained model
        
        Args:
            model_id: ID of trained model
            input_data: Input features
            return_confidence: Return confidence scores
            return_explanation: Return feature importance
        
        Returns:
            PredictResult with predictions
        """
        start_time = time.time()
        
        if model_id not in self.models:
            raise ValueError(f"Model {model_id} not found")
        
        model = self.models[model_id]
        model.eval()
        
        with torch.no_grad():
            X_tensor = torch.FloatTensor(input_data).to(self.device)
            outputs = model(X_tensor)
            
            # Get predictions
            if outputs.dim() > 1 and outputs.shape[1] > 1:
                # Classification
                probabilities = torch.softmax(outputs, dim=1)
                confidence, predicted = torch.max(probabilities, 1)
                prediction = predicted.cpu().numpy()
                confidence_score = confidence.cpu().numpy().mean()
            else:
                # Regression
                prediction = outputs.cpu().numpy()
                confidence_score = 1.0
        
        # Generate explanation if requested
        explanation = {}
        if return_explanation:
            explanation = self._generate_feature_importance(model, input_data)
        
        inference_time = int((time.time() - start_time) * 1000)
        
        return PredictResult(
            prediction=prediction,
            confidence=float(confidence_score),
            explanation=explanation,
            inference_time_ms=inference_time
        )
    
    def _generate_feature_importance(
        self,
        model: nn.Module,
        input_data: np.ndarray
    ) -> Dict[str, float]:
        """Generate feature importance scores (simplified)"""
        # Simplified implementation - in production, use SHAP or LIME
        importance = {}
        
        for i in range(min(10, input_data.shape[1])):
            importance[f"feature_{i}"] = float(np.random.rand())
        
        return importance
    
    def evaluate_model(
        self,
        model_id: str,
        test_data: np.ndarray,
        test_labels: np.ndarray,
        metrics_list: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """
        Evaluate model performance
        
        Args:
            model_id: ID of trained model
            test_data: Test features
            test_labels: Test labels
            metrics_list: List of metrics to compute
        
        Returns:
            Dictionary of metric values
        """
        if metrics_list is None:
            metrics_list = ['accuracy', 'precision', 'recall', 'f1']
        
        result = self.predict(model_id, test_data, return_confidence=False)
        predictions = result.prediction
        
        # Compute metrics
        metrics_dict = {}
        
        if 'accuracy' in metrics_list:
            accuracy = np.mean(predictions == test_labels)
            metrics_dict['accuracy'] = float(accuracy)
        
        if any(m in metrics_list for m in ['precision', 'recall', 'f1']):
            # Simplified - compute for binary classification
            tp = np.sum((predictions == 1) & (test_labels == 1))
            fp = np.sum((predictions == 1) & (test_labels == 0))
            fn = np.sum((predictions == 0) & (test_labels == 1))
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            
            if 'precision' in metrics_list:
                metrics_dict['precision'] = precision
            if 'recall' in metrics_list:
                metrics_dict['recall'] = recall
            if 'f1' in metrics_list:
                metrics_dict['f1'] = f1
        
        return metrics_dict
    
    def get_model_weights(self, model_id: str, encrypt: bool = False) -> bytes:
        """Get model weights (optionally encrypted)"""
        if model_id not in self.models:
            raise ValueError(f"Model {model_id} not found")
        
        model = self.models[model_id]
        state_dict = model.state_dict()
        
        # Serialize
        weights_bytes = pickle.dumps(state_dict)
        
        if encrypt:
            # Encrypt with homomorphic encryption
            weights_bytes = self.he.encrypt(np.frombuffer(weights_bytes, dtype=np.uint8))
        
        return weights_bytes
    
    def set_model_weights(self, model_id: str, weights_bytes: bytes, encrypted: bool = False):
        """Set model weights (handle encrypted weights)"""
        if encrypted:
            # Decrypt
            decrypted = self.he.decrypt(weights_bytes)
            weights_bytes = decrypted.tobytes()
        
        state_dict = pickle.loads(weights_bytes)
        
        if model_id in self.models:
            self.models[model_id].load_state_dict(state_dict)
        else:
            logger.warning(f"Model {model_id} not found, cannot load weights")
    
    def aggregate_models(
        self,
        model_ids: List[str],
        weights: Optional[List[float]] = None
    ) -> str:
        """
        Aggregate multiple models (for federated learning)
        
        Args:
            model_ids: List of model IDs to aggregate
            weights: Weights for each model (default: equal weights)
        
        Returns:
            ID of aggregated model
        """
        if not model_ids:
            raise ValueError("No models to aggregate")
        
        if weights is None:
            weights = [1.0 / len(model_ids)] * len(model_ids)
        
        # Get first model as template
        aggregated_state = OrderedDict()
        first_model = self.models[model_ids[0]]
        
        # Initialize with zeros
        for key in first_model.state_dict():
            aggregated_state[key] = torch.zeros_like(first_model.state_dict()[key])
        
        # Weighted average
        for model_id, weight in zip(model_ids, weights):
            if model_id not in self.models:
                logger.warning(f"Model {model_id} not found, skipping")
                continue
            
            state = self.models[model_id].state_dict()
            for key in state:
                aggregated_state[key] += state[key] * weight
        
        # Create new model with aggregated weights
        aggregated_id = f"aggregated_{int(time.time())}"
        
        # Clone architecture from first model
        import copy
        aggregated_model = copy.deepcopy(first_model)
        aggregated_model.load_state_dict(aggregated_state)
        
        self.models[aggregated_id] = aggregated_model
        
        logger.info(f"Aggregated {len(model_ids)} models into {aggregated_id}")
        
        return aggregated_id
