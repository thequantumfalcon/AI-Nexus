"""
MLflow Experiment Tracking Integration
=======================================

Track experiments, log metrics, save models, and manage ML lifecycle.
"""

import mlflow
import mlflow.pytorch
from mlflow.tracking import MlflowClient
from typing import Dict, Any, Optional, List
import torch
import numpy as np
from datetime import datetime
import json

from core.logger import get_logger
from core.config import get_config

logger = get_logger(__name__)
config = get_config()


class MLflowTracker:
    """
    Comprehensive MLflow integration for experiment tracking
    """
    
    def __init__(
        self,
        tracking_uri: Optional[str] = None,
        experiment_name: str = "AI-Nexus-Experiments"
    ):
        import os
        import pathlib
        import warnings
        
        # Suppress MLflow filesystem deprecation warning
        warnings.filterwarnings('ignore', message='.*Filesystem tracking backend.*')
        
        # Handle tracking URI and ensure directory exists
        if tracking_uri:
            self.tracking_uri = tracking_uri
            # Extract path from file:// URI
            if tracking_uri.startswith("file:"):
                path_part = tracking_uri.replace("file:", "").lstrip("/").lstrip("\\")
                # Ensure directory exists with proper structure
                mlruns_dir = pathlib.Path(path_part).resolve()
                mlruns_dir.mkdir(parents=True, exist_ok=True)
                # Create .trash directory to prevent MLflow errors
                trash_dir = mlruns_dir / ".trash"
                trash_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.tracking_uri = config.get("mlflow.tracking_uri", "file:./mlruns")
            mlruns_dir = pathlib.Path("./mlruns").resolve()
            mlruns_dir.mkdir(parents=True, exist_ok=True)
            trash_dir = mlruns_dir / ".trash"
            trash_dir.mkdir(parents=True, exist_ok=True)
        
        mlflow.set_tracking_uri(self.tracking_uri)
        
        self.experiment_name = experiment_name
        self.client = MlflowClient()
        
        # Create experiment if it doesn't exist
        try:
            self.experiment = mlflow.get_experiment_by_name(experiment_name)
            if self.experiment is None:
                self.experiment_id = mlflow.create_experiment(experiment_name)
            else:
                self.experiment_id = self.experiment.experiment_id
        except Exception as e:
            logger.error(f"Failed to create/get experiment: {e}")
            self.experiment_id = "0"
        
        try:
            mlflow.set_experiment(experiment_name)
        except Exception as e:
            logger.warning(f"Failed to set experiment: {e}")
            
        logger.info(f"MLflow initialized: {self.tracking_uri}, experiment={experiment_name}")
    
    def start_run(
        self,
        run_name: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None
    ) -> str:
        """Start a new MLflow run"""
        run = mlflow.start_run(run_name=run_name)
        
        if tags:
            for key, value in tags.items():
                mlflow.set_tag(key, value)
        
        # Default tags
        mlflow.set_tag("framework", "pytorch")
        mlflow.set_tag("platform", "AI-Nexus")
        mlflow.set_tag("timestamp", datetime.now().isoformat())
        
        logger.info(f"Started MLflow run: {run.info.run_id}")
        return run.info.run_id
    
    def log_params(self, params: Dict[str, Any]):
        """Log parameters"""
        for key, value in params.items():
            try:
                mlflow.log_param(key, value)
            except Exception as e:
                logger.warning(f"Could not log param {key}: {e}")
    
    def log_metrics(
        self,
        metrics: Dict[str, float],
        step: Optional[int] = None
    ):
        """Log metrics"""
        for key, value in metrics.items():
            try:
                if isinstance(value, (int, float, np.number)):
                    mlflow.log_metric(key, float(value), step=step)
            except Exception as e:
                logger.warning(f"Could not log metric {key}: {e}")
    
    def log_model(
        self,
        model: torch.nn.Module,
        artifact_path: str = "model",
        signature: Optional[Any] = None,
        input_example: Optional[np.ndarray] = None
    ):
        """Log PyTorch model"""
        try:
            mlflow.pytorch.log_model(
                model,
                artifact_path,
                signature=signature,
                input_example=input_example
            )
            logger.info(f"Logged model to {artifact_path}")
        except Exception as e:
            logger.error(f"Failed to log model: {e}")
    
    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None):
        """Log an artifact (file)"""
        try:
            mlflow.log_artifact(local_path, artifact_path)
        except Exception as e:
            logger.error(f"Failed to log artifact: {e}")
    
    def log_dict(self, dictionary: Dict, filename: str):
        """Log dictionary as JSON artifact"""
        import tempfile
        import os
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            filepath = os.path.join(tmp_dir, filename)
            with open(filepath, 'w') as f:
                json.dump(dictionary, f, indent=2)
            mlflow.log_artifact(filepath)
    
    def log_figure(self, figure, filename: str):
        """Log matplotlib figure"""
        import tempfile
        import os
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            filepath = os.path.join(tmp_dir, filename)
            figure.savefig(filepath)
            mlflow.log_artifact(filepath)
    
    def log_training_run(
        self,
        model: torch.nn.Module,
        params: Dict[str, Any],
        metrics_history: List[Dict[str, float]],
        final_metrics: Dict[str, float],
        run_name: Optional[str] = None
    ) -> str:
        """
        Log complete training run
        """
        run_id = self.start_run(run_name=run_name)
        
        # Log parameters
        self.log_params(params)
        
        # Log metrics history
        for step, metrics in enumerate(metrics_history):
            self.log_metrics(metrics, step=step)
        
        # Log final metrics
        self.log_metrics(final_metrics)
        
        # Log model
        self.log_model(model)
        
        # Log model summary
        model_info = self._get_model_info(model)
        self.log_dict(model_info, "model_info.json")
        
        mlflow.end_run()
        logger.info(f"Training run logged: {run_id}")
        return run_id
    
    def log_privacy_metrics(
        self,
        epsilon: float,
        delta: float,
        noise_scale: Optional[float] = None,
        mechanism: str = "gaussian"
    ):
        """Log differential privacy metrics"""
        mlflow.log_metric("privacy_epsilon", epsilon)
        mlflow.log_metric("privacy_delta", delta)
        if noise_scale:
            mlflow.log_metric("noise_scale", noise_scale)
        mlflow.set_tag("privacy_mechanism", mechanism)
    
    def log_federated_round(
        self,
        round_num: int,
        num_clients: int,
        aggregated_metrics: Dict[str, float],
        client_metrics: Optional[List[Dict[str, float]]] = None
    ):
        """Log federated learning round"""
        # Log aggregated metrics
        for key, value in aggregated_metrics.items():
            mlflow.log_metric(f"federated_{key}", value, step=round_num)
        
        mlflow.log_metric("num_clients", num_clients, step=round_num)
        
        # Log client-level metrics if provided
        if client_metrics:
            for i, client_m in enumerate(client_metrics):
                for key, value in client_m.items():
                    mlflow.log_metric(f"client_{i}_{key}", value, step=round_num)
    
    def compare_runs(
        self,
        run_ids: List[str],
        metrics: List[str]
    ) -> Dict[str, Dict[str, float]]:
        """Compare multiple runs"""
        comparison = {}
        
        for run_id in run_ids:
            run = self.client.get_run(run_id)
            comparison[run_id] = {
                metric: run.data.metrics.get(metric, None)
                for metric in metrics
            }
        
        return comparison
    
    def get_best_run(
        self,
        metric: str,
        maximize: bool = True
    ) -> Optional[str]:
        """Get best run based on metric"""
        runs = mlflow.search_runs(
            experiment_ids=[self.experiment_id],
            order_by=[f"metrics.{metric} {'DESC' if maximize else 'ASC'}"],
            max_results=1
        )
        
        if len(runs) > 0:
            return runs.iloc[0]['run_id']
        return None
    
    def load_model(self, run_id: str, artifact_path: str = "model") -> torch.nn.Module:
        """Load model from run"""
        model_uri = f"runs:/{run_id}/{artifact_path}"
        return mlflow.pytorch.load_model(model_uri)
    
    def register_model(
        self,
        run_id: str,
        model_name: str,
        artifact_path: str = "model"
    ) -> str:
        """Register model in MLflow Model Registry"""
        model_uri = f"runs:/{run_id}/{artifact_path}"
        
        result = mlflow.register_model(model_uri, model_name)
        logger.info(f"Registered model: {model_name} version {result.version}")
        return result.version
    
    def transition_model_stage(
        self,
        model_name: str,
        version: str,
        stage: str
    ):
        """
        Transition model to different stage
        Stages: None, Staging, Production, Archived
        """
        self.client.transition_model_version_stage(
            name=model_name,
            version=version,
            stage=stage
        )
        logger.info(f"Transitioned {model_name} v{version} to {stage}")
    
    def get_production_model(self, model_name: str) -> Optional[torch.nn.Module]:
        """Get latest production model"""
        try:
            model_uri = f"models:/{model_name}/Production"
            return mlflow.pytorch.load_model(model_uri)
        except Exception as e:
            logger.error(f"Could not load production model: {e}")
            return None
    
    def end_run(self):
        """End current MLflow run"""
        mlflow.end_run()
    
    def cleanup(self):
        """Clean up resources and close database connections"""
        import gc
        
        try:
            # End any active runs
            mlflow.end_run()
        except Exception:
            pass
        
        try:
            # Force close client connections by accessing the store and disposing the engine
            if hasattr(self, 'client') and self.client:
                try:
                    # Access the tracking store
                    if hasattr(self.client, '_tracking_client'):
                        tracking_client = self.client._tracking_client
                        if hasattr(tracking_client, 'store'):
                            store = tracking_client.store
                            if hasattr(store, 'engine'):
                                store.engine.dispose()
                except Exception:
                    pass
                    
                try:
                    # Also try direct store access
                    if hasattr(self.client, 'store'):
                        if hasattr(self.client.store, 'engine'):
                            self.client.store.engine.dispose()
                except Exception:
                    pass
        except Exception as e:
            logger.debug(f"Cleanup warning: {e}")
        
        # Force garbage collection to release file handles
        gc.collect()
        
        # Small delay to ensure connections are released
        import time
        time.sleep(0.05)
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.cleanup()
    
    def _get_model_info(self, model: torch.nn.Module) -> Dict[str, Any]:
        """Get model information"""
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_class': model.__class__.__name__,
            'device': str(next(model.parameters()).device),
        }
    
    def autolog(self, framework: str = "pytorch"):
        """Enable MLflow autologging"""
        if framework == "pytorch":
            mlflow.pytorch.autolog()
        logger.info(f"Autologging enabled for {framework}")
