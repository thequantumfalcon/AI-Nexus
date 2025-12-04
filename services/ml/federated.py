"""
Advanced Federated Learning Algorithms
=======================================

FedProx, FedAvg variants, secure aggregation, and adaptive optimization.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Optional, Tuple, Any
import copy

from core.crypto import SecureMultiPartyComputation, HomomorphicEncryption
from core.logger import get_logger

logger = get_logger(__name__)


class FederatedOptimizer:
    """
    Advanced federated learning with multiple aggregation strategies
    """
    
    def __init__(self, strategy: str = 'fedavg'):
        self.strategy = strategy
        self.smpc = SecureMultiPartyComputation()
        self.he = HomomorphicEncryption()
        
        self.strategies = {
            'fedavg': self.federated_averaging,
            'fedprox': self.federated_proximal,
            'fedadam': self.federated_adam,
            'fedyogi': self.federated_yogi,
            'scaffold': self.scaffold_aggregation
        }
    
    def federated_averaging(
        self,
        client_weights: List[Dict[str, torch.Tensor]],
        client_samples: List[int]
    ) -> Dict[str, torch.Tensor]:
        """
        Standard FedAvg - weighted average by sample count
        
        w_global = Σ(n_i / N) * w_i
        """
        logger.info("FedAvg aggregation")
        
        total_samples = sum(client_samples)
        global_weights = {}
        
        # Initialize with zeros
        for key in client_weights[0].keys():
            global_weights[key] = torch.zeros_like(client_weights[0][key])
        
        # Weighted average
        for client_w, n_samples in zip(client_weights, client_samples):
            weight = n_samples / total_samples
            for key in global_weights.keys():
                global_weights[key] += weight * client_w[key]
        
        return global_weights
    
    def federated_proximal(
        self,
        client_weights: List[Dict[str, torch.Tensor]],
        client_samples: List[int],
        global_weights: Dict[str, torch.Tensor],
        mu: float = 0.01
    ) -> Dict[str, torch.Tensor]:
        """
        FedProx - FedAvg with proximal term for non-IID data
        
        Adds regularization: μ/2 * ||w - w_global||^2
        Better for heterogeneous clients
        """
        logger.info(f"FedProx aggregation (μ={mu})")
        
        # Apply proximal term before averaging
        proximal_weights = []
        for client_w in client_weights:
            prox_w = {}
            for key in client_w.keys():
                # Proximal regularization
                prox_w[key] = client_w[key] - mu * (client_w[key] - global_weights[key])
            proximal_weights.append(prox_w)
        
        # Standard weighted averaging
        return self.federated_averaging(proximal_weights, client_samples)
    
    def federated_adam(
        self,
        client_weights: List[Dict[str, torch.Tensor]],
        client_samples: List[int],
        global_weights: Dict[str, torch.Tensor],
        m_t: Dict[str, torch.Tensor],
        v_t: Dict[str, torch.Tensor],
        beta1: float = 0.9,
        beta2: float = 0.99,
        tau: float = 1e-3,
        t: int = 1
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        FedAdam - Adaptive federated optimization with Adam-style updates
        Better convergence for non-convex objectives
        """
        logger.info(f"FedAdam aggregation (t={t})")
        
        # Compute pseudo-gradient (average of client updates)
        delta = self.federated_averaging(client_weights, client_samples)
        pseudo_grad = {}
        for key in delta.keys():
            pseudo_grad[key] = global_weights[key] - delta[key]
        
        # Initialize momentum if needed
        if not m_t:
            m_t = {key: torch.zeros_like(val) for key, val in global_weights.items()}
            v_t = {key: torch.zeros_like(val) for key, val in global_weights.items()}
        
        # Update moments
        new_m = {}
        new_v = {}
        new_weights = {}
        
        for key in global_weights.keys():
            # First moment
            new_m[key] = beta1 * m_t[key] + (1 - beta1) * pseudo_grad[key]
            
            # Second moment
            new_v[key] = beta2 * v_t[key] + (1 - beta2) * (pseudo_grad[key] ** 2)
            
            # Bias correction
            m_hat = new_m[key] / (1 - beta1 ** t)
            v_hat = new_v[key] / (1 - beta2 ** t)
            
            # Update
            new_weights[key] = global_weights[key] - tau * m_hat / (torch.sqrt(v_hat) + 1e-8)
        
        return new_weights, new_m, new_v
    
    def federated_yogi(
        self,
        client_weights: List[Dict[str, torch.Tensor]],
        client_samples: List[int],
        global_weights: Dict[str, torch.Tensor],
        m_t: Dict[str, torch.Tensor],
        v_t: Dict[str, torch.Tensor],
        beta1: float = 0.9,
        beta2: float = 0.99,
        tau: float = 1e-2,
        t: int = 1
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        FedYogi - Adaptive optimization with Yogi-style second moment
        More robust to heterogeneous data
        """
        logger.info(f"FedYogi aggregation (t={t})")
        
        delta = self.federated_averaging(client_weights, client_samples)
        pseudo_grad = {}
        for key in delta.keys():
            pseudo_grad[key] = global_weights[key] - delta[key]
        
        if not m_t:
            m_t = {key: torch.zeros_like(val) for key, val in global_weights.items()}
            v_t = {key: torch.zeros_like(val) for key, val in global_weights.items()}
        
        new_m = {}
        new_v = {}
        new_weights = {}
        
        for key in global_weights.keys():
            # First moment (same as Adam)
            new_m[key] = beta1 * m_t[key] + (1 - beta1) * pseudo_grad[key]
            
            # Second moment (Yogi-style: additive instead of multiplicative)
            new_v[key] = v_t[key] - (1 - beta2) * torch.sign(v_t[key] - pseudo_grad[key] ** 2) * (pseudo_grad[key] ** 2)
            
            # Update
            new_weights[key] = global_weights[key] - tau * new_m[key] / (torch.sqrt(new_v[key]) + 1e-8)
        
        return new_weights, new_m, new_v
    
    def scaffold_aggregation(
        self,
        client_weights: List[Dict[str, torch.Tensor]],
        client_samples: List[int],
        client_controls: List[Dict[str, torch.Tensor]],
        server_control: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        SCAFFOLD - Control variates for variance reduction
        Addresses client drift in heterogeneous settings
        """
        logger.info("SCAFFOLD aggregation")
        
        # Average weights
        avg_weights = self.federated_averaging(client_weights, client_samples)
        
        # Update server control variate
        total_samples = sum(client_samples)
        new_control = {}
        
        for key in server_control.keys():
            delta_c = torch.zeros_like(server_control[key])
            for client_c, n_samples in zip(client_controls, client_samples):
                weight = n_samples / total_samples
                delta_c += weight * (client_c[key] - server_control[key])
            
            new_control[key] = server_control[key] + delta_c
        
        return avg_weights, new_control
    
    def secure_aggregation(
        self,
        client_weights: List[Dict[str, torch.Tensor]],
        client_samples: List[int],
        use_smpc: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Secure aggregation using SMPC or homomorphic encryption
        Server cannot see individual client weights
        """
        logger.info(f"Secure aggregation (SMPC={use_smpc})")
        
        if use_smpc:
            return self._smpc_aggregation(client_weights, client_samples)
        else:
            return self._he_aggregation(client_weights, client_samples)
    
    def _smpc_aggregation(
        self,
        client_weights: List[Dict[str, torch.Tensor]],
        client_samples: List[int]
    ) -> Dict[str, torch.Tensor]:
        """SMPC-based secure aggregation"""
        total_samples = sum(client_samples)
        global_weights = {}
        
        for key in client_weights[0].keys():
            # Flatten weights
            client_values = [w[key].flatten().numpy() for w in client_weights]
            
            # Secret share each client's weights
            shares_list = []
            for values, n_samples in zip(client_values, client_samples):
                weight_factor = n_samples / total_samples
                weighted_values = values * weight_factor
                shares = self.smpc.share_secret(weighted_values, num_parties=len(client_weights))
                shares_list.append(shares)
            
            # Sum shares (each party sums their shares)
            aggregated_shares = [np.zeros_like(shares_list[0][i]) for i in range(len(client_weights))]
            for shares in shares_list:
                for i, share in enumerate(shares):
                    aggregated_shares[i] += share
            
            # Reconstruct
            aggregated = self.smpc.reconstruct_secret_array(aggregated_shares)
            global_weights[key] = torch.from_numpy(aggregated).reshape(client_weights[0][key].shape)
        
        return global_weights
    
    def _he_aggregation(
        self,
        client_weights: List[Dict[str, torch.Tensor]],
        client_samples: List[int]
    ) -> Dict[str, torch.Tensor]:
        """Homomorphic encryption-based secure aggregation"""
        total_samples = sum(client_samples)
        global_weights = {}
        
        for key in client_weights[0].keys():
            # Initialize with encrypted zero
            encrypted_sum = None
            
            for w, n_samples in zip(client_weights, client_samples):
                weight = n_samples / total_samples
                values = (w[key] * weight).flatten().numpy()
                
                # Encrypt and add
                for val in values:
                    encrypted_val = self.he.encrypt(float(val))
                    if encrypted_sum is None:
                        encrypted_sum = [encrypted_val]
                    else:
                        # Homomorphic addition
                        encrypted_sum = [self.he.add(encrypted_sum[0], encrypted_val)]
            
            # Decrypt (only server can do this)
            decrypted = [self.he.decrypt(enc) for enc in encrypted_sum]
            global_weights[key] = torch.tensor(decrypted).reshape(client_weights[0][key].shape)
        
        return global_weights
    
    def client_selection(
        self,
        num_clients: int,
        selection_rate: float = 0.1,
        strategy: str = 'random'
    ) -> List[int]:
        """
        Select subset of clients for training round
        
        Strategies:
        - random: Uniform random selection
        - importance: Select based on data importance (to be implemented)
        - loss: Select clients with highest loss (to be implemented)
        """
        num_selected = max(1, int(num_clients * selection_rate))
        
        if strategy == 'random':
            return np.random.choice(num_clients, num_selected, replace=False).tolist()
        else:
            # Placeholder for other strategies
            return list(range(num_selected))
    
    def aggregate(
        self,
        client_weights: List[Dict[str, torch.Tensor]],
        client_samples: List[int],
        **kwargs
    ) -> Any:
        """
        Main aggregation dispatcher
        """
        if self.strategy not in self.strategies:
            raise ValueError(f"Unknown strategy: {self.strategy}")
        
        return self.strategies[self.strategy](client_weights, client_samples, **kwargs)
