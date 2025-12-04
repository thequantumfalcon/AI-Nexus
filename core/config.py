"""
Configuration management for AI-Nexus
======================================

Handles loading and managing configuration from YAML files.
"""

import os
import yaml
from typing import Dict, Any, Optional
from pathlib import Path
import logging


class Config:
    """Configuration manager for AI-Nexus"""
    
    def __init__(self, config_path: Optional[str] = None):
        if config_path is None:
            # Default to config/config.yaml
            base_dir = Path(__file__).parent.parent
            config_path = base_dir / "config" / "config.yaml"
        
        self.config_path = Path(config_path)
        self._config: Dict[str, Any] = {}
        self._load_config()
        
    def _load_config(self):
        """Load configuration from YAML file"""
        try:
            with open(self.config_path, 'r') as f:
                self._config = yaml.safe_load(f)
            logging.info(f"Loaded configuration from {self.config_path}")
        except FileNotFoundError:
            logging.warning(f"Config file not found: {self.config_path}. Using defaults.")
            self._config = self._get_default_config()
        except yaml.YAMLError as e:
            logging.error(f"Error parsing YAML config: {e}")
            self._config = self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Return default configuration"""
        return {
            'network': {
                'node_id': 1,
                'bind_address': '0.0.0.0',
                'port': 5001,
                'max_connections': 100,
            },
            'security': {
                'encryption': {
                    'algorithm': 'kyber512',
                    'key_rotation_interval': 86400,
                },
                'privacy': {
                    'differential_privacy': {
                        'enabled': True,
                        'epsilon': 1.0,
                        'delta': 1e-5,
                    }
                }
            },
            'ai_services': {
                'nlp': {
                    'enabled': True,
                    'model': 'meta-llama/Llama-3.1-8B',
                    'max_length': 8192,
                },
                'ml': {
                    'enabled': True,
                    'framework': 'pytorch',
                }
            }
        }
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value using dot notation"""
        keys = key.split('.')
        value = self._config
        
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
                if value is None:
                    return default
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any):
        """Set configuration value using dot notation"""
        keys = key.split('.')
        config = self._config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
    
    def save(self, path: Optional[str] = None):
        """Save configuration to file"""
        save_path = Path(path) if path else self.config_path
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(save_path, 'w') as f:
            yaml.dump(self._config, f, default_flow_style=False)
        
        logging.info(f"Saved configuration to {save_path}")
    
    @property
    def network(self) -> Dict[str, Any]:
        """Get network configuration"""
        return self._config.get('network', {})
    
    @property
    def security(self) -> Dict[str, Any]:
        """Get security configuration"""
        return self._config.get('security', {})
    
    @property
    def ai_services(self) -> Dict[str, Any]:
        """Get AI services configuration"""
        return self._config.get('ai_services', {})
    
    @property
    def blockchain(self) -> Dict[str, Any]:
        """Get blockchain configuration"""
        return self._config.get('blockchain', {})
    
    @property
    def resources(self) -> Dict[str, Any]:
        """Get resource management configuration"""
        return self._config.get('resources', {})
    
    @property
    def monitoring(self) -> Dict[str, Any]:
        """Get monitoring configuration"""
        return self._config.get('monitoring', {})
    
    def __repr__(self) -> str:
        return f"Config(path={self.config_path})"


# Global config instance
_global_config: Optional[Config] = None


def get_config(config_path: Optional[str] = None) -> Config:
    """Get global configuration instance"""
    global _global_config
    if _global_config is None or config_path is not None:
        _global_config = Config(config_path)
    return _global_config
