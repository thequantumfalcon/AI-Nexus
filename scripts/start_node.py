"""
Start AI-Nexus Node
===================

Main entry point for running an AI-Nexus node.
"""

import sys
import os
import argparse
import asyncio
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.config import get_config
from core.logger import setup_logger
from core.metrics import get_metrics_collector
from services.nlp.nlp_engine import SecureNLPEngine
from services.ml.ml_engine import PrivacyPreservingMLEngine
from services.blockchain.blockchain import TokenManager, GovernanceSystem

logger = setup_logger('ainexus.node', level='INFO', log_file='logs/node.log')


class AINodeRunner:
    """Main AI-Nexus node runner"""
    
    def __init__(self, node_id: int, port: int):
        self.node_id = node_id
        self.port = port
        self.config = get_config()
        
        # Update config with CLI args
        self.config.set('network.node_id', node_id)
        self.config.set('network.port', port)
        
        logger.info(f"Initializing AI-Nexus Node {node_id} on port {port}")
        
        # Initialize components
        self.metrics = get_metrics_collector()
        self.nlp_engine = None
        self.ml_engine = None
        self.token_manager = None
        self.governance = None
        
    def initialize_services(self):
        """Initialize all services"""
        logger.info("Initializing services...")
        
        # NLP Service
        if self.config.get('ai_services.nlp.enabled', True):
            logger.info("  Starting NLP service...")
            self.nlp_engine = SecureNLPEngine()
        
        # ML Service
        if self.config.get('ai_services.ml.enabled', True):
            logger.info("  Starting ML service...")
            self.ml_engine = PrivacyPreservingMLEngine()
        
        # Blockchain & Governance
        if self.config.get('blockchain.enabled', True):
            logger.info("  Starting blockchain service...")
            self.token_manager = TokenManager()
            self.governance = GovernanceSystem(self.token_manager)
        
        logger.info("All services initialized ✓")
    
    def start_metrics_server(self):
        """Start Prometheus metrics server"""
        if self.config.get('monitoring.enabled', True):
            metrics_port = self.config.get('monitoring.metrics_port', 9090)
            logger.info(f"Starting metrics server on port {metrics_port}")
            self.metrics.start()
    
    async def run(self):
        """Run the node"""
        logger.info("=" * 60)
        logger.info(f"AI-Nexus Node {self.node_id} Starting")
        logger.info("=" * 60)
        
        # Initialize
        self.initialize_services()
        self.start_metrics_server()
        
        logger.info("")
        logger.info("Node is running! 🚀")
        logger.info(f"  Node ID: {self.node_id}")
        logger.info(f"  Port: {self.port}")
        logger.info(f"  NLP: {'Enabled' if self.nlp_engine else 'Disabled'}")
        logger.info(f"  ML: {'Enabled' if self.ml_engine else 'Disabled'}")
        logger.info(f"  Blockchain: {'Enabled' if self.token_manager else 'Disabled'}")
        logger.info("")
        logger.info("Press Ctrl+C to stop")
        logger.info("")
        
        # Demo: Run some sample tasks
        await self.run_demo_tasks()
        
        # Keep running
        try:
            while True:
                await asyncio.sleep(10)
                self.metrics.update_system_metrics()
        except KeyboardInterrupt:
            logger.info("\nShutting down node...")
            self.shutdown()
    
    async def run_demo_tasks(self):
        """Run demonstration tasks"""
        logger.info("Running demonstration tasks...")
        
        # NLP Demo
        if self.nlp_engine:
            logger.info("\n--- NLP Demo ---")
            result = self.nlp_engine.process_text(
                "AI-Nexus is an amazing decentralized AI platform!",
                task_type="sentiment"
            )
            logger.info(f"Sentiment: {result.result} (confidence: {result.confidence:.2f})")
            logger.info(f"Processing time: {result.processing_time_ms}ms")
        
        # ML Demo
        if self.ml_engine:
            logger.info("\n--- ML Demo ---")
            import numpy as np
            
            # Generate sample data
            X_train = np.random.randn(100, 10)
            y_train = np.random.randint(0, 2, 100)
            
            # Train model
            result = self.ml_engine.train_model(
                "neural_net",
                X_train,
                y_train,
                hyperparameters={'epochs': 5, 'batch_size': 16},
                use_differential_privacy=True
            )
            logger.info(f"Model trained: {result.model_id}")
            logger.info(f"Final accuracy: {result.final_accuracy:.4f}")
            logger.info(f"Training time: {result.training_time_ms}ms")
        
        # Blockchain Demo
        if self.token_manager:
            logger.info("\n--- Blockchain Demo ---")
            
            # Create sample transaction
            self.token_manager.reward_node(f"node_{self.node_id}", 100)
            balance = self.token_manager.get_balance(f"node_{self.node_id}")
            logger.info(f"Node balance: {balance} AINEX tokens")
            
            # Validate blockchain
            is_valid = self.token_manager.blockchain.is_chain_valid()
            logger.info(f"Blockchain valid: {is_valid}")
        
        logger.info("\n--- Demo Complete ---\n")
    
    def shutdown(self):
        """Shutdown the node"""
        logger.info("Node shutdown complete")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Start AI-Nexus Node')
    parser.add_argument('--node-id', type=int, default=1, help='Node ID')
    parser.add_argument('--port', type=int, default=5001, help='Port number')
    parser.add_argument('--config', type=str, help='Path to config file')
    
    args = parser.parse_args()
    
    # Load config if specified
    if args.config:
        get_config(args.config)
    
    # Create and run node
    runner = AINodeRunner(args.node_id, args.port)
    
    # Run async
    asyncio.run(runner.run())


if __name__ == "__main__":
    main()
