"""
Setup script for AI-Nexus
==========================

Initializes the environment and downloads required models.
"""

import os
import sys
from pathlib import Path
import subprocess
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def check_python_version():
    """Ensure Python 3.10+"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 10):
        logger.error("Python 3.10 or higher is required")
        sys.exit(1)
    logger.info(f"Python version: {version.major}.{version.minor}.{version.micro}")


def create_directories():
    """Create necessary directories"""
    dirs = [
        'logs',
        'data',
        'models',
        'blockchain_data',
        'state'
    ]
    
    for dir_name in dirs:
        Path(dir_name).mkdir(parents=True, exist_ok=True)
        logger.info(f"Created directory: {dir_name}")


def install_dependencies():
    """Install Python dependencies"""
    logger.info("Installing Python dependencies...")
    
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "--upgrade", "pip"
        ])
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
        ])
        logger.info("Dependencies installed successfully")
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to install dependencies: {e}")
        sys.exit(1)


def generate_proto_files():
    """Generate Python code from proto files"""
    logger.info("Generating gRPC code from proto files...")
    
    proto_dir = Path("proto")
    if not proto_dir.exists():
        logger.warning("Proto directory not found, skipping code generation")
        return
    
    try:
        for proto_file in proto_dir.glob("*.proto"):
            logger.info(f"Processing {proto_file.name}")
            subprocess.check_call([
                sys.executable, "-m", "grpc_tools.protoc",
                f"-I{proto_dir}",
                f"--python_out=.",
                f"--grpc_python_out=.",
                str(proto_file)
            ])
        logger.info("Proto files generated successfully")
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to generate proto files: {e}")


def download_models():
    """Download required AI models"""
    logger.info("Downloading AI models (this may take a while)...")
    
    try:
        # Import here to avoid issues if transformers isn't installed yet
        from transformers import AutoTokenizer, AutoModel
        
        models = [
            'distilbert-base-uncased',
            'distilbert-base-uncased-finetuned-sst-2-english',
        ]
        
        for model_name in models:
            logger.info(f"Downloading {model_name}...")
            try:
                AutoTokenizer.from_pretrained(model_name)
                AutoModel.from_pretrained(model_name)
                logger.info(f"  ✓ {model_name}")
            except Exception as e:
                logger.warning(f"  ✗ Failed to download {model_name}: {e}")
        
        logger.info("Model download complete")
    except ImportError:
        logger.warning("Transformers not installed, skipping model download")


def create_env_file():
    """Create .env file with default settings"""
    env_file = Path(".env")
    if env_file.exists():
        logger.info(".env file already exists, skipping")
        return
    
    env_content = """# AI-Nexus Environment Variables

# Node Configuration
NODE_ID=1
NODE_ADDRESS=0.0.0.0
NODE_PORT=5001

# API Configuration
GRPC_PORT=50051
REST_PORT=8080

# Security
ENCRYPTION_KEY=CHANGE_ME_IN_PRODUCTION

# Blockchain
ETHEREUM_NODE_URL=http://localhost:8545

# Monitoring
METRICS_PORT=9090

# Development
DEBUG=false
LOG_LEVEL=INFO
"""
    
    with open(env_file, 'w') as f:
        f.write(env_content)
    
    logger.info("Created .env file")


def run_tests():
    """Run basic tests to verify installation"""
    logger.info("Running verification tests...")
    
    try:
        # Test imports
        import torch
        import transformers
        import grpc
        import flask
        
        logger.info("  ✓ Core dependencies imported successfully")
        
        # Test CUDA availability
        if torch.cuda.is_available():
            logger.info(f"  ✓ CUDA available: {torch.cuda.get_device_name(0)}")
        else:
            logger.info("  ℹ CUDA not available, will use CPU")
        
        logger.info("Verification complete")
        
    except ImportError as e:
        logger.error(f"Import test failed: {e}")
        sys.exit(1)


def main():
    """Main setup function"""
    logger.info("=" * 60)
    logger.info("AI-Nexus Setup")
    logger.info("=" * 60)
    
    check_python_version()
    create_directories()
    create_env_file()
    install_dependencies()
    generate_proto_files()
    download_models()
    run_tests()
    
    logger.info("=" * 60)
    logger.info("Setup complete! 🚀")
    logger.info("=" * 60)
    logger.info("")
    logger.info("Next steps:")
    logger.info("  1. Review and update config/config.yaml")
    logger.info("  2. Edit .env file with your settings")
    logger.info("  3. Run: python scripts/start_node.py")
    logger.info("")


if __name__ == "__main__":
    main()
