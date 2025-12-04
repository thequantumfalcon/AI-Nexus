"""
Setup configuration for AI-Nexus
"""
from setuptools import setup, find_packages

setup(
    name="ai-nexus",
    version="0.1.0",
    description="Decentralized AI Platform with Privacy-Preserving NLP and Federated ML",
    author="AI-Nexus Team",
    packages=find_packages(where="."),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.30.0",
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "scikit-learn>=1.3.0",
        "pandas>=2.0.0",
        "grpcio>=1.50.0",
        "grpcio-tools>=1.50.0",
        "protobuf>=4.21.0",
        "flask>=3.0.0",
        "flask-cors>=4.0.0",
        "aiohttp>=3.8.0",
        "cryptography>=41.0.0",
        "pyyaml>=6.0",
        "phe>=1.5.0",
        "web3>=6.0.0",
        "prometheus-client>=0.17.0",
        "psutil>=5.9.0",
        "pytest>=7.4.0",
        "pytest-asyncio>=0.21.0",
        "pytest-cov>=4.1.0",
    ],
    extras_require={
        'explainability': ['shap>=0.42.0', 'lime>=0.2.0'],
        'dev': ['black', 'flake8', 'mypy', 'rich', 'typer'],
    },
    entry_points={
        'console_scripts': [
            'ainexus-node=scripts.start_node:main',
            'ainexus-validate=scripts.validate:main',
        ],
    },
)
