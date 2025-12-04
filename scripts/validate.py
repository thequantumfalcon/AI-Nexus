"""
AI-Nexus Validation Script
===========================

Validates the entire system and runs comprehensive tests.
"""

import sys
import subprocess
from pathlib import Path
import importlib
from typing import List, Tuple
import time

# Colors for terminal output
class Colors:
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BLUE = '\033[94m'
    BOLD = '\033[1m'
    END = '\033[0m'


def print_header(text: str):
    """Print formatted header"""
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{text:^60}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.END}\n")


def print_success(text: str):
    """Print success message"""
    print(f"{Colors.GREEN}✓ {text}{Colors.END}")


def print_warning(text: str):
    """Print warning message"""
    print(f"{Colors.YELLOW}⚠ {text}{Colors.END}")


def print_error(text: str):
    """Print error message"""
    print(f"{Colors.RED}✗ {text}{Colors.END}")


def check_python_version() -> bool:
    """Check Python version"""
    version = sys.version_info
    if version.major >= 3 and version.minor >= 10:
        print_success(f"Python version: {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print_error(f"Python 3.10+ required, found {version.major}.{version.minor}")
        return False


def check_dependencies() -> Tuple[bool, List[str]]:
    """Check if all dependencies are installed"""
    required_packages = [
        'torch',
        'transformers',
        'numpy',
        'scipy',
        'sklearn',  # Import name for scikit-learn
        'grpcio',
        'flask',
        'cryptography',
        'yaml',  # Import name for pyyaml
        'pytest'
    ]
    
    missing = []
    for package in required_packages:
        try:
            importlib.import_module(package)
            print_success(f"Package '{package}' installed")
        except ImportError:
            print_error(f"Package '{package}' missing")
            missing.append(package)
    
    return len(missing) == 0, missing


def check_directory_structure() -> bool:
    """Validate directory structure"""
    required_dirs = [
        'core',
        'services',
        'services/nlp',
        'services/ml',
        'services/blockchain',
        'proto',
        'config',
        'tests',
        'scripts'
    ]
    
    all_exist = True
    for dir_path in required_dirs:
        if Path(dir_path).exists():
            print_success(f"Directory '{dir_path}' exists")
        else:
            print_error(f"Directory '{dir_path}' missing")
            all_exist = False
    
    return all_exist


def check_core_modules() -> bool:
    """Test importing core modules"""
    modules = [
        'core.crypto',
        'core.config',
        'core.logger',
        'core.metrics',
        'services.nlp.nlp_engine',
        'services.ml.ml_engine',
        'services.blockchain.blockchain'
    ]
    
    all_imported = True
    for module in modules:
        try:
            importlib.import_module(module)
            print_success(f"Module '{module}' imports successfully")
        except Exception as e:
            print_error(f"Module '{module}' failed to import: {e}")
            all_imported = False
    
    return all_imported


def run_unit_tests() -> bool:
    """Run pytest unit tests"""
    print("\nRunning unit tests...")
    try:
        result = subprocess.run(
            [sys.executable, '-m', 'pytest', 'tests/', '-v', '--tb=short'],
            capture_output=True,
            text=True,
            timeout=300
        )
        
        print(result.stdout)
        if result.returncode == 0:
            print_success("All unit tests passed")
            return True
        else:
            print_error("Some unit tests failed")
            print(result.stderr)
            return False
    except subprocess.TimeoutExpired:
        print_error("Tests timed out")
        return False
    except Exception as e:
        print_error(f"Failed to run tests: {e}")
        return False


def validate_crypto_functionality() -> bool:
    """Validate cryptographic operations"""
    print("\nValidating cryptography...")
    try:
        from core.crypto import (
            EncryptionManager,
            DifferentialPrivacy,
            HomomorphicEncryption
        )
        
        # Test encryption
        em = EncryptionManager()
        em.generate_symmetric_key()
        plaintext = b"Test encryption"
        ciphertext = em.encrypt_symmetric(plaintext)
        decrypted = em.decrypt_symmetric(ciphertext)
        
        if decrypted == plaintext:
            print_success("Symmetric encryption works")
        else:
            print_error("Symmetric encryption failed")
            return False
        
        # Test DP
        import numpy as np
        dp = DifferentialPrivacy()
        data = np.array([1.0, 2.0, 3.0])
        noisy = dp.add_laplace_noise(data)
        
        if noisy.shape == data.shape:
            print_success("Differential privacy works")
        else:
            print_error("Differential privacy failed")
            return False
        
        return True
        
    except Exception as e:
        print_error(f"Crypto validation failed: {e}")
        return False


def validate_nlp_engine() -> bool:
    """Validate NLP engine"""
    print("\nValidating NLP engine...")
    try:
        from services.nlp.nlp_engine import SecureNLPEngine
        
        engine = SecureNLPEngine()
        
        # Test sentiment analysis
        result = engine.analyze_sentiment("This is great!")
        if 'sentiment' in result and 'score' in result:
            print_success(f"NLP sentiment analysis works: {result['sentiment']}")
        else:
            print_error("NLP sentiment analysis failed")
            return False
        
        # Test capabilities
        caps = engine.get_capabilities()
        if 'supported_tasks' in caps:
            print_success(f"NLP supports {len(caps['supported_tasks'])} tasks")
        
        return True
        
    except Exception as e:
        print_error(f"NLP validation failed: {e}")
        return False


def validate_ml_engine() -> bool:
    """Validate ML engine"""
    print("\nValidating ML engine...")
    try:
        from services.ml.ml_engine import PrivacyPreservingMLEngine
        import numpy as np
        
        engine = PrivacyPreservingMLEngine()
        
        # Generate test data
        X = np.random.randn(50, 10)
        y = np.random.randint(0, 2, 50)
        
        # Train model
        result = engine.train_model(
            "neural_net",
            X,
            y,
            hyperparameters={'epochs': 2, 'batch_size': 16},
            use_differential_privacy=True
        )
        
        if result.model_id and result.final_accuracy >= 0:
            print_success(f"ML training works: accuracy={result.final_accuracy:.4f}")
        else:
            print_error("ML training failed")
            return False
        
        # Test prediction
        pred_result = engine.predict(result.model_id, X[:5])
        if pred_result.prediction is not None:
            print_success("ML prediction works")
        else:
            print_error("ML prediction failed")
            return False
        
        return True
        
    except Exception as e:
        print_error(f"ML validation failed: {e}")
        return False


def validate_blockchain() -> bool:
    """Validate blockchain functionality"""
    print("\nValidating blockchain...")
    try:
        from services.blockchain.blockchain import (
            AIBlockchain,
            TokenManager,
            GovernanceSystem
        )
        
        # Test blockchain
        blockchain = AIBlockchain(difficulty=2)
        blockchain.add_block({'test': 'data'})
        
        if blockchain.is_chain_valid():
            print_success("Blockchain validation works")
        else:
            print_error("Blockchain validation failed")
            return False
        
        # Test token manager
        tm = TokenManager()
        tm.balances['test'] = 1000
        success = tm.transfer('test', 'recipient', 100)
        
        if success and tm.get_balance('recipient') == 100:
            print_success("Token transfers work")
        else:
            print_error("Token transfers failed")
            return False
        
        # Test governance
        gov = GovernanceSystem(tm)
        tm.balances['proposer'] = 2000
        proposal_id = gov.create_proposal(
            'proposer',
            'Test',
            'Test proposal'
        )
        
        if proposal_id in gov.proposals:
            print_success("Governance proposals work")
        else:
            print_error("Governance proposals failed")
            return False
        
        return True
        
    except Exception as e:
        print_error(f"Blockchain validation failed: {e}")
        return False


def generate_validation_report(results: dict) -> str:
    """Generate validation report"""
    report = f"\n{Colors.BOLD}{'='*60}\n"
    report += f"{'AI-NEXUS VALIDATION REPORT':^60}\n"
    report += f"{'='*60}{Colors.END}\n\n"
    
    total = len(results)
    passed = sum(1 for v in results.values() if v)
    
    for test, result in results.items():
        status = f"{Colors.GREEN}PASS{Colors.END}" if result else f"{Colors.RED}FAIL{Colors.END}"
        report += f"  {test:<40} {status}\n"
    
    report += f"\n{Colors.BOLD}{'='*60}\n"
    report += f"Total: {total} | Passed: {passed} | Failed: {total - passed}\n"
    
    if passed == total:
        report += f"{Colors.GREEN}✓ ALL VALIDATIONS PASSED!{Colors.END}\n"
    else:
        report += f"{Colors.RED}✗ SOME VALIDATIONS FAILED{Colors.END}\n"
    
    report += f"{'='*60}{Colors.END}\n"
    
    return report


def main():
    """Main validation function"""
    print_header("AI-NEXUS SYSTEM VALIDATION")
    
    start_time = time.time()
    
    results = {}
    
    # Run validations
    print_header("1. Environment Check")
    results['Python Version'] = check_python_version()
    deps_ok, missing = check_dependencies()
    results['Dependencies'] = deps_ok
    
    print_header("2. Structure Check")
    results['Directory Structure'] = check_directory_structure()
    results['Core Modules'] = check_core_modules()
    
    print_header("3. Component Validation")
    results['Cryptography'] = validate_crypto_functionality()
    results['NLP Engine'] = validate_nlp_engine()
    results['ML Engine'] = validate_ml_engine()
    results['Blockchain'] = validate_blockchain()
    
    print_header("4. Unit Tests")
    results['Unit Tests'] = run_unit_tests()
    
    # Generate report
    elapsed = time.time() - start_time
    report = generate_validation_report(results)
    print(report)
    
    print(f"\nValidation completed in {elapsed:.2f} seconds\n")
    
    # Exit code
    all_passed = all(results.values())
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
