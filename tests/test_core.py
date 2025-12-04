"""
Test suite for AI-Nexus Core Components
========================================

Tests cryptography, configuration, and utilities.
"""

import pytest
import numpy as np
from pathlib import Path
import tempfile

from core.crypto import (
    EncryptionManager,
    DifferentialPrivacy,
    HomomorphicEncryption,
    SecureMultiPartyComputation,
    ZeroKnowledgeProof,
    hash_data,
    derive_key
)
from core.config import Config
from core.logger import setup_logger
from core.metrics import MetricsCollector


class TestEncryptionManager:
    """Test encryption functionality"""
    
    def test_symmetric_encryption(self):
        """Test AES encryption/decryption"""
        em = EncryptionManager()
        em.generate_symmetric_key()
        
        plaintext = b"Secret message for AI-Nexus"
        ciphertext = em.encrypt_symmetric(plaintext)
        decrypted = em.decrypt_symmetric(ciphertext)
        
        assert decrypted == plaintext
        assert ciphertext != plaintext
    
    def test_asymmetric_encryption(self):
        """Test RSA encryption/decryption"""
        em = EncryptionManager()
        em.generate_keypair()
        
        plaintext = b"Asymmetric test"
        ciphertext = em.encrypt_asymmetric(plaintext)
        decrypted = em.decrypt_asymmetric(ciphertext)
        
        assert decrypted == plaintext
    
    def test_key_derivation(self):
        """Test password-based key derivation"""
        password = "secure_password_123"
        key1, salt = derive_key(password)
        key2, _ = derive_key(password, salt)
        
        assert key1 == key2
        assert len(key1) == 32  # 256 bits


class TestDifferentialPrivacy:
    """Test differential privacy mechanisms"""
    
    def test_laplace_noise(self):
        """Test Laplace noise addition"""
        dp = DifferentialPrivacy(epsilon=1.0)
        
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        noisy_data = dp.add_laplace_noise(data)
        
        assert noisy_data.shape == data.shape
        assert not np.array_equal(noisy_data, data)
    
    def test_gaussian_noise(self):
        """Test Gaussian noise for (ε, δ)-DP"""
        dp = DifferentialPrivacy(epsilon=1.0, delta=1e-5)
        
        data = np.array([10.0, 20.0, 30.0])
        noisy_data = dp.add_gaussian_noise(data)
        
        assert noisy_data.shape == data.shape
        assert not np.array_equal(noisy_data, data)
    
    def test_gradient_clipping(self):
        """Test gradient clipping"""
        dp = DifferentialPrivacy()
        
        gradients = np.array([10.0, 20.0, 30.0])
        clipped = dp.clip_gradients(gradients, clip_norm=1.0)
        
        assert np.linalg.norm(clipped) <= 1.0
    
    def test_privacy_budget(self):
        """Test privacy budget tracking"""
        dp = DifferentialPrivacy(epsilon=2.0)
        
        assert dp.check_privacy_budget()
        dp.consume_budget(1.0)
        assert dp.privacy_budget == 1.0
        dp.consume_budget(1.0)
        assert dp.privacy_budget == 0.0


class TestHomomorphicEncryption:
    """Test homomorphic encryption (mock)"""
    
    def test_encrypt_decrypt(self):
        """Test basic HE encryption"""
        he = HomomorphicEncryption()
        
        plaintext = np.array([1.0, 2.0, 3.0])
        ciphertext = he.encrypt(plaintext)
        decrypted = he.decrypt(ciphertext)
        
        np.testing.assert_array_almost_equal(decrypted, plaintext)
    
    def test_homomorphic_addition(self):
        """Test addition on encrypted values"""
        he = HomomorphicEncryption()
        
        a = np.array([5.0])
        b = np.array([3.0])
        
        ct_a = he.encrypt(a)
        ct_b = he.encrypt(b)
        ct_sum = he.add_encrypted(ct_a, ct_b)
        
        result = he.decrypt(ct_sum)
        np.testing.assert_array_almost_equal(result, a + b)


class TestSecureMultiPartyComputation:
    """Test SMPC functionality"""
    
    def test_secret_sharing(self):
        """Test Shamir secret sharing"""
        smpc = SecureMultiPartyComputation(num_parties=5, threshold=3)
        
        secret = 42
        shares = smpc.secret_share(secret)
        
        assert len(shares) == 5
    
    def test_secret_reconstruction(self):
        """Test secret reconstruction"""
        smpc = SecureMultiPartyComputation(num_parties=5, threshold=3)
        
        secret = 12345
        shares = smpc.secret_share(secret)
        
        # Use first 3 shares
        shares_with_indices = [(i+1, shares[i]) for i in range(3)]
        reconstructed = smpc.reconstruct_secret(shares_with_indices)
        
        assert reconstructed == secret


class TestZeroKnowledgeProof:
    """Test ZKP functionality"""
    
    def test_commitment(self):
        """Test Pedersen commitment"""
        zkp = ZeroKnowledgeProof()
        
        value = 100
        commitment, randomness = zkp.generate_commitment(value)
        
        is_valid = zkp.verify_commitment(commitment, value, randomness)
        assert is_valid
    
    def test_range_proof(self):
        """Test range proof"""
        zkp = ZeroKnowledgeProof()
        
        value = 50
        proof = zkp.create_range_proof(value, 0, 100)
        
        assert proof['in_range'] == True


class TestConfig:
    """Test configuration management"""
    
    def test_load_config(self):
        """Test loading configuration"""
        config = Config()
        
        assert 'network' in config._config
        assert 'security' in config._config
    
    def test_get_config_value(self):
        """Test getting config values"""
        config = Config()
        
        node_id = config.get('network.node_id')
        assert node_id is not None
    
    def test_set_config_value(self):
        """Test setting config values"""
        config = Config()
        
        config.set('network.node_id', 999)
        assert config.get('network.node_id') == 999
    
    def test_save_config(self):
        """Test saving configuration"""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.yaml') as f:
            temp_path = f.name
        
        try:
            config = Config()
            config.set('test.value', 123)
            config.save(temp_path)
            
            # Load it back
            config2 = Config(temp_path)
            assert config2.get('test.value') == 123
        finally:
            Path(temp_path).unlink(missing_ok=True)


class TestMetrics:
    """Test metrics collection"""
    
    def test_metrics_collector_init(self):
        """Test metrics collector initialization"""
        mc = MetricsCollector(port=9091)
        assert mc.port == 9091
        assert not mc.started
    
    def test_record_task(self):
        """Test recording task metrics"""
        mc = MetricsCollector()
        mc.record_task('test_task', 1.5, 'success')
        # No assertion - just verify it doesn't crash
    
    def test_update_node_health(self):
        """Test node health update"""
        mc = MetricsCollector()
        mc.update_node_health(1, True)
        mc.update_node_health(2, False)


class TestHashFunctions:
    """Test hashing utilities"""
    
    def test_sha256(self):
        """Test SHA-256 hashing"""
        data = b"test data"
        hash1 = hash_data(data, "sha256")
        hash2 = hash_data(data, "sha256")
        
        assert hash1 == hash2
        assert len(hash1) == 64  # 256 bits in hex
    
    def test_different_algorithms(self):
        """Test different hash algorithms"""
        data = b"test"
        
        sha256 = hash_data(data, "sha256")
        sha512 = hash_data(data, "sha512")
        blake2b = hash_data(data, "blake2b")
        
        assert len(sha256) == 64
        assert len(sha512) == 128
        assert len(blake2b) == 128


class TestPerformance:
    """Performance tests (non-benchmark)"""
    
    def test_encryption_speed(self):
        """Test encryption performance"""
        em = EncryptionManager()
        em.generate_symmetric_key()
        data = b"x" * 1024  # 1KB
        
        import time
        start = time.time()
        result = em.encrypt_symmetric(data)
        duration = time.time() - start
        
        assert len(result) > 0
        assert duration < 1.0  # Should complete in under 1 second
    
    def test_dp_noise_speed(self):
        """Test differential privacy noise addition performance"""
        dp = DifferentialPrivacy()
        data = np.random.randn(1000)
        
        import time
        start = time.time()
        result = dp.add_laplace_noise(data, sensitivity=1.0)
        duration = time.time() - start
        
        assert result.shape == data.shape
        assert duration < 1.0  # Should complete in under 1 second


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
