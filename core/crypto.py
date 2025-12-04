"""
Cryptographic utilities for AI-Nexus
=====================================

Provides encryption, homomorphic encryption, differential privacy,
and secure multi-party computation primitives.
"""

import hashlib
import secrets
from typing import List, Tuple, Optional, Any
import numpy as np
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64


class EncryptionManager:
    """Manages various encryption schemes for AI-Nexus"""
    
    def __init__(self, key_size: int = 256):
        self.key_size = key_size
        self._private_key = None
        self._public_key = None
        self._symmetric_key = None
        
    def generate_keypair(self):
        """Generate RSA keypair for asymmetric encryption"""
        self._private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=4096,
            backend=default_backend()
        )
        self._public_key = self._private_key.public_key()
        return self._public_key
    
    def generate_symmetric_key(self) -> bytes:
        """Generate AES symmetric key"""
        self._symmetric_key = secrets.token_bytes(self.key_size // 8)
        return self._symmetric_key
    
    def encrypt_symmetric(self, plaintext: bytes, key: Optional[bytes] = None) -> bytes:
        """Encrypt data using AES-GCM"""
        if key is None:
            if self._symmetric_key is None:
                self.generate_symmetric_key()
            key = self._symmetric_key
            
        iv = secrets.token_bytes(12)
        cipher = Cipher(
            algorithms.AES(key),
            modes.GCM(iv),
            backend=default_backend()
        )
        encryptor = cipher.encryptor()
        ciphertext = encryptor.update(plaintext) + encryptor.finalize()
        return iv + encryptor.tag + ciphertext
    
    def decrypt_symmetric(self, ciphertext: bytes, key: Optional[bytes] = None) -> bytes:
        """Decrypt data using AES-GCM"""
        if key is None:
            if self._symmetric_key is None:
                raise ValueError("No symmetric key available")
            key = self._symmetric_key
            
        iv = ciphertext[:12]
        tag = ciphertext[12:28]
        actual_ciphertext = ciphertext[28:]
        
        cipher = Cipher(
            algorithms.AES(key),
            modes.GCM(iv, tag),
            backend=default_backend()
        )
        decryptor = cipher.decryptor()
        return decryptor.update(actual_ciphertext) + decryptor.finalize()
    
    def encrypt_asymmetric(self, plaintext: bytes, public_key=None) -> bytes:
        """Encrypt using RSA public key"""
        if public_key is None:
            public_key = self._public_key
        if public_key is None:
            raise ValueError("No public key available")
            
        ciphertext = public_key.encrypt(
            plaintext,
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None
            )
        )
        return ciphertext
    
    def decrypt_asymmetric(self, ciphertext: bytes) -> bytes:
        """Decrypt using RSA private key"""
        if self._private_key is None:
            raise ValueError("No private key available")
            
        plaintext = self._private_key.decrypt(
            ciphertext,
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None
            )
        )
        return plaintext


class DifferentialPrivacy:
    """Implements differential privacy mechanisms"""
    
    def __init__(self, epsilon: float = 1.0, delta: float = 1e-5):
        self.epsilon = epsilon
        self.delta = delta
        self.privacy_budget = epsilon
        
    def add_laplace_noise(self, data: np.ndarray, sensitivity: float = 1.0) -> np.ndarray:
        """Add Laplace noise for differential privacy"""
        scale = sensitivity / self.epsilon
        noise = np.random.laplace(0, scale, data.shape)
        return data + noise
    
    def add_gaussian_noise(self, data: np.ndarray, sensitivity: float = 1.0) -> np.ndarray:
        """Add Gaussian noise for (ε, δ)-differential privacy"""
        sigma = sensitivity * np.sqrt(2 * np.log(1.25 / self.delta)) / self.epsilon
        noise = np.random.normal(0, sigma, data.shape)
        return data + noise
    
    def clip_gradients(self, gradients: np.ndarray, clip_norm: float = 1.0) -> np.ndarray:
        """Clip gradients to bound sensitivity"""
        norm = np.linalg.norm(gradients)
        if norm > clip_norm:
            return gradients * (clip_norm / norm)
        return gradients
    
    def check_privacy_budget(self) -> bool:
        """Check if privacy budget is exhausted"""
        return self.privacy_budget > 0
    
    def consume_budget(self, amount: float):
        """Consume privacy budget"""
        self.privacy_budget -= amount
        if self.privacy_budget < 0:
            raise ValueError("Privacy budget exhausted")


class HomomorphicEncryption:
    """Simplified homomorphic encryption wrapper (mock for SEAL/CKKS)"""
    
    def __init__(self, scheme: str = "ckks"):
        self.scheme = scheme
        # In production, initialize actual HE library (Microsoft SEAL, etc.)
        self._context = None
        
    def encrypt(self, plaintext: np.ndarray) -> bytes:
        """Encrypt data homomorphically (simplified)"""
        # Mock implementation - in production, use actual HE library
        # This would use Microsoft SEAL's CKKS scheme
        return base64.b64encode(plaintext.tobytes())
    
    def decrypt(self, ciphertext: bytes) -> np.ndarray:
        """Decrypt homomorphically encrypted data"""
        # Mock implementation
        data = base64.b64decode(ciphertext)
        return np.frombuffer(data, dtype=np.float64)
    
    def add_encrypted(self, ct1: bytes, ct2: bytes) -> bytes:
        """Add two encrypted values"""
        # Mock - in production, use HE library's add operation
        v1 = self.decrypt(ct1)
        v2 = self.decrypt(ct2)
        return self.encrypt(v1 + v2)
    
    def multiply_encrypted(self, ct1: bytes, ct2: bytes) -> bytes:
        """Multiply two encrypted values"""
        # Mock - in production, use HE library's multiply operation
        v1 = self.decrypt(ct1)
        v2 = self.decrypt(ct2)
        return self.encrypt(v1 * v2)


class SecureMultiPartyComputation:
    """Simplified SMPC implementation"""
    
    def __init__(self, num_parties: int = 3, threshold: int = 2):
        self.num_parties = num_parties
        self.threshold = threshold
        
    def secret_share(self, secret: int, prime: int = 2**31 - 1) -> List[int]:
        """Shamir's secret sharing"""
        shares = []
        coefficients = [secret] + [secrets.randbelow(prime) for _ in range(self.threshold - 1)]
        
        for i in range(1, self.num_parties + 1):
            share = sum(coef * pow(i, power, prime) for power, coef in enumerate(coefficients)) % prime
            shares.append(share)
        
        return shares
    
    def share_secret(self, secret_array: np.ndarray, num_parties: Optional[int] = None, prime: int = 2**31 - 1) -> List[np.ndarray]:
        """
        Shamir's secret sharing for numpy arrays
        
        Args:
            secret_array: Array of secrets to share
            num_parties: Number of parties (defaults to self.num_parties)
            prime: Prime modulus
            
        Returns:
            List of share arrays, one per party
        """
        if num_parties is None:
            num_parties = self.num_parties
            
        # Flatten array if needed
        original_shape = secret_array.shape
        secret_flat = secret_array.flatten()
        
        # Generate shares for each element
        all_shares = [[] for _ in range(num_parties)]
        
        for secret_val in secret_flat:
            # Convert to integer for modular arithmetic
            secret_int = int(secret_val * 1000000) % prime  # Scale for precision
            
            # Generate polynomial coefficients
            coefficients = [secret_int] + [secrets.randbelow(prime) for _ in range(self.threshold - 1)]
            
            # Evaluate polynomial at different points
            for i in range(1, num_parties + 1):
                share = sum(coef * pow(i, power, prime) for power, coef in enumerate(coefficients)) % prime
                all_shares[i-1].append(share)
        
        # Convert back to numpy arrays
        share_arrays = [np.array(shares).reshape(original_shape) for shares in all_shares]
        
        return share_arrays
    
    def reconstruct_secret(self, shares: List[Tuple[int, int]], prime: int = 2**31 - 1) -> int:
        """Reconstruct secret from shares using Lagrange interpolation"""
        if len(shares) < self.threshold:
            raise ValueError(f"Need at least {self.threshold} shares")
        
        secret = 0
        for i, (x_i, y_i) in enumerate(shares[:self.threshold]):
            numerator = 1
            denominator = 1
            for j, (x_j, _) in enumerate(shares[:self.threshold]):
                if i != j:
                    numerator = (numerator * (-x_j)) % prime
                    denominator = (denominator * (x_i - x_j)) % prime
            
            lagrange = (numerator * pow(denominator, -1, prime)) % prime
            secret = (secret + y_i * lagrange) % prime
        
        return secret
    
    def reconstruct_secret_array(self, share_arrays: List[np.ndarray], prime: int = 2**31 - 1) -> np.ndarray:
        """
        Reconstruct secret array from shares using Lagrange interpolation
        
        Args:
            share_arrays: List of share arrays from each party
            prime: Prime modulus
            
        Returns:
            Reconstructed secret array
        """
        if len(share_arrays) < self.threshold:
            raise ValueError(f"Need at least {self.threshold} shares")
        
        # Get shape
        shape = share_arrays[0].shape
        
        # Flatten all shares
        flat_shares = [s.flatten() for s in share_arrays[:self.threshold]]
        
        # Reconstruct each element
        reconstructed = []
        for idx in range(len(flat_shares[0])):
            # Get shares for this element (x=party_id, y=share_value)
            element_shares = [(i+1, int(flat_shares[i][idx])) for i in range(self.threshold)]
            
            # Lagrange interpolation
            secret = 0
            for i, (x_i, y_i) in enumerate(element_shares):
                numerator = 1
                denominator = 1
                for j, (x_j, _) in enumerate(element_shares):
                    if i != j:
                        numerator = (numerator * (-x_j)) % prime
                        denominator = (denominator * (x_i - x_j)) % prime
                
                lagrange = (numerator * pow(denominator, -1, prime)) % prime
                secret = (secret + y_i * lagrange) % prime
            
            # Convert back to float (unscale)
            reconstructed.append(float(secret) / 1000000.0)
        
        return np.array(reconstructed).reshape(shape)


class ZeroKnowledgeProof:
    """Simplified Zero-Knowledge Proof implementation"""
    
    def __init__(self):
        self.commitments = {}
        
    def generate_commitment(self, value: int, randomness: Optional[int] = None) -> Tuple[int, int]:
        """Generate Pedersen commitment"""
        if randomness is None:
            randomness = secrets.randbelow(2**256)
        
        # Simplified - in production, use elliptic curve points
        g = 7  # Generator
        h = 11  # Another generator
        p = 2**31 - 1  # Prime
        
        commitment = (pow(g, value, p) * pow(h, randomness, p)) % p
        return commitment, randomness
    
    def verify_commitment(self, commitment: int, value: int, randomness: int) -> bool:
        """Verify a commitment"""
        g = 7
        h = 11
        p = 2**31 - 1
        
        expected = (pow(g, value, p) * pow(h, randomness, p)) % p
        return commitment == expected
    
    def create_range_proof(self, value: int, min_val: int, max_val: int) -> dict:
        """Create proof that value is in range [min_val, max_val]"""
        # Simplified implementation
        commitment, randomness = self.generate_commitment(value)
        return {
            'commitment': commitment,
            'in_range': min_val <= value <= max_val,
            'proof_data': {'randomness': randomness}  # In production, don't reveal this!
        }


def hash_data(data: bytes, algorithm: str = "sha256") -> str:
    """Hash data using specified algorithm"""
    if algorithm == "sha256":
        return hashlib.sha256(data).hexdigest()
    elif algorithm == "sha512":
        return hashlib.sha512(data).hexdigest()
    elif algorithm == "blake2b":
        return hashlib.blake2b(data).hexdigest()
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}")


def derive_key(password: str, salt: Optional[bytes] = None) -> Tuple[bytes, bytes]:
    """Derive encryption key from password using PBKDF2"""
    if salt is None:
        salt = secrets.token_bytes(16)
    
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=100000,
        backend=default_backend()
    )
    key = kdf.derive(password.encode())
    return key, salt
