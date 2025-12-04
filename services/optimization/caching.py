"""
Model caching and optimization.

Features:
- LRU cache for embeddings and model outputs
- Disk-backed persistent cache
- Cache warming
- TTL (Time-To-Live) support
- Memory-aware eviction
- Cache statistics
"""

import pickle
import json
import hashlib
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Callable
from dataclasses import dataclass, field
from collections import OrderedDict
from functools import wraps
import threading


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class CacheConfig:
    """Cache configuration."""
    # Memory cache
    max_memory_items: int = 1000
    max_memory_size_mb: int = 500
    
    # Disk cache
    enable_disk_cache: bool = True
    disk_cache_dir: str = "data/cache"
    max_disk_size_mb: int = 5000
    
    # TTL
    default_ttl_seconds: int = 3600  # 1 hour
    
    # Performance
    enable_compression: bool = True
    async_write: bool = True


# ============================================================================
# Cache Entry
# ============================================================================

@dataclass
class CacheEntry:
    """Cache entry with metadata."""
    key: str
    value: Any
    size_bytes: int
    created_at: float
    accessed_at: float
    access_count: int = 0
    ttl: Optional[float] = None
    
    def is_expired(self) -> bool:
        """Check if entry is expired."""
        if self.ttl is None:
            return False
        return time.time() > (self.created_at + self.ttl)
    
    def update_access(self):
        """Update access metadata."""
        self.accessed_at = time.time()
        self.access_count += 1


# ============================================================================
# LRU Cache
# ============================================================================

class LRUCache:
    """
    Thread-safe LRU cache with size limits.
    
    Features:
    - LRU eviction policy
    - Size-based limits
    - TTL support
    - Thread-safe
    """
    
    def __init__(
        self,
        max_items: int = 1000,
        max_size_mb: int = 500,
        default_ttl: Optional[int] = None
    ):
        self.max_items = max_items
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.default_ttl = default_ttl
        
        self.cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self.current_size = 0
        self.lock = threading.RLock()
        
        # Stats
        self.hits = 0
        self.misses = 0
        self.evictions = 0
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        with self.lock:
            if key not in self.cache:
                self.misses += 1
                return None
            
            entry = self.cache[key]
            
            # Check expiration
            if entry.is_expired():
                self.remove(key)
                self.misses += 1
                return None
            
            # Update access
            entry.update_access()
            
            # Move to end (most recently used)
            self.cache.move_to_end(key)
            
            self.hits += 1
            return entry.value
    
    def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None
    ):
        """Set value in cache."""
        with self.lock:
            # Calculate size
            size = self._estimate_size(value)
            
            # Remove existing entry
            if key in self.cache:
                self.remove(key)
            
            # Evict if necessary
            while (
                len(self.cache) >= self.max_items or
                self.current_size + size > self.max_size_bytes
            ):
                if not self.cache:
                    break
                self._evict_lru()
            
            # Create entry
            entry = CacheEntry(
                key=key,
                value=value,
                size_bytes=size,
                created_at=time.time(),
                accessed_at=time.time(),
                ttl=ttl or self.default_ttl,
            )
            
            # Add to cache
            self.cache[key] = entry
            self.current_size += size
    
    def remove(self, key: str) -> bool:
        """Remove key from cache."""
        with self.lock:
            if key not in self.cache:
                return False
            
            entry = self.cache.pop(key)
            self.current_size -= entry.size_bytes
            return True
    
    def clear(self):
        """Clear entire cache."""
        with self.lock:
            self.cache.clear()
            self.current_size = 0
    
    def _evict_lru(self):
        """Evict least recently used item."""
        if not self.cache:
            return
        
        # Get first item (least recently used)
        key = next(iter(self.cache))
        entry = self.cache.pop(key)
        self.current_size -= entry.size_bytes
        self.evictions += 1
    
    def _estimate_size(self, value: Any) -> int:
        """Estimate size of value in bytes."""
        try:
            return len(pickle.dumps(value))
        except Exception:
            # Fallback: rough estimate
            return 1024
    
    def get_stats(self) -> Dict:
        """Get cache statistics."""
        with self.lock:
            hit_rate = self.hits / (self.hits + self.misses) if (self.hits + self.misses) > 0 else 0
            
            return {
                "items": len(self.cache),
                "size_bytes": self.current_size,
                "size_mb": self.current_size / (1024 * 1024),
                "hits": self.hits,
                "misses": self.misses,
                "evictions": self.evictions,
                "hit_rate": hit_rate,
            }


# ============================================================================
# Disk Cache
# ============================================================================

class DiskCache:
    """
    Disk-backed persistent cache.
    
    Features:
    - Persistent storage
    - Automatic cleanup
    - Compression support
    """
    
    def __init__(
        self,
        cache_dir: str = "data/cache",
        max_size_mb: int = 5000,
        enable_compression: bool = True
    ):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.enable_compression = enable_compression
        
        # Metadata file
        self.metadata_file = self.cache_dir / "metadata.json"
        self.metadata = self._load_metadata()
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from disk cache."""
        cache_file = self._get_cache_file(key)
        
        if not cache_file.exists():
            return None
        
        # Check metadata for expiration
        if key in self.metadata:
            meta = self.metadata[key]
            if "ttl" in meta and time.time() > meta["created_at"] + meta["ttl"]:
                self.remove(key)
                return None
        
        try:
            with open(cache_file, 'rb') as f:
                value = pickle.load(f)
            
            # Update access time
            if key in self.metadata:
                self.metadata[key]["accessed_at"] = time.time()
                self._save_metadata()
            
            return value
        except Exception as e:
            print(f"Error loading cache: {e}")
            return None
    
    def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None
    ):
        """Set value in disk cache."""
        cache_file = self._get_cache_file(key)
        
        try:
            # Serialize
            with open(cache_file, 'wb') as f:
                pickle.dump(value, f)
            
            # Update metadata
            size = cache_file.stat().st_size
            self.metadata[key] = {
                "size": size,
                "created_at": time.time(),
                "accessed_at": time.time(),
                "ttl": ttl,
            }
            self._save_metadata()
            
            # Cleanup if over limit
            self._cleanup_if_needed()
            
        except Exception as e:
            print(f"Error saving cache: {e}")
    
    def remove(self, key: str) -> bool:
        """Remove key from disk cache."""
        cache_file = self._get_cache_file(key)
        
        if cache_file.exists():
            cache_file.unlink()
        
        if key in self.metadata:
            del self.metadata[key]
            self._save_metadata()
            return True
        
        return False
    
    def clear(self):
        """Clear entire disk cache."""
        for file in self.cache_dir.glob("*.pkl"):
            file.unlink()
        
        self.metadata.clear()
        self._save_metadata()
    
    def _get_cache_file(self, key: str) -> Path:
        """Get cache file path for key."""
        # Hash key to create filename
        key_hash = hashlib.sha256(key.encode()).hexdigest()
        return self.cache_dir / f"{key_hash}.pkl"
    
    def _load_metadata(self) -> Dict:
        """Load metadata from disk."""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r') as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}
    
    def _save_metadata(self):
        """Save metadata to disk."""
        try:
            with open(self.metadata_file, 'w') as f:
                json.dump(self.metadata, f)
        except Exception as e:
            print(f"Error saving metadata: {e}")
    
    def _cleanup_if_needed(self):
        """Cleanup old entries if over size limit."""
        total_size = sum(meta["size"] for meta in self.metadata.values())
        
        if total_size <= self.max_size_bytes:
            return
        
        # Sort by access time (oldest first)
        sorted_keys = sorted(
            self.metadata.keys(),
            key=lambda k: self.metadata[k]["accessed_at"]
        )
        
        # Remove oldest entries
        for key in sorted_keys:
            if total_size <= self.max_size_bytes:
                break
            
            meta = self.metadata[key]
            total_size -= meta["size"]
            self.remove(key)
    
    def get_stats(self) -> Dict:
        """Get cache statistics."""
        total_size = sum(meta["size"] for meta in self.metadata.values())
        
        return {
            "items": len(self.metadata),
            "size_bytes": total_size,
            "size_mb": total_size / (1024 * 1024),
        }


# ============================================================================
# Multi-Level Cache
# ============================================================================

class MultiLevelCache:
    """
    Multi-level cache (Memory + Disk).
    
    Features:
    - L1 (memory) + L2 (disk) caching
    - Automatic promotion
    - Unified interface
    """
    
    def __init__(self, config: CacheConfig = None):
        self.config = config or CacheConfig()
        
        # L1: Memory cache
        self.memory_cache = LRUCache(
            max_items=self.config.max_memory_items,
            max_size_mb=self.config.max_memory_size_mb,
            default_ttl=self.config.default_ttl_seconds,
        )
        
        # L2: Disk cache
        self.disk_cache = None
        if self.config.enable_disk_cache:
            self.disk_cache = DiskCache(
                cache_dir=self.config.disk_cache_dir,
                max_size_mb=self.config.max_disk_size_mb,
                enable_compression=self.config.enable_compression,
            )
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache (checks L1 then L2)."""
        # Try L1 (memory)
        value = self.memory_cache.get(key)
        if value is not None:
            return value
        
        # Try L2 (disk)
        if self.disk_cache:
            value = self.disk_cache.get(key)
            if value is not None:
                # Promote to L1
                self.memory_cache.set(key, value)
                return value
        
        return None
    
    def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None
    ):
        """Set value in cache (both L1 and L2)."""
        # Set in L1 (memory)
        self.memory_cache.set(key, value, ttl=ttl)
        
        # Set in L2 (disk)
        if self.disk_cache:
            self.disk_cache.set(key, value, ttl=ttl)
    
    def remove(self, key: str):
        """Remove key from all cache levels."""
        self.memory_cache.remove(key)
        if self.disk_cache:
            self.disk_cache.remove(key)
    
    def clear(self):
        """Clear all cache levels."""
        self.memory_cache.clear()
        if self.disk_cache:
            self.disk_cache.clear()
    
    def get_stats(self) -> Dict:
        """Get statistics from all cache levels."""
        stats = {
            "l1_memory": self.memory_cache.get_stats(),
        }
        
        if self.disk_cache:
            stats["l2_disk"] = self.disk_cache.get_stats()
        
        return stats


# ============================================================================
# Global Instance
# ============================================================================

_model_cache: Optional[MultiLevelCache] = None


def get_model_cache() -> MultiLevelCache:
    """Get global model cache instance."""
    global _model_cache
    if _model_cache is None:
        _model_cache = MultiLevelCache()
    return _model_cache


# ============================================================================
# Decorators
# ============================================================================

def cached(
    ttl: Optional[int] = None,
    key_func: Optional[Callable] = None
):
    """
    Cache decorator for functions.
    
    Args:
        ttl: Time-to-live in seconds
        key_func: Function to generate cache key from args
    
    Example:
        @cached(ttl=3600, key_func=lambda text: f"embed:{text}")
        def embed_text(text):
            ...
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                # Default: use function name + args
                cache_key = f"{func.__name__}:{str(args)}:{str(kwargs)}"
                cache_key = hashlib.sha256(cache_key.encode()).hexdigest()
            
            # Try cache
            cache = get_model_cache()
            cached_value = cache.get(cache_key)
            
            if cached_value is not None:
                return cached_value
            
            # Call function
            result = func(*args, **kwargs)
            
            # Store in cache
            cache.set(cache_key, result, ttl=ttl)
            
            return result
        
        return wrapper
    return decorator
