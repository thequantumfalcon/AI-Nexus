"""
Rate limiting for API endpoints.

Features:
- Redis-backed distributed rate limiting
- In-memory fallback for testing
- Token bucket algorithm
- Per-user and per-endpoint limits
- Sliding window implementation
"""

import time
from typing import Optional, Dict, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import asyncio
from functools import wraps

try:
    import redis.asyncio as redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class RateLimitConfig:
    """Rate limit configuration."""
    # Default limits
    requests_per_minute: int = 60
    requests_per_hour: int = 1000
    requests_per_day: int = 10000
    
    # Burst allowance
    burst_size: int = 10  # Allow bursts up to this many requests
    
    # Redis settings
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    redis_password: Optional[str] = None
    
    # Fallback to in-memory if Redis unavailable
    use_memory_fallback: bool = True


@dataclass
class RateLimitTier:
    """Rate limit tier for different user roles."""
    name: str
    requests_per_minute: int
    requests_per_hour: int
    requests_per_day: int
    burst_size: int


# Predefined tiers
RATE_LIMIT_TIERS = {
    "free": RateLimitTier("free", 10, 100, 1000, 5),
    "developer": RateLimitTier("developer", 60, 1000, 10000, 10),
    "premium": RateLimitTier("premium", 300, 10000, 100000, 50),
    "enterprise": RateLimitTier("enterprise", 1000, 50000, 500000, 100),
}


# ============================================================================
# Rate Limiter Implementation
# ============================================================================

class RateLimiter:
    """
    Token bucket rate limiter with Redis backend.
    
    Supports:
    - Sliding window rate limiting
    - Multiple time windows (minute, hour, day)
    - Burst allowance
    - Distributed rate limiting via Redis
    """
    
    def __init__(self, config: RateLimitConfig = None):
        self.config = config or RateLimitConfig()
        self.redis_client: Optional[redis.Redis] = None
        self.memory_store: Dict[str, Dict] = {}
        self.use_redis = False
    
    async def initialize(self):
        """Initialize Redis connection."""
        if not REDIS_AVAILABLE:
            print("⚠️  Redis not available, using in-memory rate limiting")
            return
        
        try:
            self.redis_client = redis.Redis(
                host=self.config.redis_host,
                port=self.config.redis_port,
                db=self.config.redis_db,
                password=self.config.redis_password,
                decode_responses=True,
            )
            # Test connection
            await self.redis_client.ping()
            self.use_redis = True
            print("✅ Redis rate limiting enabled")
        except Exception as e:
            print(f"⚠️  Redis connection failed: {e}")
            if not self.config.use_memory_fallback:
                raise
            print("⚠️  Falling back to in-memory rate limiting")
    
    async def close(self):
        """Close Redis connection."""
        if self.redis_client:
            await self.redis_client.close()
    
    async def check_rate_limit(
        self,
        key: str,
        tier: str = "developer"
    ) -> Tuple[bool, Dict]:
        """
        Check if request is within rate limit.
        
        Args:
            key: Unique identifier (e.g., user_id, ip_address)
            tier: Rate limit tier name
        
        Returns:
            (allowed, info) where info contains:
                - remaining: requests remaining
                - reset_at: when limit resets
                - limit: total limit
        """
        limit_tier = RATE_LIMIT_TIERS.get(tier, RATE_LIMIT_TIERS["developer"])
        
        if self.use_redis and self.redis_client:
            return await self._check_redis(key, limit_tier)
        else:
            return await self._check_memory(key, limit_tier)
    
    async def _check_redis(
        self,
        key: str,
        tier: RateLimitTier
    ) -> Tuple[bool, Dict]:
        """Check rate limit using Redis."""
        now = time.time()
        windows = {
            "minute": (60, tier.requests_per_minute),
            "hour": (3600, tier.requests_per_hour),
            "day": (86400, tier.requests_per_day),
        }
        
        # Check all windows
        for window_name, (window_size, limit) in windows.items():
            redis_key = f"ratelimit:{key}:{window_name}"
            window_start = now - window_size
            
            # Use sorted set to track requests with timestamps
            pipe = self.redis_client.pipeline()
            
            # Remove old entries
            pipe.zremrangebyscore(redis_key, 0, window_start)
            
            # Count requests in window
            pipe.zcard(redis_key)
            
            # Add current request
            pipe.zadd(redis_key, {str(now): now})
            
            # Set expiration
            pipe.expire(redis_key, window_size)
            
            results = await pipe.execute()
            count = results[1]
            
            if count >= limit:
                # Calculate reset time
                oldest = await self.redis_client.zrange(redis_key, 0, 0, withscores=True)
                reset_at = oldest[0][1] + window_size if oldest else now + window_size
                
                return False, {
                    "allowed": False,
                    "limit": limit,
                    "remaining": 0,
                    "reset_at": datetime.fromtimestamp(reset_at),
                    "window": window_name,
                }
        
        # All windows passed
        return True, {
            "allowed": True,
            "limit": tier.requests_per_minute,
            "remaining": tier.requests_per_minute - 1,
            "reset_at": datetime.fromtimestamp(now + 60),
        }
    
    async def _check_memory(
        self,
        key: str,
        tier: RateLimitTier
    ) -> Tuple[bool, Dict]:
        """Check rate limit using in-memory storage."""
        now = time.time()
        
        if key not in self.memory_store:
            self.memory_store[key] = {
                "minute": [],
                "hour": [],
                "day": [],
            }
        
        store = self.memory_store[key]
        windows = {
            "minute": (60, tier.requests_per_minute),
            "hour": (3600, tier.requests_per_hour),
            "day": (86400, tier.requests_per_day),
        }
        
        # Check all windows
        for window_name, (window_size, limit) in windows.items():
            window_start = now - window_size
            
            # Remove old entries
            store[window_name] = [
                ts for ts in store[window_name] if ts > window_start
            ]
            
            if len(store[window_name]) >= limit:
                reset_at = store[window_name][0] + window_size
                return False, {
                    "allowed": False,
                    "limit": limit,
                    "remaining": 0,
                    "reset_at": datetime.fromtimestamp(reset_at),
                    "window": window_name,
                }
        
        # Add current request to all windows
        for window_name in windows:
            store[window_name].append(now)
        
        return True, {
            "allowed": True,
            "limit": tier.requests_per_minute,
            "remaining": tier.requests_per_minute - len(store["minute"]),
            "reset_at": datetime.fromtimestamp(now + 60),
        }
    
    async def reset_limit(self, key: str):
        """Reset rate limit for a key (admin function)."""
        if self.use_redis and self.redis_client:
            for window in ["minute", "hour", "day"]:
                await self.redis_client.delete(f"ratelimit:{key}:{window}")
        else:
            if key in self.memory_store:
                del self.memory_store[key]
    
    async def get_usage(self, key: str) -> Dict:
        """Get current usage statistics for a key."""
        now = time.time()
        windows = {
            "minute": 60,
            "hour": 3600,
            "day": 86400,
        }
        
        usage = {}
        
        if self.use_redis and self.redis_client:
            for window_name, window_size in windows.items():
                redis_key = f"ratelimit:{key}:{window_name}"
                window_start = now - window_size
                count = await self.redis_client.zcount(redis_key, window_start, now)
                usage[window_name] = count
        else:
            if key in self.memory_store:
                store = self.memory_store[key]
                for window_name, window_size in windows.items():
                    window_start = now - window_size
                    count = sum(1 for ts in store[window_name] if ts > window_start)
                    usage[window_name] = count
            else:
                usage = {w: 0 for w in windows}
        
        return usage


# ============================================================================
# Global Instance
# ============================================================================

_rate_limiter: Optional[RateLimiter] = None


async def get_rate_limiter() -> RateLimiter:
    """Get global rate limiter instance."""
    global _rate_limiter
    if _rate_limiter is None:
        _rate_limiter = RateLimiter()
        await _rate_limiter.initialize()
    return _rate_limiter


# ============================================================================
# Decorator
# ============================================================================

def rate_limit(tier: str = "developer", key_func=None):
    """
    Rate limiting decorator for async functions.
    
    Args:
        tier: Rate limit tier name
        key_func: Function to extract key from args (default: first arg)
    
    Example:
        @rate_limit(tier="premium", key_func=lambda user_id: user_id)
        async def api_endpoint(user_id: str):
            ...
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Extract key
            if key_func:
                key = key_func(*args, **kwargs)
            else:
                key = str(args[0]) if args else "anonymous"
            
            # Check rate limit
            limiter = await get_rate_limiter()
            allowed, info = await limiter.check_rate_limit(key, tier)
            
            if not allowed:
                raise RateLimitExceeded(
                    f"Rate limit exceeded. Reset at {info['reset_at']}"
                )
            
            # Call function
            return await func(*args, **kwargs)
        
        return wrapper
    return decorator


# ============================================================================
# Exceptions
# ============================================================================

class RateLimitExceeded(Exception):
    """Raised when rate limit is exceeded."""
    pass
