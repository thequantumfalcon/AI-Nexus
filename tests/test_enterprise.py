"""
Comprehensive enterprise security and integration tests.

Tests:
- Authentication (JWT, API keys)
- Rate limiting
- Input validation
- Cost tracking
- Caching
- Batch processing
- Metrics
"""

import pytest
import asyncio
import time
from datetime import datetime, timedelta

# Security components
from core.security import (
    User, Role, Permission, get_password_manager, get_jwt_manager,
    get_api_key_manager, authenticate_user, UserInDB
)
from core.rate_limiting import get_rate_limiter, RateLimitExceeded
from core.validation import (
    Validator, Sanitizer, ValidatedTextInput, ValidatedUserInput,
    is_safe_input, ValidationError
)

# Monitoring components
from core.usage_tracking import get_token_tracker, TokenUsage
from core.metrics import get_metrics_collector
from core.alerting import get_alert_manager, AlertSeverity

# Optimization components
from services.optimization.caching import get_model_cache, cached
from services.optimization.batching import BatchProcessor, Priority


# ============================================================================
# Security Tests
# ============================================================================

class TestAuthentication:
    """Test JWT authentication system."""
    
    def test_password_hashing(self):
        """Test password hashing and verification."""
        password_manager = get_password_manager()
        
        # Use password within bcrypt 72-byte limit
        password = "SecurePass123!"
        hashed = password_manager.hash_password(password)
        
        # Verify correct password
        assert password_manager.verify_password(password, hashed)
        
        # Verify incorrect password
        assert not password_manager.verify_password("WrongPass", hashed)
    
    def test_password_strength_validation(self):
        """Test password strength validation."""
        password_manager = get_password_manager()
        
        # Valid password
        is_valid, error = password_manager.validate_password_strength("SecurePass123!")
        assert is_valid
        assert error is None
        
        # Too short
        is_valid, error = password_manager.validate_password_strength("Short1!")
        assert not is_valid
        assert "at least" in error
        
        # No uppercase
        is_valid, error = password_manager.validate_password_strength("nouppercas3!")
        assert not is_valid
        
        # No digit
        is_valid, error = password_manager.validate_password_strength("NoDigits!")
        assert not is_valid
    
    def test_jwt_token_creation(self):
        """Test JWT token creation."""
        jwt_manager = get_jwt_manager()
        
        user = User(
            username="testuser",
            email="test@example.com",
            role=Role.DEVELOPER
        )
        
        # Create token
        token = jwt_manager.create_access_token(user)
        assert token is not None
        assert isinstance(token, str)
        
        # Decode token
        token_data = jwt_manager.decode_token(token)
        assert token_data.username == "testuser"
        assert token_data.role == Role.DEVELOPER
        assert Permission.MODEL_INFERENCE in token_data.permissions
    
    def test_jwt_token_expiration(self):
        """Test JWT token expiration."""
        jwt_manager = get_jwt_manager()
        
        user = User(username="testuser", email="test@example.com")
        
        # Create token with very short expiration
        token = jwt_manager.create_access_token(
            user,
            expires_delta=timedelta(seconds=1)
        )
        
        # Token should be valid immediately
        token_data = jwt_manager.decode_token(token)
        assert token_data.username == "testuser"
        
        # Wait for expiration
        time.sleep(2)
        
        # Token should now be expired
        from jose import JWTError
        with pytest.raises(JWTError):
            jwt_manager.decode_token(token)
    
    def test_refresh_token(self):
        """Test refresh token mechanism."""
        jwt_manager = get_jwt_manager()
        
        user = User(username="testuser", email="test@example.com")
        
        # Create token pair
        token_pair = jwt_manager.create_token_pair(user)
        assert token_pair.access_token is not None
        assert token_pair.refresh_token is not None
        
        # Use refresh token to get new access token
        new_access_token = jwt_manager.refresh_access_token(
            token_pair.refresh_token,
            user
        )
        assert new_access_token is not None
    
    def test_api_key_generation(self):
        """Test API key generation and validation."""
        api_key_manager = get_api_key_manager()
        
        # Generate API key
        raw_key, api_key = api_key_manager.generate_api_key(
            user_id="user123",
            name="Test API Key",
            role=Role.DEVELOPER
        )
        
        assert raw_key.startswith("nexus_")
        assert api_key.user_id == "user123"
        
        # Validate API key
        is_valid = api_key_manager.validate_api_key(raw_key, api_key)
        assert is_valid
        
        # Invalid key should fail
        is_valid = api_key_manager.validate_api_key("nexus_invalid_key", api_key)
        assert not is_valid
    
    def test_role_based_access_control(self):
        """Test RBAC permissions."""
        admin = User(username="admin", email="admin@example.com", role=Role.ADMIN)
        developer = User(username="dev", email="dev@example.com", role=Role.DEVELOPER)
        guest = User(username="guest", email="guest@example.com", role=Role.GUEST)
        
        # Admin should have all permissions
        assert admin.has_permission(Permission.USER_CREATE)
        assert admin.has_permission(Permission.MODEL_FINE_TUNE)
        assert admin.has_permission(Permission.SYSTEM_CONFIG)
        
        # Developer should have limited permissions
        assert developer.has_permission(Permission.MODEL_INFERENCE)
        assert developer.has_permission(Permission.DATA_READ)
        assert not developer.has_permission(Permission.USER_CREATE)
        
        # Guest should have minimal permissions
        assert guest.has_permission(Permission.MODEL_INFERENCE)
        assert not guest.has_permission(Permission.DATA_WRITE)
        assert not guest.has_permission(Permission.SYSTEM_CONFIG)


class TestRateLimiting:
    """Test rate limiting system."""
    
    @pytest.mark.asyncio
    async def test_rate_limit_basic(self):
        """Test basic rate limiting."""
        limiter = await get_rate_limiter()
        
        # First request should be allowed
        allowed, info = await limiter.check_rate_limit("user123", tier="free")
        assert allowed
        assert info["allowed"]
        
        # Multiple requests should eventually hit limit
        for i in range(20):
            allowed, info = await limiter.check_rate_limit("user123", tier="free")
        
        # Should hit limit (free tier: 10 req/min)
        assert not allowed
        assert not info["allowed"]
        assert info["limit"] == 10
    
    @pytest.mark.asyncio
    async def test_rate_limit_tiers(self):
        """Test different rate limit tiers."""
        limiter = await get_rate_limiter()
        
        # Free tier: 10 req/min
        for i in range(5):
            allowed, _ = await limiter.check_rate_limit("free_user", tier="free")
            assert allowed
        
        # Premium tier: 300 req/min
        for i in range(50):
            allowed, _ = await limiter.check_rate_limit("premium_user", tier="premium")
            assert allowed
    
    @pytest.mark.asyncio
    async def test_rate_limit_reset(self):
        """Test rate limit reset."""
        limiter = await get_rate_limiter()
        
        # Reset limit
        await limiter.reset_limit("user123")
        
        # Should be able to make requests again
        allowed, info = await limiter.check_rate_limit("user123", tier="developer")
        assert allowed


class TestInputValidation:
    """Test input validation and sanitization."""
    
    def test_xss_detection(self):
        """Test XSS attack detection."""
        assert Validator.detect_xss("<script>alert('xss')</script>")
        assert Validator.detect_xss("javascript:alert(1)")
        assert Validator.detect_xss("<img src=x onerror=alert(1)>")
        assert not Validator.detect_xss("This is safe text")
    
    def test_sql_injection_detection(self):
        """Test SQL injection detection."""
        assert Validator.detect_sql_injection("' OR '1'='1")
        assert Validator.detect_sql_injection("admin'--")
        assert Validator.detect_sql_injection("1; DROP TABLE users")
        assert not Validator.detect_sql_injection("safe query text")
    
    def test_path_traversal_detection(self):
        """Test path traversal detection."""
        assert Validator.detect_path_traversal("../../../etc/passwd")
        assert Validator.detect_path_traversal("..\\..\\windows\\system32")
        assert not Validator.detect_path_traversal("normal/path/file.txt")
    
    def test_string_sanitization(self):
        """Test string sanitization."""
        dirty_string = "  text with \x00 null bytes  "
        clean_string = Sanitizer.sanitize_string(dirty_string)
        assert clean_string == "text with  null bytes"
        assert "\x00" not in clean_string
    
    def test_html_escaping(self):
        """Test HTML escaping."""
        html_string = "<script>alert('xss')</script>"
        escaped = Sanitizer.sanitize_html(html_string)
        assert "<script>" not in escaped
        assert "&lt;script&gt;" in escaped
    
    def test_validated_model(self):
        """Test Pydantic validation."""
        # Valid input
        valid_input = ValidatedTextInput(text="Safe text")
        assert valid_input.text == "Safe text"
        
        # Invalid input (XSS)
        with pytest.raises(ValidationError):
            ValidatedTextInput(text="<script>alert(1)</script>")
    
    def test_safe_input_check(self):
        """Test comprehensive safety check."""
        assert is_safe_input("This is completely safe text")
        assert not is_safe_input("' OR '1'='1")
        assert not is_safe_input("<script>alert(1)</script>")
        assert not is_safe_input("../../../etc/passwd")


# ============================================================================
# Monitoring Tests
# ============================================================================

class TestUsageTracking:
    """Test token usage and cost tracking."""
    
    def test_cost_calculation(self):
        """Test cost calculation for different models."""
        tracker = get_token_tracker()
        
        # GPT-4 cost
        cost_gpt4 = tracker.calculate_cost("gpt-4", input_tokens=1000, output_tokens=500)
        assert cost_gpt4 > 0
        
        # LLaMA cost (should be cheaper)
        cost_llama = tracker.calculate_cost("llama-3.3-70b", input_tokens=1000, output_tokens=500)
        assert cost_llama > 0
        assert cost_llama < cost_gpt4
    
    def test_usage_recording(self):
        """Test usage recording."""
        tracker = get_token_tracker()
        
        # Record usage
        usage = tracker.record_usage(
            user_id="user123",
            model="llama-3.3-70b",
            operation="generate",
            input_tokens=100,
            output_tokens=50
        )
        
        assert usage.user_id == "user123"
        assert usage.total_tokens == 150
        assert usage.cost > 0
    
    def test_usage_stats(self):
        """Test usage statistics."""
        tracker = get_token_tracker()
        
        # Record some usage
        for i in range(5):
            tracker.record_usage(
                user_id="user123",
                model="llama-3.3-70b",
                operation="generate",
                input_tokens=100,
                output_tokens=50
            )
        
        # Get stats
        stats = tracker.get_usage_stats(user_id="user123")
        assert stats.total_requests >= 5
        assert stats.total_tokens >= 750
        assert stats.total_cost > 0


class TestMetrics:
    """Test Prometheus metrics."""
    
    def test_metrics_collector(self):
        """Test metrics collector."""
        collector = get_metrics_collector()
        
        # Record HTTP request
        collector.record_http_request("POST", "/api/generate", "200", 0.5)
        
        # Record LLM inference
        collector.record_llm_inference(
            model="llama-3.3-70b",
            operation="generate",
            status="success",
            duration=1.5,
            input_tokens=100,
            output_tokens=50,
            user="user123"
        )
        
        # Record vector search
        collector.record_vector_search("documents", "success", 0.1)
    
    def test_gpu_metrics(self):
        """Test GPU metrics update."""
        collector = get_metrics_collector()
        
        # Update GPU metrics
        collector.update_gpu_metrics(
            gpu_id="0",
            utilization=85.0,
            memory_used=12000000000,
            temperature=75.0
        )


class TestAlerting:
    """Test error alerting system."""
    
    def test_error_recording(self):
        """Test error recording."""
        alert_manager = get_alert_manager()
        
        # Record error
        alert_manager.record_error(
            message="Test error",
            severity=AlertSeverity.ERROR,
            endpoint="/api/test"
        )
        
        # Get stats
        stats = alert_manager.get_error_stats()
        assert stats["total_errors_last_hour"] >= 1
    
    def test_critical_alert(self):
        """Test critical error alert."""
        alert_manager = get_alert_manager()
        
        # Record critical error
        try:
            raise ValueError("Critical test error")
        except Exception as e:
            alert_manager.record_error(
                message="Critical failure",
                severity=AlertSeverity.CRITICAL,
                exception=e,
                endpoint="/api/critical"
            )


# ============================================================================
# Optimization Tests
# ============================================================================

class TestCaching:
    """Test model caching system."""
    
    def test_lru_cache(self):
        """Test LRU cache."""
        cache = get_model_cache()
        
        # Set value
        cache.set("key1", "value1")
        
        # Get value
        value = cache.get("key1")
        assert value == "value1"
        
        # Non-existent key
        value = cache.get("nonexistent")
        assert value is None
    
    def test_cache_ttl(self):
        """Test cache TTL."""
        cache = get_model_cache()
        
        # Set with short TTL
        cache.set("ttl_key", "ttl_value", ttl=1)
        
        # Should exist immediately
        value = cache.get("ttl_key")
        assert value == "ttl_value"
        
        # Wait for expiration
        time.sleep(2)
        
        # Should be expired
        value = cache.get("ttl_key")
        assert value is None
    
    def test_cache_decorator(self):
        """Test cache decorator basic functionality."""
        import uuid
        
        # Use completely unique key to avoid any cache pollution
        test_id = str(uuid.uuid4())
        
        @cached(ttl=60, key_func=lambda x: f"unique_{test_id}_{x}")
        def compute(x):
            return {"value": x * 2, "computed": True}
        
        # First call - should compute
        result1 = compute(42)
        assert result1["value"] == 84
        assert result1["computed"] is True
        
        # Second call with same input - should use cache
        result2 = compute(42)
        assert result2["value"] == 84
        
        # Different input - should compute again  
        result3 = compute(99)
        assert result3["value"] == 198


class TestBatchProcessing:
    """Test batch request optimization."""
    
    @pytest.mark.asyncio
    async def test_batch_processor(self):
        """Test batch processor."""
        # Define processing function
        def process_batch(inputs):
            return [x * 2 for x in inputs]
        
        # Create processor
        processor = BatchProcessor(process_batch)
        await processor.start()
        
        # Submit requests
        result = await processor.submit(5)
        assert result == 10
        
        await processor.stop()
    
    @pytest.mark.asyncio
    async def test_batch_priority(self):
        """Test priority queueing."""
        def process_batch(inputs):
            return inputs
        
        processor = BatchProcessor(process_batch)
        await processor.start()
        
        # Submit with different priorities
        low_task = asyncio.create_task(processor.submit(1, priority=Priority.LOW))
        high_task = asyncio.create_task(processor.submit(2, priority=Priority.HIGH))
        
        # High priority should be processed first
        results = await asyncio.gather(high_task, low_task)
        
        await processor.stop()


# ============================================================================
# Integration Tests
# ============================================================================

class TestEndToEndIntegration:
    """Test end-to-end integration."""
    
    @pytest.mark.asyncio
    async def test_authenticated_request_with_rate_limiting(self):
        """Test full request flow with auth and rate limiting."""
        # Create user
        password_manager = get_password_manager()
        jwt_manager = get_jwt_manager()
        
        user = User(
            username="testuser",
            email="test@example.com",
            role=Role.DEVELOPER
        )
        
        # Create token
        token = jwt_manager.create_access_token(user)
        
        # Verify token
        token_data = jwt_manager.decode_token(token)
        assert token_data.username == "testuser"
        
        # Check rate limit
        limiter = await get_rate_limiter()
        allowed, info = await limiter.check_rate_limit("testuser", tier="developer")
        assert allowed
    
    def test_full_monitoring_stack(self):
        """Test complete monitoring stack."""
        # Track usage
        tracker = get_token_tracker()
        usage = tracker.record_usage(
            user_id="user123",
            model="llama-3.3-70b",
            operation="generate",
            input_tokens=100,
            output_tokens=50
        )
        
        # Record metrics
        collector = get_metrics_collector()
        collector.record_llm_inference(
            model="llama-3.3-70b",
            operation="generate",
            status="success",
            duration=1.0,
            input_tokens=100,
            output_tokens=50,
            user="user123"
        )
        
        # Record cost
        collector.record_cost("llama-3.3-70b", "user123", usage.cost)
        
        # Get stats
        stats = tracker.get_usage_stats(user_id="user123")
        assert stats.total_cost > 0


# ============================================================================
# Performance Tests
# ============================================================================

class TestPerformance:
    """Performance benchmarks."""
    
    @pytest.mark.asyncio
    async def test_rate_limiter_performance(self):
        """Test rate limiter performance."""
        limiter = await get_rate_limiter()
        
        start = time.time()
        for i in range(100):
            await limiter.check_rate_limit(f"user{i}", tier="developer")
        duration = time.time() - start
        
        # Should handle 100 checks in under 1 second
        assert duration < 1.0
    
    def test_cache_performance(self):
        """Test cache performance."""
        cache = get_model_cache()
        
        # Write performance
        start = time.time()
        for i in range(1000):
            cache.set(f"key{i}", f"value{i}")
        write_duration = time.time() - start
        
        # Read performance
        start = time.time()
        for i in range(1000):
            cache.get(f"key{i}")
        read_duration = time.time() - start
        
        print(f"Cache write: {write_duration:.3f}s for 1000 items")
        print(f"Cache read: {read_duration:.3f}s for 1000 items")
        
        # Should be reasonably fast (relaxed for Windows disk I/O)
        assert write_duration < 10.0  # Allow time for disk operations
        assert read_duration < 0.5   # In-memory reads should be fast


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
