"""Tests for the CachedEngine wrapper."""

from typing import final, override
import pytest
from unittest.mock import Mock
from dataclasses import dataclass

from notarius.application.ports.outbound.cached_engine import (
    CachedEngine,
    CacheKeyGenerator,
    CacheBackend,
)
from notarius.application.ports.outbound.engine import ConfigurableEngine
from notarius.domain.protocols import BaseRequest, BaseResponse
from pydantic import BaseModel


# Test fixtures
@dataclass(frozen=True)
class TestRequest(BaseRequest[str]):
    """Test request type."""

    pass


@dataclass(frozen=True)
class TestResponse(BaseResponse[str]):
    """Test structured_response type."""

    pass


class TestConfig(BaseModel):
    """Test configuration."""

    value: str


@final
class TestEngine(ConfigurableEngine[TestConfig, TestRequest, TestResponse]):
    """Test engine implementation."""

    def __init__(self, config: TestConfig):
        self._init_stats()
        self.config = config
        self.process_count = 0

    @classmethod
    @override
    def from_config(cls, config: TestConfig):
        return cls(config)

    @override
    def process(self, request: TestRequest) -> TestResponse:
        self.process_count += 1
        return TestResponse(output=f"processed: {request.input}")


class TestCacheKeyGenerator(CacheKeyGenerator[TestRequest]):
    """Test key generator."""

    def generate_key(self, request: TestRequest) -> str:
        return f"key_{request.input}"


@final
class TestCacheBackend(CacheBackend[TestResponse]):
    """Test cache backend."""

    def __init__(self):
        self.cache: dict[str, TestResponse] = {}

    @override
    def get(self, key: str) -> TestResponse | None:
        return self.cache.get(key)

    @override
    def set(self, key: str, value: TestResponse) -> bool:
        self.cache[key] = value
        return True


class TestCachedEngine:
    """Test suite for CachedEngine."""

    def test_cache_miss_processes_request(self):
        """Test that cache miss sample in processing the request."""
        # Setup
        base_engine = TestEngine(TestConfig(value="test"))
        cache_backend = TestCacheBackend()
        key_generator = TestCacheKeyGenerator()

        cached_engine = CachedEngine(
            engine=base_engine,
            cache_backend=cache_backend,
            key_generator=key_generator,
            enabled=True,
        )

        request = TestRequest(input="test_data")

        # Execute
        response = cached_engine.process(request)

        # Verify
        assert response.output == "processed: test_data"
        assert base_engine.process_count == 1
        assert cached_engine.stats["misses"] == 1
        assert cached_engine.stats["hits"] == 0

    def test_cache_hit_skips_processing(self):
        """Test that cache hit returns cached structured_response without processing."""
        # Setup
        base_engine = TestEngine(TestConfig(value="test"))
        cache_backend = TestCacheBackend()
        key_generator = TestCacheKeyGenerator()

        cached_engine = CachedEngine(
            engine=base_engine,
            cache_backend=cache_backend,
            key_generator=key_generator,
            enabled=True,
        )

        request = TestRequest(input="test_data")

        # First call - cache miss
        response1 = cached_engine.process(request)

        # Second call - cache hit
        response2 = cached_engine.process(request)

        # Verify
        assert response1.output == response2.output
        assert base_engine.process_count == 1  # Only processed once
        assert cached_engine.stats["misses"] == 1
        assert cached_engine.stats["hits"] == 1

    def test_cache_disabled_always_processes(self):
        """Test that disabled cache always processes requests."""
        # Setup
        base_engine = TestEngine(TestConfig(value="test"))
        cache_backend = TestCacheBackend()
        key_generator = TestCacheKeyGenerator()

        cached_engine = CachedEngine(
            engine=base_engine,
            cache_backend=cache_backend,
            key_generator=key_generator,
            enabled=False,  # Cache disabled
        )

        request = TestRequest(input="test_data")

        # Execute multiple times
        response1 = cached_engine.process(request)
        response2 = cached_engine.process(request)

        # Verify
        assert response1.output == response2.output
        assert base_engine.process_count == 2  # Processed both times
        assert len(cache_backend.cache) == 0  # Nothing cached

    def test_cache_error_falls_back_to_processing(self):
        """Test that cache errors don't break processing."""
        # Setup
        base_engine = TestEngine(TestConfig(value="test"))
        cache_backend = Mock()
        cache_backend.get.side_effect = Exception("Cache error")
        key_generator = TestCacheKeyGenerator()

        cached_engine = CachedEngine(
            engine=base_engine,
            cache_backend=cache_backend,
            key_generator=key_generator,
            enabled=True,
        )

        request = TestRequest(input="test_data")

        # Execute
        response = cached_engine.process(request)

        # Verify
        assert response.output == "processed: test_data"
        assert base_engine.process_count == 1
        assert cached_engine.stats["errors"] == 1

    def test_from_config_raises_not_implemented(self):
        """Test that from_config raises NotImplementedError."""
        with pytest.raises(NotImplementedError):
            CachedEngine.from_config(TestConfig(value="test"))

    def test_clear_stats(self):
        """Test that clear_stats resets statistics."""
        # Setup
        base_engine = TestEngine(TestConfig(value="test"))
        cache_backend = TestCacheBackend()
        key_generator = TestCacheKeyGenerator()

        cached_engine = CachedEngine(
            engine=base_engine,
            cache_backend=cache_backend,
            key_generator=key_generator,
            enabled=True,
        )

        # Generate some stats
        cached_engine.process(TestRequest(input="test1"))
        cached_engine.process(TestRequest(input="test1"))  # Cache hit

        # Clear stats
        cached_engine.clear_stats()

        # Verify
        assert cached_engine.stats == {"calls": 0, "hits": 0, "misses": 0, "errors": 0}

    def test_wrapped_engine_access(self):
        """Test that wrapped_engine property provides access to base engine."""
        # Setup
        base_engine = TestEngine(TestConfig(value="test"))
        cache_backend = TestCacheBackend()
        key_generator = TestCacheKeyGenerator()

        cached_engine = CachedEngine(
            engine=base_engine,
            cache_backend=cache_backend,
            key_generator=key_generator,
            enabled=True,
        )

        # Verify
        assert cached_engine.wrapped_engine is base_engine

    def test_different_requests_have_different_keys(self):
        """Test that different requests generate different cache keys."""
        # Setup
        base_engine = TestEngine(TestConfig(value="test"))
        cache_backend = TestCacheBackend()
        key_generator = TestCacheKeyGenerator()

        cached_engine = CachedEngine(
            engine=base_engine,
            cache_backend=cache_backend,
            key_generator=key_generator,
            enabled=True,
        )

        # Process different requests
        response1 = cached_engine.process(TestRequest(input="data1"))
        response2 = cached_engine.process(TestRequest(input="data2"))

        # Verify
        assert response1.output == "processed: data1"
        assert response2.output == "processed: data2"
        assert base_engine.process_count == 2  # Both processed
        assert len(cache_backend.cache) == 2  # Both cached
        assert cached_engine.stats["misses"] == 2
        assert cached_engine.stats["hits"] == 0
