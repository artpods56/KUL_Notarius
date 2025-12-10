"""Cache decorator for ConfigurableEngine implementations.

This module provides a generic caching wrapper that can be applied to any
ConfigurableEngine, moving caching logic out of use cases and into the
infrastructure layer where it belongs.
"""

from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Protocol, TypedDict, Unpack, final, runtime_checkable, Any

from pydantic import BaseModel
from structlog import get_logger

from notarius.application.ports.outbound.engine import ConfigurableEngine
from notarius.domain.protocols import BaseRequest, BaseResponse

logger = get_logger(__name__)

@runtime_checkable
class CacheKeyGenerator[RequestT: BaseRequest](Protocol):
    """Protocol for generating cache keys from requests."""

    def generate_key(self, request: RequestT) -> str:
        """Generate a unique cache key for the given request."""
        ...


@runtime_checkable
class CacheBackend[ResponseT: BaseResponse](Protocol):
    """Protocol for cache storage backends."""

    def get(self, key: str) -> ResponseT | None:
        """Retrieve cached response by key."""
        ...

    def set(self, key: str, value: ResponseT) -> bool:
        """Store response in cache."""
        ...


@final
class CachedEngine[
    ConfigT: BaseModel,
    RequestT: BaseRequest,
    ResponseT: BaseResponse,
](ConfigurableEngine[ConfigT, RequestT, ResponseT]):
    """
    Decorator that adds caching capabilities to any ConfigurableEngine.

    This wrapper intercepts the process() method, checks cache before
    delegating to the wrapped engine, and caches the results.

    Example:
        ```python
        # Create the base engine
        llm_engine = LLMEngine.from_config(config)

        # Wrap it with caching
        cached_engine = CachedEngine(
            engine=llm_engine,
            cache_backend=llm_cache,
            key_generator=llm_key_generator,
            enabled=True
        )

        # Use it exactly like the original engine
        response = cached_engine.process(request)
        ```
    """

    def __init__(
        self,
        engine: ConfigurableEngine[ConfigT, RequestT, ResponseT],
        cache_backend: CacheBackend[ResponseT],
        key_generator: CacheKeyGenerator[RequestT],
        enabled: bool = True,
    ):
        """
        Initialize the cached engine wrapper.

        Args:
            engine: The underlying engine to wrap
            cache_backend: Cache storage implementation
            key_generator: Strategy for generating cache keys
            enabled: Whether caching is enabled
        """
        self._engine = engine
        self._cache = cache_backend
        self._key_generator = key_generator
        self._enabled = enabled
        self._stats = {"hits": 0, "misses": 0, "errors": 0}

    @classmethod
    def from_config(cls, config: ConfigT, *args, **kwargs):
        """This method should not be called on the wrapper."""
        raise NotImplementedError(
            "CachedEngine should be instantiated with an existing engine, ",
            "not from config directly",
        )

    def process(self, request: RequestT) -> ResponseT:
        """
        Process request with caching.

        Checks cache first, delegates to wrapped engine on miss,
        and stores results in cache.
        """
        if not self._enabled:
            return self._engine.process(request)

        try:
            # Generate cache key
            cache_key = self._key_generator.generate_key(request)

            # Check cache
            cached_response = self._cache.get(cache_key)
            if cached_response is not None:
                self._stats["hits"] += 1
                logger.debug(
                    "Cache hit",
                    key=cache_key[:16],
                    engine_type=type(self._engine).__name__,
                )
                return cached_response

            # Cache miss - process with underlying engine
            self._stats["misses"] += 1
            logger.debug(
                "Cache miss",
                key=cache_key[:16],
                engine_type=type(self._engine).__name__,
            )

            response = self._engine.process(request)

            # Store in cache
            success = self._cache.set(cache_key, response)
            if success:
                logger.debug(
                    "Cached response",
                    key=cache_key[:16],
                    engine_type=type(self._engine).__name__,
                )

            return response

        except Exception as e:
            # Log error but don't fail the request
            self._stats["errors"] += 1
            logger.warning(
                "Cache error, falling back to direct processing",
                error=str(e),
                engine_type=type(self._engine).__name__,
            )
            return self._engine.process(request)

    @property
    def stats(self) -> dict[str, int]:
        """Get cache statistics."""
        return self._stats.copy()

    @property
    def wrapped_engine(self) -> ConfigurableEngine[ConfigT, RequestT, ResponseT]:
        """Access the underlying engine."""
        return self._engine

    def clear_stats(self) -> None:
        """Reset cache statistics."""
        self._stats = {"hits": 0, "misses": 0, "errors": 0}
