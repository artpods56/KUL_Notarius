"""Cache decorator for ConfigurableEngine implementations.

This module provides a generic caching wrapper that can be applied to any
ConfigurableEngine, moving caching logic out of use cases and into the
infrastructure layer where it belongs.
"""

from typing import Never, Protocol, final, runtime_checkable, Any, cast, override

from pydantic import BaseModel
from structlog import get_logger

from notarius.application.ports.outbound.engine import ConfigurableEngine
from notarius.domain.protocols import BaseRequest, BaseResponse
from notarius.shared.logger import Logger

logger = cast(Logger, get_logger(__name__))


@runtime_checkable
class CacheKeyGenerator[RequestT: BaseRequest[Any]](Protocol):
    """Protocol for generating cache keys from requests."""

    def generate_key(self, request: RequestT) -> str:
        """Generate a unique cache key for the given request."""
        ...


@runtime_checkable
class CacheBackend[ResponseT: BaseResponse[Any]](Protocol):
    """Protocol for cache storage backends."""

    def get(self, key: str) -> ResponseT | None:
        """Retrieve cached structured_response by key."""
        ...

    def set(self, key: str, value: ResponseT) -> bool:
        """Store structured_response in cache."""
        ...


@final
class CachedEngine[
    ConfigT: BaseModel,
    RequestT: BaseRequest[object],
    ResponseT: BaseResponse[object],
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
        structured_response = cached_engine.process(request)
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
    @override
    def from_config(cls, config: ConfigT) -> Never:
        """This method should not be called on the wrapper."""
        raise NotImplementedError(
            "CachedEngine should be instantiated with an existing engine, ",
            "not from config directly",
        )

    @override
    def process(self, request: RequestT) -> ResponseT:
        """
        Process request with caching.

        Checks cache first, delegates to wrapped engine on miss,
        and stores results in cache.
        """
        if not self._enabled:
            return self._engine.process(request)

        try:
            cache_key = self._key_generator.generate_key(request)

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
                    "Cached structured_response",
                    key=cache_key[:16],
                    engine_type=type(self._engine).__name__,
                )

            return response

        except Exception as e:
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
