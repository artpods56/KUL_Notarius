"""LayoutLMv3 cache backend and key generator for cached engine pattern."""

from __future__ import annotations

from typing import final, override

from structlog import get_logger

from notarius.application.ports.outbound.cached_engine import (
    CacheBackend,
    CacheKeyGenerator,
)
from notarius.infrastructure.cache.adapters.lmv3 import LMv3Cache
from notarius.infrastructure.cache.storage.utils import get_image_hash
from notarius.infrastructure.ml_models.lmv3.engine_adapter import (
    LMv3Request,
    LMv3Response,
)

logger = get_logger(__name__)


@final
class LMv3CacheKeyGenerator(CacheKeyGenerator[LMv3Request]):
    """Generate deterministic cache keys for LayoutLMv3 requests.

    Keys are based on image content (hash) only, since the model checkpoint
    is already encoded in the cache instance (different checkpoint = different cache).
    """

    @override
    def generate_key(self, request: LMv3Request) -> str:
        """Generate cache key from image.

        Args:
            request: LMv3 request containing image

        Returns:
            Image hash as cache key
        """
        return get_image_hash(request.input)


@final
class LMv3CacheBackend(CacheBackend[LMv3Response]):
    """Cache backend adapter for LayoutLMv3 responses.

    This adapter bridges the CachedEngine protocol with LMv3Cache storage,
    using pickle serialization for automatic handling of complex types.

    The cache stores complete LMv3Response objects including:
    - SchematismPage predictions with all entries
    - Optional fields and multilingual data
    """

    def __init__(self, cache: LMv3Cache, key_generator: LMv3CacheKeyGenerator):
        """Initialize the cache backend.

        Args:
            cache: LMv3Cache instance for storage
            key_generator: Key generator for creating cache keys from requests
        """
        self.cache = cache
        self.key_generator = key_generator

    @override
    def get(self, key: str) -> LMv3Response | None:
        """Retrieve LMv3Response from cache.

        Args:
            key: Cache key

        Returns:
            Cached LMv3Response if found, None otherwise
        """
        return self.cache.get(key)

    @override
    def set(self, key: str, value: LMv3Response) -> bool:
        """Store LMv3Response in cache.

        Args:
            key: Cache key
            value: LMv3Response to cache

        Returns:
            True if cached successfully
        """
        return self.cache.set(key, value)


def create_lmv3_cache_backend(
    checkpoint: str,
) -> tuple[LMv3CacheBackend, LMv3CacheKeyGenerator]:
    """Create an LMv3 cache backend with key generator.

    This is a convenience factory for setting up LayoutLMv3 caching.

    Args:
        checkpoint: Model checkpoint name for cache directory namespacing

    Returns:
        Tuple of (cache_backend, key_generator)

    Example:
        >>> backend, keygen = create_lmv3_cache_backend("layoutlmv3-base")
        >>> # Use with CachedEngine
        >>> cached_engine = CachedEngine(
        ...     engine=lmv3_engine,
        ...     cache_backend=backend,
        ...     key_generator=keygen,
        ...     enabled=True
        ... )
    """
    cache = LMv3Cache(checkpoint=checkpoint)
    key_generator = LMv3CacheKeyGenerator()
    backend = LMv3CacheBackend(cache=cache, key_generator=key_generator)

    return backend, key_generator
