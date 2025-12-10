"""OCR cache backend and key generator for cached engine pattern."""

from __future__ import annotations

from typing import final, override

from structlog import get_logger

from notarius.application.ports.outbound.cached_engine import (
    CacheBackend,
    CacheKeyGenerator,
)
from notarius.infrastructure.cache.adapters.ocr import PyTesseractCache
from notarius.infrastructure.cache.storage.utils import get_image_hash
from notarius.infrastructure.ocr.engine_adapter import OCRRequest, OCRResponse

logger = get_logger(__name__)


@final
class OCRCacheKeyGenerator(CacheKeyGenerator[OCRRequest]):
    """Generate deterministic cache keys for OCR requests.

    Keys are based on image content (hash) only, since OCR mode is
    already encoded in the cache instance (different language = different cache).
    """

    @override
    def generate_key(self, request: OCRRequest) -> str:
        """Generate cache key from image.

        Args:
            request: OCR request containing image and mode

        Returns:
            Image hash as cache key
        """
        return get_image_hash(request.input)


@final
class OCRCacheBackend(CacheBackend[OCRResponse]):
    """Cache backend adapter for OCR responses.

    This adapter bridges the CachedEngine protocol with PyTesseractCache storage,
    using pickle serialization for automatic handling of OCR response types.

    The cache stores complete OCRResponse objects including:
    - Simple text-only results
    - Structured results with words and bounding boxes
    """

    def __init__(self, cache: PyTesseractCache, key_generator: OCRCacheKeyGenerator):
        """Initialize the cache backend.

        Args:
            cache: PyTesseractCache instance for storage
            key_generator: Key generator for creating cache keys from requests
        """
        self.cache = cache
        self.key_generator = key_generator

    @override
    def get(self, key: str) -> OCRResponse | None:
        """Retrieve OCRResponse from cache.

        Args:
            key: Cache key

        Returns:
            Cached OCRResponse if found, None otherwise
        """
        return self.cache.get(key)

    @override
    def set(self, key: str, value: OCRResponse) -> bool:
        """Store OCRResponse in cache.

        Args:
            key: Cache key
            value: OCRResponse to cache

        Returns:
            True if cached successfully
        """
        return self.cache.set(key, value)


def create_ocr_cache_backend(
    language: str = "lat+pol+rus",
) -> tuple[OCRCacheBackend, OCRCacheKeyGenerator]:
    """Create an OCR cache backend with key generator.

    This is a convenience factory for setting up OCR caching.

    Args:
        language: OCR language configuration (default: "lat+pol+rus")

    Returns:
        Tuple of (cache_backend, key_generator)

    Example:
        >>> backend, keygen = create_ocr_cache_backend("eng")
        >>> # Use with CachedEngine
        >>> cached_engine = CachedEngine(
        ...     engine=ocr_engine,
        ...     cache_backend=backend,
        ...     key_generator=keygen,
        ...     enabled=True
        ... )
    """
    cache = PyTesseractCache(language=language)
    key_generator = OCRCacheKeyGenerator()
    backend = OCRCacheBackend(cache=cache, key_generator=key_generator)

    return backend, key_generator
