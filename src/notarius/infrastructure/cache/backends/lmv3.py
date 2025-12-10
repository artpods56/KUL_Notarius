"""LayoutLMv3 cache backend and key generator for cached engine pattern."""

from typing import final, override

from structlog import get_logger

from notarius.application.ports.outbound.cached_engine import CacheBackend
from notarius.infrastructure.cache.storage import LMv3Cache, get_image_hash
from notarius.infrastructure.ml_models.lmv3.engine_adapter import (
    LMv3Request,
    LMv3Response,
)

logger = get_logger(__name__)


@final
class LMv3CacheKeyGenerator:
    """Generate cache keys for LayoutLMv3 requests."""

    def generate_key(self, request: LMv3Request) -> str:
        """Generate cache key from image.

        Args:
            request: LMv3 request containing image

        Returns:
            Cache key string (image hash)
        """
        return get_image_hash(request.input)


@final
class LMv3CacheBackend(CacheBackend[LMv3Response]):
    """Cache backend for LayoutLMv3 responses using LMv3Cache directly."""

    def __init__(self, cache: LMv3Cache):
        """Initialize with LMv3Cache instance.

        Args:
            cache: The LMv3Cache instance for storage operations
        """
        self.cache = cache

    @override
    def get(self, key: str) -> LMv3Response | None:
        """Retrieve cached LMv3 response.

        Args:
            key: Cache key to retrieve

        Returns:
            LMv3Response if found, None otherwise
        """
        cached_item = self.cache.get(key)
        if not cached_item:
            return None

        try:
            # Return LMv3Response with the structured predictions
            return LMv3Response(
                output=cached_item.content.structured_predictions,
            )

        except Exception as e:
            logger.error(
                "Failed to reconstruct LMv3Response from cache",
                key=key[:16],
                error=str(e),
            )
            return None

    @override
    def set(self, key: str, value: LMv3Response) -> bool:
        """Store LMv3 response in cache.

        Args:
            key: Cache key
            value: LMv3Response to cache

        Returns:
            True if successfully cached, False otherwise
        """
        try:
            from notarius.schemas.data.cache import LMv3CacheItem, LMv3Content

            # Create cache item with structured predictions
            cache_item = LMv3CacheItem(
                content=LMv3Content(
                    raw_predictions=None,  # We don't cache raw predictions
                    structured_predictions=value.output,
                )
            )

            # Store in cache
            return self.cache.set(key=key, value=cache_item)

        except Exception as e:
            logger.error(
                "Failed to cache LMv3Response",
                key=key[:16],
                error=str(e),
            )
            return False
