"""Repository for LMv3 cache operations."""

from pathlib import Path
from typing import Any, Self, final

from PIL.Image import Image as PILImage
from structlog import get_logger

from notarius.domain.entities.schematism import SchematismPage
from notarius.infrastructure.cache.storage import LMv3Cache, get_image_hash
from notarius.schemas.data.cache import LMv3CacheItem, LMv3Content
from notarius.shared.logger import Logger

logger: Logger = get_logger(__name__)


@final
class LMv3CacheRepository:
    """Repository for managing LMv3 cache operations.

    Encapsulates all cache-related logic including key generation,
    retrieval, storage, and cache invalidation.
    """

    def __init__(self, cache: LMv3Cache):
        """Initialize the repository with a cache instance.

        Args:
            cache: LMv3Cache instance to use for storage
        """
        self.cache = cache
        self.logger = logger.bind(checkpoint=cache.checkpoint)

    @classmethod
    def create(cls, checkpoint: str, caches_dir: Path | None = None) -> Self:
        """Factory method to create a repository with a new cache.

        Args:
            checkpoint: Checkpoint name for the LMv3 model
            caches_dir: Optional custom cache directory

        Returns:
            New LMv3CacheRepository instance
        """
        cache = LMv3Cache(checkpoint=checkpoint, caches_dir=caches_dir)
        return cls(cache=cache)

    def generate_key(self, image: PILImage) -> str:
        """Generate a cache key from input parameters.

        Args:
            image: PIL Image object to hash

        Returns:
            SHA-256 hash key for cache lookup
        """
        image_hash = get_image_hash(image)
        return self.cache.generate_hash(image_hash=image_hash)

    def get(self, key: str) -> LMv3CacheItem | None:
        """Retrieve an item from cache.

        Args:
            key: Cache key to retrieve

        Returns:
            LMv3CacheItem if found and valid, None otherwise
        """
        cache_item = self.cache.get(key=key)
        if cache_item is not None:
            self.logger.debug("Cache hit", key=key[:16])
            return cache_item

        self.logger.debug("Cache miss", key=key[:16])
        return None

    def set(
        self,
        key: str,
        structured_predictions: SchematismPage,
        raw_predictions: tuple[list[Any], list[Any], list[Any]] | None = None,
    ) -> bool:
        """Store an item in cache.

        Args:
            key: Cache key for storage
            raw_predictions: Tuple of (bboxes, prediction_ids, words)
            structured_predictions: Parsed SchematismPage predictions

        Returns:
            True if successful, False otherwise
        """
        content = LMv3Content(
            raw_predictions=raw_predictions or None,
            structured_predictions=structured_predictions,
        )
        cache_item = LMv3CacheItem(content=content)

        success = self.cache.set(key=key, value=cache_item)

        if success:
            self.logger.debug("Cache set", key=key[:16])

        return success

    def delete(self, key: str) -> bool:
        """Delete an item from cache.

        Args:
            key: Cache key to delete

        Returns:
            True if deleted, False otherwise
        """
        success = self.cache.delete(key)
        if success:
            self.logger.debug("Cache invalidated", key=key[:16])
        return success
