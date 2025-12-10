"""Repository for LLM cache operations."""

from pathlib import Path
from typing import Any, Self, final

from PIL.Image import Image as PILImage
from structlog import get_logger

from notarius.infrastructure.cache.storage import LLMCache, get_image_hash
from notarius.infrastructure.cache.storage.utils import get_text_hash
from notarius.schemas.data.cache import LLMCacheItem, LLMContent
from notarius.shared.logger import Logger

logger: Logger = get_logger(__name__)


@final
class LLMCacheRepository:
    """Repository for managing LLM cache operations.

    Encapsulates all cache-related logic including key generation,
    retrieval, storage, and cache invalidation.
    """

    def __init__(self, cache: LLMCache):
        """Initialize the repository with a cache instance.

        Args:
            cache: LLMCache instance to use for storage
        """
        self.cache = cache
        self.logger = logger.bind(model=cache.model_name)

    @classmethod
    def create(cls, model_name: str, caches_dir: Path | None = None) -> Self:
        """Factory method to create a repository with a new cache.

        Args:
            model_name: Name of the LLM model
            caches_dir: Optional custom cache directory

        Returns:
            New LLMCacheRepository instance
        """
        cache = LLMCache(model_name=model_name, caches_dir=caches_dir)
        return cls(cache=cache)

    def generate_key(
        self,
        image: PILImage | None = None,
        text: str | None = None,
        messages: str | None = None,
        hints: dict[str, Any] | None = None,
    ) -> str:
        """Generate a cache key from input parameters.

        Args:
            image: PIL Image object to hash
            text: Text string to hash
            messages: Serialized messages to hash
            hints: Additional context hints

        Returns:
            SHA-256 hash key for cache lookup
        """
        return self.cache.generate_hash(
            image_hash=get_image_hash(image) if image is not None else None,
            text_hash=get_text_hash(text),
            messages_hash=get_text_hash(messages),
            hints=hints,
        )

    def get(self, key: str) -> LLMCacheItem | None:
        """Retrieve an item from cache.

        Args:
            key: Cache key to retrieve

        Returns:
            LLMCacheItem if found and valid, None otherwise
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
        response: dict[str, Any],
        hints: dict[str, Any] | None = None,
    ) -> bool:
        """Store an item in cache.

        Args:
            key: Cache key for storage
            response: LLM response data
            hints: Optional hints used for generation

        Returns:
            True if successful, False otherwise
        """
        content = LLMContent(
            response=response,
            hints=hints,
        )
        cache_item = LLMCacheItem(content=content)

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
