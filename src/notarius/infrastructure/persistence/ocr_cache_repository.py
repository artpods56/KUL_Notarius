"""Repository for OCR cache operations."""

from pathlib import Path
from typing import Self, final

from PIL.Image import Image as PILImage
from structlog import get_logger

from notarius.infrastructure.cache.storage import PyTesseractCache, get_image_hash
from notarius.schemas.data.cache import PyTesseractCacheItem, PyTesseractContent
from notarius.schemas.data.structs import BBox
from notarius.shared.logger import Logger

logger: Logger = get_logger(__name__)


@final
class OCRCacheRepository:
    """Repository for managing OCR cache operations.

    This repository encapsulates all OCR cache interactions following the
    repository pattern, providing a clean abstraction over the cache backend.
    """

    def __init__(self, cache: PyTesseractCache):
        """Initialize repository with cache backend.

        Args:
            cache: PyTesseract cache instance
        """
        self.cache = cache
        self.logger = logger.bind(language=cache.language)

    @classmethod
    def create(
        cls, language: str = "lat+pol+rus", caches_dir: Path | None = None
    ) -> Self:
        """Factory method to create repository with new cache instance.

        Args:
            language: Language configuration for OCR
            caches_dir: Optional custom cache directory

        Returns:
            Initialized OCRCacheRepository
        """
        cache = PyTesseractCache(language=language, caches_dir=caches_dir)
        return cls(cache=cache)

    def generate_key(self, image: PILImage) -> str:
        """Generate cache key for an image.

        Args:
            image: PIL Image to generate key for

        Returns:
            Cache key string (SHA-256 hash)
        """
        image_hash = get_image_hash(image)
        return self.cache.generate_hash(image_hash=image_hash)

    def get(self, key: str) -> PyTesseractCacheItem | None:
        """Retrieve OCR result from cache.

        Args:
            key: Cache key to retrieve

        Returns:
            PyTesseractCacheItem if found and valid, None otherwise
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
        language: str,
        text: str,
        words: list[str] | None = None,
        bbox: list[BBox] | None = None,
    ) -> bool:
        """Store OCR result in cache.

        Args:
            key: Cache key to store under
            text: OCR extracted text
            bbox: Bounding boxes for detected words
            words: List of detected words
            language: Language used for OCR

        Returns:
            True if successful, False otherwise
        """
        content = PyTesseractContent(
            text=text,
            bbox=bbox,
            words=words,
            language=language,
        )
        cache_item = PyTesseractCacheItem(content=content)

        success = self.cache.set(key=key, value=cache_item)

        if success:
            self.logger.debug(
                "Cached OCR result",
                key=key[:16],
            )

        return success

    def delete(self, key: str) -> bool:
        """Delete cache entry.

        Args:
            key: Cache key to delete

        Returns:
            True if deleted, False otherwise
        """
        success = self.cache.delete(key=key)
        if success:
            self.logger.debug("Deleted cache entry", key=key[:16])
        return success
