"""OCR cache backend and key generator for cached engine pattern."""

import hashlib
import json

from PIL.Image import Image
from typing import final, override, TypedDict, Unpack

from structlog import get_logger

from notarius.application.ports.outbound.cached_engine import CacheBackend
from notarius.infrastructure.cache.storage import PyTesseractCache, get_image_hash
from notarius.infrastructure.cache.storage.utils import get_text_hash
from notarius.infrastructure.ocr.engine_adapter import OCRRequest, OCRResponse

logger = get_logger(__name__)


OCRCacheKeyGeneratorPayload = TypedDict(
    "OCRCacheKeyGeneratorPayload",
    {"image": Image, "config": str},
)


@final
class OCRCacheKeyGenerator:
    """Generate cache keys for OCR requests."""

    def generate_key(self, **payload: Unpack[OCRCacheKeyGeneratorPayload]) -> str:
        """Generate cache key from image and mode.

        Args:
            request: OCR request containing image and mode

        Returns:
            Cache key string
        """
        image = payload["image"]
        config = payload["config"]

        generator_payload = {"image": image, "config": config}

        payload_str = json.dumps(generator_payload, sort_keys=True)
        return hashlib.sha256(payload_str.encode()).hexdigest()


@final
class OCRCacheBackend(CacheBackend[OCRResponse]):
    """Cache backend for OCR responses using PyTesseractCache directly."""

    def __init__(self, cache: OCRResponse):
        """Initialize with PyTesseractCache instance.

        Args:
            cache: The PyTesseractCache instance for storage operations
        """
        self.cache = cache

    @override
    def get(self, key: str) -> OCRResponse | None:
        """Retrieve cached OCR response.

        Args:
            key: Cache key to retrieve

        Returns:
            OCRResponse if found, None otherwise
        """
        try:
            cached_item = self.cache.get(key)
            if not cached_item:
                return None
            else:
                return cached_item

        except Exception as e:
            logger.error(
                "Failed to reconstruct OCRResponse from cache",
                key=key[:16],
                error=str(e),
            )
            return None

    @override
    def set(self, key: str, value: OCRResponse) -> bool:
        """Store OCR response in cache.

        Args:
            key: Cache key
            value: OCRResponse to cache

        Returns:
            True if successfully cached, False otherwise
        """
        try:
            # Extract text from the response
            text = ""
            words = None
            bbox = None

            if hasattr(value.output, "text"):
                text = value.output.text
            if hasattr(value.output, "words"):
                words = value.output.words
            if hasattr(value.output, "bboxes"):
                bbox = value.output.bboxes

            from notarius.schemas.data.cache import (
                PyTesseractCacheItem,
                PyTesseractContent,
            )

            # Create cache item
            cache_item = PyTesseractCacheItem(
                content=PyTesseractContent(
                    text=text,
                    bbox=bbox,
                    words=words,
                    language=self.cache.language,
                )
            )

            # Store in cache
            return self.cache.set(key=key, value=cache_item)

        except Exception as e:
            logger.error(
                "Failed to cache OCRResponse",
                key=key[:16],
                error=str(e),
            )
            return False
