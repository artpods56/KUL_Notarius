"""Storage layer for cache implementations.

This module contains the actual cache storage implementations that handle
persistence operations (disk/database).
"""

from notarius.infrastructure.cache.adapters.llm import LLMCache
from notarius.infrastructure.cache.adapters.ocr import PyTesseractCache, OCRCacheKeyParams
from notarius.infrastructure.cache.adapters.lmv3 import LMv3Cache, LMv3CacheKeyParams
from notarius.infrastructure.cache.storage.utils import get_image_hash

__all__ = [
    "LLMCache",
    "PyTesseractCache",
    "OCRCacheKeyParams",
    "LMv3Cache",
    "LMv3CacheKeyParams",
    "get_image_hash",
]
