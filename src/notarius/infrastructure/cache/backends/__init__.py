"""Cache backends for engine caching pattern.

This module contains CacheBackend implementations and key generators
for different engine types (LLM, OCR, LMv3).
"""

from notarius.infrastructure.cache.backends.factory import (
    create_cached_llm_engine,
    create_cached_lmv3_engine,
    create_cached_ocr_engine,
)
from notarius.infrastructure.cache.backends.llm import (
    LLMCacheBackend,
    LLMCacheKeyGenerator,
)
from notarius.infrastructure.cache.backends.lmv3 import (
    LMv3CacheBackend,
    LMv3CacheKeyGenerator,
)
from notarius.infrastructure.cache.backends.ocr import (
    OCRCacheBackend,
    OCRCacheKeyGenerator,
)

__all__ = [
    # LLM
    "LLMCacheBackend",
    "LLMCacheKeyGenerator",
    # OCR
    "OCRCacheBackend",
    "OCRCacheKeyGenerator",
    # LMv3
    "LMv3CacheBackend",
    "LMv3CacheKeyGenerator",
    # Factories
    "create_cached_llm_engine",
    "create_cached_ocr_engine",
    "create_cached_lmv3_engine",
]
