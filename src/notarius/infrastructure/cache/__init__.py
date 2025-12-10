"""Cache infrastructure for the notarius application.

This module provides a two-layer caching architecture:

1. **Storage Layer** (`cache.storage`):
   - Handles actual persistence (disk/database operations)
   - Implementations: LLMCache, PyTesseractCache, LMv3Cache

2. **Backend Layer** (`cache.backends`):
   - Implements CacheBackend protocol for engine caching pattern
   - Handles serialization/deserialization between domain and storage
   - Implementations: LLMCacheBackend, OCRCacheBackend, LMv3CacheBackend

Usage:
    ```python
    # Create storage layer
    from notarius.infrastructure.cache.storage import LLMCache
    llm_cache = LLMCache(model_name="gpt-4")

    # Create cached engine
    from notarius.infrastructure.cache.backends import create_cached_llm_engine
    cached_engine = create_cached_llm_engine(
        llm_engine=my_engine,
        llm_cache=llm_cache,
        enabled=True
    )
    ```
"""

# Storage layer exports
from notarius.infrastructure.cache.storage import (
    LLMCache,
    LMv3Cache,
    LMv3CacheKeyParams,
    PyTesseractCache,
    OCRCacheKeyParams,
    get_image_hash,
)

# Backend layer exports
from notarius.infrastructure.cache.backends import (
    LLMCacheBackend,
    LLMCacheKeyGenerator,
    LMv3CacheBackend,
    LMv3CacheKeyGenerator,
    OCRCacheBackend,
    OCRCacheKeyGenerator,
    create_cached_llm_engine,
    create_cached_lmv3_engine,
    create_cached_ocr_engine,
)

__all__ = [
    # Storage layer
    "LLMCache",
    "PyTesseractCache",
    "OCRCacheKeyParams",
    "LMv3Cache",
    "LMv3CacheKeyParams",
    "get_image_hash",
    # Backend layer
    "LLMCacheBackend",
    "LLMCacheKeyGenerator",
    "OCRCacheBackend",
    "OCRCacheKeyGenerator",
    "LMv3CacheBackend",
    "LMv3CacheKeyGenerator",
    # Factory functions
    "create_cached_llm_engine",
    "create_cached_ocr_engine",
    "create_cached_lmv3_engine",
]
