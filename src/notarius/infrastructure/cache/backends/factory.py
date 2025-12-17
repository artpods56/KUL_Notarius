"""Factory functions for creating cached engine instances.

This module provides convenient factory functions that combine engines
with their cache backends to create cached versions.
"""

from typing import Any

from notarius.application.ports.outbound.cached_engine import CachedEngine
from notarius.infrastructure.cache.backends.llm import (
    LLMCacheBackend,
    LLMCacheKeyGenerator,
)
from notarius.infrastructure.cache.backends.ocr import (
    OCRCacheBackend,
    OCRCacheKeyGenerator,
)
from notarius.infrastructure.cache.backends.lmv3 import (
    LMv3CacheBackend,
    LMv3CacheKeyGenerator,
)
from notarius.infrastructure.cache.storage import LLMCache, PyTesseractCache, LMv3Cache
from notarius.infrastructure.llm.engine_adapter import LLMEngine


def create_cached_llm_engine(
    llm_engine: LLMEngine,
    llm_cache: LLMCache,
    enabled: bool = True,
):
    """Create a cached LLM engine.

    Args:
        llm_engine: Base LLM engine instance
        llm_cache: LLM cache storage instance
        enabled: Whether caching is enabled

    Returns:
        CachedEngine wrapping the LLM engine
    """
    return CachedEngine(
        engine=llm_engine,
        cache_backend=LLMCacheBackend[Any](llm_cache),
        key_generator=LLMCacheKeyGenerator(),
        enabled=enabled,
    )


def create_cached_ocr_engine(
    ocr_engine,
    ocr_cache: PyTesseractCache,
    enabled: bool = True,
):
    """Create a cached OCR engine.

    Args:
        ocr_engine: Base OCR engine instance
        ocr_cache: OCR cache storage instance
        enabled: Whether caching is enabled

    Returns:
        CachedEngine wrapping the OCR engine
    """
    return CachedEngine(
        engine=ocr_engine,
        cache_backend=OCRCacheBackend(ocr_cache),
        key_generator=OCRCacheKeyGenerator(),
        enabled=enabled,
    )


def create_cached_lmv3_engine(
    lmv3_engine,
    lmv3_cache: LMv3Cache,
    enabled: bool = True,
):
    """Create a cached LayoutLMv3 engine.

    Args:
        lmv3_engine: Base LMv3 engine instance
        lmv3_cache: LMv3 cache storage instance
        enabled: Whether caching is enabled

    Returns:
        CachedEngine wrapping the LMv3 engine
    """
    return CachedEngine(
        engine=lmv3_engine,
        cache_backend=LMv3CacheBackend(lmv3_cache),
        key_generator=LMv3CacheKeyGenerator(),
        enabled=enabled,
    )
