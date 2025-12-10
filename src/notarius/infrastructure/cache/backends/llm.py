"""LLM cache backend and key generator for cached engine pattern."""

from __future__ import annotations

import hashlib
import json
from typing import final, override

from pydantic import BaseModel
from structlog import get_logger

from notarius.application.ports.outbound.cached_engine import (
    CacheBackend,
    CacheKeyGenerator,
)
from notarius.infrastructure.cache.adapters.llm import LLMCache
from notarius.infrastructure.llm.conversation import Conversation
from notarius.infrastructure.llm.engine_adapter import (
    CompletionResult,
    CompletionRequest,
)

logger = get_logger(__name__)


@final
class LLMCacheKeyGenerator(CacheKeyGenerator[CompletionRequest]):
    """Generate deterministic cache keys for LLM completion requests.

    Keys are based on:
    - Conversation messages (content and roles)
    - Whether structured output is requested
    """

    @override
    def generate_key(self, request: CompletionRequest) -> str:
        """Generate a unique cache key from the request.

        Args:
            request: CompletionRequest containing conversation and config

        Returns:
            SHA-256 hash of the request parameters
        """
        # Serialize conversation to dict
        conversation_dict = request.input.to_dict()

        # Create payload for hashing
        payload = {
            "messages": conversation_dict,
            "structured_output": request.structured_output is not None,
        }

        # Generate deterministic hash
        payload_str = json.dumps(payload, sort_keys=True)
        return hashlib.sha256(payload_str.encode()).hexdigest()


@final
class LLMCacheBackend[T: BaseModel](CacheBackend[CompletionResult[T]]):
    """Cache backend adapter for LLM responses.

    This adapter bridges the CachedEngine protocol with the LLMCache storage,
    using pickle serialization for automatic handling of complex types.

    The cache stores complete CompletionResult objects including:
    - Full conversation history
    - Provider responses
    - Structured output (Pydantic models)
    """

    def __init__(self, cache: LLMCache, key_generator: LLMCacheKeyGenerator):
        """Initialize the cache backend.

        Args:
            cache: LLMCache instance for storage
            key_generator: Key generator for creating cache keys from requests
        """
        self.cache = cache
        self.key_generator = key_generator

    @override
    def get(self, key: str) -> CompletionResult[T] | None:
        """Retrieve CompletionResult from cache.

        Args:
            key: Cache key

        Returns:
            Cached CompletionResult if found, None otherwise
        """
        return self.cache.get(key)

    @override
    def set(self, key: str, value: CompletionResult[T]) -> bool:
        """Store CompletionResult in cache.

        Args:
            key: Cache key
            value: CompletionResult to cache

        Returns:
            True if cached successfully
        """
        return self.cache.set(key, value)


def create_llm_cache_backend(
    model_name: str,
) -> tuple[LLMCacheBackend[BaseModel], LLMCacheKeyGenerator]:
    """Create an LLM cache backend with key generator.

    This is a convenience factory for setting up LLM caching.

    Args:
        model_name: Model name for cache directory namespacing

    Returns:
        Tuple of (cache_backend, key_generator)

    Example:
        >>> backend, keygen = create_llm_cache_backend("gpt-4")
        >>> # Use with CachedEngine
        >>> cached_engine = CachedEngine(
        ...     engine=llm_engine,
        ...     cache_backend=backend,
        ...     key_generator=keygen,
        ...     enabled=True
        ... )
    """
    cache = LLMCache(model_name=model_name)
    key_generator = LLMCacheKeyGenerator()
    backend = LLMCacheBackend[BaseModel](cache=cache, key_generator=key_generator)

    return backend, key_generator
