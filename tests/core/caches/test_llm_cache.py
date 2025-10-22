from pathlib import Path

from core.caches.llm_cache import LLMCache
from schemas.data.cache import LLMCacheItem


def test_llm_cache_persistence(tmp_path: Path):
    """Verify that values stored in LLMCache persist across instances and are retrievable."""

    cache_root = tmp_path / "cache_root"

    cache1 = LLMCache(model_name="gpt-4", caches_dir=cache_root)
    key = cache1.generate_hash(
        image_hash="test_image_hash",
        text_hash="test_text_hash",
        messages_hash="test_messages_hash",
        hints={"hint1": "value1"}
    )

    payload = LLMCacheItem(
        response={"content": "Test response", "role": "assistant"},
        hints={"hint1": "value1"}
    )
    cache1.set(key, payload.model_dump())

    # Same instance retrieval
    result1 = cache1.get(key)
    assert result1["response"]["content"] == "Test response"
    assert result1["hints"]["hint1"] == "value1"

    # Create a new instance  should read the same data
    cache2 = LLMCache(model_name="gpt-4", caches_dir=cache_root)
    result2 = cache2.get(key)
    assert result2["response"]["content"] == "Test response"
    assert result2["hints"]["hint1"] == "value1"


def test_llm_cache_model_separation(tmp_path: Path):
    """Test that different models get separate cache directories."""

    cache_root = tmp_path / "cache_root"

    # Create caches for different models
    gpt4_cache = LLMCache(model_name="gpt-4", caches_dir=cache_root)
    claude_cache = LLMCache(model_name="claude-3", caches_dir=cache_root)

    # Generate proper hash keys based on the cache's normalize_kwargs
    gpt4_key = gpt4_cache.generate_hash(
        image_hash="test_image",
        text_hash="test_text",
        messages_hash="test_messages",
        hints={}
    )
    claude_key = claude_cache.generate_hash(
        image_hash="test_image",
        text_hash="test_text",
        messages_hash="test_messages",
        hints={}
    )

    gpt4_payload = LLMCacheItem(
        response={"content": "GPT-4 response", "role": "assistant"}
    )

    claude_payload = LLMCacheItem(
        response={"content": "Claude response", "role": "assistant"}
    )

    gpt4_cache.set(gpt4_key, gpt4_payload.model_dump())
    claude_cache.set(claude_key, claude_payload.model_dump())

    # Verify they're stored separately
    gpt4_result = gpt4_cache.get(gpt4_key)
    claude_result = claude_cache.get(claude_key)
    assert gpt4_result["response"]["content"] == "GPT-4 response"
    assert claude_result["response"]["content"] == "Claude response"

    # Verify cache directories are separate
    assert gpt4_cache.model_cache_dir != claude_cache.model_cache_dir


def test_llm_cache_hit_miss(tmp_path: Path):
    """Test that LLM cache hits and misses work correctly."""

    cache_root = tmp_path / "cache_root"
    cache = LLMCache(model_name="gpt-4", caches_dir=cache_root)

    # Test cache miss
    key = cache.generate_hash(
        image_hash="test_image_hash",
        text_hash="test_text_hash",
        messages_hash="test_messages_hash",
        hints={}
    )

    value = cache.get(key)
    assert value is None

    # Test cache hit
    payload = LLMCacheItem(
        response={"content": "Cached response", "role": "assistant"},
        hints={"test": "hint"}
    )
    cache.set(key, payload.model_dump())

    result = cache.get(key)
    assert result["response"]["content"] == "Cached response"
    assert result["hints"]["test"] == "hint"


def test_llm_cache_model_name_parsing(tmp_path: Path):
    """Test that model names are parsed correctly for cache directory naming."""

    cache_root = tmp_path / "cache_root"

    # Test with path-like model name
    cache1 = LLMCache(model_name="/models/gemma-3-27b-it-Q4_K_M.gguf", caches_dir=cache_root)
    assert "gemma-3-27b-it-Q4_K_M" in str(cache1.model_cache_dir)

    # Test with simple model name
    cache2 = LLMCache(model_name="gpt-4-turbo", caches_dir=cache_root)
    assert "gpt-4-turbo" in str(cache2.model_cache_dir)

    # Test with forward slash in model name (e.g., "openai/gpt-4")
    cache3 = LLMCache(model_name="openai/gpt-4", caches_dir=cache_root)
    assert "openai_gpt-4" in str(cache3.model_cache_dir)
