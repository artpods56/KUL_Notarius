"""Tests for LLM cache adapter and repository."""

import pytest

from notarius.infrastructure.cache.llm_adapter import LLMCache
from notarius.infrastructure.persistence.llm_cache_repository import LLMCacheRepository
from notarius.schemas.data.cache import LLMCacheItem


class TestLLMCache:
    """Tests for LLMCache adapter."""

    def test_cache_initialization(self, temp_cache_dir):
        """Test cache initializes correctly."""
        cache = LLMCache(model_name="gpt-4", caches_dir=temp_cache_dir)

        assert cache.model_name == "gpt-4"
        assert cache._cache_loaded is True
        assert len(cache) == 0

    def test_model_name_parsing_absolute_path(self, temp_cache_dir):
        """Test model name parsing from absolute path."""
        cache = LLMCache(
            model_name="/models/gemma-3-27b-it-Q4_K_M.gguf", caches_dir=temp_cache_dir
        )

        assert cache.model_name == "gemma-3-27b-it-Q4_K_M"

    def test_model_name_parsing_with_slashes(self, temp_cache_dir):
        """Test model name parsing replaces slashes."""
        cache = LLMCache(model_name="gpt-4/turbo", caches_dir=temp_cache_dir)

        assert cache.model_name == "gpt-4_turbo"

    def test_normalize_kwargs_with_image_hash(self, temp_cache_dir):
        """Test normalize_kwargs with image_hash."""
        cache = LLMCache(model_name="gpt-4", caches_dir=temp_cache_dir)

        result = cache.normalize_kwargs(image_hash="abc123")

        assert result == {
            "model": "gpt-4",
            "image_hash": "abc123",
        }

    def test_normalize_kwargs_with_text_hash(self, temp_cache_dir):
        """Test normalize_kwargs with text_hash."""
        cache = LLMCache(model_name="gpt-4", caches_dir=temp_cache_dir)

        result = cache.normalize_kwargs(text_hash="def456")

        assert result == {
            "model": "gpt-4",
            "text_hash": "def456",
        }

    def test_normalize_kwargs_with_messages_hash(self, temp_cache_dir):
        """Test normalize_kwargs with messages_hash."""
        cache = LLMCache(model_name="gpt-4", caches_dir=temp_cache_dir)

        result = cache.normalize_kwargs(messages_hash="ghi789")

        assert result == {
            "model": "gpt-4",
            "messages_hash": "ghi789",
        }

    def test_normalize_kwargs_with_hints(self, temp_cache_dir):
        """Test normalize_kwargs with hints."""
        cache = LLMCache(model_name="gpt-4", caches_dir=temp_cache_dir)

        result = cache.normalize_kwargs(hints={"context": "test"})

        assert "model" in result
        assert "hints" in result
        assert '"context": "test"' in result["hints"]

    def test_normalize_kwargs_with_multiple_params(self, temp_cache_dir):
        """Test normalize_kwargs with multiple parameters."""
        cache = LLMCache(model_name="gpt-4", caches_dir=temp_cache_dir)

        result = cache.normalize_kwargs(
            image_hash="abc123",
            text_hash="def456",
            messages_hash="ghi789",
            hints={"key": "value"},
        )

        assert result["model"] == "gpt-4"
        assert result["image_hash"] == "abc123"
        assert result["text_hash"] == "def456"
        assert result["messages_hash"] == "ghi789"
        assert "hints" in result

    def test_normalize_kwargs_no_params_raises_error(self, temp_cache_dir):
        """Test normalize_kwargs raises error when no params provided."""
        cache = LLMCache(model_name="gpt-4", caches_dir=temp_cache_dir)

        with pytest.raises(ValueError, match="At least one cache key parameter"):
            cache.normalize_kwargs()

    def test_generate_hash_deterministic(self, temp_cache_dir):
        """Test that generate_hash produces deterministic results."""
        cache = LLMCache(model_name="gpt-4", caches_dir=temp_cache_dir)

        hash1 = cache.generate_hash(text_hash="abc123")
        hash2 = cache.generate_hash(text_hash="abc123")

        assert hash1 == hash2
        assert len(hash1) == 64  # SHA-256 hex digest

    def test_generate_hash_different_for_different_inputs(self, temp_cache_dir):
        """Test that different inputs produce different hashes."""
        cache = LLMCache(model_name="gpt-4", caches_dir=temp_cache_dir)

        hash1 = cache.generate_hash(text_hash="abc123")
        hash2 = cache.generate_hash(text_hash="def456")

        assert hash1 != hash2

    def test_set_and_get_cache_item(self, temp_cache_dir, sample_llm_cache_item):
        """Test setting and retrieving cache item."""
        cache = LLMCache(model_name="gpt-4", caches_dir=temp_cache_dir)

        key = cache.generate_hash(text_hash="test123")
        success = cache.set(key, sample_llm_cache_item)

        assert success is True
        assert len(cache) == 1

        retrieved = cache.get(key)
        assert retrieved is not None
        assert retrieved.content.response == sample_llm_cache_item.content.response
        assert retrieved.content.hints == sample_llm_cache_item.content.hints

    def test_get_nonexistent_key(self, temp_cache_dir):
        """Test getting a key that doesn't exist."""
        cache = LLMCache(model_name="gpt-4", caches_dir=temp_cache_dir)

        result = cache.get("nonexistent_key")

        assert result is None

    def test_delete_cache_item(self, temp_cache_dir, sample_llm_cache_item):
        """Test deleting cache item."""
        cache = LLMCache(model_name="gpt-4", caches_dir=temp_cache_dir)

        key = cache.generate_hash(text_hash="test123")
        cache.set(key, sample_llm_cache_item)

        assert len(cache) == 1

        success = cache.delete(key)
        assert success is True
        assert len(cache) == 0

    def test_cache_survives_reinitialization(
        self, temp_cache_dir, sample_llm_cache_item
    ):
        """Test that cache persists after reinitialization."""
        # Create cache and store item
        cache1 = LLMCache(model_name="gpt-4", caches_dir=temp_cache_dir)
        key = cache1.generate_hash(text_hash="test123")
        cache1.set(key, sample_llm_cache_item)

        # Create new cache instance with same directory
        cache2 = LLMCache(model_name="gpt-4", caches_dir=temp_cache_dir)

        # Should retrieve the item from disk
        retrieved = cache2.get(key)
        assert retrieved is not None
        assert retrieved.content.response == sample_llm_cache_item.content.response

    def test_invalid_cache_data_is_deleted(self, temp_cache_dir):
        """Test that invalid cache data is automatically deleted."""
        cache = LLMCache(model_name="gpt-4", caches_dir=temp_cache_dir)

        # Manually insert invalid JSON
        key = "test_key"
        cache.cache.set(key, '{"invalid": "structure"}')

        # Should return None and delete the entry
        result = cache.get(key)
        assert result is None
        assert key not in cache.cache


class TestLLMCacheRepository:
    """Tests for LLMCacheRepository."""

    def test_repository_creation(self, temp_cache_dir):
        """Test repository factory method."""
        repo = LLMCacheRepository.create(model_name="gpt-4", caches_dir=temp_cache_dir)

        assert repo.cache.model_name == "gpt-4"
        assert isinstance(repo.cache, LLMCache)

    def test_generate_key_from_image(self, temp_cache_dir, sample_image):
        """Test generating cache key from image."""
        repo = LLMCacheRepository.create(model_name="gpt-4", caches_dir=temp_cache_dir)

        key = repo.generate_key(image=sample_image)

        assert isinstance(key, str)
        assert len(key) == 64  # SHA-256 hex

    def test_generate_key_from_text(self, temp_cache_dir):
        """Test generating cache key from text."""
        repo = LLMCacheRepository.create(model_name="gpt-4", caches_dir=temp_cache_dir)

        key = repo.generate_key(text="Test text")

        assert isinstance(key, str)
        assert len(key) == 64

    def test_generate_key_from_messages(self, temp_cache_dir):
        """Test generating cache key from messages."""
        repo = LLMCacheRepository.create(model_name="gpt-4", caches_dir=temp_cache_dir)

        key = repo.generate_key(messages='[{"role": "user", "content": "test"}]')

        assert isinstance(key, str)
        assert len(key) == 64

    def test_generate_key_with_hints(self, temp_cache_dir):
        """Test generating cache key with hints."""
        repo = LLMCacheRepository.create(model_name="gpt-4", caches_dir=temp_cache_dir)

        key = repo.generate_key(text="Test", hints={"context": "test"})

        assert isinstance(key, str)
        assert len(key) == 64

    def test_generate_key_deterministic(self, temp_cache_dir):
        """Test that same inputs produce same key."""
        repo = LLMCacheRepository.create(model_name="gpt-4", caches_dir=temp_cache_dir)

        key1 = repo.generate_key(text="Test text")
        key2 = repo.generate_key(text="Test text")

        assert key1 == key2

    def test_set_and_get(self, temp_cache_dir):
        """Test setting and getting LLM results."""
        repo = LLMCacheRepository.create(model_name="gpt-4", caches_dir=temp_cache_dir)
        key = repo.generate_key(text="Test text")

        # Set LLM result
        success = repo.set(
            key=key,
            response={"page_number": "42", "entries": []},
            hints={"context": "test"},
        )

        assert success is True

        # Get LLM result
        result = repo.get(key)
        assert result is not None
        assert result.content.response["page_number"] == "42"
        assert result.content.hints == {"context": "test"}

    def test_get_cache_miss(self, temp_cache_dir):
        """Test cache miss returns None."""
        repo = LLMCacheRepository.create(model_name="gpt-4", caches_dir=temp_cache_dir)

        result = repo.get("nonexistent_key")

        assert result is None

    def test_delete(self, temp_cache_dir):
        """Test deleting cache entry."""
        repo = LLMCacheRepository.create(model_name="gpt-4", caches_dir=temp_cache_dir)
        key = repo.generate_key(text="Test text")

        # Store item
        repo.set(key=key, response={"test": "data"})

        # Delete it
        success = repo.delete(key)
        assert success is True

        # Verify deleted
        assert repo.get(key) is None

    def test_cache_isolation_by_model(self, temp_cache_dir):
        """Test that different models use different cache directories."""
        repo1 = LLMCacheRepository.create(model_name="gpt-4", caches_dir=temp_cache_dir)
        repo2 = LLMCacheRepository.create(
            model_name="gpt-3.5-turbo", caches_dir=temp_cache_dir
        )

        # Same text should produce different keys due to model difference
        key1 = repo1.generate_key(text="Test text")
        key2 = repo2.generate_key(text="Test text")

        # Keys should be different because model is part of the hash
        assert key1 != key2
