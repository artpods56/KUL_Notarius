"""Tests for LMv3 cache adapter and repository."""

import pytest

from notarius.infrastructure.cache.lmv3_adapter import LMv3Cache
from notarius.infrastructure.persistence.lmv3_cache_repository import (
    LMv3CacheRepository,
)
from notarius.schemas.data.cache import LMv3CacheItem


class TestLMv3Cache:
    """Tests for LMv3Cache adapter."""

    def test_cache_initialization(self, temp_cache_dir):
        """Test cache initializes correctly."""
        cache = LMv3Cache(checkpoint="test-checkpoint", caches_dir=temp_cache_dir)

        assert cache.checkpoint == "test-checkpoint"
        assert cache._cache_loaded is True
        assert len(cache) == 0

    def test_normalize_kwargs_valid(self, temp_cache_dir):
        """Test normalize_kwargs with valid parameters."""
        cache = LMv3Cache(checkpoint="test-checkpoint", caches_dir=temp_cache_dir)

        result = cache.normalize_kwargs(image_hash="abc123")

        assert result == {
            "image_hash": "abc123",
            "checkpoint": "test-checkpoint",
        }

    def test_normalize_kwargs_missing_image_hash(self, temp_cache_dir):
        """Test normalize_kwargs raises error when image_hash is missing."""
        cache = LMv3Cache(checkpoint="test-checkpoint", caches_dir=temp_cache_dir)

        with pytest.raises(ValueError, match="image_hash is required"):
            cache.normalize_kwargs()

    def test_normalize_kwargs_empty_image_hash(self, temp_cache_dir):
        """Test normalize_kwargs raises error when image_hash is empty."""
        cache = LMv3Cache(checkpoint="test-checkpoint", caches_dir=temp_cache_dir)

        with pytest.raises(ValueError, match="image_hash is required"):
            cache.normalize_kwargs(image_hash="")

    def test_generate_hash_deterministic(self, temp_cache_dir):
        """Test that generate_hash produces deterministic results."""
        cache = LMv3Cache(checkpoint="test-checkpoint", caches_dir=temp_cache_dir)

        hash1 = cache.generate_hash(image_hash="abc123")
        hash2 = cache.generate_hash(image_hash="abc123")

        assert hash1 == hash2
        assert len(hash1) == 64  # SHA-256 hex digest

    def test_generate_hash_different_for_different_inputs(self, temp_cache_dir):
        """Test that different inputs produce different hashes."""
        cache = LMv3Cache(checkpoint="test-checkpoint", caches_dir=temp_cache_dir)

        hash1 = cache.generate_hash(image_hash="abc123")
        hash2 = cache.generate_hash(image_hash="def456")

        assert hash1 != hash2

    def test_set_and_get_cache_item(self, temp_cache_dir, sample_lmv3_cache_item):
        """Test setting and retrieving cache item."""
        cache = LMv3Cache(checkpoint="test-checkpoint", caches_dir=temp_cache_dir)

        key = cache.generate_hash(image_hash="test123")
        success = cache.set(key, sample_lmv3_cache_item)

        assert success is True
        assert len(cache) == 1

        retrieved = cache.get(key)
        assert retrieved is not None
        assert (
            retrieved.content.structured_predictions.page_number
            == sample_lmv3_cache_item.content.structured_predictions.page_number
        )
        assert len(retrieved.content.structured_predictions.entries) == len(
            sample_lmv3_cache_item.content.structured_predictions.entries
        )

    def test_get_nonexistent_key(self, temp_cache_dir):
        """Test getting a key that doesn't exist."""
        cache = LMv3Cache(checkpoint="test-checkpoint", caches_dir=temp_cache_dir)

        result = cache.get("nonexistent_key")

        assert result is None

    def test_delete_cache_item(self, temp_cache_dir, sample_lmv3_cache_item):
        """Test deleting cache item."""
        cache = LMv3Cache(checkpoint="test-checkpoint", caches_dir=temp_cache_dir)

        key = cache.generate_hash(image_hash="test123")
        cache.set(key, sample_lmv3_cache_item)

        assert len(cache) == 1

        success = cache.delete(key)
        assert success is True
        assert len(cache) == 0

    def test_cache_survives_reinitialization(
        self, temp_cache_dir, sample_lmv3_cache_item
    ):
        """Test that cache persists after reinitialization."""
        # Create cache and store item
        cache1 = LMv3Cache(checkpoint="test-checkpoint", caches_dir=temp_cache_dir)
        key = cache1.generate_hash(image_hash="test123")
        cache1.set(key, sample_lmv3_cache_item)

        # Create new cache instance with same directory
        cache2 = LMv3Cache(checkpoint="test-checkpoint", caches_dir=temp_cache_dir)

        # Should retrieve the item from disk
        retrieved = cache2.get(key)
        assert retrieved is not None
        assert (
            retrieved.content.structured_predictions.page_number
            == sample_lmv3_cache_item.content.structured_predictions.page_number
        )

    def test_invalid_cache_data_is_deleted(self, temp_cache_dir):
        """Test that invalid cache data is automatically deleted."""
        cache = LMv3Cache(checkpoint="test-checkpoint", caches_dir=temp_cache_dir)

        # Manually insert invalid JSON
        key = "test_key"
        cache.cache.set(key, '{"invalid": "structure"}')

        # Should return None and delete the entry
        result = cache.get(key)
        assert result is None
        assert key not in cache.cache


class TestLMv3CacheRepository:
    """Tests for LMv3CacheRepository."""

    def test_repository_creation(self, temp_cache_dir):
        """Test repository factory method."""
        repo = LMv3CacheRepository.create(
            checkpoint="test-checkpoint", caches_dir=temp_cache_dir
        )

        assert repo.cache.checkpoint == "test-checkpoint"
        assert isinstance(repo.cache, LMv3Cache)

    def test_generate_key_from_image(self, temp_cache_dir, sample_image):
        """Test generating cache key from image."""
        repo = LMv3CacheRepository.create(
            checkpoint="test-checkpoint", caches_dir=temp_cache_dir
        )

        key = repo.generate_key(sample_image)

        assert isinstance(key, str)
        assert len(key) == 64  # SHA-256 hex

    def test_generate_key_deterministic(self, temp_cache_dir, sample_image):
        """Test that same image produces same key."""
        repo = LMv3CacheRepository.create(
            checkpoint="test-checkpoint", caches_dir=temp_cache_dir
        )

        key1 = repo.generate_key(sample_image)
        key2 = repo.generate_key(sample_image)

        assert key1 == key2

    def test_set_and_get(self, temp_cache_dir, sample_image, sample_schematism_page):
        """Test setting and getting LMv3 results."""
        repo = LMv3CacheRepository.create(
            checkpoint="test-checkpoint", caches_dir=temp_cache_dir
        )
        key = repo.generate_key(sample_image)

        # Set LMv3 result
        success = repo.set(
            key=key,
            raw_predictions=(
                [(0, 0, 100, 20)],
                [1, 2, 3],
                ["Test", "Parish"],
            ),
            structured_predictions=sample_schematism_page,
        )

        assert success is True

        # Get LMv3 result
        result = repo.get(key)
        assert result is not None
        assert (
            result.content.structured_predictions.page_number
            == sample_schematism_page.page_number
        )
        assert len(result.content.raw_predictions) == 3

    def test_get_cache_miss(self, temp_cache_dir):
        """Test cache miss returns None."""
        repo = LMv3CacheRepository.create(
            checkpoint="test-checkpoint", caches_dir=temp_cache_dir
        )

        result = repo.get("nonexistent_key")

        assert result is None

    def test_delete(self, temp_cache_dir, sample_image, sample_schematism_page):
        """Test deleting cache entry."""
        repo = LMv3CacheRepository.create(
            checkpoint="test-checkpoint", caches_dir=temp_cache_dir
        )
        key = repo.generate_key(sample_image)

        # Store item
        repo.set(
            key=key,
            raw_predictions=([], [], []),
            structured_predictions=sample_schematism_page,
        )

        # Delete it
        success = repo.delete(key)
        assert success is True

        # Verify deleted
        assert repo.get(key) is None

    def test_cache_isolation_by_checkpoint(self, temp_cache_dir, sample_image):
        """Test that different checkpoints use different cache directories."""
        repo1 = LMv3CacheRepository.create(
            checkpoint="checkpoint-v1", caches_dir=temp_cache_dir
        )
        repo2 = LMv3CacheRepository.create(
            checkpoint="checkpoint-v2", caches_dir=temp_cache_dir
        )

        # Same image should produce different keys due to checkpoint difference
        key1 = repo1.generate_key(sample_image)
        key2 = repo2.generate_key(sample_image)

        # Keys should be different because checkpoint is part of the hash
        assert key1 != key2
