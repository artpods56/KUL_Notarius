"""Tests for OCR cache adapter and repository."""

import pytest

from notarius.infrastructure.cache.ocr_adapter import PyTesseractCache, OCRCacheKeyParams
from notarius.infrastructure.cache.utils import get_image_hash
from notarius.infrastructure.persistence.ocr_cache_repository import OCRCacheRepository
from notarius.schemas.data.cache import PyTesseractCacheItem


class TestPyTesseractCache:
    """Tests for PyTesseractCache adapter."""

    def test_cache_initialization(self, temp_cache_dir):
        """Test cache initializes correctly."""
        cache = PyTesseractCache(language="eng", caches_dir=temp_cache_dir)

        assert cache.language == "eng"
        assert cache._cache_loaded is True
        assert len(cache) == 0

    def test_cache_initialization_with_default_language(self, temp_cache_dir):
        """Test cache initializes with default language."""
        cache = PyTesseractCache(caches_dir=temp_cache_dir)

        assert cache.language == "lat+pol+rus"

    def test_normalize_kwargs_valid(self, temp_cache_dir):
        """Test normalize_kwargs with valid parameters."""
        cache = PyTesseractCache(language="eng", caches_dir=temp_cache_dir)

        result = cache.normalize_kwargs(image_hash="abc123")

        assert result == {
            "image_hash": "abc123",
            "language": "eng",
        }

    def test_normalize_kwargs_missing_image_hash(self, temp_cache_dir):
        """Test normalize_kwargs raises error when image_hash is missing."""
        cache = PyTesseractCache(language="eng", caches_dir=temp_cache_dir)

        with pytest.raises(ValueError, match="image_hash is required"):
            cache.normalize_kwargs()

    def test_normalize_kwargs_empty_image_hash(self, temp_cache_dir):
        """Test normalize_kwargs raises error when image_hash is empty."""
        cache = PyTesseractCache(language="eng", caches_dir=temp_cache_dir)

        with pytest.raises(ValueError, match="image_hash is required"):
            cache.normalize_kwargs(image_hash="")

    def test_generate_hash_deterministic(self, temp_cache_dir):
        """Test that generate_hash produces deterministic results."""
        cache = PyTesseractCache(language="eng", caches_dir=temp_cache_dir)

        hash1 = cache.generate_hash(image_hash="abc123")
        hash2 = cache.generate_hash(image_hash="abc123")

        assert hash1 == hash2
        assert len(hash1) == 64  # SHA-256 hex digest

    def test_generate_hash_different_for_different_inputs(self, temp_cache_dir):
        """Test that different inputs produce different hashes."""
        cache = PyTesseractCache(language="eng", caches_dir=temp_cache_dir)

        hash1 = cache.generate_hash(image_hash="abc123")
        hash2 = cache.generate_hash(image_hash="def456")

        assert hash1 != hash2

    def test_set_and_get_cache_item(self, temp_cache_dir, sample_ocr_cache_item):
        """Test setting and retrieving cache item."""
        cache = PyTesseractCache(language="eng", caches_dir=temp_cache_dir)

        key = cache.generate_hash(image_hash="test123")
        success = cache.set(key, sample_ocr_cache_item)

        assert success is True
        assert len(cache) == 1

        retrieved = cache.get(key)
        assert retrieved is not None
        assert retrieved.content.text == sample_ocr_cache_item.content.text
        assert retrieved.content.words == sample_ocr_cache_item.content.words
        assert retrieved.content.language == sample_ocr_cache_item.content.language

    def test_get_nonexistent_key(self, temp_cache_dir):
        """Test getting a key that doesn't exist."""
        cache = PyTesseractCache(language="eng", caches_dir=temp_cache_dir)

        result = cache.get("nonexistent_key")

        assert result is None

    def test_delete_cache_item(self, temp_cache_dir, sample_ocr_cache_item):
        """Test deleting cache item."""
        cache = PyTesseractCache(language="eng", caches_dir=temp_cache_dir)

        key = cache.generate_hash(image_hash="test123")
        cache.set(key, sample_ocr_cache_item)

        assert len(cache) == 1

        success = cache.delete(key)
        assert success is True
        assert len(cache) == 0

        # Verify item is gone
        assert cache.get(key) is None

    def test_cache_survives_reinitialization(
        self, temp_cache_dir, sample_ocr_cache_item
    ):
        """Test that cache persists after reinitialization."""
        # Create cache and store item
        cache1 = PyTesseractCache(language="eng", caches_dir=temp_cache_dir)
        key = cache1.generate_hash(image_hash="test123")
        cache1.set(key, sample_ocr_cache_item)

        # Create new cache instance with same directory
        cache2 = PyTesseractCache(language="eng", caches_dir=temp_cache_dir)

        # Should retrieve the item from disk
        retrieved = cache2.get(key)
        assert retrieved is not None
        assert retrieved.content.text == sample_ocr_cache_item.content.text

    def test_invalid_cache_data_is_deleted(self, temp_cache_dir):
        """Test that invalid cache data is automatically deleted."""
        cache = PyTesseractCache(language="eng", caches_dir=temp_cache_dir)

        # Manually insert invalid JSON
        key = "test_key"
        cache.cache.set(key, '{"invalid": "structure"}')

        # Should return None and delete the entry
        result = cache.get(key)
        assert result is None
        assert key not in cache.cache


class TestOCRCacheRepository:
    """Tests for OCRCacheRepository."""

    def test_repository_creation(self, temp_cache_dir):
        """Test repository factory method."""
        repo = OCRCacheRepository.create(language="eng", caches_dir=temp_cache_dir)

        assert repo.cache.language == "eng"
        assert isinstance(repo.cache, PyTesseractCache)

    def test_generate_key_from_image(self, temp_cache_dir, sample_image):
        """Test generating cache key from image."""
        repo = OCRCacheRepository.create(language="eng", caches_dir=temp_cache_dir)

        key = repo.generate_key(sample_image)

        assert isinstance(key, str)
        assert len(key) == 64  # SHA-256 hex

    def test_generate_key_deterministic(self, temp_cache_dir, sample_image):
        """Test that same image produces same key."""
        repo = OCRCacheRepository.create(language="eng", caches_dir=temp_cache_dir)

        key1 = repo.generate_key(sample_image)
        key2 = repo.generate_key(sample_image)

        assert key1 == key2

    def test_set_and_get(self, temp_cache_dir, sample_image):
        """Test setting and getting OCR results."""
        repo = OCRCacheRepository.create(language="eng", caches_dir=temp_cache_dir)
        key = repo.generate_key(sample_image)

        # Set OCR result
        success = repo.set(
            key=key,
            text="Test text",
            bbox=[(0, 0, 100, 20)],
            words=["Test", "text"],
            language="eng",
        )

        assert success is True

        # Get OCR result
        result = repo.get(key)
        assert result is not None
        assert result.content.text == "Test text"
        assert result.content.words == ["Test", "text"]
        assert result.content.language == "eng"
        assert len(result.content.bbox) == 1

    def test_get_cache_miss(self, temp_cache_dir):
        """Test cache miss returns None."""
        repo = OCRCacheRepository.create(language="eng", caches_dir=temp_cache_dir)

        result = repo.get("nonexistent_key")

        assert result is None

    def test_delete(self, temp_cache_dir, sample_image):
        """Test deleting cache entry."""
        repo = OCRCacheRepository.create(language="eng", caches_dir=temp_cache_dir)
        key = repo.generate_key(sample_image)

        # Store item
        repo.set(
            key=key,
            text="Test",
            bbox=[],
            words=["Test"],
            language="eng",
        )

        # Delete it
        success = repo.delete(key)
        assert success is True

        # Verify deleted
        assert repo.get(key) is None

    def test_cache_isolation_by_language(self, temp_cache_dir, sample_image):
        """Test that different languages use different cache directories."""
        repo_eng = OCRCacheRepository.create(language="eng", caches_dir=temp_cache_dir)
        repo_lat = OCRCacheRepository.create(
            language="lat+pol+rus", caches_dir=temp_cache_dir
        )

        # Same image should produce different keys due to language difference
        key_eng = repo_eng.generate_key(sample_image)
        key_lat = repo_lat.generate_key(sample_image)

        # Keys should be different because language is part of the hash
        assert key_eng != key_lat
