import hashlib
import json
from collections.abc import Generator
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import pytest
from PIL import Image
from PIL.Image import Image as PILImage

from notarius.infrastructure.cache.adapters.ocr import (
    PyTesseractCache,
)
from notarius.infrastructure.cache.storage.utils import get_image_hash


# NOTE: TestBaseOcrCache is commented out as BaseOcrCache no longer exists
# The OCR cache now uses pickle serialization via BaseCache directly
# class TestBaseOcrCache:
#     """Test suite for BaseOcrCache class."""
#     ...


class TestPyTesseractCache:
    """Test suite for PyTesseractCache class."""

    @pytest.fixture
    def tmp_path(self) -> Generator[Path, None, None]:
        """Create a temporary cache directory for testing."""
        with TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def tesseract_cache(self, tmp_path: Path) -> PyTesseractCache:
        """Create a PyTesseractCache instance for testing."""
        cache = PyTesseractCache(language="lat+pol+rus", caches_dir=tmp_path)
        return cache

    @pytest.fixture
    def sample_image(self) -> PILImage:
        """Create a sample PIL image for testing."""
        img: PILImage = Image.new("RGB", (100, 100), color="red")
        return img

    def test_init_default_language(self, tmp_path: Path) -> None:
        """Test initialization with default language."""
        cache: PyTesseractCache = PyTesseractCache(caches_dir=tmp_path)
        assert cache.language == "lat+pol+rus"
        assert cache._cache_loaded

    def test_init_custom_language(self, tmp_path: Path) -> None:
        """Test initialization with custom language."""
        cache: PyTesseractCache = PyTesseractCache(language="eng", caches_dir=tmp_path)
        assert cache.language == "eng"
        assert cache._cache_loaded

    def test_cache_directory_created(self, tmp_path: Path) -> None:
        """Test that cache directory is created on initialization."""
        cache: PyTesseractCache = PyTesseractCache(language="eng", caches_dir=tmp_path)

        expected_dir: Path = tmp_path / "PyTesseractCache" / "eng"
        assert expected_dir.exists()
        assert expected_dir.is_dir()

    def test_normalize_kwargs_with_language(
        self, tesseract_cache: PyTesseractCache
    ) -> None:
        """Test normalize_kwargs includes language."""
        result: dict[str, Any] = tesseract_cache.normalize_kwargs(
            image_hash="abc123", extra_param="ignored"
        )
        assert result == {"image_hash": "abc123", "language": "lat+pol+rus"}

    def test_normalize_kwargs_missing_image_hash(
        self, tesseract_cache: PyTesseractCache
    ) -> None:
        """Test normalize_kwargs with missing image_hash."""
        result: dict[str, Any] = tesseract_cache.normalize_kwargs(other_param="value")
        assert result == {"image_hash": None, "language": "lat+pol+rus"}

    def test_set_and_get_cache(
        self, tesseract_cache: PyTesseractCache, sample_image: PILImage
    ) -> None:
        """Test setting and getting cache entries."""
        image_hash: str = get_image_hash(sample_image)
        cache_key: str = tesseract_cache.generate_hash(image_hash=image_hash)

        test_value: dict[str, Any] = {"text": "Sample OCR text", "confidence": 0.95}

        tesseract_cache.set(cache_key, test_value)
        retrieved = tesseract_cache.get(cache_key)

        assert retrieved == test_value

    def test_set_cache_with_tags(
        self, tesseract_cache: PyTesseractCache, sample_image: PILImage
    ) -> None:
        """Test setting cache with schematism and filename tags."""
        image_hash: str = get_image_hash(sample_image)
        cache_key: str = tesseract_cache.generate_hash(image_hash=image_hash)

        test_value: dict[str, str] = {"text": "Tagged entry"}

        tesseract_cache.set(
            cache_key, test_value, schematism="schematism_2023", filename="page_001.jpg"
        )

        retrieved = tesseract_cache.get(cache_key)
        assert retrieved == test_value

    def test_set_cache_with_null_tags(
        self, tesseract_cache: PyTesseractCache, sample_image: PILImage
    ) -> None:
        """Test setting cache with None tags converts to 'null'."""
        image_hash: str = get_image_hash(sample_image)
        cache_key: str = tesseract_cache.generate_hash(image_hash=image_hash)

        test_value: dict[str, str] = {"text": "Null tagged entry"}

        tesseract_cache.set(cache_key, test_value, schematism=None, filename=None)
        retrieved = tesseract_cache.get(cache_key)
        assert retrieved == test_value

    def test_delete_cache_entry(
        self, tesseract_cache: PyTesseractCache, sample_image: PILImage
    ) -> None:
        """Test deleting cache entries."""
        image_hash: str = get_image_hash(sample_image)
        cache_key: str = tesseract_cache.generate_hash(image_hash=image_hash)

        test_value: dict[str, str] = {"text": "To be deleted"}

        tesseract_cache.set(cache_key, test_value)
        assert tesseract_cache.get(cache_key) == test_value

        tesseract_cache.delete(cache_key)
        assert tesseract_cache.get(cache_key) is None

    def test_cache_length(
        self, tesseract_cache: PyTesseractCache, sample_image: PILImage
    ) -> None:
        """Test __len__ returns correct cache size."""
        initial_length: int = len(tesseract_cache)

        image_hash: str = get_image_hash(sample_image)
        cache_key: str = tesseract_cache.generate_hash(image_hash=image_hash)

        tesseract_cache.set(cache_key, {"text": "Entry 1"})
        assert len(tesseract_cache) == initial_length + 1

    def test_cache_miss_returns_none(self, tesseract_cache: PyTesseractCache) -> None:
        """Test that cache miss returns None."""
        nonexistent_key: str = "nonexistent_hash_123"
        result = tesseract_cache.get(nonexistent_key)
        assert result is None

    def test_multiple_languages_separate_caches(self, tmp_path: Path) -> None:
        """Test that different languages create separate cache directories."""
        cache_eng: PyTesseractCache = PyTesseractCache(
            language="eng", caches_dir=tmp_path
        )
        cache_rus: PyTesseractCache = PyTesseractCache(
            language="rus", caches_dir=tmp_path
        )

        eng_dir: Path = tmp_path / "PyTesseractCache" / "eng"
        rus_dir: Path = tmp_path / "PyTesseractCache" / "rus"

        assert eng_dir.exists()
        assert rus_dir.exists()
        assert eng_dir != rus_dir

    def test_cache_persists_across_instances(
        self, tmp_path: Path, sample_image: PILImage
    ) -> None:
        """Test that cache persists when creating new instances."""
        # Create first instance and add entry
        cache1: PyTesseractCache = PyTesseractCache(language="eng", caches_dir=tmp_path)
        image_hash: str = get_image_hash(sample_image)
        cache_key: str = cache1.generate_hash(image_hash=image_hash)
        test_value: dict[str, str] = {"text": "Persistent data"}
        cache1.set(cache_key, test_value)

        # Create second instance and retrieve entry
        cache2: PyTesseractCache = PyTesseractCache(language="eng", caches_dir=tmp_path)
        retrieved = cache2.get(cache_key)

        assert retrieved == test_value

    def test_hash_generation_with_same_image(
        self, tesseract_cache: PyTesseractCache, sample_image: PILImage
    ) -> None:
        """Test that same image generates same hash."""
        hash1: str = get_image_hash(sample_image)
        hash2: str = get_image_hash(sample_image)
        assert hash1 == hash2

    def test_hash_generation_with_different_images(
        self, tesseract_cache: PyTesseractCache
    ) -> None:
        """Test that different images generate different hashes."""
        img1: PILImage = Image.new("RGB", (100, 100), color="red")
        img2: PILImage = Image.new("RGB", (100, 100), color="blue")

        hash1: str = get_image_hash(img1)
        hash2: str = get_image_hash(img2)
        assert hash1 != hash2

    def test_cache_overwrite_existing_entry(
        self, tesseract_cache: PyTesseractCache, sample_image: PILImage
    ) -> None:
        """Test overwriting an existing cache entry."""
        image_hash: str = get_image_hash(sample_image)
        cache_key: str = tesseract_cache.generate_hash(image_hash=image_hash)

        # Set initial value
        initial_value: dict[str, str] = {"text": "Initial"}
        tesseract_cache.set(cache_key, initial_value)
        assert tesseract_cache.get(cache_key) == initial_value

        # Overwrite with new value
        new_value: dict[str, str] = {"text": "Updated"}
        tesseract_cache.set(cache_key, new_value)
        assert tesseract_cache.get(cache_key) == new_value


class TestCacheUtils:
    """Test suite for cache utility functions."""

    def test_get_image_hash_md5(self) -> None:
        """Test that get_image_hash returns MD5 hash."""
        img: PILImage = Image.new("RGB", (10, 10), color="white")
        hash_result: str = get_image_hash(img)

        # MD5 hash should be 32 characters
        assert len(hash_result) == 32
        assert isinstance(hash_result, str)

    def test_get_image_hash_deterministic(self) -> None:
        """Test that same image produces same hash."""
        img: PILImage = Image.new("RGB", (10, 10), color="white")
        hash1: str = get_image_hash(img)
        hash2: str = get_image_hash(img)
        assert hash1 == hash2

    def test_get_image_hash_different_for_different_images(self) -> None:
        """Test that different images produce different hashes."""
        img1: PILImage = Image.new("RGB", (10, 10), color="white")
        img2: PILImage = Image.new("RGB", (10, 10), color="black")

        hash1: str = get_image_hash(img1)
        hash2: str = get_image_hash(img2)
        assert hash1 != hash2

    def test_get_image_hash_different_sizes(self) -> None:
        """Test that images of different sizes produce different hashes."""
        img1: PILImage = Image.new("RGB", (10, 10), color="red")
        img2: PILImage = Image.new("RGB", (20, 20), color="red")

        hash1: str = get_image_hash(img1)
        hash2: str = get_image_hash(img2)
        assert hash1 != hash2
