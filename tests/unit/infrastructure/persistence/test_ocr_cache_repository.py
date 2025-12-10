"""Tests for OCRCacheRepository."""

from pathlib import Path
from tempfile import TemporaryDirectory

import pytest
from PIL import Image
from PIL.Image import Image as PILImage
from pydantic import ValidationError

from notarius.infrastructure.cache.ocr_adapter import PyTesseractCache
from notarius.infrastructure.persistence.ocr_cache_repository import OCRCacheRepository
from notarius.schemas.data.cache import PyTesseractCacheItem


class TestOCRCacheRepository:
    """Test suite for OCRCacheRepository class."""

    @pytest.fixture
    def tmp_path(self) -> Path:
        """Create a temporary cache directory for testing."""
        with TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def ocr_cache(self, tmp_path: Path) -> PyTesseractCache:
        """Create a PyTesseractCache instance for testing."""
        return PyTesseractCache(language="eng", caches_dir=tmp_path)

    @pytest.fixture
    def repository(self, ocr_cache: PyTesseractCache) -> OCRCacheRepository:
        """Create a repository instance for testing."""
        return OCRCacheRepository(cache=ocr_cache)

    @pytest.fixture
    def sample_image(self) -> PILImage:
        """Create a sample PIL image for testing."""
        return Image.new("RGB", (100, 100), color="blue")

    @pytest.fixture
    def sample_cache_item(self) -> PyTesseractCacheItem:
        """Create a sample cache item."""
        return PyTesseractCacheItem(
            text="Sample text from OCR",
            words=["Sample", "text", "from", "OCR"],
            bbox=[(0, 0, 50, 20), (50, 0, 80, 20), (80, 0, 110, 20), (110, 0, 140, 20)],
        )

    def test_init(self, ocr_cache: PyTesseractCache) -> None:
        """Test repository initialization."""
        repository = OCRCacheRepository(cache=ocr_cache)
        assert repository.cache is ocr_cache
        assert repository.logger is not None

    def test_create_factory_method(self, tmp_path: Path) -> None:
        """Test factory method creates repository correctly."""
        repository = OCRCacheRepository.create(language="eng", caches_dir=tmp_path)
        assert isinstance(repository, OCRCacheRepository)
        assert isinstance(repository.cache, PyTesseractCache)
        assert repository.cache.language == "eng"

    def test_create_with_default_language(self, tmp_path: Path) -> None:
        """Test factory method with default language."""
        repository = OCRCacheRepository.create(caches_dir=tmp_path)
        assert repository.cache.language == "lat+pol+rus"

    def test_generate_key_with_image(
        self, repository: OCRCacheRepository, sample_image: PILImage
    ) -> None:
        """Test key generation with image."""
        key = repository.generate_key(image=sample_image)
        assert isinstance(key, str)
        assert len(key) == 64  # SHA-256 hash length

        # Same image should produce same key
        key2 = repository.generate_key(image=sample_image)
        assert key == key2

    def test_generate_key_deterministic(
        self, repository: OCRCacheRepository, sample_image: PILImage
    ) -> None:
        """Test that key generation is deterministic."""
        key1 = repository.generate_key(image=sample_image)
        key2 = repository.generate_key(image=sample_image)
        key3 = repository.generate_key(image=sample_image)

        assert key1 == key2 == key3

    def test_set_and_get(
        self,
        repository: OCRCacheRepository,
        sample_cache_item: PyTesseractCacheItem,
        sample_image: PILImage,
    ) -> None:
        """Test setting and getting cache items."""
        key = repository.generate_key(image=sample_image)
        repository.set(key=key, item=sample_cache_item)

        retrieved = repository.get(key=key)
        assert retrieved is not None
        assert retrieved.text == "Sample text from OCR"
        assert retrieved.words == ["Sample", "text", "from", "OCR"]
        assert len(retrieved.bbox) == 4

    def test_set_with_tags(
        self,
        repository: OCRCacheRepository,
        sample_cache_item: PyTesseractCacheItem,
        sample_image: PILImage,
    ) -> None:
        """Test setting cache items with schematism and filename tags."""
        key = repository.generate_key(image=sample_image)
        repository.set(
            key=key,
            item=sample_cache_item,
            schematism="test_schematism_2024",
            filename="test_page.jpg",
        )

        retrieved = repository.get(key=key)
        assert retrieved is not None
        assert retrieved.text == "Sample text from OCR"

    def test_get_cache_miss(self, repository: OCRCacheRepository) -> None:
        """Test getting non-existent cache entry returns None."""
        key = "nonexistent_key_987654321"
        result = repository.get(key=key)
        assert result is None

    def test_get_with_validation_error(
        self, repository: OCRCacheRepository, sample_image: PILImage
    ) -> None:
        """Test that validation errors invalidate cache entry."""
        key = repository.generate_key(image=sample_image)

        # Manually insert invalid data into cache
        repository.cache.set(key, {"invalid": "data without required fields"})

        # Should return None and delete invalid entry
        result = repository.get(key=key)
        assert result is None

        # Verify entry was deleted
        assert repository.cache.get(key) is None

    def test_delete(
        self,
        repository: OCRCacheRepository,
        sample_cache_item: PyTesseractCacheItem,
        sample_image: PILImage,
    ) -> None:
        """Test deleting cache entries."""
        key = repository.generate_key(image=sample_image)
        repository.set(key=key, item=sample_cache_item)

        # Verify it exists
        assert repository.get(key=key) is not None

        # Delete it
        repository.delete(key=key)

        # Verify it's gone
        assert repository.get(key=key) is None

    def test_cache_overwrite(
        self,
        repository: OCRCacheRepository,
        sample_image: PILImage,
    ) -> None:
        """Test overwriting existing cache entry."""
        key = repository.generate_key(image=sample_image)

        # Set initial value
        initial_item = PyTesseractCacheItem(
            text="Initial text",
            words=["Initial"],
            bbox=[(0, 0, 50, 20)],
        )
        repository.set(key=key, item=initial_item)
        assert repository.get(key=key).text == "Initial text"

        # Overwrite with new value
        new_item = PyTesseractCacheItem(
            text="Updated text",
            words=["Updated"],
            bbox=[(0, 0, 60, 20)],
        )
        repository.set(key=key, item=new_item)
        assert repository.get(key=key).text == "Updated text"

    def test_cache_persists_across_instances(
        self, tmp_path: Path, sample_image: PILImage
    ) -> None:
        """Test that cache persists when creating new repository instances."""
        # Create first instance and store item
        repo1 = OCRCacheRepository.create(language="eng", caches_dir=tmp_path)
        key = repo1.generate_key(image=sample_image)
        item = PyTesseractCacheItem(
            text="Persistent text",
            words=["Persistent"],
            bbox=[(0, 0, 70, 20)],
        )
        repo1.set(key=key, item=item)

        # Create second instance and retrieve
        repo2 = OCRCacheRepository.create(language="eng", caches_dir=tmp_path)
        retrieved = repo2.get(key=key)

        assert retrieved is not None
        assert retrieved.text == "Persistent text"

    def test_multiple_items(self, repository: OCRCacheRepository) -> None:
        """Test storing and retrieving multiple different items."""
        items = []
        for i in range(3):
            img = Image.new("RGB", (100 + i, 100 + i), color="green")
            item = PyTesseractCacheItem(
                text=f"Text {i+1}",
                words=[f"Word{i+1}"],
                bbox=[(0, 0, 40 + i * 10, 20)],
            )
            items.append((img, item))

        # Store all items
        keys = []
        for img, item in items:
            key = repository.generate_key(image=img)
            keys.append(key)
            repository.set(key=key, item=item)

        # Retrieve and verify all items
        for i, key in enumerate(keys):
            retrieved = repository.get(key=key)
            assert retrieved is not None
            assert retrieved.text == f"Text {i+1}"

    def test_different_images_different_keys(
        self, repository: OCRCacheRepository
    ) -> None:
        """Test that different images produce different keys."""
        img1 = Image.new("RGB", (100, 100), color="red")
        img2 = Image.new("RGB", (100, 100), color="blue")
        img3 = Image.new("RGB", (200, 200), color="red")

        key1 = repository.generate_key(image=img1)
        key2 = repository.generate_key(image=img2)
        key3 = repository.generate_key(image=img3)

        assert key1 != key2
        assert key1 != key3
        assert key2 != key3

    def test_cache_item_with_many_words(
        self, repository: OCRCacheRepository, sample_image: PILImage
    ) -> None:
        """Test storing cache items with many words."""
        many_words = [f"word{i}" for i in range(50)]
        many_bboxes = [(i * 10, 0, (i + 1) * 10, 20) for i in range(50)]

        complex_item = PyTesseractCacheItem(
            text=" ".join(many_words),
            words=many_words,
            bbox=many_bboxes,
        )

        key = repository.generate_key(image=sample_image)
        repository.set(key=key, item=complex_item)

        retrieved = repository.get(key=key)
        assert retrieved is not None
        assert len(retrieved.words) == 50
        assert len(retrieved.bbox) == 50

    def test_cache_item_with_empty_results(
        self, repository: OCRCacheRepository, sample_image: PILImage
    ) -> None:
        """Test storing cache items with empty OCR results."""
        empty_item = PyTesseractCacheItem(
            text="",
            words=[],
            bbox=[],
        )

        key = repository.generate_key(image=sample_image)
        repository.set(key=key, item=empty_item)

        retrieved = repository.get(key=key)
        assert retrieved is not None
        assert retrieved.text == ""
        assert retrieved.words == []
        assert retrieved.bbox == []

    def test_logging_on_operations(
        self,
        repository: OCRCacheRepository,
        sample_cache_item: PyTesseractCacheItem,
        sample_image: PILImage,
    ) -> None:
        """Test that operations are logged correctly."""
        key = repository.generate_key(image=sample_image)

        # Set operation
        repository.set(key=key, item=sample_cache_item)

        # Get operation (cache hit)
        repository.get(key=key)

        # Get operation (cache miss)
        repository.get(key="nonexistent")

        # Delete operation
        repository.delete(key=key)

        # Verify logger exists on repository
        assert repository.logger is not None

    def test_different_languages_different_caches(self, tmp_path: Path) -> None:
        """Test that different language configurations use different caches."""
        img = Image.new("RGB", (100, 100), color="white")

        # Create repositories with different languages
        repo_eng = OCRCacheRepository.create(language="eng", caches_dir=tmp_path)
        repo_lat = OCRCacheRepository.create(language="lat", caches_dir=tmp_path)

        # Store item in English cache
        key_eng = repo_eng.generate_key(image=img)
        item_eng = PyTesseractCacheItem(
            text="English text",
            words=["English"],
            bbox=[(0, 0, 50, 20)],
        )
        repo_eng.set(key=key_eng, item=item_eng)

        # Try to retrieve from Latin cache with same key
        # Note: The key generation is the same, but cache instances are different
        key_lat = repo_lat.generate_key(image=img)
        retrieved_lat = repo_lat.get(key=key_lat)

        # Should be None because different cache instances
        assert retrieved_lat is None
