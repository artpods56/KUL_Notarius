"""Tests for LMv3CacheRepository."""

from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import pytest
from PIL import Image
from PIL.Image import Image as PILImage
from pydantic import ValidationError

from notarius.domain.entities.schematism import SchematismPage, SchematismEntry
from notarius.infrastructure.cache.lmv3_adapter import LMv3Cache
from notarius.infrastructure.cache.utils import get_image_hash
from notarius.infrastructure.persistence.lmv3_cache_repository import (
    LMv3CacheRepository,
)
from notarius.schemas.data.cache import LMv3CacheItem


class TestLMv3CacheRepository:
    """Test suite for LMv3CacheRepository class."""

    @pytest.fixture
    def tmp_path(self) -> Path:
        """Create a temporary cache directory for testing."""
        with TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def lmv3_cache(self, tmp_path: Path) -> LMv3Cache:
        """Create an LMv3Cache instance for testing."""
        return LMv3Cache(checkpoint="test-checkpoint", caches_dir=tmp_path)

    @pytest.fixture
    def repository(self, lmv3_cache: LMv3Cache) -> LMv3CacheRepository:
        """Create a repository instance for testing."""
        return LMv3CacheRepository(cache=lmv3_cache)

    @pytest.fixture
    def sample_image(self) -> PILImage:
        """Create a sample PIL image for testing."""
        return Image.new("RGB", (100, 100), color="green")

    @pytest.fixture
    def sample_cache_item(self) -> LMv3CacheItem:
        """Create a sample cache item."""
        return LMv3CacheItem(
            raw_predictions=(
                ["word1", "word2"],
                [[0, 0, 10, 10], [10, 0, 20, 10]],
                ["B-PARISH", "I-PARISH"],
            ),
            structured_predictions=SchematismPage(
                page_number="1",
                entries=[
                    SchematismEntry(
                        parish="Test Parish",
                        deanery=None,
                        dedication=None,
                        building_material=None,
                    )
                ],
            ),
        )

    def test_init(self, lmv3_cache: LMv3Cache) -> None:
        """Test repository initialization."""
        repository = LMv3CacheRepository(cache=lmv3_cache)
        assert repository.cache is lmv3_cache
        assert repository.logger is not None

    def test_create_factory_method(self, tmp_path: Path) -> None:
        """Test factory method creates repository correctly."""
        repository = LMv3CacheRepository.create(
            checkpoint="microsoft/layoutlmv3-base", caches_dir=tmp_path
        )
        assert isinstance(repository, LMv3CacheRepository)
        assert isinstance(repository.cache, LMv3Cache)
        assert repository.cache.checkpoint == "microsoft/layoutlmv3-base"

    def test_generate_key_with_image(
        self, repository: LMv3CacheRepository, sample_image: PILImage
    ) -> None:
        """Test key generation with image."""
        key = repository.generate_key(image=sample_image)
        assert isinstance(key, str)
        assert len(key) == 64  # SHA-256 hash length

        # Same image should produce same key
        key2 = repository.generate_key(image=sample_image)
        assert key == key2

    def test_generate_key_with_raw_predictions_flag(
        self, repository: LMv3CacheRepository, sample_image: PILImage
    ) -> None:
        """Test key generation with raw_predictions parameter."""
        key_raw = repository.generate_key(image=sample_image, raw_predictions=True)
        key_structured = repository.generate_key(
            image=sample_image, raw_predictions=False
        )

        assert isinstance(key_raw, str)
        assert isinstance(key_structured, str)
        assert len(key_raw) == 64
        assert len(key_structured) == 64

    def test_generate_key_deterministic(
        self, repository: LMv3CacheRepository, sample_image: PILImage
    ) -> None:
        """Test that key generation is deterministic."""
        key1 = repository.generate_key(image=sample_image)
        key2 = repository.generate_key(image=sample_image)
        key3 = repository.generate_key(image=sample_image)

        assert key1 == key2 == key3

    def test_set_and_get(
        self,
        repository: LMv3CacheRepository,
        sample_cache_item: LMv3CacheItem,
        sample_image: PILImage,
    ) -> None:
        """Test setting and getting cache items."""
        key = repository.generate_key(image=sample_image)
        repository.set(key=key, item=sample_cache_item)

        retrieved = repository.get(key=key)
        assert retrieved is not None
        assert retrieved.raw_predictions == sample_cache_item.raw_predictions
        assert len(retrieved.structured_predictions.entries) == 1
        assert retrieved.structured_predictions.entries[0].parish == "Test Parish"

    def test_set_with_tags(
        self,
        repository: LMv3CacheRepository,
        sample_cache_item: LMv3CacheItem,
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
        assert len(retrieved.structured_predictions.entries) == 1
        assert retrieved.structured_predictions.entries[0].parish == "Test Parish"

    def test_get_cache_miss(self, repository: LMv3CacheRepository) -> None:
        """Test getting non-existent cache entry returns None."""
        key = "nonexistent_key_987654321"
        result = repository.get(key=key)
        assert result is None

    def test_get_with_validation_error(
        self, repository: LMv3CacheRepository, sample_image: PILImage
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
        repository: LMv3CacheRepository,
        sample_cache_item: LMv3CacheItem,
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
        repository: LMv3CacheRepository,
        sample_image: PILImage,
    ) -> None:
        """Test overwriting existing cache entry."""
        key = repository.generate_key(image=sample_image)

        # Set initial value
        initial_item = LMv3CacheItem(
            raw_predictions=(["word"], [[0, 0, 10, 10]], ["O"]),
            structured_predictions=SchematismPage(
                entries=[
                    SchematismEntry(
                        parish="Initial Parish",
                    )
                ],
            ),
        )
        repository.set(key=key, item=initial_item)
        assert (
            repository.get(key=key).structured_predictions.entries[0].parish
            == "Initial Parish"
        )

        # Overwrite with new value
        new_item = LMv3CacheItem(
            raw_predictions=(["word"], [[0, 0, 10, 10]], ["O"]),
            structured_predictions=SchematismPage(
                entries=[
                    SchematismEntry(
                        parish="Updated Parish",
                    )
                ],
            ),
        )
        repository.set(key=key, item=new_item)
        assert (
            repository.get(key=key).structured_predictions.entries[0].parish
            == "Updated Parish"
        )

    def test_cache_persists_across_instances(
        self, tmp_path: Path, sample_image: PILImage
    ) -> None:
        """Test that cache persists when creating new repository instances."""
        # Create first instance and store item
        repo1 = LMv3CacheRepository.create(
            checkpoint="test-checkpoint", caches_dir=tmp_path
        )
        key = repo1.generate_key(image=sample_image)
        item = LMv3CacheItem(
            raw_predictions=(["persistent"], [[0, 0, 10, 10]], ["O"]),
            structured_predictions=SchematismPage(
                entries=[
                    SchematismEntry(
                        parish="Persistent Parish",
                    )
                ],
            ),
        )
        repo1.set(key=key, item=item)

        # Create second instance and retrieve
        repo2 = LMv3CacheRepository.create(
            checkpoint="test-checkpoint", caches_dir=tmp_path
        )
        retrieved = repo2.get(key=key)

        assert retrieved is not None
        assert retrieved.structured_predictions.entries[0].parish == "Persistent Parish"

    def test_multiple_items(self, repository: LMv3CacheRepository) -> None:
        """Test storing and retrieving multiple different items."""
        items = []
        for i in range(3):
            img = Image.new("RGB", (100 + i, 100 + i), color="blue")
            item = LMv3CacheItem(
                raw_predictions=(["word"], [[0, 0, 10, 10]], ["O"]),
                structured_predictions=SchematismPage(
                    entries=[
                        SchematismEntry(
                            parish=f"Parish {i+1}",
                        )
                    ],
                ),
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
            assert retrieved.structured_predictions.entries[0].parish == f"Parish {i+1}"

    def test_different_images_different_keys(
        self, repository: LMv3CacheRepository
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

    def test_cache_item_with_complex_predictions(
        self, repository: LMv3CacheRepository, sample_image: PILImage
    ) -> None:
        """Test storing cache items with complex predictions."""
        complex_item = LMv3CacheItem(
            raw_predictions=(
                ["St.", "Mary's", "Church", "Krakow", "Diocese"],
                [
                    [0, 0, 20, 10],
                    [20, 0, 50, 10],
                    [50, 0, 80, 10],
                    [0, 10, 30, 20],
                    [30, 10, 60, 20],
                ],
                [
                    "B-DEDICATION",
                    "I-DEDICATION",
                    "I-DEDICATION",
                    "B-PARISH",
                    "B-DEANERY",
                ],
            ),
            structured_predictions=SchematismPage(
                entries=[
                    SchematismEntry(
                        parish="Krakow",
                        deanery="Diocese",
                        dedication="St. Mary's Church",
                        building_material=None,
                    )
                ],
            ),
        )

        key = repository.generate_key(image=sample_image)
        repository.set(key=key, item=complex_item)

        retrieved = repository.get(key=key)
        assert retrieved is not None
        assert len(retrieved.structured_predictions.entries) == 1
        assert retrieved.structured_predictions.entries[0].parish == "Krakow"
        assert retrieved.structured_predictions.entries[0].deanery == "Diocese"
        assert (
            retrieved.structured_predictions.entries[0].dedication
            == "St. Mary's Church"
        )
        assert len(retrieved.raw_predictions[0]) == 5

    def test_logging_on_operations(
        self,
        repository: LMv3CacheRepository,
        sample_cache_item: LMv3CacheItem,
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
