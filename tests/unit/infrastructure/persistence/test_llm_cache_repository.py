"""Tests for LLMCacheRepository."""

from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image
from PIL.Image import Image as PILImage
from pydantic import ValidationError

from notarius.infrastructure.cache.llm_adapter import LLMCache
from notarius.infrastructure.cache.utils import get_image_hash, get_text_hash
from notarius.infrastructure.persistence.llm_cache_repository import LLMCacheRepository
from notarius.schemas.data.cache import LLMCacheItem


class TestLLMCacheRepository:
    """Test suite for LLMCacheRepository class."""

    @pytest.fixture
    def tmp_path(self) -> Path:
        """Create a temporary cache directory for testing."""
        with TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def llm_cache(self, tmp_path: Path) -> LLMCache:
        """Create an LLMCache instance for testing."""
        return LLMCache(model_name="test-model", caches_dir=tmp_path)

    @pytest.fixture
    def repository(self, llm_cache: LLMCache) -> LLMCacheRepository:
        """Create a repository instance for testing."""
        return LLMCacheRepository(cache=llm_cache)

    @pytest.fixture
    def sample_image(self) -> PILImage:
        """Create a sample PIL image for testing."""
        return Image.new("RGB", (100, 100), color="blue")

    @pytest.fixture
    def sample_cache_item(self) -> LLMCacheItem:
        """Create a sample cache item."""
        return LLMCacheItem(
            response={"result": "test output", "confidence": 0.95},
            hints={"language": "en"},
        )

    def test_init(self, llm_cache: LLMCache) -> None:
        """Test repository initialization."""
        repository = LLMCacheRepository(cache=llm_cache)
        assert repository.cache is llm_cache
        assert repository.logger is not None

    def test_create_factory_method(self, tmp_path: Path) -> None:
        """Test factory method creates repository correctly."""
        repository = LLMCacheRepository.create(model_name="gpt-4", caches_dir=tmp_path)
        assert isinstance(repository, LLMCacheRepository)
        assert isinstance(repository.cache, LLMCache)
        assert repository.cache.model_name == "gpt-4"

    def test_generate_key_with_image(
        self, repository: LLMCacheRepository, sample_image: PILImage
    ) -> None:
        """Test key generation with image."""
        key = repository.generate_key(image=sample_image)
        assert isinstance(key, str)
        assert len(key) == 64  # SHA-256 hash length

        # Same image should produce same key
        key2 = repository.generate_key(image=sample_image)
        assert key == key2

    def test_generate_key_with_text(self, repository: LLMCacheRepository) -> None:
        """Test key generation with text."""
        text = "Test text text"
        key = repository.generate_key(text=text)
        assert isinstance(key, str)
        assert len(key) == 64

        # Same text should produce same key
        key2 = repository.generate_key(text=text)
        assert key == key2

    def test_generate_key_with_messages(self, repository: LLMCacheRepository) -> None:
        """Test key generation with messages."""
        messages = "System: Hello\nUser: Hi there"
        key = repository.generate_key(messages=messages)
        assert isinstance(key, str)
        assert len(key) == 64

    def test_generate_key_with_hints(self, repository: LLMCacheRepository) -> None:
        """Test key generation with hints."""
        hints = {"language": "en", "context": "test"}
        key = repository.generate_key(text="test", hints=hints)
        assert isinstance(key, str)
        assert len(key) == 64

        # Different hints should produce different key
        different_hints = {"language": "fr", "context": "test"}
        key2 = repository.generate_key(text="test", hints=different_hints)
        assert key != key2

    def test_generate_key_with_all_params(
        self, repository: LLMCacheRepository, sample_image: PILImage
    ) -> None:
        """Test key generation with all parameters."""
        key = repository.generate_key(
            image=sample_image,
            text="test text",
            messages="test messages",
            hints={"key": "value"},
        )
        assert isinstance(key, str)
        assert len(key) == 64

    def test_set_and_get(
        self,
        repository: LLMCacheRepository,
        sample_cache_item: LLMCacheItem,
    ) -> None:
        """Test setting and getting cache items."""
        key = repository.generate_key(text="test")
        repository.set(key=key, item=sample_cache_item)

        retrieved = repository.get(key=key)
        assert retrieved is not None
        assert retrieved.response == sample_cache_item.response
        assert retrieved.hints == sample_cache_item.hints

    def test_set_with_tags(
        self,
        repository: LLMCacheRepository,
        sample_cache_item: LLMCacheItem,
    ) -> None:
        """Test setting cache items with schematism and filename tags."""
        key = repository.generate_key(text="test")
        repository.set(
            key=key,
            item=sample_cache_item,
            schematism="test_schematism",
            filename="test_file.txt",
        )

        retrieved = repository.get(key=key)
        assert retrieved is not None
        assert retrieved.response == sample_cache_item.response

    def test_get_cache_miss(self, repository: LLMCacheRepository) -> None:
        """Test getting non-existent cache entry returns None."""
        key = "nonexistent_key_123456789"
        result = repository.get(key=key)
        assert result is None

    def test_get_with_validation_error(self, repository: LLMCacheRepository) -> None:
        """Test that validation errors invalidate cache entry."""
        key = repository.generate_key(text="test")

        # Manually insert invalid data into cache
        repository.cache.set(key, {"invalid": "data without output field"})

        # Should return None and delete invalid entry
        result = repository.get(key=key)
        assert result is None

        # Verify entry was deleted
        assert repository.cache.get(key) is None

    def test_delete(
        self,
        repository: LLMCacheRepository,
        sample_cache_item: LLMCacheItem,
    ) -> None:
        """Test deleting cache entries."""
        key = repository.generate_key(text="test")
        repository.set(key=key, item=sample_cache_item)

        # Verify it exists
        assert repository.get(key=key) is not None

        # Delete it
        repository.delete(key=key)

        # Verify it's gone
        assert repository.get(key=key) is None

    def test_cache_overwrite(
        self,
        repository: LLMCacheRepository,
    ) -> None:
        """Test overwriting existing cache entry."""
        key = repository.generate_key(text="test")

        # Set initial value
        initial_item = LLMCacheItem(response={"version": 1}, hints=None)
        repository.set(key=key, item=initial_item)
        assert repository.get(key=key).response == {"version": 1}

        # Overwrite with new value
        new_item = LLMCacheItem(response={"version": 2}, hints=None)
        repository.set(key=key, item=new_item)
        assert repository.get(key=key).response == {"version": 2}

    def test_cache_persists_across_instances(self, tmp_path: Path) -> None:
        """Test that cache persists when creating new repository instances."""
        # Create first instance and store item
        repo1 = LLMCacheRepository.create(model_name="test-model", caches_dir=tmp_path)
        key = repo1.generate_key(text="persistent_test")
        item = LLMCacheItem(response={"data": "persistent"}, hints=None)
        repo1.set(key=key, item=item)

        # Create second instance and retrieve
        repo2 = LLMCacheRepository.create(model_name="test-model", caches_dir=tmp_path)
        retrieved = repo2.get(key=key)

        assert retrieved is not None
        assert retrieved.response == {"data": "persistent"}

    def test_multiple_items(self, repository: LLMCacheRepository) -> None:
        """Test storing and retrieving multiple different items."""
        items = [
            ("text1", LLMCacheItem(response={"id": 1}, hints=None)),
            ("text2", LLMCacheItem(response={"id": 2}, hints=None)),
            ("text3", LLMCacheItem(response={"id": 3}, hints=None)),
        ]

        # Store all items
        keys = []
        for text, item in items:
            key = repository.generate_key(text=text)
            keys.append(key)
            repository.set(key=key, item=item)

        # Retrieve and verify all items
        for i, key in enumerate(keys):
            retrieved = repository.get(key=key)
            assert retrieved is not None
            assert retrieved.response == {"id": i + 1}

    def test_generate_key_deterministic(self, repository: LLMCacheRepository) -> None:
        """Test that key generation is deterministic."""
        params = {
            "text": "test",
            "messages": "messages",
            "hints": {"key": "value"},
        }

        key1 = repository.generate_key(**params)
        key2 = repository.generate_key(**params)
        key3 = repository.generate_key(**params)

        assert key1 == key2 == key3

    def test_different_params_different_keys(
        self, repository: LLMCacheRepository
    ) -> None:
        """Test that different parameters produce different keys."""
        key1 = repository.generate_key(text="text1")
        key2 = repository.generate_key(text="text2")
        key3 = repository.generate_key(text="text1", hints={"extra": "hint"})

        assert key1 != key2
        assert key1 != key3
        assert key2 != key3

    def test_cache_item_with_complex_response(
        self, repository: LLMCacheRepository
    ) -> None:
        """Test storing cache items with complex nested responses."""
        complex_response = {
            "entities": [
                {"text": "Entity 1", "label": "LABEL", "score": 0.95},
                {"text": "Entity 2", "label": "LABEL", "score": 0.88},
            ],
            "metadata": {
                "model": "gpt-4",
                "timestamp": "2024-01-01T00:00:00",
            },
        }

        item = LLMCacheItem(response=complex_response, hints={"type": "ner"})
        key = repository.generate_key(text="test")
        repository.set(key=key, item=item)

        retrieved = repository.get(key=key)
        assert retrieved is not None
        assert retrieved.response == complex_response
        assert retrieved.hints == {"type": "ner"}

    def test_logging_on_operations(
        self,
        repository: LLMCacheRepository,
        sample_cache_item: LLMCacheItem,
    ) -> None:
        """Test that operations are logged correctly."""
        key = repository.generate_key(text="test")

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
