"""Tests for LayoutLMv3 cache adapter with pickle serialization."""

import pickle
from pathlib import Path
from unittest.mock import patch

import pytest

from notarius.domain.entities.schematism import SchematismPage, SchematismEntry
from notarius.infrastructure.cache.adapters.lmv3 import LMv3Cache
from notarius.infrastructure.ml_models.lmv3.engine_adapter import LMv3Response


# Test fixtures


@pytest.fixture
def tmp_cache_dir(tmp_path: Path) -> Path:
    """Create a temporary cache directory."""
    return tmp_path / "test_caches"


@pytest.fixture
def lmv3_cache(tmp_cache_dir: Path) -> LMv3Cache:
    """Create a LMv3Cache instance for testing."""
    return LMv3Cache(checkpoint="test-checkpoint", caches_dir=tmp_cache_dir)


@pytest.fixture
def sample_schematism_page() -> SchematismPage:
    """Create a sample SchematismPage with entries."""
    entries = [
        SchematismEntry(
            parish="St. Mary's Church",
            deanery="Central Deanery",
            dedication="Virgin Mary",
            building_material="Brick",
        ),
        SchematismEntry(
            parish="St. John's Cathedral",
            deanery=None,
            dedication="John the Baptist",
            building_material="Stone",
        ),
    ]
    return SchematismPage(page_number="42", entries=entries)


@pytest.fixture
def sample_lmv3_response(sample_schematism_page: SchematismPage) -> LMv3Response:
    """Create a sample LMv3Response."""
    return LMv3Response(output=sample_schematism_page)


@pytest.fixture
def empty_lmv3_response() -> LMv3Response:
    """Create an LMv3Response with no entries."""
    page = SchematismPage(page_number="1", entries=[])
    return LMv3Response(output=page)


# Test cases


class TestLMv3CacheInitialization:
    """Test LMv3 cache initialization."""

    def test_init_creates_cache_directory(self, tmp_cache_dir: Path) -> None:
        """Test that cache directory is created on initialization."""
        cache = LMv3Cache(checkpoint="model-v1", caches_dir=tmp_cache_dir)
        expected_path = tmp_cache_dir / "LMv3Cache" / "model-v1"

        assert expected_path.exists()
        assert expected_path.is_dir()

    def test_init_with_checkpoint(self, tmp_cache_dir: Path) -> None:
        """Test initialization with checkpoint name."""
        cache = LMv3Cache(checkpoint="layoutlmv3-base", caches_dir=tmp_cache_dir)
        assert cache.checkpoint == "layoutlmv3-base"
        assert cache.cache_type == "LMv3Cache"
        assert cache.cache_name == "layoutlmv3-base"

    def test_cache_type_property(self, lmv3_cache: LMv3Cache) -> None:
        """Test that cache_type returns correct value."""
        assert lmv3_cache.cache_type == "LMv3Cache"

    def test_different_checkpoints_use_separate_caches(
        self, tmp_cache_dir: Path
    ) -> None:
        """Test that different checkpoints create separate cache directories."""
        cache_v1 = LMv3Cache(checkpoint="model-v1", caches_dir=tmp_cache_dir)
        cache_v2 = LMv3Cache(checkpoint="model-v2", caches_dir=tmp_cache_dir)

        v1_dir = tmp_cache_dir / "LMv3Cache" / "model-v1"
        v2_dir = tmp_cache_dir / "LMv3Cache" / "model-v2"

        assert v1_dir.exists()
        assert v2_dir.exists()
        assert v1_dir != v2_dir


class TestLMv3CacheSetAndGet:
    """Test basic cache set and get operations."""

    def test_set_and_get_response(
        self,
        lmv3_cache: LMv3Cache,
        sample_lmv3_response: LMv3Response,
    ) -> None:
        """Test caching and retrieving an LMv3Response."""
        key = "test_image_hash_1"

        # Cache the result
        success = lmv3_cache.set(key, sample_lmv3_response)
        assert success is True

        # Retrieve the result
        cached_result = lmv3_cache.get(key)
        assert cached_result is not None
        assert cached_result.output.page_number == "42"
        assert len(cached_result.output.entries) == 2

    def test_get_nonexistent_key_returns_none(self, lmv3_cache: LMv3Cache) -> None:
        """Test that getting a nonexistent key returns None."""
        result = lmv3_cache.get("nonexistent_key")
        assert result is None

    def test_cache_overwrites_existing_entry(
        self,
        lmv3_cache: LMv3Cache,
        sample_lmv3_response: LMv3Response,
        empty_lmv3_response: LMv3Response,
    ) -> None:
        """Test that setting the same key twice overwrites the first value."""
        key = "overwrite_test"

        # Set initial value
        lmv3_cache.set(key, sample_lmv3_response)
        initial = lmv3_cache.get(key)
        assert initial is not None
        assert len(initial.output.entries) == 2

        # Overwrite with new value
        lmv3_cache.set(key, empty_lmv3_response)
        updated = lmv3_cache.get(key)

        assert updated is not None
        assert len(updated.output.entries) == 0

    def test_delete_cache_entry(
        self,
        lmv3_cache: LMv3Cache,
        sample_lmv3_response: LMv3Response,
    ) -> None:
        """Test deleting a cache entry."""
        key = "delete_test"

        lmv3_cache.set(key, sample_lmv3_response)
        assert lmv3_cache.get(key) is not None

        lmv3_cache.delete(key)
        assert lmv3_cache.get(key) is None


class TestLMv3CacheSchematismData:
    """Test caching with SchematismPage data."""

    def test_cache_preserves_schematism_entries(
        self,
        lmv3_cache: LMv3Cache,
        sample_lmv3_response: LMv3Response,
    ) -> None:
        """Test that schematism entries are preserved through caching."""
        key = "schematism_test"

        lmv3_cache.set(key, sample_lmv3_response)
        cached = lmv3_cache.get(key)

        assert cached is not None
        entries = cached.output.entries

        # Verify first entry
        assert entries[0].parish == "St. Mary's Church"
        assert entries[0].deanery == "Central Deanery"
        assert entries[0].dedication == "Virgin Mary"
        assert entries[0].building_material == "Brick"

        # Verify second entry
        assert entries[1].parish == "St. John's Cathedral"
        assert entries[1].deanery is None
        assert entries[1].dedication == "John the Baptist"
        assert entries[1].building_material == "Stone"

    def test_cache_with_empty_entries(
        self,
        lmv3_cache: LMv3Cache,
        empty_lmv3_response: LMv3Response,
    ) -> None:
        """Test caching a page with no entries."""
        key = "empty_entries_test"

        lmv3_cache.set(key, empty_lmv3_response)
        cached = lmv3_cache.get(key)

        assert cached is not None
        assert cached.output.page_number == "1"
        assert cached.output.entries == []

    def test_cache_with_many_entries(
        self,
        lmv3_cache: LMv3Cache,
    ) -> None:
        """Test caching a page with many entries."""
        entries = [
            SchematismEntry(
                parish=f"Church {i}",
                deanery=f"Deanery {i % 5}",
                dedication=f"Saint {i}",
                building_material="Brick" if i % 2 == 0 else "Stone",
            )
            for i in range(50)
        ]
        page = SchematismPage(page_number="100", entries=entries)
        response = LMv3Response(output=page)

        key = "many_entries_test"
        lmv3_cache.set(key, response)
        cached = lmv3_cache.get(key)

        assert cached is not None
        assert len(cached.output.entries) == 50

    def test_cache_with_optional_fields(
        self,
        lmv3_cache: LMv3Cache,
    ) -> None:
        """Test caching entries with None values in optional fields."""
        entries = [
            SchematismEntry(
                parish="Minimal Church",
                deanery=None,
                dedication=None,
                building_material=None,
            )
        ]
        page = SchematismPage(page_number="5", entries=entries)
        response = LMv3Response(output=page)

        key = "optional_fields_test"
        lmv3_cache.set(key, response)
        cached = lmv3_cache.get(key)

        assert cached is not None
        assert cached.output.entries[0].parish == "Minimal Church"
        assert cached.output.entries[0].deanery is None
        assert cached.output.entries[0].dedication is None
        assert cached.output.entries[0].building_material is None


class TestLMv3CachePersistence:
    """Test cache persistence across instances."""

    def test_cache_persists_across_instances(
        self,
        tmp_cache_dir: Path,
        sample_lmv3_response: LMv3Response,
    ) -> None:
        """Test that cached data persists when creating new cache instances."""
        key = "persistence_test"

        # Cache with first instance
        cache1 = LMv3Cache(checkpoint="model-v1", caches_dir=tmp_cache_dir)
        cache1.set(key, sample_lmv3_response)

        # Retrieve with second instance
        cache2 = LMv3Cache(checkpoint="model-v1", caches_dir=tmp_cache_dir)
        cached = cache2.get(key)

        assert cached is not None
        assert cached.output.page_number == "42"
        assert len(cached.output.entries) == 2

    def test_different_checkpoints_use_separate_caches(
        self,
        tmp_cache_dir: Path,
        sample_lmv3_response: LMv3Response,
    ) -> None:
        """Test that different checkpoint caches are isolated."""
        key = "same_key"

        cache_v1 = LMv3Cache(checkpoint="model-v1", caches_dir=tmp_cache_dir)
        cache_v2 = LMv3Cache(checkpoint="model-v2", caches_dir=tmp_cache_dir)

        # Cache in v1
        cache_v1.set(key, sample_lmv3_response)

        # Should not exist in v2 cache
        assert cache_v2.get(key) is None

        # Should exist in v1 cache
        assert cache_v1.get(key) is not None


class TestLMv3CacheErrorHandling:
    """Test error handling and edge cases."""

    def test_pickle_error_during_serialization(
        self,
        lmv3_cache: LMv3Cache,
    ) -> None:
        """Test handling of pickle errors during serialization."""
        with patch.object(
            lmv3_cache.cache, "set", side_effect=pickle.PickleError("Mock error")
        ):
            result = lmv3_cache.set("test_key", lambda: None)  # type: ignore
            assert result is False

    def test_pickle_error_during_deserialization(
        self,
        lmv3_cache: LMv3Cache,
    ) -> None:
        """Test handling of pickle errors during deserialization."""
        with patch.object(
            lmv3_cache.cache,
            "get",
            side_effect=pickle.PickleError("Corrupt data"),
        ):
            result = lmv3_cache.get("test_key")
            assert result is None

    def test_attribute_error_during_deserialization(
        self,
        lmv3_cache: LMv3Cache,
    ) -> None:
        """Test handling of AttributeError (e.g., class definition changed)."""
        with patch.object(
            lmv3_cache.cache,
            "get",
            side_effect=AttributeError("Class not found"),
        ):
            result = lmv3_cache.get("test_key")
            assert result is None

    def test_cache_length(
        self,
        lmv3_cache: LMv3Cache,
        sample_lmv3_response: LMv3Response,
    ) -> None:
        """Test that __len__ returns correct cache size."""
        initial_len = len(lmv3_cache)

        lmv3_cache.set("key1", sample_lmv3_response)
        assert len(lmv3_cache) == initial_len + 1

        lmv3_cache.set("key2", sample_lmv3_response)
        assert len(lmv3_cache) == initial_len + 2


class TestLMv3CacheRealWorldScenarios:
    """Test real-world usage scenarios."""

    def test_cache_multiple_pages_same_document(
        self,
        lmv3_cache: LMv3Cache,
    ) -> None:
        """Test caching multiple pages from the same document."""
        for page_num in range(1, 6):
            entries = [
                SchematismEntry(
                    parish=f"Church on page {page_num}",
                    deanery="Main Deanery",
                    dedication="Various",
                    building_material="Mixed",
                )
            ]
            page = SchematismPage(page_number=str(page_num), entries=entries)
            response = LMv3Response(output=page)

            lmv3_cache.set(f"doc_page_{page_num}", response)

        # Verify all pages cached
        for page_num in range(1, 6):
            cached = lmv3_cache.get(f"doc_page_{page_num}")
            assert cached is not None
            assert cached.output.page_number == str(page_num)

    def test_cache_with_complex_entry_data(
        self,
        lmv3_cache: LMv3Cache,
    ) -> None:
        """Test caching with complex, real-world entry data."""
        entries = [
            SchematismEntry(
                parish="Parafia św. Wojciecha",
                deanery="Dekanat Krakowski",
                dedication="Św. Wojciech, Biskup i Męczennik",
                building_material="Kamień i cegła",
            ),
            SchematismEntry(
                parish="Cathedral of St. John",
                deanery="Urban Deanery Chapter III",
                dedication="John the Baptist and John the Evangelist",
                building_material="Sandstone with Gothic elements",
            ),
        ]
        page = SchematismPage(page_number="XLII", entries=entries)
        response = LMv3Response(output=page)

        key = "complex_data_test"
        lmv3_cache.set(key, response)
        cached = lmv3_cache.get(key)

        assert cached is not None
        assert cached.output.entries[0].parish == "Parafia św. Wojciecha"
        assert (
            cached.output.entries[1].dedication
            == "John the Baptist and John the Evangelist"
        )
