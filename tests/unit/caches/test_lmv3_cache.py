from collections.abc import Generator
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock, patch
import pytest
from PIL import Image

from notarius.infrastructure.cache.lmv3_adapter import LMv3Cache
from notarius.infrastructure.cache.utils import get_image_hash
from typing import Any, cast
from PIL.Image import Image as PILImage


class TestLMv3Cache:
    """Test suite for LMv3Cache class."""

    @pytest.fixture
    def lmv3_cache(self, tmp_path: Path) -> LMv3Cache:
        """Create an LMv3Cache instance for testing."""
        cache = LMv3Cache(checkpoint="microsoft/layoutlmv3-base", caches_dir=tmp_path)
        return cache

    @pytest.fixture
    def sample_image(self) -> PILImage:
        """Create a sample PIL image for testing."""
        img = Image.new("RGB", (100, 100), color="green")
        return img

    @pytest.fixture
    def sample_predictions(self) -> dict[str, Any]:
        """Create sample structured predictions."""
        return {
            "entities": [
                {"text": "John Doe", "label": "PER", "score": 0.95},
                {"text": "New York", "label": "LOC", "score": 0.88},
            ]
        }

    def test_init_with_checkpoint(self, tmp_path: Path) -> None:
        """Test initialization with checkpoint name."""
        cache = LMv3Cache(checkpoint="microsoft/layoutlmv3-base", caches_dir=tmp_path)
        assert cache.checkpoint == "microsoft/layoutlmv3-base"
        assert cache._cache_loaded

    def test_init_with_custom_checkpoint(self, tmp_path: Path) -> None:
        """Test initialization with custom checkpoint."""
        cache = LMv3Cache(checkpoint="custom/layoutlmv3-large", caches_dir=tmp_path)
        assert cache.checkpoint == "custom/layoutlmv3-large"
        assert cache._cache_loaded

    def test_cache_directory_created(self, tmp_path: Path) -> None:
        """Test that cache directory is created on initialization."""
        cache = LMv3Cache(checkpoint="microsoft/layoutlmv3-base", caches_dir=tmp_path)
        expected_dir = tmp_path / "LMv3Cache" / "microsoft/layoutlmv3-base"
        assert expected_dir.exists()
        assert expected_dir.is_dir()

    def test_normalize_kwargs_with_all_params(
        self, lmv3_cache: LMv3Cache, sample_predictions: dict[str, Any]
    ) -> None:
        """Test normalize_kwargs with all parameters."""
        result = lmv3_cache.normalize_kwargs(
            image_hash="img123",
            structured_predictions=sample_predictions,
            extra_param="ignored",
        )
        assert result == {
            "image_hash": "img123",
            "structured_predictions": sample_predictions,
        }

    def test_normalize_kwargs_with_image_hash_only(self, lmv3_cache: LMv3Cache) -> None:
        """Test normalize_kwargs with only image hash."""
        result = lmv3_cache.normalize_kwargs(image_hash="img123")
        assert result == {"image_hash": "img123", "structured_predictions": None}

    def test_normalize_kwargs_missing_params(self, lmv3_cache: LMv3Cache) -> None:
        """Test normalize_kwargs with missing parameters."""
        result = lmv3_cache.normalize_kwargs()
        assert result == {"image_hash": None, "structured_predictions": None}

    def test_set_and_get_cache_with_image(
        self, lmv3_cache: LMv3Cache, sample_image: PILImage
    ) -> None:
        """Test setting and getting cache entries with image hash."""
        image_hash = get_image_hash(sample_image)
        cache_key = lmv3_cache.generate_hash(image_hash=image_hash)

        test_value = {
            "predictions": [
                {"label": "PARISH", "box": [100, 200, 300, 400]},
                {"label": "PRIEST", "box": [150, 250, 350, 450]},
            ],
            "confidence": 0.92,
        }

        lmv3_cache.set(cache_key, test_value)
        retrieved = lmv3_cache.get(cache_key)

        assert retrieved == test_value

    def test_set_and_get_cache_with_predictions(
        self,
        lmv3_cache: LMv3Cache,
        sample_image: PILImage,
        sample_predictions: dict[str, Any],
    ) -> None:
        """Test setting and getting cache entries with structured predictions."""
        image_hash = get_image_hash(sample_image)
        cache_key = lmv3_cache.generate_hash(
            image_hash=image_hash, structured_predictions=sample_predictions
        )

        test_value = {"result": "processed", "entities": sample_predictions["entities"]}

        lmv3_cache.set(cache_key, test_value)
        retrieved = lmv3_cache.get(cache_key)

        assert retrieved == test_value

    def test_set_cache_with_tags(
        self, lmv3_cache: LMv3Cache, sample_image: PILImage
    ) -> None:
        """Test setting cache with schematism and filename tags."""
        image_hash = get_image_hash(sample_image)
        cache_key = lmv3_cache.generate_hash(image_hash=image_hash)

        test_value = {"predictions": "Tagged LMv3 predictions"}

        lmv3_cache.set(
            cache_key,
            test_value,
            schematism="schematism_2024",
            filename="document_001.pdf",
        )

        retrieved = lmv3_cache.get(cache_key)
        assert retrieved == test_value

    def test_set_cache_with_null_tags(
        self, lmv3_cache: LMv3Cache, sample_image: PILImage
    ) -> None:
        """Test setting cache with None tags converts to 'null'."""
        image_hash = get_image_hash(sample_image)
        cache_key = lmv3_cache.generate_hash(image_hash=image_hash)

        test_value = {"predictions": "Null tagged entry"}

        lmv3_cache.set(cache_key, test_value, schematism=None, filename=None)
        retrieved = lmv3_cache.get(cache_key)
        assert retrieved == test_value

    def test_delete_cache_entry(
        self, lmv3_cache: LMv3Cache, sample_image: PILImage
    ) -> None:
        """Test deleting cache entries."""
        image_hash = get_image_hash(sample_image)
        cache_key = lmv3_cache.generate_hash(image_hash=image_hash)

        test_value = {"predictions": "To be deleted"}

        lmv3_cache.set(cache_key, test_value)
        assert lmv3_cache.get(cache_key) == test_value

        lmv3_cache.delete(cache_key)
        assert lmv3_cache.get(cache_key) is None

    def test_cache_length(self, lmv3_cache: LMv3Cache, sample_image: PILImage) -> None:
        """Test __len__ returns correct cache size."""
        initial_length = len(lmv3_cache)

        image_hash = get_image_hash(sample_image)
        cache_key = lmv3_cache.generate_hash(image_hash=image_hash)

        lmv3_cache.set(cache_key, {"predictions": "Entry 1"})
        assert len(lmv3_cache) == initial_length + 1

    def test_cache_miss_returns_none(self, lmv3_cache: LMv3Cache) -> None:
        """Test that cache miss returns None."""
        nonexistent_key = "nonexistent_hash_789"
        result = lmv3_cache.get(nonexistent_key)
        assert result is None

    def test_multiple_checkpoints_separate_caches(self, tmp_path: Path) -> None:
        """Test that different checkpoints create separate cache directories."""
        cache_base = LMv3Cache(
            checkpoint="microsoft/layoutlmv3-base", caches_dir=tmp_path
        )
        cache_large = LMv3Cache(
            checkpoint="microsoft/layoutlmv3-large", caches_dir=tmp_path
        )

        base_dir = tmp_path / "LMv3Cache" / "microsoft/layoutlmv3-base"
        large_dir = tmp_path / "LMv3Cache" / "microsoft/layoutlmv3-large"

        assert base_dir.exists()
        assert large_dir.exists()
        assert base_dir != large_dir

    def test_cache_persists_across_instances(
        self, tmp_path: Path, sample_image: PILImage
    ) -> None:
        """Test that cache persists when creating new instances."""
        # Create first instance and add entry
        cache1 = LMv3Cache(checkpoint="microsoft/layoutlmv3-base", caches_dir=tmp_path)
        image_hash = get_image_hash(sample_image)
        cache_key = cache1.generate_hash(image_hash=image_hash)
        test_value = {"predictions": "Persistent LMv3 data"}
        cache1.set(cache_key, test_value)

        # Create second instance and retrieve entry
        cache2 = LMv3Cache(checkpoint="microsoft/layoutlmv3-base", caches_dir=tmp_path)
        retrieved = cache2.get(cache_key)

        assert retrieved == test_value

    def test_hash_generation_deterministic(
        self, lmv3_cache: LMv3Cache, sample_predictions: dict[str, Any]
    ) -> None:
        """Test that hash generation is deterministic."""
        hash1 = lmv3_cache.generate_hash(
            image_hash="test_img", structured_predictions=sample_predictions
        )
        hash2 = lmv3_cache.generate_hash(
            image_hash="test_img", structured_predictions=sample_predictions
        )
        assert hash1 == hash2

    def test_hash_generation_different_for_different_inputs(
        self, lmv3_cache: LMv3Cache
    ) -> None:
        """Test that different inputs generate different hashes."""
        hash1 = lmv3_cache.generate_hash(image_hash="img1")
        hash2 = lmv3_cache.generate_hash(image_hash="img2")
        assert hash1 != hash2

    def test_hash_generation_with_complex_predictions(
        self, lmv3_cache: LMv3Cache
    ) -> None:
        """Test hash generation with complex structured predictions."""
        complex_predictions = {
            "entities": [
                {
                    "text": "Sample Entity",
                    "label": "LABEL",
                    "box": [10, 20, 30, 40],
                    "confidence": 0.95,
                }
            ],
            "metadata": {"model": "layoutlmv3", "version": "1.0"},
        }

        hash1 = lmv3_cache.generate_hash(
            image_hash="test", structured_predictions=complex_predictions
        )
        hash2 = lmv3_cache.generate_hash(
            image_hash="test", structured_predictions=complex_predictions
        )

        assert hash1 == hash2
        assert len(hash1) == 64  # SHA-256

    def test_cache_overwrite_existing_entry(
        self, lmv3_cache: LMv3Cache, sample_image: PILImage
    ) -> None:
        """Test overwriting an existing cache entry."""
        image_hash = get_image_hash(sample_image)
        cache_key = lmv3_cache.generate_hash(image_hash=image_hash)

        # Set initial value
        initial_value = {"predictions": "Initial"}
        lmv3_cache.set(cache_key, initial_value)
        assert lmv3_cache.get(cache_key) == initial_value

        # Overwrite with new value
        new_value = {"predictions": "Updated"}
        lmv3_cache.set(cache_key, new_value)
        assert lmv3_cache.get(cache_key) == new_value

    def test_cache_with_different_prediction_structures(
        self, lmv3_cache: LMv3Cache, sample_image: PILImage
    ) -> None:
        """Test caching with different prediction structures."""
        image_hash = get_image_hash(sample_image)

        # Prediction structure 1
        pred1 = {"type": "tokens", "data": ["token1", "token2"]}
        key1 = lmv3_cache.generate_hash(
            image_hash=image_hash, structured_predictions=pred1
        )
        lmv3_cache.set(key1, {"result": "tokens_result"})

        # Prediction structure 2
        pred2 = {"type": "boxes", "data": [[0, 0, 10, 10]]}
        key2 = lmv3_cache.generate_hash(
            image_hash=image_hash, structured_predictions=pred2
        )
        lmv3_cache.set(key2, {"result": "boxes_result"})

        # Keys should be different
        assert key1 != key2

        # Both should be retrievable
        assert lmv3_cache.get(key1) == {"result": "tokens_result"}
        assert lmv3_cache.get(key2) == {"result": "boxes_result"}

    def test_cache_with_none_predictions(
        self, lmv3_cache: LMv3Cache, sample_image: PILImage
    ) -> None:
        """Test caching when predictions are None."""
        image_hash = get_image_hash(sample_image)
        cache_key = lmv3_cache.generate_hash(
            image_hash=image_hash, structured_predictions=None
        )

        test_value = {"status": "no_predictions"}
        lmv3_cache.set(cache_key, test_value)
        retrieved = lmv3_cache.get(cache_key)

        assert retrieved == test_value

    def test_cache_stores_complex_nested_data(
        self, lmv3_cache: LMv3Cache, sample_image: PILImage
    ) -> None:
        """Test that cache can store complex nested data structures."""
        image_hash = get_image_hash(sample_image)
        cache_key = lmv3_cache.generate_hash(image_hash=image_hash)

        complex_value = {
            "predictions": {
                "entities": [
                    {
                        "id": 1,
                        "text": "Entity 1",
                        "metadata": {"confidence": 0.95, "source": "model"},
                    }
                ],
                "relationships": [{"from": 1, "to": 2, "type": "references"}],
            },
            "statistics": {"total": 10, "processed": 8},
        }

        lmv3_cache.set(cache_key, complex_value)
        retrieved = cast(dict[str, Any], lmv3_cache.get(cache_key))

        assert retrieved == complex_value
        assert retrieved["predictions"]["entities"][0]["metadata"]["confidence"] == 0.95


class TestLMv3CacheIntegration:
    """Integration tests for LMv3Cache."""

    @pytest.fixture
    def tmp_path(self) -> Generator[Path, None, None]:
        """Create a temporary cache directory for testing."""
        with TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    def test_full_workflow(self, tmp_path: Path) -> None:
        """Test complete workflow: create, store, retrieve, update, delete."""
        with patch(
            "notarius.infrastructure.cache.lmv3_adapter.get_logger"
        ) as mock_logger:
            mock_logger.return_value.bind.return_value = MagicMock()

            # Create cache
            cache = LMv3Cache(checkpoint="test/checkpoint", caches_dir=tmp_path)

            # Store initial data
            img = Image.new("RGB", (50, 50), color="yellow")
            image_hash = get_image_hash(img)
            key = cache.generate_hash(image_hash=image_hash)

            initial_data = {"version": 1, "predictions": ["entity1"]}
            cache.set(key, initial_data, schematism="test_schema", filename="test.pdf")

            # Retrieve
            retrieved = cache.get(key)
            assert retrieved == initial_data

            # Update
            updated_data = {"version": 2, "predictions": ["entity1", "entity2"]}
            cache.set(key, updated_data)
            assert cache.get(key) == updated_data

            # Delete
            cache.delete(key)
            assert cache.get(key) is None
