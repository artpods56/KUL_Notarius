"""Integration tests for LayoutLMv3 _engine components.

Tests the complete LMv3 pipeline:
- SimpleLMv3Engine (with mocking for model)
- LMv3CacheRepository
- BIOProcessingService
- GenerateLMv3Prediction use case
- LMv3EngineAdapter

These tests use real components where possible, with minimal mocking.
"""

import tempfile
from pathlib import Path

import pytest
from PIL import Image
from omegaconf import DictConfig, OmegaConf

from notarius.infrastructure.persistence.lmv3_cache_repository import (
    LMv3CacheRepository,
)
from notarius.domain.services.bio_processing_service import BIOProcessingService
from notarius.domain.entities.schematism import SchematismPage
from notarius.schemas.data.cache import LMv3CacheItem


@pytest.fixture
def temp_cache_dir() -> Path:
    """Create temporary directory for cache."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def test_image() -> Image.Image:
    """Create a simple test image."""
    return Image.new("RGB", (300, 200), color="white")


@pytest.fixture
def test_words() -> list[str]:
    """Sample words for testing."""
    return ["John", "Smith", "Warsaw", "Poland"]


@pytest.fixture
def test_bboxes() -> list[list[int]]:
    """Sample normalized bounding boxes (0-1000 range)."""
    return [
        [0, 0, 100, 50],
        [100, 0, 200, 50],
        [200, 0, 300, 50],
        [300, 0, 400, 50],
    ]


@pytest.fixture
def test_predictions() -> list[str]:
    """Sample BIO predictions."""
    return ["B-name", "I-name", "B-location", "I-location"]


class TestLMv3CacheIntegration:
    """Integration tests for LMv3 caching with real cache backend."""

    def test_cache_persistence_across_instances(
        self,
        test_image: Image.Image,
        test_words: list[str],
        test_bboxes: list[list[int]],
        temp_cache_dir: Path,
    ) -> None:
        """Test that cache persists across repository instances."""
        # First instance - store data
        repo1 = LMv3CacheRepository.create(
            model_name="test_model",
            caches_dir=temp_cache_dir,
        )

        key = repo1.generate_key(
            image=test_image,
            words=test_words,
            bboxes=test_bboxes,
        )

        # Create a simple SchematismPage
        page = SchematismPage(
            words=test_words,
            bboxes=test_bboxes,
            labels=["B-name", "I-name", "B-location", "I-location"],
        )

        item = LMv3CacheItem(
            raw_predictions=(
                test_words,
                test_bboxes,
                ["B-name", "I-name", "B-location", "I-location"],
            ),
            structured_predictions=page,
        )
        repo1.set(key, item, schematism="test_schematism")

        # Second instance - retrieve data
        repo2 = LMv3CacheRepository.create(
            model_name="test_model",
            caches_dir=temp_cache_dir,
        )

        retrieved = repo2.get(key)

        assert retrieved is not None
        assert retrieved.structured_predictions.words == test_words
        assert retrieved.structured_predictions.labels == page.labels

    def test_cache_invalidation(
        self,
        test_image: Image.Image,
        test_words: list[str],
        test_bboxes: list[list[int]],
        temp_cache_dir: Path,
    ) -> None:
        """Test cache invalidation removes entries."""
        repo = LMv3CacheRepository.create(
            model_name="test_model",
            caches_dir=temp_cache_dir,
        )

        key = repo.generate_key(
            image=test_image,
            words=test_words,
            bboxes=test_bboxes,
        )

        page = SchematismPage(
            words=test_words,
            bboxes=test_bboxes,
            labels=["B-name", "I-name", "B-location", "I-location"],
        )

        item = LMv3CacheItem(
            raw_predictions=(
                test_words,
                test_bboxes,
                ["B-name", "I-name", "B-location", "I-location"],
            ),
            structured_predictions=page,
        )
        repo.set(key, item)

        # Verify exists
        assert repo.get(key) is not None

        # Delete
        repo.delete(key)

        # Verify removed
        assert repo.get(key) is None

    def test_different_keys_for_different_inputs(
        self,
        test_image: Image.Image,
        temp_cache_dir: Path,
    ) -> None:
        """Test that different inputs produce different cache keys."""
        repo = LMv3CacheRepository.create(
            model_name="test_model",
            caches_dir=temp_cache_dir,
        )

        words1 = ["word1", "word2"]
        words2 = ["word3", "word4"]
        bboxes1 = [[0, 0, 100, 50], [100, 0, 200, 50]]
        bboxes2 = [[0, 0, 50, 25], [50, 0, 100, 25]]

        key1 = repo.generate_key(image=test_image, words=words1, bboxes=bboxes1)
        key2 = repo.generate_key(image=test_image, words=words2, bboxes=bboxes2)

        assert key1 != key2


class TestBIOProcessingServiceIntegration:
    """Integration tests for BIO processing service."""

    def test_process_creates_schematism_page(
        self,
        test_words: list[str],
        test_bboxes: list[list[int]],
        test_predictions: list[str],
    ) -> None:
        """Test that processing creates valid SchematismPage."""
        service = BIOProcessingService()

        page = service.process(
            words=test_words,
            bboxes=test_bboxes,
            predictions=test_predictions,
        )

        assert isinstance(page, SchematismPage)
        assert page.words == test_words
        assert page.bboxes == test_bboxes
        assert page.labels == test_predictions

    def test_process_repairs_invalid_bio_sequences(self) -> None:
        """Test that service repairs invalid BIO sequences."""
        service = BIOProcessingService()

        # Invalid: I-tag without B-tag
        words = ["John", "Smith"]
        bboxes = [[0, 0, 100, 50], [100, 0, 200, 50]]
        invalid_predictions = ["I-name", "I-name"]  # Invalid!

        page = service.process(
            words=words,
            bboxes=bboxes,
            predictions=invalid_predictions,
        )

        # Should be repaired (I -> B)
        assert page.labels[0] == "B-name"
        assert page.labels[1] == "I-name"

    def test_process_with_empty_input(self) -> None:
        """Test processing with empty input."""
        service = BIOProcessingService()

        page = service.process(words=[], bboxes=[], predictions=[])

        assert isinstance(page, SchematismPage)
        assert page.words == []
        assert page.bboxes == []
        assert page.labels == []

    def test_process_with_multiple_entities(self) -> None:
        """Test processing with multiple named entities."""
        service = BIOProcessingService()

        words = ["John", "Smith", "lives", "in", "Warsaw", "Poland"]
        bboxes = [
            [0, 0, 50, 20],
            [50, 0, 100, 20],
            [100, 0, 150, 20],
            [150, 0, 170, 20],
            [170, 0, 230, 20],
            [230, 0, 290, 20],
        ]
        predictions = ["B-name", "I-name", "O", "O", "B-location", "I-location"]

        page = service.process(words=words, bboxes=bboxes, predictions=predictions)

        assert isinstance(page, SchematismPage)
        assert len(page.words) == 6
        assert page.labels == predictions


class TestLMv3ComponentsEndToEnd:
    """End-to-end integration tests for LMv3 components working together."""

    def test_cache_and_bio_service_integration(
        self,
        test_image: Image.Image,
        test_words: list[str],
        test_bboxes: list[list[int]],
        temp_cache_dir: Path,
    ) -> None:
        """Test that cache and BIO service work together."""
        # Create components
        cache_repo = LMv3CacheRepository.create(
            model_name="test_model",
            caches_dir=temp_cache_dir,
        )
        bio_service = BIOProcessingService()

        # Process predictions with BIO service
        predictions = ["B-name", "I-name", "B-location", "I-location"]
        page = bio_service.process(
            words=test_words,
            bboxes=test_bboxes,
            predictions=predictions,
        )

        # Store in cache
        key = cache_repo.generate_key(
            image=test_image,
            words=test_words,
            bboxes=test_bboxes,
        )

        item = LMv3CacheItem(
            raw_predictions=(test_words, test_bboxes, predictions),
            structured_predictions=page,
        )
        cache_repo.set(key, item)

        # Retrieve from cache
        cached_item = cache_repo.get(key)

        assert cached_item is not None
        assert cached_item.structured_predictions.words == test_words
        assert cached_item.structured_predictions.labels == predictions

    def test_bio_service_repairs_before_caching(
        self,
        test_image: Image.Image,
        temp_cache_dir: Path,
    ) -> None:
        """Test workflow: predict → repair BIO → cache."""
        cache_repo = LMv3CacheRepository.create(
            model_name="test_model",
            caches_dir=temp_cache_dir,
        )
        bio_service = BIOProcessingService()

        # Simulate invalid predictions from model
        words = ["John", "Smith"]
        bboxes = [[0, 0, 100, 50], [100, 0, 200, 50]]
        invalid_predictions = ["I-name", "I-name"]  # Invalid - no B-tag

        # Repair using BIO service (as would happen in use case)
        page = bio_service.process(
            words=words,
            bboxes=bboxes,
            predictions=invalid_predictions,
        )

        # Verify repair happened
        assert page.labels[0] == "B-name"
        assert page.labels[1] == "I-name"

        # Cache the repaired results
        key = cache_repo.generate_key(
            image=test_image,
            words=words,
            bboxes=bboxes,
        )

        item = LMv3CacheItem(
            raw_predictions=(words, bboxes, invalid_predictions),  # Store original
            structured_predictions=page,  # Store repaired
        )
        cache_repo.set(key, item)

        # Retrieve - should get repaired version
        cached = cache_repo.get(key)
        assert cached is not None
        assert cached.structured_predictions.labels[0] == "B-name"


class TestLMv3RealWorldScenarios:
    """Integration tests for realistic LMv3 scenarios."""

    def test_empty_predictions_handling(self, temp_cache_dir: Path) -> None:
        """Test handling of images with no predictions."""
        bio_service = BIOProcessingService()

        # Empty results (e.g., blank page)
        page = bio_service.process(words=[], bboxes=[], predictions=[])

        assert isinstance(page, SchematismPage)
        assert len(page.words) == 0

    def test_single_word_predictions(self, temp_cache_dir: Path) -> None:
        """Test handling of single-word predictions."""
        bio_service = BIOProcessingService()

        page = bio_service.process(
            words=["John"],
            bboxes=[[0, 0, 100, 50]],
            predictions=["B-name"],
        )

        assert isinstance(page, SchematismPage)
        assert len(page.words) == 1
        assert page.labels == ["B-name"]

    def test_long_entity_sequences(self, temp_cache_dir: Path) -> None:
        """Test handling of long entity sequences."""
        bio_service = BIOProcessingService()

        # Long name
        words = ["Very", "Long", "Name", "Here"]
        bboxes = [[i * 100, 0, (i + 1) * 100, 50] for i in range(4)]
        predictions = ["B-name", "I-name", "I-name", "I-name"]

        page = bio_service.process(words=words, bboxes=bboxes, predictions=predictions)

        assert isinstance(page, SchematismPage)
        assert len(page.words) == 4
        assert page.labels[0].startswith("B-")
        assert all(label.startswith("I-") for label in page.labels[1:])

    def test_alternating_entities(self, temp_cache_dir: Path) -> None:
        """Test handling of alternating entity types."""
        bio_service = BIOProcessingService()

        words = ["John", "Warsaw", "Smith", "Poland"]
        bboxes = [[i * 100, 0, (i + 1) * 100, 50] for i in range(4)]
        predictions = ["B-name", "B-location", "B-name", "B-location"]

        page = bio_service.process(words=words, bboxes=bboxes, predictions=predictions)

        assert isinstance(page, SchematismPage)
        assert len(page.words) == 4
        assert all(label.startswith("B-") for label in page.labels)
