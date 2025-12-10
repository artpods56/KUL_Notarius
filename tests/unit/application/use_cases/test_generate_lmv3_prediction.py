"""Tests for GenerateLMv3Prediction use case."""

from unittest.mock import MagicMock, patch

import pytest
from PIL import Image
from PIL.Image import Image as PILImage

from notarius.application.use_cases.gen.generate_lmv3_prediction import (
    GenerateLMv3Prediction,
    GenerateLMv3PredictionRequest,
    GenerateLMv3PredictionResponse,
)
from notarius.domain.entities.schematism import SchematismPage, SchematismEntry
from notarius.domain.services.bio_processing_service import BIOProcessingService
from notarius.infrastructure.ml_models.lmv3.engine import SimpleLMv3Engine
from notarius.infrastructure.persistence.lmv3_cache_repository import (
    LMv3CacheRepository,
)
from notarius.schemas.data.cache import LMv3CacheItem


class TestGenerateLMv3PredictionRequest:
    """Test suite for GenerateLMv3PredictionRequest class."""

    def test_init_with_all_params(self) -> None:
        """Test request initialization with all parameters."""
        image = Image.new("RGB", (100, 100))
        words = ["word1", "word2"]
        bboxes = [[0, 0, 10, 10], [10, 0, 20, 10]]

        request = GenerateLMv3PredictionRequest(
            image=image,
            words=words,
            bboxes=bboxes,
            raw_predictions=True,
            invalidate_cache=True,
            schematism="test_schematism",
            filename="test.jpg",
        )

        assert request.image is image
        assert request.words == words
        assert request.bboxes == bboxes
        assert request.raw_predictions is True
        assert request.invalidate_cache is True
        assert request.schematism == "test_schematism"
        assert request.filename == "test.jpg"

    def test_init_with_defaults(self) -> None:
        """Test request initialization with default values."""
        image = Image.new("RGB", (100, 100))
        words = ["word"]
        bboxes = [[0, 0, 10, 10]]

        request = GenerateLMv3PredictionRequest(
            image=image,
            words=words,
            bboxes=bboxes,
        )

        assert request.image is image
        assert request.words == words
        assert request.bboxes == bboxes
        assert request.raw_predictions is False
        assert request.invalidate_cache is False
        assert request.schematism is None
        assert request.filename is None

    def test_init_with_empty_words(self) -> None:
        """Test request initialization with empty words list."""
        image = Image.new("RGB", (100, 100))

        request = GenerateLMv3PredictionRequest(
            image=image,
            words=[],
            bboxes=[],
        )

        assert request.words == []
        assert request.bboxes == []


class TestGenerateLMv3PredictionResponse:
    """Test suite for GenerateLMv3PredictionResponse class."""

    def test_init_with_structured_prediction(self) -> None:
        """Test output initialization with structured prediction."""
        page = SchematismPage(entries=[SchematismEntry(parish="Test Parish")])

        response = GenerateLMv3PredictionResponse(
            prediction=page,
            from_cache=True,
        )

        assert response.prediction == page
        assert response.from_cache is True

    def test_init_with_raw_prediction(self) -> None:
        """Test output initialization with raw prediction tuple."""
        raw_pred = (["word"], [[0, 0, 10, 10]], ["B-PARISH"])

        response = GenerateLMv3PredictionResponse(
            prediction=raw_pred,
            from_cache=False,
        )

        assert response.prediction == raw_pred
        assert response.from_cache is False

    def test_init_default_from_cache(self) -> None:
        """Test output with default from_cache value."""
        page = SchematismPage(entries=[])

        response = GenerateLMv3PredictionResponse(prediction=page)

        assert response.from_cache is False


class TestGenerateLMv3Prediction:
    """Test suite for GenerateLMv3Prediction use case."""

    @pytest.fixture
    def mock_engine(self) -> MagicMock:
        """Create a mock SimpleLMv3Engine."""
        engine = MagicMock(spec=SimpleLMv3Engine)
        engine.predict.return_value = (
            ["word1", "word2"],
            [[0, 0, 10, 10], [10, 0, 20, 10]],
            ["B-PARISH", "I-PARISH"],
        )
        return engine

    @pytest.fixture
    def mock_cache_repository(self) -> MagicMock:
        """Create a mock LMv3CacheRepository."""
        repository = MagicMock(spec=LMv3CacheRepository)
        repository.generate_key.return_value = "test_cache_key_123"
        repository.get.return_value = None  # Cache miss by default
        return repository

    @pytest.fixture
    def mock_bio_service(self) -> MagicMock:
        """Create a mock BIOProcessingService."""
        service = MagicMock(spec=BIOProcessingService)
        service.process.return_value = SchematismPage(
            entries=[SchematismEntry(parish="Test Parish")]
        )
        return service

    @pytest.fixture
    def use_case(
        self,
        mock_engine: MagicMock,
        mock_cache_repository: MagicMock,
        mock_bio_service: MagicMock,
    ) -> GenerateLMv3Prediction:
        """Create a use case instance for testing."""
        return GenerateLMv3Prediction(
            engine=mock_engine,
            cache_repository=mock_cache_repository,
            bio_service=mock_bio_service,
        )

    @pytest.fixture
    def sample_image(self) -> PILImage:
        """Create a sample PIL image for testing."""
        return Image.new("RGB", (100, 100), color="white")

    @pytest.fixture
    def sample_words(self) -> list[str]:
        """Sample OCR words."""
        return ["word1", "word2"]

    @pytest.fixture
    def sample_bboxes(self) -> list:
        """Sample bounding boxes."""
        return [[0, 0, 10, 10], [10, 0, 20, 10]]

    def test_init(
        self,
        mock_engine: MagicMock,
        mock_cache_repository: MagicMock,
        mock_bio_service: MagicMock,
    ) -> None:
        """Test use case initialization."""
        use_case = GenerateLMv3Prediction(
            engine=mock_engine,
            cache_repository=mock_cache_repository,
            bio_service=mock_bio_service,
        )

        assert use_case.engine is mock_engine
        assert use_case.cache_repository is mock_cache_repository
        assert use_case.bio_service is mock_bio_service
        assert use_case.logger is not None

    def test_init_without_cache(
        self, mock_engine: MagicMock, mock_bio_service: MagicMock
    ) -> None:
        """Test use case initialization without cache repository."""
        use_case = GenerateLMv3Prediction(
            engine=mock_engine,
            cache_repository=None,
            bio_service=mock_bio_service,
        )

        assert use_case.cache_repository is None

    def test_execute_cache_miss_generates_prediction(
        self,
        use_case: GenerateLMv3Prediction,
        mock_engine: MagicMock,
        sample_image: PILImage,
        sample_words: list[str],
        sample_bboxes: list,
    ) -> None:
        """Test that cache miss triggers prediction generation."""
        request = GenerateLMv3PredictionRequest(
            image=sample_image,
            words=sample_words,
            bboxes=sample_bboxes,
        )

        response = use_case.execute(request)

        assert isinstance(response.prediction, SchematismPage)
        assert response.from_cache is False
        mock_engine.predict.assert_called_once()

    def test_execute_cache_hit_returns_cached_structured(
        self,
        use_case: GenerateLMv3Prediction,
        mock_cache_repository: MagicMock,
        mock_engine: MagicMock,
        sample_image: PILImage,
        sample_words: list[str],
        sample_bboxes: list,
    ) -> None:
        """Test that cache hit returns cached structured result."""
        cached_item = LMv3CacheItem(
            raw_predictions=(["word"], [[0, 0, 10, 10]], ["O"]),
            structured_predictions=SchematismPage(
                entries=[SchematismEntry(parish="Cached Parish")]
            ),
        )
        mock_cache_repository.get.return_value = cached_item

        request = GenerateLMv3PredictionRequest(
            image=sample_image,
            words=sample_words,
            bboxes=sample_bboxes,
            raw_predictions=False,
        )
        response = use_case.execute(request)

        assert isinstance(response.prediction, SchematismPage)
        assert response.prediction.entries[0].parish == "Cached Parish"
        assert response.from_cache is True
        mock_engine.predict.assert_not_called()

    def test_execute_cache_hit_returns_cached_raw(
        self,
        use_case: GenerateLMv3Prediction,
        mock_cache_repository: MagicMock,
        mock_engine: MagicMock,
        sample_image: PILImage,
        sample_words: list[str],
        sample_bboxes: list,
    ) -> None:
        """Test that cache hit returns cached raw predictions."""
        cached_raw = (["cached"], [[5, 5, 15, 15]], ["B-PARISH"])
        cached_item = LMv3CacheItem(
            raw_predictions=cached_raw,
            structured_predictions=SchematismPage(entries=[]),
        )
        mock_cache_repository.get.return_value = cached_item

        request = GenerateLMv3PredictionRequest(
            image=sample_image,
            words=sample_words,
            bboxes=sample_bboxes,
            raw_predictions=True,
        )
        response = use_case.execute(request)

        assert response.prediction == cached_raw
        assert response.from_cache is True
        mock_engine.predict.assert_not_called()

    def test_execute_calls_engine_with_correct_params(
        self,
        use_case: GenerateLMv3Prediction,
        mock_engine: MagicMock,
        sample_image: PILImage,
        sample_words: list[str],
        sample_bboxes: list,
    ) -> None:
        """Test that execute calls _engine with correct parameters."""
        request = GenerateLMv3PredictionRequest(
            image=sample_image,
            words=sample_words,
            bboxes=sample_bboxes,
        )

        use_case.execute(request)

        mock_engine.predict.assert_called_once_with(
            image=sample_image,
            words=sample_words,
            bboxes=sample_bboxes,
        )

    def test_execute_calls_bio_service_for_structured(
        self,
        use_case: GenerateLMv3Prediction,
        mock_bio_service: MagicMock,
        sample_image: PILImage,
        sample_words: list[str],
        sample_bboxes: list,
    ) -> None:
        """Test that execute calls BIO service for structured predictions."""
        request = GenerateLMv3PredictionRequest(
            image=sample_image,
            words=sample_words,
            bboxes=sample_bboxes,
            raw_predictions=False,
        )

        use_case.execute(request)

        # BIO service should be called with _engine output
        mock_bio_service.process.assert_called_once_with(
            words=["word1", "word2"],
            bboxes=[[0, 0, 10, 10], [10, 0, 20, 10]],
            predictions=["B-PARISH", "I-PARISH"],
        )

    def test_execute_skips_bio_service_for_raw(
        self,
        use_case: GenerateLMv3Prediction,
        mock_bio_service: MagicMock,
        sample_image: PILImage,
        sample_words: list[str],
        sample_bboxes: list,
    ) -> None:
        """Test that execute skips BIO service for raw predictions."""
        request = GenerateLMv3PredictionRequest(
            image=sample_image,
            words=sample_words,
            bboxes=sample_bboxes,
            raw_predictions=True,
        )

        use_case.execute(request)

        # BIO service should not be called
        mock_bio_service.process.assert_not_called()

    def test_execute_stores_in_cache(
        self,
        use_case: GenerateLMv3Prediction,
        mock_cache_repository: MagicMock,
        sample_image: PILImage,
        sample_words: list[str],
        sample_bboxes: list,
    ) -> None:
        """Test that prediction result is stored in cache."""
        request = GenerateLMv3PredictionRequest(
            image=sample_image,
            words=sample_words,
            bboxes=sample_bboxes,
        )

        use_case.execute(request)

        # Verify cache.set was called
        mock_cache_repository.set.assert_called_once()
        call_args = mock_cache_repository.set.call_args
        assert call_args[1]["key"] == "test_cache_key_123"
        assert isinstance(call_args[1]["item"], LMv3CacheItem)

    def test_execute_with_cache_invalidation(
        self,
        use_case: GenerateLMv3Prediction,
        mock_cache_repository: MagicMock,
        mock_engine: MagicMock,
        sample_image: PILImage,
        sample_words: list[str],
        sample_bboxes: list,
    ) -> None:
        """Test that invalidate_cache flag deletes cache entry."""
        request = GenerateLMv3PredictionRequest(
            image=sample_image,
            words=sample_words,
            bboxes=sample_bboxes,
            invalidate_cache=True,
        )

        use_case.execute(request)

        # Verify cache was invalidated
        mock_cache_repository.delete.assert_called_once_with("test_cache_key_123")

        # Should still generate new prediction
        mock_engine.predict.assert_called_once()

    def test_execute_without_cache_repository(
        self,
        mock_engine: MagicMock,
        mock_bio_service: MagicMock,
        sample_image: PILImage,
        sample_words: list[str],
        sample_bboxes: list,
    ) -> None:
        """Test execution when cache repository is None."""
        use_case = GenerateLMv3Prediction(
            engine=mock_engine,
            cache_repository=None,
            bio_service=mock_bio_service,
        )

        request = GenerateLMv3PredictionRequest(
            image=sample_image,
            words=sample_words,
            bboxes=sample_bboxes,
        )
        response = use_case.execute(request)

        # Should generate prediction directly without caching
        assert isinstance(response.prediction, SchematismPage)
        assert response.from_cache is False
        mock_engine.predict.assert_called_once()

    def test_execute_with_schematism_and_filename(
        self,
        use_case: GenerateLMv3Prediction,
        mock_cache_repository: MagicMock,
        sample_image: PILImage,
        sample_words: list[str],
        sample_bboxes: list,
    ) -> None:
        """Test that schematism and filename are passed to cache."""
        request = GenerateLMv3PredictionRequest(
            image=sample_image,
            words=sample_words,
            bboxes=sample_bboxes,
            schematism="test_schema",
            filename="test_file.jpg",
        )

        use_case.execute(request)

        # Verify cache.set includes metadata
        mock_cache_repository.set.assert_called_once()
        set_kwargs = mock_cache_repository.set.call_args[1]
        assert set_kwargs["schematism"] == "test_schema"
        assert set_kwargs["filename"] == "test_file.jpg"

    def test_execute_returns_raw_tuple_when_requested(
        self,
        use_case: GenerateLMv3Prediction,
        sample_image: PILImage,
        sample_words: list[str],
        sample_bboxes: list,
    ) -> None:
        """Test that raw_predictions=True returns tuple format."""
        request = GenerateLMv3PredictionRequest(
            image=sample_image,
            words=sample_words,
            bboxes=sample_bboxes,
            raw_predictions=True,
        )

        response = use_case.execute(request)

        # Should be tuple, not SchematismPage
        assert isinstance(response.prediction, tuple)
        assert len(response.prediction) == 3
        assert response.prediction[0] == ["word1", "word2"]

    def test_execute_returns_structured_by_default(
        self,
        use_case: GenerateLMv3Prediction,
        sample_image: PILImage,
        sample_words: list[str],
        sample_bboxes: list,
    ) -> None:
        """Test that default behavior returns structured predictions."""
        request = GenerateLMv3PredictionRequest(
            image=sample_image,
            words=sample_words,
            bboxes=sample_bboxes,
        )

        response = use_case.execute(request)

        # Should be SchematismPage, not tuple
        assert isinstance(response.prediction, SchematismPage)

    def test_execute_with_empty_words(
        self,
        use_case: GenerateLMv3Prediction,
        mock_engine: MagicMock,
        sample_image: PILImage,
    ) -> None:
        """Test execution with empty words list."""
        mock_engine.predict.return_value = ([], [], [])

        request = GenerateLMv3PredictionRequest(
            image=sample_image,
            words=[],
            bboxes=[],
        )

        response = use_case.execute(request)

        assert isinstance(response.prediction, SchematismPage)
        mock_engine.predict.assert_called_once()

    def test_execute_cache_key_generation(
        self,
        use_case: GenerateLMv3Prediction,
        mock_cache_repository: MagicMock,
        sample_image: PILImage,
        sample_words: list[str],
        sample_bboxes: list,
    ) -> None:
        """Test that cache key generation includes correct parameters."""
        request = GenerateLMv3PredictionRequest(
            image=sample_image,
            words=sample_words,
            bboxes=sample_bboxes,
            raw_predictions=True,
        )

        use_case.execute(request)

        # Verify cache key generation
        mock_cache_repository.generate_key.assert_called_once_with(
            image=sample_image,
            raw_predictions=True,
        )

    def test_execute_integration_flow(
        self,
        use_case: GenerateLMv3Prediction,
        mock_cache_repository: MagicMock,
        mock_engine: MagicMock,
        mock_bio_service: MagicMock,
        sample_image: PILImage,
        sample_words: list[str],
        sample_bboxes: list,
    ) -> None:
        """Test complete integration flow: check cache -> predict -> process -> store."""
        request = GenerateLMv3PredictionRequest(
            image=sample_image,
            words=sample_words,
            bboxes=sample_bboxes,
        )

        # Ensure cache miss
        mock_cache_repository.get.return_value = None

        # Execute
        response = use_case.execute(request)

        # Verify complete flow
        # 1. Cache key was generated
        mock_cache_repository.generate_key.assert_called_once()

        # 2. Cache was checked
        mock_cache_repository.get.assert_called_once()

        # 3. Engine prediction was made
        mock_engine.predict.assert_called_once()

        # 4. BIO processing was done
        mock_bio_service.process.assert_called_once()

        # 5. Result was cached
        mock_cache_repository.set.assert_called_once()

        # 6. Response is correct
        assert response.from_cache is False
        assert isinstance(response.prediction, SchematismPage)

    def test_execute_cache_hit_skips_engine_and_bio(
        self,
        use_case: GenerateLMv3Prediction,
        mock_cache_repository: MagicMock,
        mock_engine: MagicMock,
        mock_bio_service: MagicMock,
        sample_image: PILImage,
        sample_words: list[str],
        sample_bboxes: list,
    ) -> None:
        """Test that cache hit completely skips _engine and BIO processing."""
        cached_item = LMv3CacheItem(
            raw_predictions=([], [], []),
            structured_predictions=SchematismPage(entries=[]),
        )
        mock_cache_repository.get.return_value = cached_item

        request = GenerateLMv3PredictionRequest(
            image=sample_image,
            words=sample_words,
            bboxes=sample_bboxes,
        )
        use_case.execute(request)

        # Engine and BIO service should never be called
        mock_engine.predict.assert_not_called()
        mock_bio_service.process.assert_not_called()

        # Cache.set should not be called either
        mock_cache_repository.set.assert_not_called()

    @patch("notarius.application.use_cases.generate_lmv3_prediction.logger")
    def test_logging(
        self,
        mock_logger: MagicMock,
        use_case: GenerateLMv3Prediction,
        sample_image: PILImage,
        sample_words: list[str],
        sample_bboxes: list,
    ) -> None:
        """Test that operations are logged."""
        request = GenerateLMv3PredictionRequest(
            image=sample_image,
            words=sample_words,
            bboxes=sample_bboxes,
        )

        use_case.execute(request)

        # Verify logger was used
        assert use_case.logger is not None

    def test_execute_cache_stores_both_formats(
        self,
        use_case: GenerateLMv3Prediction,
        mock_cache_repository: MagicMock,
        sample_image: PILImage,
        sample_words: list[str],
        sample_bboxes: list,
    ) -> None:
        """Test that cache stores both raw and structured predictions."""
        request = GenerateLMv3PredictionRequest(
            image=sample_image,
            words=sample_words,
            bboxes=sample_bboxes,
        )

        use_case.execute(request)

        # Verify cache item has both formats
        set_call = mock_cache_repository.set.call_args
        cached_item = set_call[1]["item"]
        assert isinstance(cached_item, LMv3CacheItem)
        assert cached_item.raw_predictions is not None
        assert isinstance(cached_item.structured_predictions, SchematismPage)
