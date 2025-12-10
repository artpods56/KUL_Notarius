"""Tests for GenerateOCRExtraction use case."""

from unittest.mock import MagicMock, patch

import pytest
from PIL import Image
from PIL.Image import Image as PILImage

from notarius.application.use_cases.generate_ocr_extraction import (
    GenerateOCRExtraction,
    GenerateOCRExtractionRequest,
    GenerateOCRExtractionResponse,
)
from notarius.infrastructure.ocr.engine import SimpleOCREngine

import notarius.application.use_cases.inference.add_ocr_to_dataset
from notarius.infrastructure.persistence.ocr_cache_repository import OCRCacheRepository
from notarius.schemas.data.cache import PyTesseractCacheItem


class TestGenerateOCRExtractionRequest:
    """Test suite for GenerateOCRExtractionRequest class."""

    def test_init_with_all_params(self) -> None:
        """Test request initialization with all parameters."""
        image = Image.new("RGB", (100, 100))

        request = GenerateOCRExtractionRequest(
            image=image,
            text_only=True,
            invalidate_cache=True,
            schematism="test_schema",
            filename="test.jpg",
        )

        assert request.image is image
        assert request.text_only is True
        assert request.invalidate_cache is True
        assert request.schematism == "test_schema"
        assert request.filename == "test.jpg"

    def test_init_with_defaults(self) -> None:
        """Test request initialization with default values."""
        image = Image.new("RGB", (100, 100))

        request = GenerateOCRExtractionRequest(image=image)

        assert request.image is image
        assert request.text_only is False
        assert request.invalidate_cache is False
        assert request.schematism is None
        assert request.filename is None


class TestGenerateOCRExtractionResponse:
    """Test suite for GenerateOCRExtractionResponse class."""

    def test_init_with_text_result(self) -> None:
        """Test output initialization with text result."""
        response = GenerateOCRExtractionResponse(
            result="Sample text",
            from_cache=True,
        )

        assert response.result == "Sample text"
        assert response.from_cache is True

    def test_init_with_tuple_result(self) -> None:
        """Test output initialization with tuple result."""
        words = ["word1", "word2"]
        bboxes = [[0, 0, 100, 50], [100, 0, 200, 50]]

        response = GenerateOCRExtractionResponse(
            result=(words, bboxes),
            from_cache=False,
        )

        assert response.result == (words, bboxes)
        assert response.from_cache is False

    def test_init_default_from_cache(self) -> None:
        """Test output with default from_cache value."""
        response = GenerateOCRExtractionResponse(result="text")

        assert response.from_cache is False


class TestGenerateOCRExtraction:
    """Test suite for GenerateOCRExtraction use case."""

    @pytest.fixture
    def mock_engine(self) -> MagicMock:
        """Create a mock SimpleOCREngine."""
        engine = MagicMock(spec=SimpleOCREngine)
        engine.extract_text.return_value = "Sample OCR text"
        engine.extract_words_and_boxes.return_value = (
            ["Sample", "OCR", "text"],
            [[0, 0, 100, 50], [100, 0, 200, 50], [200, 0, 300, 50]],
        )
        return engine

    @pytest.fixture
    def mock_cache_repository(self) -> MagicMock:
        """Create a mock OCRCacheRepository."""
        repository = MagicMock(spec=OCRCacheRepository)
        repository.generate_key.return_value = "test_cache_key_123"
        repository.get.return_value = None  # Cache miss by default
        return repository

    @pytest.fixture
    def use_case(
        self,
        mock_engine: MagicMock,
        mock_cache_repository: MagicMock,
    ) -> GenerateOCRExtraction:
        """Create a use case instance for testing."""
        return GenerateOCRExtraction(
            engine=mock_engine,
            cache_repository=mock_cache_repository,
        )

    @pytest.fixture
    def sample_image(self) -> PILImage:
        """Create a sample PIL image for testing."""
        return Image.new("RGB", (200, 100), color="white")

    def test_init(
        self,
        mock_engine: MagicMock,
        mock_cache_repository: MagicMock,
    ) -> None:
        """Test use case initialization."""
        use_case = GenerateOCRExtraction(
            engine=mock_engine,
            cache_repository=mock_cache_repository,
        )

        assert use_case._engine is mock_engine
        assert use_case.cache_repository is mock_cache_repository
        assert (
            notarius.application.use_cases.inference.add_ocr_to_dataset.logger
            is not None
        )

    def test_init_without_cache(self, mock_engine: MagicMock) -> None:
        """Test use case initialization without cache repository."""
        use_case = GenerateOCRExtraction(
            engine=mock_engine,
            cache_repository=None,
        )

        assert use_case.cache_repository is None

    def test_execute_cache_miss_generates_extraction(
        self,
        use_case: GenerateOCRExtraction,
        mock_engine: MagicMock,
        sample_image: PILImage,
    ) -> None:
        """Test that cache miss triggers OCR extraction."""
        request = GenerateOCRExtractionRequest(image=sample_image)

        response = use_case.execute(request)

        assert isinstance(response.result, tuple)
        assert response.from_cache is False
        mock_engine.extract_text.assert_called_once()
        mock_engine.extract_words_and_boxes.assert_called_once()

    def test_execute_cache_hit_returns_cached_text(
        self,
        use_case: GenerateOCRExtraction,
        mock_cache_repository: MagicMock,
        mock_engine: MagicMock,
        sample_image: PILImage,
    ) -> None:
        """Test that cache hit returns cached text result."""
        cached_item = PyTesseractCacheItem(
            text="Cached text",
            words=["Cached"],
            bbox=[[0, 0, 50, 20]],
        )
        mock_cache_repository.get.return_value = cached_item

        request = GenerateOCRExtractionRequest(image=sample_image, text_only=True)
        response = use_case.execute(request)

        assert response.result == "Cached text"
        assert response.from_cache is True
        mock_engine.extract_text.assert_not_called()

    def test_execute_cache_hit_returns_cached_words_boxes(
        self,
        use_case: GenerateOCRExtraction,
        mock_cache_repository: MagicMock,
        mock_engine: MagicMock,
        sample_image: PILImage,
    ) -> None:
        """Test that cache hit returns cached words/bboxes."""
        cached_item = PyTesseractCacheItem(
            text="Cached text",
            words=["Cached", "text"],
            bbox=[[0, 0, 50, 20], [50, 0, 100, 20]],
        )
        mock_cache_repository.get.return_value = cached_item

        request = GenerateOCRExtractionRequest(image=sample_image, text_only=False)
        response = use_case.execute(request)

        # PyTesseractCacheItem stores bbox as tuples internally
        words, bboxes = response.result
        assert words == ["Cached", "text"]
        assert len(bboxes) == 2
        assert response.from_cache is True
        mock_engine.extract_words_and_boxes.assert_not_called()

    def test_execute_calls_engine_with_correct_params(
        self,
        use_case: GenerateOCRExtraction,
        mock_engine: MagicMock,
        sample_image: PILImage,
    ) -> None:
        """Test that execute calls _engine with correct parameters."""
        request = GenerateOCRExtractionRequest(image=sample_image)

        use_case.execute(request)

        mock_engine.extract_text.assert_called_once_with(sample_image)
        mock_engine.extract_words_and_boxes.assert_called_once_with(sample_image)

    def test_execute_stores_in_cache(
        self,
        use_case: GenerateOCRExtraction,
        mock_cache_repository: MagicMock,
        sample_image: PILImage,
    ) -> None:
        """Test that extraction result is stored in cache."""
        request = GenerateOCRExtractionRequest(image=sample_image)

        use_case.execute(request)

        # Verify cache.set was called
        mock_cache_repository.set.assert_called_once()
        call_args = mock_cache_repository.set.call_args
        assert call_args[1]["key"] == "test_cache_key_123"
        assert isinstance(call_args[1]["item"], PyTesseractCacheItem)

    def test_execute_with_cache_invalidation(
        self,
        use_case: GenerateOCRExtraction,
        mock_cache_repository: MagicMock,
        mock_engine: MagicMock,
        sample_image: PILImage,
    ) -> None:
        """Test that invalidate_cache flag deletes cache entry."""
        request = GenerateOCRExtractionRequest(
            image=sample_image,
            invalidate_cache=True,
        )

        use_case.execute(request)

        # Verify cache was invalidated
        mock_cache_repository.delete.assert_called_once_with("test_cache_key_123")

        # Should still generate new extraction
        mock_engine.extract_text.assert_called_once()

    def test_execute_without_cache_repository(
        self,
        mock_engine: MagicMock,
        sample_image: PILImage,
    ) -> None:
        """Test execution when cache repository is None."""
        use_case = GenerateOCRExtraction(
            engine=mock_engine,
            cache_repository=None,
        )

        request = GenerateOCRExtractionRequest(image=sample_image)
        response = use_case.execute(request)

        # Should generate extraction directly without caching
        assert isinstance(response.result, tuple)
        assert response.from_cache is False
        mock_engine.extract_text.assert_called_once()

    def test_execute_with_schematism_and_filename(
        self,
        use_case: GenerateOCRExtraction,
        mock_cache_repository: MagicMock,
        sample_image: PILImage,
    ) -> None:
        """Test that schematism and filename are passed to cache."""
        request = GenerateOCRExtractionRequest(
            image=sample_image,
            schematism="test_schema",
            filename="test_file.jpg",
        )

        use_case.execute(request)

        # Verify cache.set includes metadata
        mock_cache_repository.set.assert_called_once()
        set_kwargs = mock_cache_repository.set.call_args[1]
        assert set_kwargs["schematism"] == "test_schema"
        assert set_kwargs["filename"] == "test_file.jpg"

    def test_execute_returns_text_when_text_only(
        self,
        use_case: GenerateOCRExtraction,
        sample_image: PILImage,
    ) -> None:
        """Test that text_only=True returns string format."""
        request = GenerateOCRExtractionRequest(image=sample_image, text_only=True)

        response = use_case.execute(request)

        # Should be string, not tuple
        assert isinstance(response.result, str)
        assert response.result == "Sample OCR text"

    def test_execute_returns_tuple_by_default(
        self,
        use_case: GenerateOCRExtraction,
        sample_image: PILImage,
    ) -> None:
        """Test that default behavior returns words/bboxes tuple."""
        request = GenerateOCRExtractionRequest(image=sample_image)

        response = use_case.execute(request)

        # Should be tuple, not string
        assert isinstance(response.result, tuple)
        assert len(response.result) == 2

    def test_execute_cache_stores_both_formats(
        self,
        use_case: GenerateOCRExtraction,
        mock_cache_repository: MagicMock,
        sample_image: PILImage,
    ) -> None:
        """Test that cache stores both text and words/bboxes."""
        request = GenerateOCRExtractionRequest(image=sample_image)

        use_case.execute(request)

        # Verify cache item has both formats
        set_call = mock_cache_repository.set.call_args
        cached_item = set_call[1]["item"]
        assert isinstance(cached_item, PyTesseractCacheItem)
        assert cached_item.text is not None
        assert cached_item.words is not None
        assert cached_item.bbox is not None

    def test_execute_integration_flow(
        self,
        use_case: GenerateOCRExtraction,
        mock_cache_repository: MagicMock,
        mock_engine: MagicMock,
        sample_image: PILImage,
    ) -> None:
        """Test complete integration flow: check cache -> extract -> store."""
        request = GenerateOCRExtractionRequest(image=sample_image)

        # Ensure cache miss
        mock_cache_repository.get.return_value = None

        # Execute
        response = use_case.execute(request)

        # Verify complete flow
        # 1. Cache key was generated
        mock_cache_repository.generate_key.assert_called_once()

        # 2. Cache was checked
        mock_cache_repository.get.assert_called_once()

        # 3. OCR extraction was performed
        mock_engine.extract_text.assert_called_once()
        mock_engine.extract_words_and_boxes.assert_called_once()

        # 4. Result was cached
        mock_cache_repository.set.assert_called_once()

        # 5. Response is correct
        assert response.from_cache is False
        assert isinstance(response.result, tuple)

    def test_execute_cache_hit_skips_engine(
        self,
        use_case: GenerateOCRExtraction,
        mock_cache_repository: MagicMock,
        mock_engine: MagicMock,
        sample_image: PILImage,
    ) -> None:
        """Test that cache hit completely skips _engine extraction."""
        cached_item = PyTesseractCacheItem(
            text="Cached",
            words=["Cached"],
            bbox=[[0, 0, 50, 20]],
        )
        mock_cache_repository.get.return_value = cached_item

        request = GenerateOCRExtractionRequest(image=sample_image)
        use_case.execute(request)

        # Engine should never be called
        mock_engine.extract_text.assert_not_called()
        mock_engine.extract_words_and_boxes.assert_not_called()

        # Cache.set should not be called either
        mock_cache_repository.set.assert_not_called()

    @patch("notarius.application.use_cases.generate_ocr_extraction.logger")
    def test_logging(
        self,
        mock_logger: MagicMock,
        use_case: GenerateOCRExtraction,
        sample_image: PILImage,
    ) -> None:
        """Test that operations are logged."""
        request = GenerateOCRExtractionRequest(image=sample_image)

        use_case.execute(request)

        # Verify logger was used
        assert (
            notarius.application.use_cases.inference.add_ocr_to_dataset.logger
            is not None
        )

    def test_execute_extracts_both_even_if_text_only(
        self,
        use_case: GenerateOCRExtraction,
        mock_engine: MagicMock,
        sample_image: PILImage,
    ) -> None:
        """Test that both text and words/boxes are extracted for caching."""
        request = GenerateOCRExtractionRequest(image=sample_image, text_only=True)

        use_case.execute(request)

        # Both methods should be called for cache storage
        mock_engine.extract_text.assert_called_once()
        mock_engine.extract_words_and_boxes.assert_called_once()
