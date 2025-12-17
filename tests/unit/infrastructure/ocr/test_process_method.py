"""Tests for OCREngine process method (new unified interface)."""

from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from notarius.infrastructure.ocr.engine_adapter import (
    OCREngine,
    OCRRequest,
    OCRResponse,
)
from notarius.infrastructure.ocr.types import (
    PytesseractOCRResultDict,
    SimpleOCRResult,
    StructuredOCRResult,
)
from notarius.schemas.configs import PytesseractOCRConfig


@pytest.fixture
def ocr_config() -> PytesseractOCRConfig:
    """Create OCR configuration for testing."""
    return PytesseractOCRConfig(
        language="eng",
        psm_mode=6,
        oem_mode=3,
    )


@pytest.fixture
def sample_image() -> Image.Image:
    """Create a sample PIL image for testing."""
    return Image.new("RGB", (200, 100), color="white")


@pytest.fixture
def sample_ocr_dict() -> PytesseractOCRResultDict:
    """Create sample OCR result dictionary."""
    return {
        "level": [1, 2, 3, 4, 5, 5, 5, 5],
        "page_num": [1, 1, 1, 1, 1, 1, 1, 1],
        "block_num": [0, 1, 1, 1, 1, 1, 1, 1],
        "par_num": [0, 0, 1, 1, 1, 1, 1, 1],
        "line_num": [0, 0, 0, 1, 1, 1, 1, 1],
        "word_num": [0, 0, 0, 0, 1, 2, 3, 4],
        "text": ["", "", "", "", "Hello", "World", "", "Test"],
        "conf": [-1, -1, -1, -1, 95, 90, -1, 85],
        "left": [0, 0, 0, 0, 10, 60, 110, 150],
        "top": [0, 0, 0, 0, 20, 20, 20, 20],
        "width": [200, 200, 200, 200, 40, 50, 30, 40],
        "height": [100, 100, 100, 100, 30, 30, 30, 30],
    }


class TestOcrEngineProcessMethod:
    """Test suite for OCREngine.process() method."""

    @pytest.fixture
    def engine(self, ocr_config: PytesseractOCRConfig) -> OCREngine:
        """Create an _engine instance for testing."""
        return OCREngine.from_config(config=ocr_config)

    def test_process_method_exists(self, engine: OCREngine) -> None:
        """Test that process method exists and has correct signature."""
        assert hasattr(engine, "process")
        assert callable(engine.process)

    @patch("notarius.infrastructure.ocr.engine_adapter.pytesseract.image_to_string")
    def test_process_text_mode(
        self,
        mock_image_to_string: MagicMock,
        engine: OCREngine,
        sample_image: Image.Image,
    ) -> None:
        """Test process method with text mode."""
        mock_image_to_string.return_value = "Extracted text content"

        request = OCRRequest(input=sample_image, mode="text")
        response = engine.process(request)

        # Verify structured_response structure
        assert isinstance(response, OCRResponse)
        assert isinstance(response.output, SimpleOCRResult)
        assert response.output.text == "Extracted text content"

        # Verify pytesseract was called
        mock_image_to_string.assert_called_once()

    @patch("notarius.infrastructure.ocr.engine_adapter.pytesseract.image_to_data")
    def test_process_structured_mode(
        self,
        mock_image_to_data: MagicMock,
        engine: OCREngine,
        sample_image: Image.Image,
        sample_ocr_dict: PytesseractOCRResultDict,
    ) -> None:
        """Test process method with structured mode."""
        mock_image_to_data.return_value = sample_ocr_dict

        request = OCRRequest(input=sample_image, mode="structured")
        response = engine.process(request)

        # Verify structured_response structure
        assert isinstance(response, OCRResponse)
        assert isinstance(response.output, StructuredOCRResult)

        # Check words and bboxes
        assert len(response.output.words) == 3  # "Hello", "World", "Test"
        assert response.output.words == ["Hello", "World", "Test"]
        assert len(response.output.bboxes) == 3

        # Verify bounding boxes are normalized (0-1000 range)
        for bbox in response.output.bboxes:
            assert isinstance(bbox, tuple)
            assert len(bbox) == 4
            xmin, ymin, xmax, ymax = bbox
            assert 0 <= xmin <= 1000
            assert 0 <= ymin <= 1000
            assert 0 <= xmax <= 1000
            assert 0 <= ymax <= 1000
            assert xmin < xmax
            assert ymin < ymax

        # Verify pytesseract was called
        mock_image_to_data.assert_called_once()

    @patch("notarius.infrastructure.ocr.engine_adapter.pytesseract.image_to_data")
    def test_process_structured_filters_correctly(
        self,
        mock_image_to_data: MagicMock,
        engine: OCREngine,
        sample_image: Image.Image,
    ) -> None:
        """Test that structured mode filters low confidence and non-word entries."""
        ocr_dict: PytesseractOCRResultDict = {
            "level": [1, 5, 5, 5, 5],
            "page_num": [1, 1, 1, 1, 1],
            "block_num": [0, 1, 1, 1, 1],
            "par_num": [0, 1, 1, 1, 1],
            "line_num": [0, 1, 1, 1, 1],
            "word_num": [0, 1, 2, 3, 4],
            "text": ["Page", "Valid", "", "LowConf", "Good"],
            "conf": [95, 90, 85, -1, 95],  # LowConf has conf < 0
            "left": [0, 10, 50, 100, 150],
            "top": [0, 20, 20, 20, 20],
            "width": [200, 40, 30, 40, 40],
            "height": [100, 30, 30, 30, 30],
        }
        mock_image_to_data.return_value = ocr_dict

        request = OCRRequest(input=sample_image, mode="structured")
        response = engine.process(request)

        # Should only extract "Valid" and "Good"
        # - "Page" filtered (level != 5)
        # - empty string filtered
        # - "LowConf" filtered (conf < 0)
        assert response.output.words == ["Valid", "Good"]
        assert len(response.output.bboxes) == 2

    @patch("notarius.infrastructure.ocr.engine_adapter.pytesseract.image_to_data")
    def test_process_structured_bbox_normalization(
        self, mock_image_to_data: MagicMock, engine: OCREngine
    ) -> None:
        """Test that bounding boxes are correctly normalized to 0-1000 range."""
        ocr_dict: PytesseractOCRResultDict = {
            "level": [5, 5],
            "page_num": [1, 1],
            "block_num": [1, 1],
            "par_num": [1, 1],
            "line_num": [1, 1],
            "word_num": [1, 2],
            "text": ["Word1", "Word2"],
            "conf": [95, 90],
            "left": [0, 100],  # pixels
            "top": [0, 50],  # pixels
            "width": [100, 100],  # pixels
            "height": [50, 50],  # pixels
        }
        mock_image_to_data.return_value = ocr_dict

        # Image size: 200x100
        image = Image.new("RGB", (200, 100), color="white")
        request = OCRRequest(input=image, mode="structured")
        response = engine.process(request)

        # First word: left=0, top=0, width=100, height=50
        # Normalized to 200x100 image: (0, 0, 500, 500)
        assert response.output.bboxes[0] == (0, 0, 500, 500)

        # Second word: left=100, top=50, width=100, height=50
        # Normalized: (500, 500, 1000, 1000)
        assert response.output.bboxes[1] == (500, 500, 1000, 1000)

    @patch("notarius.infrastructure.ocr.engine_adapter.pytesseract.image_to_data")
    def test_process_structured_empty_result(
        self,
        mock_image_to_data: MagicMock,
        engine: OCREngine,
        sample_image: Image.Image,
    ) -> None:
        """Test process with empty OCR result."""
        empty_dict: PytesseractOCRResultDict = {
            "level": [],
            "page_num": [],
            "block_num": [],
            "par_num": [],
            "line_num": [],
            "word_num": [],
            "text": [],
            "conf": [],
            "left": [],
            "top": [],
            "width": [],
            "height": [],
        }
        mock_image_to_data.return_value = empty_dict

        request = OCRRequest(input=sample_image, mode="structured")
        response = engine.process(request)

        assert response.output.words == []
        assert response.output.bboxes == []

    @patch("notarius.infrastructure.ocr.engine_adapter.pytesseract.image_to_string")
    def test_process_text_mode_with_grayscale_conversion(
        self, mock_image_to_string: MagicMock, engine: OCREngine
    ) -> None:
        """Test that text mode converts image to grayscale."""
        rgb_image = Image.new("RGB", (100, 100), color=(255, 0, 0))
        mock_image_to_string.return_value = "Text"

        request = OCRRequest(input=rgb_image, mode="text")
        _ = engine.process(request)

        # Verify image was processed as grayscale
        mock_image_to_string.assert_called_once()
        call_args = mock_image_to_string.call_args

        # Check that image array was passed
        import numpy as np

        assert isinstance(call_args[0][0], np.ndarray)

    def test_process_with_different_language_config(
        self, sample_image: Image.Image
    ) -> None:
        """Test process with different language configuration."""
        config = PytesseractOCRConfig(language="pol", psm_mode=3, oem_mode=1)
        engine = OCREngine.from_config(config)

        assert engine.config.language == "pol"
        assert engine.config.tesseract_config == "--psm 3 --oem 1"

    @patch("notarius.infrastructure.ocr.engine_adapter.pytesseract.image_to_data")
    @patch("notarius.infrastructure.ocr.engine_adapter.pytesseract.image_to_string")
    def test_process_integration_workflow(
        self,
        mock_image_to_string: MagicMock,
        mock_image_to_data: MagicMock,
        ocr_config: PytesseractOCRConfig,
        sample_ocr_dict: PytesseractOCRResultDict,
    ) -> None:
        """Test full integration workflow with both modes."""
        mock_image_to_string.return_value = "Full text"
        mock_image_to_data.return_value = sample_ocr_dict

        engine = OCREngine.from_config(config=ocr_config)
        image = Image.new("RGB", (200, 100), color="white")

        # Test text mode
        text_request = OCRRequest(input=image, mode="text")
        text_response = engine.process(text_request)
        assert isinstance(text_response.output, SimpleOCRResult)
        assert text_response.output.text == "Full text"

        # Test structured mode
        struct_request = OCRRequest(input=image, mode="structured")
        struct_response = engine.process(struct_request)
        assert isinstance(struct_response.output, StructuredOCRResult)
        assert len(struct_response.output.words) == 3

    def test_request_response_dataclasses_are_frozen(
        self, sample_image: Image.Image
    ) -> None:
        """Test that Request and Response dataclasses are immutable."""
        from dataclasses import FrozenInstanceError

        request = OCRRequest(input=sample_image, mode="text")

        with pytest.raises(FrozenInstanceError):
            request.mode = "structured"  # type: ignore[misc]

        result = SimpleOCRResult(text="test")
        response = OCRResponse(output=result)

        with pytest.raises(FrozenInstanceError):
            response.output = SimpleOCRResult(text="changed")  # type: ignore[misc]

    @patch("notarius.infrastructure.ocr.engine_adapter.pytesseract.image_to_data")
    def test_process_structured_strips_whitespace(
        self,
        mock_image_to_data: MagicMock,
        engine: OCREngine,
        sample_image: Image.Image,
    ) -> None:
        """Test that words are stripped of leading/trailing whitespace."""
        ocr_dict: PytesseractOCRResultDict = {
            "level": [5, 5],
            "page_num": [1, 1],
            "block_num": [1, 1],
            "par_num": [1, 1],
            "line_num": [1, 1],
            "word_num": [1, 2],
            "text": ["  Leading  ", "Trailing  "],
            "conf": [95, 90],
            "left": [0, 100],
            "top": [0, 0],
            "width": [100, 100],
            "height": [50, 50],
        }
        mock_image_to_data.return_value = ocr_dict

        request = OCRRequest(input=sample_image, mode="structured")
        response = engine.process(request)

        assert response.output.words == ["Leading", "Trailing"]
