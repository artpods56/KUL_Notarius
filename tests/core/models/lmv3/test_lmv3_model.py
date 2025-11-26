import PIL
import pytest
import json
import respx
import io
from unittest.mock import Mock, patch, MagicMock

from PIL.Image import Image

from core.models.lmv3.model import LMv3Model
from schemas.data.schematism import SchematismPage


@pytest.fixture(scope="module")
def mock_model_and_processor():
    """Mock the model and processor to avoid loading heavy models"""
    mock_model = MagicMock()
    mock_model.hf_device_map = {"": 0}
    mock_model.config.id2label = {
        0: "O",
        1: "B-PARISH",
        2: "I-PARISH",
        3: "B-DEANERY",
        4: "I-DEANERY"
    }
    mock_model.config.label2id = {
        "O": 0,
        "B-PARISH": 1,
        "I-PARISH": 2,
        "B-DEANERY": 3,
        "I-DEANERY": 4
    }

    mock_processor = MagicMock()

    return mock_model, mock_processor


@pytest.fixture(scope="module")
def lmv3_model(lmv3_model_config, mock_model_and_processor):
    """Fixture that provides an LMv3Model instance for testing with mocked dependencies"""
    mock_model, mock_processor = mock_model_and_processor

    with patch('core.models.lmv3.model.get_model_and_processor', return_value=(mock_model, mock_processor)):
        model = LMv3Model(config=lmv3_model_config)

    return model


@pytest.fixture
def lmv3_model_no_cache(lmv3_model_config, mock_model_and_processor):
    """LMv3 model with cache disabled"""
    mock_model, mock_processor = mock_model_and_processor

    with patch('core.models.lmv3.model.get_model_and_processor', return_value=(mock_model, mock_processor)):
        model = LMv3Model(config=lmv3_model_config, enable_cache=False)

    return model


class TestLMv3Model:

    def test_initialization(self, lmv3_model):
        """Test proper initialization of LMv3Model"""
        assert lmv3_model
        assert isinstance(lmv3_model, LMv3Model)
        assert lmv3_model.id2label is not None
        assert lmv3_model.label2id is not None

    def test_no_input_prediction(self, lmv3_model):
        """Test that predict raises TypeError when no image is provided"""
        with pytest.raises(TypeError):
            lmv3_model.predict()

    @patch('core.models.lmv3.model.ocr_page')
    @patch('core.models.lmv3.model.retrieve_predictions')
    @patch('core.models.lmv3.model.repair_bio_labels')
    @patch('core.models.lmv3.model.build_page_json')
    def test_predict_with_mocked_operations(
        self,
        mock_build_page_json,
        mock_repair_bio_labels,
        mock_retrieve_predictions,
        mock_ocr_page,
        lmv3_model: LMv3Model,
        sample_pil_image: Image
    ):
        """Test that the LMv3 model prediction pipeline works with mocked operations."""
        # Mock cache to return None (cache miss) to force prediction pipeline execution
        with patch.object(lmv3_model.cache, 'get', return_value=None):
            # Mock OCR output
            mock_words = ["Sample", "Parish", "Name"]
            mock_bboxes = [[10, 10, 100, 30], [110, 10, 200, 30], [210, 10, 300, 30]]
            mock_ocr_page.return_value = (mock_words, mock_bboxes)

            # Mock predictions
            mock_prediction_ids = [1, 2, 0]  # B-PARISH, I-PARISH, O
            mock_retrieve_predictions.return_value = (mock_bboxes, mock_prediction_ids, None)

            # Mock repaired predictions
            mock_repaired = ["B-PARISH", "I-PARISH", "O"]
            mock_repair_bio_labels.return_value = mock_repaired

            # Mock structured output
            mock_structured = {
                "page_number": "1",
                "entries": [
                    {"parish": "Sample Parish", "deanery": None, "dedication": None, "building_material": None}
                ]
            }
            mock_build_page_json.return_value = SchematismPage(**mock_structured)

            # Execute prediction
            result = lmv3_model.predict(pil_image=sample_pil_image)

            # Assertions
            assert result is not None
            # Result should be a SchematismPage object (not cached) or dict (if cached)
            # Since we're testing with cache enabled, check for SchematismPage
            assert isinstance(result, (dict, SchematismPage))

            # Convert to dict for uniform checking
            result_dict = result if isinstance(result, dict) else result.model_dump()
            assert "page_number" in result_dict
            assert "entries" in result_dict

            # Verify mocks were called
            mock_ocr_page.assert_called_once()
            mock_retrieve_predictions.assert_called_once()
            mock_repair_bio_labels.assert_called_once()
            mock_build_page_json.assert_called_once()

    @patch('core.models.lmv3.model.ocr_page')
    @patch('core.models.lmv3.model.retrieve_predictions')
    @patch('core.models.lmv3.model.repair_bio_labels')
    def test_predict_raw_predictions(
        self,
        mock_repair_bio_labels,
        mock_retrieve_predictions,
        mock_ocr_page,
        lmv3_model_no_cache: LMv3Model,
        sample_pil_image: Image
    ):
        """Test that raw predictions mode returns tuple of words, bboxes, predictions"""
        # Mock OCR output
        mock_words = ["Test", "Word"]
        mock_bboxes = [[10, 10, 100, 30], [110, 10, 200, 30]]
        mock_ocr_page.return_value = (mock_words, mock_bboxes)

        # Mock predictions
        mock_prediction_ids = [1, 0]
        mock_retrieve_predictions.return_value = (mock_bboxes, mock_prediction_ids, None)

        # Mock repaired predictions
        mock_repaired = ["B-PARISH", "O"]
        mock_repair_bio_labels.return_value = mock_repaired

        # Execute prediction with raw_predictions=True
        words, bboxes, predictions = lmv3_model_no_cache.predict(
            pil_image=sample_pil_image,
            raw_predictions=True
        )

        # Assertions
        assert words == mock_words
        assert bboxes == mock_bboxes
        assert predictions == mock_repaired

    def test_cache_disabled(self, lmv3_model_no_cache):
        """Test that cache is properly disabled when enable_cache=False"""
        assert lmv3_model_no_cache.enable_cache is False
        assert not hasattr(lmv3_model_no_cache, 'cache') or lmv3_model_no_cache.cache is None

