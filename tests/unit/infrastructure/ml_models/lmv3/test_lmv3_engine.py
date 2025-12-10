"""Tests for SimpleLMv3Engine."""

from unittest.mock import MagicMock, patch

import pytest
from PIL import Image
from PIL.Image import Image as PILImage

from notarius.infrastructure.ml_models.lmv3.engine import SimpleLMv3Engine

import notarius.application.use_cases.inference.add_ocr_to_dataset


class TestSimpleLMv3Engine:
    """Test suite for SimpleLMv3Engine class."""

    @pytest.fixture
    def mock_model(self) -> MagicMock:
        """Create a mock LMv3 model."""
        model = MagicMock()
        model.config.name_or_path = "microsoft/layoutlmv3-base"
        model.config.id2label = {
            0: "O",
            1: "B-PARISH",
            2: "I-PARISH",
            3: "B-DEANERY",
            4: "I-DEANERY",
        }
        return model

    @pytest.fixture
    def mock_processor(self) -> MagicMock:
        """Create a mock processor."""
        return MagicMock()

    @pytest.fixture
    def id2label(self) -> dict[int, str]:
        """Create id2label mapping for testing."""
        return {
            0: "O",
            1: "B-PARISH",
            2: "I-PARISH",
            3: "B-DEANERY",
            4: "I-DEANERY",
        }

    @pytest.fixture
    def engine(
        self, mock_model: MagicMock, mock_processor: MagicMock, id2label: dict[int, str]
    ) -> SimpleLMv3Engine:
        """Create an _engine instance for testing."""
        return SimpleLMv3Engine(
            model=mock_model, processor=mock_processor, id2label=id2label
        )

    @pytest.fixture
    def sample_image(self) -> PILImage:
        """Create a sample PIL image for testing."""
        return Image.new("RGB", (200, 100), color="white")

    @pytest.fixture
    def sample_words(self) -> list[str]:
        """Sample OCR words."""
        return ["St.", "Mary's", "Church"]

    @pytest.fixture
    def sample_bboxes(self) -> list:
        """Sample bounding boxes."""
        return [[0, 0, 20, 10], [20, 0, 50, 10], [50, 0, 80, 10]]

    def test_init(
        self, mock_model: MagicMock, mock_processor: MagicMock, id2label: dict[int, str]
    ) -> None:
        """Test _engine initialization."""
        engine = SimpleLMv3Engine(
            model=mock_model, processor=mock_processor, id2label=id2label
        )

        assert engine.model is mock_model
        assert engine.processor is mock_processor
        assert engine.id2label == id2label
        assert (
            notarius.application.use_cases.inference.add_ocr_to_dataset.logger
            is not None
        )

    @patch("notarius.infrastructure.ml_models.lmv3._engine.retrieve_predictions")
    def test_predict_success(
        self,
        mock_retrieve: MagicMock,
        engine: SimpleLMv3Engine,
        sample_image: PILImage,
        sample_words: list[str],
        sample_bboxes: list,
    ) -> None:
        """Test successful prediction."""
        # Mock retrieve_predictions return value
        mock_retrieve.return_value = (
            [[0, 0, 20, 10], [20, 0, 50, 10], [50, 0, 80, 10]],  # bboxes
            [1, 2, 0],  # prediction IDs
            None,  # other
        )

        words, bboxes, predictions = engine.predict(
            image=sample_image, words=sample_words, bboxes=sample_bboxes
        )

        assert words == sample_words
        assert len(bboxes) == 3
        assert predictions == ["B-PARISH", "I-PARISH", "O"]
        mock_retrieve.assert_called_once()

    @patch("notarius.infrastructure.ml_models.lmv3._engine.retrieve_predictions")
    def test_predict_converts_to_grayscale(
        self,
        mock_retrieve: MagicMock,
        engine: SimpleLMv3Engine,
        sample_image: PILImage,
        sample_words: list[str],
        sample_bboxes: list,
    ) -> None:
        """Test that image is converted to grayscale before prediction."""
        mock_retrieve.return_value = ([[0, 0, 10, 10]], [0], None)

        engine.predict(image=sample_image, words=sample_words, bboxes=sample_bboxes)

        # Verify the image passed to retrieve_predictions is grayscale->RGB
        call_args = mock_retrieve.call_args
        image_arg = call_args[1]["image"]

        # The image should be in RGB mode (after grayscale conversion)
        assert image_arg.mode == "RGB"

    @patch("notarius.infrastructure.ml_models.lmv3._engine.retrieve_predictions")
    def test_predict_maps_ids_to_labels(
        self,
        mock_retrieve: MagicMock,
        engine: SimpleLMv3Engine,
        sample_image: PILImage,
        sample_words: list[str],
        sample_bboxes: list,
    ) -> None:
        """Test that prediction IDs are correctly mapped to labels."""
        # Return various prediction IDs
        mock_retrieve.return_value = (
            [[0, 0, 10, 10], [10, 0, 20, 10], [20, 0, 30, 10]],
            [1, 3, 4],  # B-PARISH, B-DEANERY, I-DEANERY
            None,
        )

        _, _, predictions = engine.predict(
            image=sample_image, words=sample_words, bboxes=sample_bboxes
        )

        assert predictions == ["B-PARISH", "B-DEANERY", "I-DEANERY"]

    @patch("notarius.infrastructure.ml_models.lmv3._engine.retrieve_predictions")
    def test_predict_with_empty_words(
        self,
        mock_retrieve: MagicMock,
        engine: SimpleLMv3Engine,
        sample_image: PILImage,
    ) -> None:
        """Test prediction with empty words list."""
        mock_retrieve.return_value = ([], [], None)

        words, bboxes, predictions = engine.predict(
            image=sample_image, words=[], bboxes=[]
        )

        assert words == []
        assert bboxes == []
        assert predictions == []

    @patch("notarius.infrastructure.ml_models.lmv3._engine.retrieve_predictions")
    def test_predict_preserves_word_order(
        self,
        mock_retrieve: MagicMock,
        engine: SimpleLMv3Engine,
        sample_image: PILImage,
    ) -> None:
        """Test that word order is preserved."""
        input_words = ["word1", "word2", "word3", "word4"]
        input_bboxes = [[0, 0, 10, 10]] * 4

        mock_retrieve.return_value = (
            input_bboxes,
            [0, 1, 2, 3],
            None,
        )

        words, _, _ = engine.predict(
            image=sample_image, words=input_words, bboxes=input_bboxes
        )

        assert words == input_words

    @patch("notarius.infrastructure.ml_models.lmv3._engine.retrieve_predictions")
    def test_predict_calls_retrieve_predictions_correctly(
        self,
        mock_retrieve: MagicMock,
        engine: SimpleLMv3Engine,
        sample_image: PILImage,
        sample_words: list[str],
        sample_bboxes: list,
    ) -> None:
        """Test that retrieve_predictions is called with correct parameters."""
        mock_retrieve.return_value = ([[0, 0, 10, 10]], [0], None)

        engine.predict(image=sample_image, words=sample_words, bboxes=sample_bboxes)

        # Verify retrieve_predictions was called with correct args
        mock_retrieve.assert_called_once()
        call_kwargs = mock_retrieve.call_args[1]

        assert "image" in call_kwargs
        assert call_kwargs["processor"] is engine.processor
        assert call_kwargs["model"] is engine.model
        assert call_kwargs["words"] == sample_words
        assert call_kwargs["bboxes"] == sample_bboxes

    def test_predict_returns_correct_tuple_structure(
        self, engine: SimpleLMv3Engine, sample_image: PILImage
    ) -> None:
        """Test that predict returns tuple of correct type."""
        with patch(
            "notarius.infrastructure.ml_models.lmv3._engine.retrieve_predictions"
        ) as mock_retrieve:
            mock_retrieve.return_value = ([[0, 0, 10, 10]], [0], None)

            result = engine.predict(
                image=sample_image, words=["word"], bboxes=[[0, 0, 10, 10]]
            )

            assert isinstance(result, tuple)
            assert len(result) == 3
            assert isinstance(result[0], list)  # words
            assert isinstance(result[1], list)  # bboxes
            assert isinstance(result[2], list)  # predictions

    @patch(
        "notarius.infrastructure.ml_models.lmv3.inference_utils.get_model_and_processor"
    )
    def test_from_config_factory_method(
        self, mock_get_model: MagicMock, mock_model: MagicMock
    ) -> None:
        """Test from_config factory method."""
        mock_processor = MagicMock()
        mock_get_model.return_value = (mock_model, mock_processor)

        mock_config = MagicMock()
        mock_config.inference.checkpoint = "test/checkpoint"

        engine = SimpleLMv3Engine.from_config(mock_config)

        assert isinstance(engine, SimpleLMv3Engine)
        assert engine.model is mock_model
        assert engine.processor is mock_processor
        mock_get_model.assert_called_once_with(mock_config)

    @patch("notarius.infrastructure.ml_models.lmv3._engine.logger")
    def test_logging_on_prediction(
        self,
        mock_logger: MagicMock,
        engine: SimpleLMv3Engine,
        sample_image: PILImage,
        sample_words: list[str],
        sample_bboxes: list,
    ) -> None:
        """Test that predictions are logged."""
        with patch(
            "notarius.infrastructure.ml_models.lmv3._engine.retrieve_predictions"
        ) as mock_retrieve:
            mock_retrieve.return_value = ([[0, 0, 10, 10]], [0], None)

            engine.predict(image=sample_image, words=sample_words, bboxes=sample_bboxes)

            # Verify logger exists
            assert (
                notarius.application.use_cases.inference.add_ocr_to_dataset.logger
                is not None
            )
