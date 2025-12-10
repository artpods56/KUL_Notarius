"""Tests for LMv3EngineAdapter."""

from unittest.mock import MagicMock, patch

import pytest
from omegaconf import DictConfig
from PIL import Image
from PIL.Image import Image as PILImage

from notarius.domain.entities.schematism import SchematismPage, SchematismEntry
from notarius.infrastructure.ml_models.lmv3.engine_adapter import LMv3EngineAdapter
from notarius.application.use_cases.gen.generate_lmv3_prediction import (
    GenerateLMv3PredictionResponse,
)


class TestLMv3EngineAdapter:
    """Test suite for LMv3EngineAdapter class."""

    @pytest.fixture
    def mock_config(self) -> DictConfig:
        """Create a mock configuration."""
        config = DictConfig(
            {
                "inference": {
                    "checkpoint": "test/checkpoint",
                },
                "ocr": {
                    "language": "eng",
                    "enable_cache": True,
                    "psm_mode": 6,
                    "oem_mode": 3,
                },
                "enable_cache": True,
            }
        )
        return config

    @pytest.fixture
    def mock_ocr_engine(self) -> MagicMock:
        """Create a mock OCR _engine."""
        engine = MagicMock()
        engine.predict.return_value = (
            ["word1", "word2"],
            [[0, 0, 100, 50], [100, 0, 200, 50]],
        )
        return engine

    @pytest.fixture
    def sample_image(self) -> PILImage:
        """Create a sample PIL image for testing."""
        return Image.new("RGB", (200, 100), color="white")

    @patch("notarius.infrastructure.ml_models.lmv3.engine_adapter.SimpleLMv3Engine")
    @patch("notarius.infrastructure.ml_models.lmv3.engine_adapter.LMv3CacheRepository")
    @patch(
        "notarius.infrastructure.ml_models.lmv3.engine_adapter.GenerateLMv3Prediction"
    )
    def test_init_with_ocr_engine(
        self,
        mock_use_case_class: MagicMock,
        mock_cache_repo_class: MagicMock,
        mock_engine_class: MagicMock,
        mock_config: DictConfig,
        mock_ocr_engine: MagicMock,
    ) -> None:
        """Test adapter initialization with provided OCR _engine."""
        adapter = LMv3EngineAdapter(
            config=mock_config,
            enable_cache=True,
            ocr_engine=mock_ocr_engine,
        )

        assert adapter.ocr_engine is mock_ocr_engine
        assert adapter.enable_cache is True
        assert adapter.config is mock_config

    @patch("notarius.infrastructure.ml_models.lmv3.engine_adapter.SimpleLMv3Engine")
    @patch("notarius.infrastructure.ml_models.lmv3.engine_adapter.LMv3CacheRepository")
    @patch(
        "notarius.infrastructure.ml_models.lmv3.engine_adapter.GenerateLMv3Prediction"
    )
    @patch("notarius.infrastructure.ml_models.lmv3.engine_adapter.OCREngine")
    def test_init_without_ocr_engine(
        self,
        mock_ocr_class: MagicMock,
        mock_use_case_class: MagicMock,
        mock_cache_repo_class: MagicMock,
        mock_engine_class: MagicMock,
        mock_config: DictConfig,
    ) -> None:
        """Test adapter initialization creates OCR _engine from provider_config."""
        mock_ocr_instance = MagicMock()
        mock_ocr_class.from_config.return_value = mock_ocr_instance

        adapter = LMv3EngineAdapter(config=mock_config, enable_cache=True)

        # Verify OCR _engine was created from provider_config
        mock_ocr_class.from_config.assert_called_once_with(mock_config.ocr)
        assert adapter.ocr_engine is mock_ocr_instance

    @patch("notarius.infrastructure.ml_models.lmv3.engine_adapter.SimpleLMv3Engine")
    @patch("notarius.infrastructure.ml_models.lmv3.engine_adapter.LMv3CacheRepository")
    @patch(
        "notarius.infrastructure.ml_models.lmv3.engine_adapter.GenerateLMv3Prediction"
    )
    def test_init_without_cache(
        self,
        mock_use_case_class: MagicMock,
        mock_cache_repo_class: MagicMock,
        mock_engine_class: MagicMock,
        mock_config: DictConfig,
        mock_ocr_engine: MagicMock,
    ) -> None:
        """Test adapter initialization with caching disabled."""
        adapter = LMv3EngineAdapter(
            config=mock_config,
            enable_cache=False,
            ocr_engine=mock_ocr_engine,
        )

        assert adapter.cache_repository is None
        assert adapter.enable_cache is False

    @patch("notarius.infrastructure.ml_models.lmv3.engine_adapter.SimpleLMv3Engine")
    @patch("notarius.infrastructure.ml_models.lmv3.engine_adapter.LMv3CacheRepository")
    @patch("notarius.infrastructure.ml_models.lmv3.engine_adapter.BIOProcessingService")
    @patch(
        "notarius.infrastructure.ml_models.lmv3.engine_adapter.GenerateLMv3Prediction"
    )
    def test_predict_returns_structured_by_default(
        self,
        mock_use_case_class: MagicMock,
        mock_bio_class: MagicMock,
        mock_cache_repo_class: MagicMock,
        mock_engine_class: MagicMock,
        mock_config: DictConfig,
        mock_ocr_engine: MagicMock,
        sample_image: PILImage,
    ) -> None:
        """Test that predict returns structured dict by default."""
        # Setup mocks
        mock_use_case = MagicMock()
        mock_use_case_class.return_value = mock_use_case

        mock_page = SchematismPage(entries=[SchematismEntry(parish="Test Parish")])
        mock_response = GenerateLMv3PredictionResponse(
            prediction=mock_page,
            from_cache=False,
        )
        mock_use_case.execute.return_value = mock_response

        adapter = LMv3EngineAdapter(
            config=mock_config,
            enable_cache=True,
            ocr_engine=mock_ocr_engine,
        )

        # Execute prediction
        result = adapter.predict(sample_image)

        # Verify result is a dict (from model_dump())
        assert isinstance(result, dict)
        mock_ocr_engine.predict.assert_called_once_with(sample_image, text_only=False)

    @patch("notarius.infrastructure.ml_models.lmv3.engine_adapter.SimpleLMv3Engine")
    @patch("notarius.infrastructure.ml_models.lmv3.engine_adapter.LMv3CacheRepository")
    @patch("notarius.infrastructure.ml_models.lmv3.engine_adapter.BIOProcessingService")
    @patch(
        "notarius.infrastructure.ml_models.lmv3.engine_adapter.GenerateLMv3Prediction"
    )
    def test_predict_returns_raw_when_requested(
        self,
        mock_use_case_class: MagicMock,
        mock_bio_class: MagicMock,
        mock_cache_repo_class: MagicMock,
        mock_engine_class: MagicMock,
        mock_config: DictConfig,
        mock_ocr_engine: MagicMock,
        sample_image: PILImage,
    ) -> None:
        """Test that predict returns raw tuple when raw_predictions=True."""
        # Setup mocks
        mock_use_case = MagicMock()
        mock_use_case_class.return_value = mock_use_case

        raw_pred = (["word"], [[0, 0, 10, 10]], ["B-PARISH"])
        mock_response = GenerateLMv3PredictionResponse(
            prediction=raw_pred,
            from_cache=False,
        )
        mock_use_case.execute.return_value = mock_response

        adapter = LMv3EngineAdapter(
            config=mock_config,
            enable_cache=True,
            ocr_engine=mock_ocr_engine,
        )

        # Execute prediction with raw_predictions=True
        result = adapter.predict(sample_image, raw_predictions=True)

        # Verify result is a tuple
        assert isinstance(result, tuple)
        assert len(result) == 3

    @patch("notarius.infrastructure.ml_models.lmv3.engine_adapter.SimpleLMv3Engine")
    @patch("notarius.infrastructure.ml_models.lmv3.engine_adapter.LMv3CacheRepository")
    @patch("notarius.infrastructure.ml_models.lmv3.engine_adapter.BIOProcessingService")
    @patch(
        "notarius.infrastructure.ml_models.lmv3.engine_adapter.GenerateLMv3Prediction"
    )
    def test_predict_passes_metadata_to_use_case(
        self,
        mock_use_case_class: MagicMock,
        mock_bio_class: MagicMock,
        mock_cache_repo_class: MagicMock,
        mock_engine_class: MagicMock,
        mock_config: DictConfig,
        mock_ocr_engine: MagicMock,
        sample_image: PILImage,
    ) -> None:
        """Test that predict passes schematism and filename metadata."""
        # Setup mocks
        mock_use_case = MagicMock()
        mock_use_case_class.return_value = mock_use_case

        mock_page = SchematismPage(entries=[])
        mock_response = GenerateLMv3PredictionResponse(
            prediction=mock_page,
            from_cache=False,
        )
        mock_use_case.execute.return_value = mock_response

        adapter = LMv3EngineAdapter(
            config=mock_config,
            enable_cache=True,
            ocr_engine=mock_ocr_engine,
        )

        # Execute prediction with metadata
        adapter.predict(
            sample_image,
            schematism="test_schema",
            filename="test_file.jpg",
        )

        # Verify use case was called with correct request
        mock_use_case.execute.assert_called_once()
        request = mock_use_case.execute.call_args[0][0]
        assert request.schematism == "test_schema"
        assert request.filename == "test_file.jpg"

    @patch("notarius.infrastructure.ml_models.lmv3.engine_adapter.SimpleLMv3Engine")
    @patch("notarius.infrastructure.ml_models.lmv3.engine_adapter.LMv3CacheRepository")
    @patch("notarius.infrastructure.ml_models.lmv3.engine_adapter.BIOProcessingService")
    @patch(
        "notarius.infrastructure.ml_models.lmv3.engine_adapter.GenerateLMv3Prediction"
    )
    def test_predict_uses_ocr_results(
        self,
        mock_use_case_class: MagicMock,
        mock_bio_class: MagicMock,
        mock_cache_repo_class: MagicMock,
        mock_engine_class: MagicMock,
        mock_config: DictConfig,
        mock_ocr_engine: MagicMock,
        sample_image: PILImage,
    ) -> None:
        """Test that predict uses OCR results for LMv3 prediction."""
        # Setup mocks
        mock_use_case = MagicMock()
        mock_use_case_class.return_value = mock_use_case

        mock_page = SchematismPage(entries=[])
        mock_response = GenerateLMv3PredictionResponse(
            prediction=mock_page,
            from_cache=False,
        )
        mock_use_case.execute.return_value = mock_response

        # Setup OCR results
        mock_ocr_engine.predict.return_value = (
            ["word1", "word2", "word3"],
            [[0, 0, 100, 50], [100, 0, 200, 50], [200, 0, 300, 50]],
        )

        adapter = LMv3EngineAdapter(
            config=mock_config,
            enable_cache=True,
            ocr_engine=mock_ocr_engine,
        )

        # Execute prediction
        adapter.predict(sample_image)

        # Verify OCR was called
        mock_ocr_engine.predict.assert_called_once_with(sample_image, text_only=False)

        # Verify use case was called with OCR results
        request = mock_use_case.execute.call_args[0][0]
        assert request.words == ["word1", "word2", "word3"]
        assert len(request.bboxes) == 3

    @patch("notarius.infrastructure.ml_models.lmv3.engine_adapter.SimpleLMv3Engine")
    @patch("notarius.infrastructure.ml_models.lmv3.engine_adapter.LMv3CacheRepository")
    @patch("notarius.infrastructure.ml_models.lmv3.engine_adapter.BIOProcessingService")
    @patch(
        "notarius.infrastructure.ml_models.lmv3.engine_adapter.GenerateLMv3Prediction"
    )
    @patch("notarius.infrastructure.ml_models.lmv3.engine_adapter.OCREngine")
    def test_from_config_factory(
        self,
        mock_ocr_class: MagicMock,
        mock_use_case_class: MagicMock,
        mock_bio_class: MagicMock,
        mock_cache_repo_class: MagicMock,
        mock_engine_class: MagicMock,
        mock_config: DictConfig,
    ) -> None:
        """Test from_config factory method."""
        adapter = LMv3EngineAdapter.from_config(mock_config)

        assert isinstance(adapter, LMv3EngineAdapter)
        assert adapter.enable_cache is True

    @patch("notarius.infrastructure.ml_models.lmv3.engine_adapter.SimpleLMv3Engine")
    @patch("notarius.infrastructure.ml_models.lmv3.engine_adapter.BIOProcessingService")
    @patch(
        "notarius.infrastructure.ml_models.lmv3.engine_adapter.GenerateLMv3Prediction"
    )
    def test_init_raises_without_ocr_config(
        self,
        mock_use_case_class: MagicMock,
        mock_bio_class: MagicMock,
        mock_engine_class: MagicMock,
    ) -> None:
        """Test that init raises error when OCR provider_config missing."""
        config = DictConfig(
            {
                "inference": {
                    "checkpoint": "test/checkpoint",
                },
                # No ocr provider_config
            }
        )

        with pytest.raises(ValueError, match="OCR configuration required"):
            LMv3EngineAdapter(config=config, enable_cache=True)
