"""Tests for LLMEngineAdapter."""

from unittest.mock import MagicMock, patch

import pytest
from omegaconf import DictConfig
from PIL import Image
from PIL.Image import Image as PILImage

from notarius.application.use_cases.gen.generate_llm_prediction import (
    GenerateLLMPredictionResponse,
)
from notarius.infrastructure.llm.engine_adapter import LLMEngineAdapter


class TestLLMEngineAdapter:
    """Test suite for LLMEngineAdapter class."""

    @pytest.fixture
    def mock_config(self) -> DictConfig:
        """Create a mock configuration."""
        config = DictConfig(
            {
                "provider": {
                    "retries": 3,
                    "template_dir": "templates",
                    "enable_cache": True,
                },
                "provider": {
                    "model": "gpt-4",
                    "api_key": "test_key",
                },
            }
        )
        return config

    @pytest.fixture
    def sample_image(self) -> PILImage:
        """Create a sample PIL image for testing."""
        return Image.new("RGB", (200, 100), color="white")

    @patch("notarius.infrastructure.llm.engine_adapter.llm_provider_factory")
    @patch("notarius.infrastructure.llm.engine_adapter.SimpleLLMEngine")
    @patch("notarius.infrastructure.llm.engine_adapter.Jinja2PromptRenderer")
    @patch("notarius.infrastructure.llm.engine_adapter.PromptConstructionService")
    @patch("notarius.infrastructure.llm.engine_adapter.LLMCacheRepository")
    @patch("notarius.infrastructure.llm.engine_adapter.GenerateLLMPrediction")
    def test_init(
        self,
        mock_use_case_class: MagicMock,
        mock_cache_repo_class: MagicMock,
        mock_prompt_service_class: MagicMock,
        mock_prompt_renderer_class: MagicMock,
        mock_engine_class: MagicMock,
        mock_factory: MagicMock,
        mock_config: DictConfig,
    ) -> None:
        """Test adapter initialization."""
        # Setup mocks
        mock_provider = MagicMock()
        mock_cache = MagicMock()
        mock_factory.return_value = (mock_provider, mock_cache)

        adapter = LLMEngineAdapter(config=mock_config, enable_cache=True)

        assert adapter.enable_cache is True
        assert adapter.config is mock_config

    @patch("notarius.infrastructure.llm.engine_adapter.llm_provider_factory")
    @patch("notarius.infrastructure.llm.engine_adapter.SimpleLLMEngine")
    @patch("notarius.infrastructure.llm.engine_adapter.Jinja2PromptRenderer")
    @patch("notarius.infrastructure.llm.engine_adapter.PromptConstructionService")
    @patch("notarius.infrastructure.llm.engine_adapter.LLMCacheRepository")
    @patch("notarius.infrastructure.llm.engine_adapter.GenerateLLMPrediction")
    def test_init_without_cache(
        self,
        mock_use_case_class: MagicMock,
        mock_cache_repo_class: MagicMock,
        mock_prompt_service_class: MagicMock,
        mock_prompt_renderer_class: MagicMock,
        mock_engine_class: MagicMock,
        mock_factory: MagicMock,
        mock_config: DictConfig,
    ) -> None:
        """Test adapter initialization with caching disabled."""
        mock_provider = MagicMock()
        mock_cache = MagicMock()
        mock_factory.return_value = (mock_provider, mock_cache)

        adapter = LLMEngineAdapter(config=mock_config, enable_cache=False)

        assert adapter.cache_repository is None
        assert adapter.enable_cache is False

    @patch("notarius.infrastructure.llm.engine_adapter.llm_provider_factory")
    @patch("notarius.infrastructure.llm.engine_adapter.SimpleLLMEngine")
    @patch("notarius.infrastructure.llm.engine_adapter.Jinja2PromptRenderer")
    @patch("notarius.infrastructure.llm.engine_adapter.PromptConstructionService")
    @patch("notarius.infrastructure.llm.engine_adapter.LLMCacheRepository")
    @patch("notarius.infrastructure.llm.engine_adapter.GenerateLLMPrediction")
    def test_predict_text_only(
        self,
        mock_use_case_class: MagicMock,
        mock_cache_repo_class: MagicMock,
        mock_prompt_service_class: MagicMock,
        mock_prompt_renderer_class: MagicMock,
        mock_engine_class: MagicMock,
        mock_factory: MagicMock,
        mock_config: DictConfig,
    ) -> None:
        """Test prediction with text-only input."""
        # Setup mocks
        mock_provider = MagicMock()
        mock_cache = MagicMock()
        mock_factory.return_value = (mock_provider, mock_cache)

        mock_use_case = MagicMock()
        mock_use_case_class.return_value = mock_use_case

        mock_response = GenerateLLMPredictionResponse(
            prediction={"parish": "Test Parish"},
            messages="messages",
            from_cache=False,
        )
        mock_use_case.execute.return_value = mock_response

        adapter = LLMEngineAdapter(config=mock_config)

        # Execute prediction
        result = adapter.predict(text="Sample OCR text")

        # Verify result
        assert result == {"parish": "Test Parish"}
        mock_use_case.execute.assert_called_once()

    @patch("notarius.infrastructure.llm.engine_adapter.llm_provider_factory")
    @patch("notarius.infrastructure.llm.engine_adapter.SimpleLLMEngine")
    @patch("notarius.infrastructure.llm.engine_adapter.Jinja2PromptRenderer")
    @patch("notarius.infrastructure.llm.engine_adapter.PromptConstructionService")
    @patch("notarius.infrastructure.llm.engine_adapter.LLMCacheRepository")
    @patch("notarius.infrastructure.llm.engine_adapter.GenerateLLMPrediction")
    def test_predict_image_only(
        self,
        mock_use_case_class: MagicMock,
        mock_cache_repo_class: MagicMock,
        mock_prompt_service_class: MagicMock,
        mock_prompt_renderer_class: MagicMock,
        mock_engine_class: MagicMock,
        mock_factory: MagicMock,
        mock_config: DictConfig,
        sample_image: PILImage,
    ) -> None:
        """Test prediction with image-only input."""
        mock_provider = MagicMock()
        mock_cache = MagicMock()
        mock_factory.return_value = (mock_provider, mock_cache)

        mock_use_case = MagicMock()
        mock_use_case_class.return_value = mock_use_case

        mock_response = GenerateLLMPredictionResponse(
            prediction={"parish": "Image Parish"},
            messages="messages",
            from_cache=False,
        )
        mock_use_case.execute.return_value = mock_response

        adapter = LLMEngineAdapter(config=mock_config)

        # Execute prediction
        result = adapter.predict(image=sample_image)

        # Verify result
        assert result == {"parish": "Image Parish"}

    @patch("notarius.infrastructure.llm.engine_adapter.llm_provider_factory")
    @patch("notarius.infrastructure.llm.engine_adapter.SimpleLLMEngine")
    @patch("notarius.infrastructure.llm.engine_adapter.Jinja2PromptRenderer")
    @patch("notarius.infrastructure.llm.engine_adapter.PromptConstructionService")
    @patch("notarius.infrastructure.llm.engine_adapter.LLMCacheRepository")
    @patch("notarius.infrastructure.llm.engine_adapter.GenerateLLMPrediction")
    def test_predict_multimodal(
        self,
        mock_use_case_class: MagicMock,
        mock_cache_repo_class: MagicMock,
        mock_prompt_service_class: MagicMock,
        mock_prompt_renderer_class: MagicMock,
        mock_engine_class: MagicMock,
        mock_factory: MagicMock,
        mock_config: DictConfig,
        sample_image: PILImage,
    ) -> None:
        """Test multimodal prediction with both image and text."""
        mock_provider = MagicMock()
        mock_cache = MagicMock()
        mock_factory.return_value = (mock_provider, mock_cache)

        mock_use_case = MagicMock()
        mock_use_case_class.return_value = mock_use_case

        mock_response = GenerateLLMPredictionResponse(
            prediction={"parish": "Multimodal Parish"},
            messages="messages",
            from_cache=False,
        )
        mock_use_case.execute.return_value = mock_response

        adapter = LLMEngineAdapter(config=mock_config)

        # Execute prediction
        result = adapter.predict(
            image=sample_image,
            text="OCR text",
            context={"hints": {"language": "en"}},
        )

        # Verify result
        assert result == {"parish": "Multimodal Parish"}

        # Verify request was created correctly
        request = mock_use_case.execute.call_args[0][0]
        assert request.image is sample_image
        assert request.text == "OCR text"
        assert request.context == {"hints": {"language": "en"}}

    @patch("notarius.infrastructure.llm.engine_adapter.llm_provider_factory")
    @patch("notarius.infrastructure.llm.engine_adapter.SimpleLLMEngine")
    @patch("notarius.infrastructure.llm.engine_adapter.Jinja2PromptRenderer")
    @patch("notarius.infrastructure.llm.engine_adapter.PromptConstructionService")
    @patch("notarius.infrastructure.llm.engine_adapter.LLMCacheRepository")
    @patch("notarius.infrastructure.llm.engine_adapter.GenerateLLMPrediction")
    def test_predict_with_custom_templates(
        self,
        mock_use_case_class: MagicMock,
        mock_cache_repo_class: MagicMock,
        mock_prompt_service_class: MagicMock,
        mock_prompt_renderer_class: MagicMock,
        mock_engine_class: MagicMock,
        mock_factory: MagicMock,
        mock_config: DictConfig,
    ) -> None:
        """Test prediction with custom template names."""
        mock_provider = MagicMock()
        mock_cache = MagicMock()
        mock_factory.return_value = (mock_provider, mock_cache)

        mock_use_case = MagicMock()
        mock_use_case_class.return_value = mock_use_case

        mock_response = GenerateLLMPredictionResponse(
            prediction={"result": "test"},
            messages="messages",
            from_cache=False,
        )
        mock_use_case.execute.return_value = mock_response

        adapter = LLMEngineAdapter(config=mock_config)

        # Execute prediction with custom templates
        adapter.predict(
            text="test",
            system_template="custom_system.j2",
            user_template="custom_user.j2",
        )

        # Verify templates were passed
        request = mock_use_case.execute.call_args[0][0]
        assert request.system_template == "custom_system.j2"
        assert request.user_template == "custom_user.j2"

    @patch("notarius.infrastructure.llm.engine_adapter.llm_provider_factory")
    @patch("notarius.infrastructure.llm.engine_adapter.SimpleLLMEngine")
    @patch("notarius.infrastructure.llm.engine_adapter.Jinja2PromptRenderer")
    @patch("notarius.infrastructure.llm.engine_adapter.PromptConstructionService")
    @patch("notarius.infrastructure.llm.engine_adapter.LLMCacheRepository")
    @patch("notarius.infrastructure.llm.engine_adapter.GenerateLLMPrediction")
    def test_predict_with_cache_invalidation(
        self,
        mock_use_case_class: MagicMock,
        mock_cache_repo_class: MagicMock,
        mock_prompt_service_class: MagicMock,
        mock_prompt_renderer_class: MagicMock,
        mock_engine_class: MagicMock,
        mock_factory: MagicMock,
        mock_config: DictConfig,
    ) -> None:
        """Test prediction with cache invalidation flag."""
        mock_provider = MagicMock()
        mock_cache = MagicMock()
        mock_factory.return_value = (mock_provider, mock_cache)

        mock_use_case = MagicMock()
        mock_use_case_class.return_value = mock_use_case

        mock_response = GenerateLLMPredictionResponse(
            prediction={"result": "test"},
            messages="messages",
            from_cache=False,
        )
        mock_use_case.execute.return_value = mock_response

        adapter = LLMEngineAdapter(config=mock_config)

        # Execute prediction with invalidate_cache=True
        adapter.predict(text="test", invalidate_cache=True)

        # Verify flag was passed
        request = mock_use_case.execute.call_args[0][0]
        assert request.invalidate_cache is True

    @patch("notarius.infrastructure.llm.engine_adapter.llm_provider_factory")
    @patch("notarius.infrastructure.llm.engine_adapter.SimpleLLMEngine")
    @patch("notarius.infrastructure.llm.engine_adapter.Jinja2PromptRenderer")
    @patch("notarius.infrastructure.llm.engine_adapter.PromptConstructionService")
    @patch("notarius.infrastructure.llm.engine_adapter.LLMCacheRepository")
    @patch("notarius.infrastructure.llm.engine_adapter.GenerateLLMPrediction")
    def test_from_config_factory(
        self,
        mock_use_case_class: MagicMock,
        mock_cache_repo_class: MagicMock,
        mock_prompt_service_class: MagicMock,
        mock_prompt_renderer_class: MagicMock,
        mock_engine_class: MagicMock,
        mock_factory: MagicMock,
        mock_config: DictConfig,
    ) -> None:
        """Test from_config factory method."""
        mock_provider = MagicMock()
        mock_cache = MagicMock()
        mock_factory.return_value = (mock_provider, mock_cache)

        adapter = LLMEngineAdapter.from_config(mock_config)

        assert isinstance(adapter, LLMEngineAdapter)
        assert adapter.enable_cache is True

    @patch("notarius.infrastructure.llm.engine_adapter.llm_provider_factory")
    @patch("notarius.infrastructure.llm.engine_adapter.SimpleLLMEngine")
    @patch("notarius.infrastructure.llm.engine_adapter.Jinja2PromptRenderer")
    @patch("notarius.infrastructure.llm.engine_adapter.PromptConstructionService")
    @patch("notarius.infrastructure.llm.engine_adapter.LLMCacheRepository")
    @patch("notarius.infrastructure.llm.engine_adapter.GenerateLLMPrediction")
    def test_init_creates_components_correctly(
        self,
        mock_use_case_class: MagicMock,
        mock_cache_repo_class: MagicMock,
        mock_prompt_service_class: MagicMock,
        mock_prompt_renderer_class: MagicMock,
        mock_engine_class: MagicMock,
        mock_factory: MagicMock,
        mock_config: DictConfig,
    ) -> None:
        """Test that initialization creates all components correctly."""
        mock_provider = MagicMock()
        mock_cache = MagicMock()
        mock_factory.return_value = (mock_provider, mock_cache)

        mock_engine_instance = MagicMock()
        mock_prompt_service_instance = MagicMock()
        mock_cache_repo_instance = MagicMock()

        mock_engine_class.return_value = mock_engine_instance
        mock_prompt_service_class.return_value = mock_prompt_service_instance
        mock_cache_repo_class.return_value = mock_cache_repo_instance

        adapter = LLMEngineAdapter(config=mock_config)

        # Verify use case was created with correct dependencies
        mock_use_case_class.assert_called_once_with(
            engine=mock_engine_instance,
            cache_repository=mock_cache_repo_instance,
            prompt_service=mock_prompt_service_instance,
        )
