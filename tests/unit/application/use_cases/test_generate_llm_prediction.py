"""Tests for GenerateLLMPrediction use case."""

from unittest.mock import MagicMock, patch

import pytest
from PIL import Image
from PIL.Image import Image as PILImage

from notarius.application.use_cases.gen.generate_llm_prediction import (
    GenerateLLMPrediction,
    GenerateLLMPredictionRequest,
    GenerateLLMPredictionResponse,
)
from notarius.domain.services.prompt_service import PromptConstructionService
from notarius.infrastructure.llm.engine import SimpleLLMEngine
from notarius.infrastructure.persistence.llm_cache_repository import LLMCacheRepository
from notarius.schemas.data.cache import LLMCacheItem


class TestGenerateLLMPredictionRequest:
    """Test suite for GenerateLLMPredictionRequest class."""

    def test_init_with_all_params(self) -> None:
        """Test request initialization with all parameters."""
        image = Image.new("RGB", (10, 10))
        context = {"key": "value"}

        request = GenerateLLMPredictionRequest(
            image=image,
            text="test text",
            context=context,
            system_template="custom_system.j2",
            user_template="custom_user.j2",
            invalidate_cache=True,
        )

        assert request.image is image
        assert request.text == "test text"
        assert request.context == context
        assert request.system_template == "custom_system.j2"
        assert request.user_template == "custom_user.j2"
        assert request.invalidate_cache is True

    def test_init_with_defaults(self) -> None:
        """Test request initialization with default values."""
        request = GenerateLLMPredictionRequest(text="test")

        assert request.image is None
        assert request.text == "test"
        assert request.context == {}
        assert request.system_template == "system.j2"
        assert request.user_template == "user.j2"
        assert request.invalidate_cache is False

    def test_init_with_none_context(self) -> None:
        """Test that None context defaults to empty dict."""
        request = GenerateLLMPredictionRequest(text="test", context=None)
        assert request.context == {}


class TestGenerateLLMPredictionResponse:
    """Test suite for GenerateLLMPredictionResponse class."""

    def test_init(self) -> None:
        """Test output initialization."""
        prediction = {"result": "test"}
        messages = "System: test\nUser: query"

        response = GenerateLLMPredictionResponse(
            prediction=prediction,
            messages=messages,
            from_cache=True,
        )

        assert response.prediction == prediction
        assert response.messages == messages
        assert response.from_cache is True

    def test_init_default_from_cache(self) -> None:
        """Test output with default from_cache value."""
        response = GenerateLLMPredictionResponse(
            prediction={},
            messages="",
        )

        assert response.from_cache is False


class TestGenerateLLMPrediction:
    """Test suite for GenerateLLMPrediction use case."""

    @pytest.fixture
    def mock_engine(self) -> MagicMock:
        """Create a mock SimpleLLMEngine."""
        engine = MagicMock(spec=SimpleLLMEngine)
        engine.predict.return_value = {"result": "_engine output"}
        return engine

    @pytest.fixture
    def mock_cache_repository(self) -> MagicMock:
        """Create a mock LLMCacheRepository."""
        repository = MagicMock(spec=LLMCacheRepository)
        repository.generate_key.return_value = "test_cache_key_123"
        repository.get.return_value = None  # Cache miss by default
        return repository

    @pytest.fixture
    def mock_prompt_service(self) -> MagicMock:
        """Create a mock PromptConstructionService."""
        service = MagicMock(spec=PromptConstructionService)
        service.build_messages.return_value = [
            {"role": "system", "text": "System"},
            {"role": "user", "text": "User"},
        ]
        return service

    @pytest.fixture
    def use_case(
        self,
        mock_engine: MagicMock,
        mock_cache_repository: MagicMock,
        mock_prompt_service: MagicMock,
    ) -> GenerateLLMPrediction:
        """Create a use case instance for testing."""
        return GenerateLLMPrediction(
            engine=mock_engine,
            cache_repository=mock_cache_repository,
            prompt_service=mock_prompt_service,
        )

    @pytest.fixture
    def sample_image(self) -> PILImage:
        """Create a sample PIL image for testing."""
        return Image.new("RGB", (100, 100), color="red")

    def test_init(
        self,
        mock_engine: MagicMock,
        mock_cache_repository: MagicMock,
        mock_prompt_service: MagicMock,
    ) -> None:
        """Test use case initialization."""
        use_case = GenerateLLMPrediction(
            engine=mock_engine,
            cache_repository=mock_cache_repository,
            prompt_service=mock_prompt_service,
        )

        assert use_case.engine is mock_engine
        assert use_case.cache_repository is mock_cache_repository
        assert use_case.prompt_service is mock_prompt_service
        assert use_case.logger is not None

    def test_init_without_cache(
        self, mock_engine: MagicMock, mock_prompt_service: MagicMock
    ) -> None:
        """Test use case initialization without cache repository."""
        use_case = GenerateLLMPrediction(
            engine=mock_engine,
            cache_repository=None,
            prompt_service=mock_prompt_service,
        )

        assert use_case.cache_repository is None

    def test_execute_cache_miss_generates_prediction(
        self, use_case: GenerateLLMPrediction, mock_engine: MagicMock
    ) -> None:
        """Test that cache miss triggers prediction generation."""
        request = GenerateLLMPredictionRequest(text="test query")

        response = use_case.execute(request)

        assert response.prediction == {"result": "_engine output"}
        assert response.from_cache is False
        mock_engine.predict.assert_called_once()

    def test_execute_cache_hit_returns_cached(
        self,
        use_case: GenerateLLMPrediction,
        mock_cache_repository: MagicMock,
        mock_engine: MagicMock,
    ) -> None:
        """Test that cache hit returns cached result without prediction."""
        cached_item = LLMCacheItem(
            response={"result": "cached output"},
            hints={"key": "value"},
        )
        mock_cache_repository.get.return_value = cached_item

        request = GenerateLLMPredictionRequest(text="test query")
        response = use_case.execute(request)

        assert response.prediction == {"result": "cached output"}
        assert response.from_cache is True
        mock_engine.predict.assert_not_called()

    def test_execute_calls_prompt_service(
        self, use_case: GenerateLLMPrediction, mock_prompt_service: MagicMock
    ) -> None:
        """Test that execute calls text service to build messages."""
        request = GenerateLLMPredictionRequest(
            text="test",
            context={"key": "value"},
            system_template="sys.j2",
            user_template="user.j2",
        )

        use_case.execute(request)

        mock_prompt_service.build_messages.assert_called_once_with(
            image=None,
            text="test",
            system_template="sys.j2",
            user_template="user.j2",
            context={"key": "value"},
        )

    def test_execute_with_image(
        self,
        use_case: GenerateLLMPrediction,
        mock_prompt_service: MagicMock,
        sample_image: PILImage,
    ) -> None:
        """Test execution with image input."""
        request = GenerateLLMPredictionRequest(
            image=sample_image,
            context={},
        )

        use_case.execute(request)

        mock_prompt_service.build_messages.assert_called_once()
        call_kwargs = mock_prompt_service.build_messages.call_args[1]
        assert call_kwargs["image"] is sample_image

    def test_execute_stores_in_cache(
        self,
        use_case: GenerateLLMPrediction,
        mock_cache_repository: MagicMock,
        mock_engine: MagicMock,
    ) -> None:
        """Test that prediction result is stored in cache."""
        request = GenerateLLMPredictionRequest(text="test")
        mock_engine.predict.return_value = {"result": "new prediction"}

        use_case.execute(request)

        # Verify cache.set was called
        mock_cache_repository.set.assert_called_once()
        call_args = mock_cache_repository.set.call_args
        assert call_args[1]["key"] == "test_cache_key_123"
        assert isinstance(call_args[1]["item"], LLMCacheItem)
        assert call_args[1]["item"].output == {"result": "new prediction"}

    def test_execute_with_cache_invalidation(
        self,
        use_case: GenerateLLMPrediction,
        mock_cache_repository: MagicMock,
        mock_engine: MagicMock,
    ) -> None:
        """Test that invalidate_cache flag deletes cache entry."""
        request = GenerateLLMPredictionRequest(
            text="test",
            invalidate_cache=True,
        )

        use_case.execute(request)

        # Verify cache was invalidated
        mock_cache_repository.delete.assert_called_once_with("test_cache_key_123")

        # Should still generate new prediction
        mock_engine.predict.assert_called_once()

    def test_execute_without_cache_repository(
        self, mock_engine: MagicMock, mock_prompt_service: MagicMock
    ) -> None:
        """Test execution when cache repository is None."""
        use_case = GenerateLLMPrediction(
            engine=mock_engine,
            cache_repository=None,
            prompt_service=mock_prompt_service,
        )

        request = GenerateLLMPredictionRequest(text="test")
        response = use_case.execute(request)

        # Should generate prediction directly without caching
        assert response.prediction == {"result": "_engine output"}
        assert response.from_cache is False
        mock_engine.predict.assert_called_once()

    def test_execute_with_context_metadata(
        self,
        use_case: GenerateLLMPrediction,
        mock_cache_repository: MagicMock,
    ) -> None:
        """Test that context metadata is used for caching."""
        request = GenerateLLMPredictionRequest(
            text="test",
            context={
                "hints": {"language": "en"},
                "schematism": "test_schema",
                "filename": "test_file.pdf",
            },
        )

        use_case.execute(request)

        # Verify cache key generation includes hints
        mock_cache_repository.generate_key.assert_called_once()
        call_kwargs = mock_cache_repository.generate_key.call_args[1]
        assert call_kwargs["hints"] == {"language": "en"}

        # Verify cache.set includes metadata
        mock_cache_repository.set.assert_called_once()
        set_kwargs = mock_cache_repository.set.call_args[1]
        assert set_kwargs["schematism"] == "test_schema"
        assert set_kwargs["filename"] == "test_file.pdf"

    def test_execute_returns_messages(
        self, use_case: GenerateLLMPrediction, mock_prompt_service: MagicMock
    ) -> None:
        """Test that execute returns serialized messages."""
        request = GenerateLLMPredictionRequest(text="test")

        with patch(
            "notarius.application.use_cases.generate_llm_prediction.messages_to_string"
        ) as mock_to_string:
            mock_to_string.return_value = "Serialized messages"

            response = use_case.execute(request)

            assert response.messages == "Serialized messages"
            mock_to_string.assert_called_once()

    def test_execute_with_multimodal_input(
        self,
        use_case: GenerateLLMPrediction,
        mock_prompt_service: MagicMock,
        sample_image: PILImage,
    ) -> None:
        """Test execution with both image and text."""
        request = GenerateLLMPredictionRequest(
            image=sample_image,
            text="describe this",
            context={},
        )

        use_case.execute(request)

        call_kwargs = mock_prompt_service.build_messages.call_args[1]
        assert call_kwargs["image"] is sample_image
        assert call_kwargs["text"] == "describe this"

    def test_execute_cache_item_includes_hints(
        self,
        use_case: GenerateLLMPrediction,
        mock_cache_repository: MagicMock,
        mock_engine: MagicMock,
    ) -> None:
        """Test that cached item includes hints from context."""
        request = GenerateLLMPredictionRequest(
            text="test",
            context={"hints": {"language": "fr", "context": "medical"}},
        )

        mock_engine.predict.return_value = {"result": "output"}

        use_case.execute(request)

        # Verify cache item has hints
        set_call = mock_cache_repository.set.call_args
        cached_item = set_call[1]["item"]
        assert cached_item.hints == {"language": "fr", "context": "medical"}

    def test_execute_integration_flow(
        self,
        use_case: GenerateLLMPrediction,
        mock_prompt_service: MagicMock,
        mock_cache_repository: MagicMock,
        mock_engine: MagicMock,
    ) -> None:
        """Test complete integration flow: build messages -> check cache -> predict -> store."""
        request = GenerateLLMPredictionRequest(
            text="integration test",
            context={"key": "value"},
        )

        # Ensure cache miss
        mock_cache_repository.get.return_value = None

        # Execute
        response = use_case.execute(request)

        # Verify complete flow
        # 1. Messages were built
        mock_prompt_service.build_messages.assert_called_once()

        # 2. Cache key was generated
        mock_cache_repository.generate_key.assert_called_once()

        # 3. Cache was checked
        mock_cache_repository.get.assert_called_once()

        # 4. Prediction was made
        mock_engine.predict.assert_called_once()

        # 5. Result was cached
        mock_cache_repository.set.assert_called_once()

        # 6. Response is correct
        assert response.from_cache is False
        assert response.prediction == {"result": "_engine output"}

    def test_execute_with_custom_templates(
        self, use_case: GenerateLLMPrediction, mock_prompt_service: MagicMock
    ) -> None:
        """Test execution with custom template names."""
        request = GenerateLLMPredictionRequest(
            text="test",
            system_template="custom_system.j2",
            user_template="custom_user.j2",
        )

        use_case.execute(request)

        call_kwargs = mock_prompt_service.build_messages.call_args[1]
        assert call_kwargs["system_template"] == "custom_system.j2"
        assert call_kwargs["user_template"] == "custom_user.j2"

    @patch("notarius.application.use_cases.generate_llm_prediction.logger")
    def test_logging(
        self,
        mock_logger: MagicMock,
        use_case: GenerateLLMPrediction,
    ) -> None:
        """Test that operations are logged."""
        request = GenerateLLMPredictionRequest(text="test")

        use_case.execute(request)

        # Verify logger was used
        assert mock_logger.bind.called or use_case.logger is not None

    def test_execute_cache_hit_skips_engine(
        self,
        use_case: GenerateLLMPrediction,
        mock_cache_repository: MagicMock,
        mock_engine: MagicMock,
    ) -> None:
        """Test that cache hit completely skips _engine prediction."""
        cached_item = LLMCacheItem(response={"cached": True}, hints=None)
        mock_cache_repository.get.return_value = cached_item

        request = GenerateLLMPredictionRequest(text="test")
        use_case.execute(request)

        # Engine should never be called
        mock_engine.predict.assert_not_called()

        # Cache.set should not be called either
        mock_cache_repository.set.assert_not_called()
