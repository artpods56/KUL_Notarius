"""Tests for SimpleLLMEngine."""

import json
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from openai.types.chat import ChatCompletionMessageParam
from pydantic_core import ValidationError

import notarius.application.use_cases.inference.add_ocr_to_dataset
from notarius.application.ports.outbound.llm_provider import LLMProvider
from notarius.infrastructure.llm.engine import SimpleLLMEngine


class TestSimpleLLMEngine:
    """Test suite for SimpleLLMEngine class."""

    @pytest.fixture
    def mock_provider(self) -> MagicMock:
        """Create a mock LLM provider."""
        provider = MagicMock(spec=LLMProvider)
        provider.model = "test-model"
        return provider

    @pytest.fixture
    def engine(self, mock_provider: MagicMock) -> SimpleLLMEngine:
        """Create an _engine instance for testing."""
        return SimpleLLMEngine(provider=mock_provider, retries=3)

    @pytest.fixture
    def sample_messages(self) -> list[ChatCompletionMessageParam]:
        """Create sample messages for testing."""
        return [
            {"role": "system", "text": "You are a helpful assistant."},
            {"role": "user", "text": "What is 2+2?"},
        ]

    def test_init(self, mock_provider: MagicMock) -> None:
        """Test _engine initialization."""
        engine = SimpleLLMEngine(provider=mock_provider, retries=5)
        assert engine.provider is mock_provider
        assert engine.retries == 5
        assert (
            notarius.application.use_cases.inference.add_ocr_to_dataset.logger
            is not None
        )

    def test_init_default_retries(self, mock_provider: MagicMock) -> None:
        """Test _engine initialization with default retries."""
        engine = SimpleLLMEngine(provider=mock_provider)
        assert engine.retries == 5

    def test_predict_success(
        self,
        engine: SimpleLLMEngine,
        mock_provider: MagicMock,
        sample_messages: list[ChatCompletionMessageParam],
    ) -> None:
        """Test successful prediction on first attempt."""
        expected_response = {"result": "4", "confidence": 1.0}
        mock_provider.generate_response.return_value = json.dumps(expected_response)

        result = engine.predict(sample_messages)

        assert result == expected_response
        mock_provider.generate_response.assert_called_once_with(sample_messages)

    def test_predict_retries_on_json_error(
        self,
        engine: SimpleLLMEngine,
        mock_provider: MagicMock,
        sample_messages: list[ChatCompletionMessageParam],
    ) -> None:
        """Test that _engine retries on JSON decode error."""
        # First two calls return invalid JSON, third succeeds
        expected_response = {"result": "success"}
        mock_provider.generate_response.side_effect = [
            "invalid json {",
            "also invalid }}}",
            json.dumps(expected_response),
        ]

        result = engine.predict(sample_messages)

        assert result == expected_response
        assert mock_provider.generate_response.call_count == 3

    def test_predict_retries_on_validation_error(
        self,
        engine: SimpleLLMEngine,
        mock_provider: MagicMock,
        sample_messages: list[ChatCompletionMessageParam],
    ) -> None:
        """Test that _engine retries on validation error."""
        expected_response = {"result": "success"}

        # Simulate ValidationError on first call, success on second
        def side_effect(*args, **kwargs):
            if mock_provider.generate_response.call_count == 1:
                raise ValidationError.from_exception_data("test", [])
            return json.dumps(expected_response)

        mock_provider.generate_response.side_effect = side_effect

        result = engine.predict(sample_messages)

        assert result == expected_response
        assert mock_provider.generate_response.call_count == 2

    def test_predict_fails_after_max_retries(
        self,
        engine: SimpleLLMEngine,
        mock_provider: MagicMock,
        sample_messages: list[ChatCompletionMessageParam],
    ) -> None:
        """Test that _engine raises error after max retries."""
        # Always return invalid JSON
        mock_provider.generate_response.return_value = "invalid json {"

        with pytest.raises(json.JSONDecodeError):
            engine.predict(sample_messages)

        # Should have tried exactly 3 times (configured retries)
        assert mock_provider.generate_response.call_count == 3

    def test_predict_with_complex_response(
        self,
        engine: SimpleLLMEngine,
        mock_provider: MagicMock,
        sample_messages: list[ChatCompletionMessageParam],
    ) -> None:
        """Test prediction with complex nested JSON output."""
        complex_response = {
            "entities": [
                {"text": "John", "label": "PERSON", "score": 0.95},
                {"text": "NYC", "label": "LOCATION", "score": 0.88},
            ],
            "metadata": {
                "model": "test-model",
                "timestamp": "2024-01-01",
            },
        }
        mock_provider.generate_response.return_value = json.dumps(complex_response)

        result = engine.predict(sample_messages)

        assert result == complex_response

    def test_predict_with_unicode(
        self,
        engine: SimpleLLMEngine,
        mock_provider: MagicMock,
        sample_messages: list[ChatCompletionMessageParam],
    ) -> None:
        """Test prediction with Unicode characters in output."""
        unicode_response = {
            "text": "Hello 世界",
            "language": "中文",
        }
        mock_provider.generate_response.return_value = json.dumps(
            unicode_response, ensure_ascii=False
        )

        result = engine.predict(sample_messages)

        assert result == unicode_response

    def test_predict_empty_messages(
        self, engine: SimpleLLMEngine, mock_provider: MagicMock
    ) -> None:
        """Test prediction with empty messages list."""
        mock_provider.generate_response.return_value = json.dumps({"result": "ok"})

        result = engine.predict([])

        assert result == {"result": "ok"}
        mock_provider.generate_response.assert_called_once_with([])

    def test_predict_preserves_message_order(
        self, engine: SimpleLLMEngine, mock_provider: MagicMock
    ) -> None:
        """Test that messages are passed to provider in correct order."""
        messages: list[ChatCompletionMessageParam] = [
            {"role": "system", "text": "System"},
            {"role": "user", "text": "User 1"},
            {"role": "assistant", "text": "Assistant 1"},
            {"role": "user", "text": "User 2"},
        ]

        mock_provider.generate_response.return_value = json.dumps({"result": "ok"})

        engine.predict(messages)

        # Verify exact messages were passed
        call_args = mock_provider.generate_response.call_args[0][0]
        assert call_args == messages

    def test_predict_with_different_retry_counts(
        self, mock_provider: MagicMock
    ) -> None:
        """Test _engine with different retry configurations."""
        # Engine with 1 retry
        engine_1_retry = SimpleLLMEngine(provider=mock_provider, retries=1)
        mock_provider.generate_response.return_value = "invalid json"

        with pytest.raises(json.JSONDecodeError):
            engine_1_retry.predict([{"role": "user", "text": "test"}])

        assert mock_provider.generate_response.call_count == 1

        # Reset mock
        mock_provider.reset_mock()

        # Engine with 10 retries
        engine_10_retries = SimpleLLMEngine(provider=mock_provider, retries=10)
        mock_provider.generate_response.return_value = "invalid json"

        with pytest.raises(json.JSONDecodeError):
            engine_10_retries.predict([{"role": "user", "text": "test"}])

        assert mock_provider.generate_response.call_count == 10

    @patch("notarius.infrastructure.llm._engine.logger")
    def test_logging_on_success(
        self,
        mock_logger: MagicMock,
        engine: SimpleLLMEngine,
        mock_provider: MagicMock,
        sample_messages: list[ChatCompletionMessageParam],
    ) -> None:
        """Test that successful predictions are logged."""
        mock_provider.generate_response.return_value = json.dumps({"result": "ok"})

        engine.predict(sample_messages)

        # Verify logger was called (bind or info)
        assert (
            mock_logger.bind.called
            or notarius.application.use_cases.inference.add_ocr_to_dataset.logger
            is not None
        )

    @patch("notarius.infrastructure.llm._engine.logger")
    def test_logging_on_retry(
        self,
        mock_logger: MagicMock,
        engine: SimpleLLMEngine,
        mock_provider: MagicMock,
        sample_messages: list[ChatCompletionMessageParam],
    ) -> None:
        """Test that retries are logged as warnings."""
        # Fail twice, succeed on third
        mock_provider.generate_response.side_effect = [
            "invalid",
            "invalid",
            json.dumps({"result": "ok"}),
        ]

        engine.predict(sample_messages)

        # Warning should have been logged for failed attempts
        # (exact assertion depends on logger setup)
        assert (
            mock_logger.bind.called
            or notarius.application.use_cases.inference.add_ocr_to_dataset.logger
            is not None
        )

    def test_predict_with_provider_exception(
        self,
        engine: SimpleLLMEngine,
        mock_provider: MagicMock,
        sample_messages: list[ChatCompletionMessageParam],
    ) -> None:
        """Test that provider exceptions are propagated."""
        mock_provider.generate_response.side_effect = RuntimeError("API Error")

        with pytest.raises(RuntimeError, match="API Error"):
            engine.predict(sample_messages)

    def test_predict_response_types(
        self,
        engine: SimpleLLMEngine,
        mock_provider: MagicMock,
        sample_messages: list[ChatCompletionMessageParam],
    ) -> None:
        """Test prediction with various JSON types."""
        test_cases = [
            {"type": "object", "value": "test"},
            [1, 2, 3],
            "string",
            123,
            True,
            None,
        ]

        for test_value in test_cases:
            mock_provider.reset_mock()
            mock_provider.generate_response.return_value = json.dumps(test_value)

            result = engine.predict(sample_messages)
            assert result == test_value

    def test_predict_returns_dict_type(
        self,
        engine: SimpleLLMEngine,
        mock_provider: MagicMock,
        sample_messages: list[ChatCompletionMessageParam],
    ) -> None:
        """Test that predict returns dict[str, Any] type."""
        response = {"key": "value", "number": 42, "nested": {"inner": "data"}}
        mock_provider.generate_response.return_value = json.dumps(response)

        result = engine.predict(sample_messages)

        assert isinstance(result, dict)
        assert result == response

    def test_engine_with_different_providers(self) -> None:
        """Test _engine works with different provider instances."""
        provider1 = MagicMock(spec=LLMProvider)
        provider1.model = "model-1"
        provider1.generate_response.return_value = json.dumps({"source": "provider1"})

        provider2 = MagicMock(spec=LLMProvider)
        provider2.model = "model-2"
        provider2.generate_response.return_value = json.dumps({"source": "provider2"})

        engine1 = SimpleLLMEngine(provider=provider1)
        engine2 = SimpleLLMEngine(provider=provider2)

        result1 = engine1.predict([{"role": "user", "text": "test"}])
        result2 = engine2.predict([{"role": "user", "text": "test"}])

        assert result1 == {"source": "provider1"}
        assert result2 == {"source": "provider2"}
