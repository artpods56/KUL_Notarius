"""Integration tests for LLM _engine components.

Tests the complete LLM pipeline:
- SimpleLLMEngine
- LLMCacheRepository
- PromptConstructionService
- GenerateLLMPrediction use case
- LLMEngineAdapter

These tests use real components (not mocks) to verify end-to-end functionality.
"""

import tempfile
from pathlib import Path
from typing import Any

import pytest
from PIL import Image
from omegaconf import DictConfig, OmegaConf

from notarius.infrastructure.llm.engine import SimpleLLMEngine
from notarius.infrastructure.llm.engine_adapter import LLMEngineAdapter
from notarius.infrastructure.llm.prompt_manager import Jinja2PromptRenderer
from notarius.infrastructure.persistence.llm_cache_repository import LLMCacheRepository
from notarius.domain.services.prompt_service import PromptConstructionService
from notarius.application.use_cases.gen.generate_llm_prediction import (
    GenerateLLMPrediction,
    GenerateLLMPredictionRequest,
)
from notarius.schemas.data.cache import LLMCacheItem


@pytest.fixture
def temp_cache_dir() -> Path:
    """Create temporary directory for cache."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def temp_template_dir() -> Path:
    """Create temporary directory for templates."""
    with tempfile.TemporaryDirectory() as tmpdir:
        template_path = Path(tmpdir)

        # Create simple test templates
        (template_path / "system.j2").write_text(
            "You are a helpful assistant. Extract entities from documents."
        )
        (template_path / "user.j2").write_text(
            "{% if text %}Text: {{ text }}{% endif %}"
            "{% if hints %}\nHints: {{ hints }}{% endif %}"
        )

        yield template_path


@pytest.fixture
def test_image() -> Image.Image:
    """Create a simple test image."""
    return Image.new("RGB", (300, 200), color="white")


@pytest.fixture
def llm_config(temp_cache_dir: Path, temp_template_dir: Path) -> DictConfig:
    """Create LLM configuration with mock provider."""
    return OmegaConf.create(
        {
            "provider": {
                "provider": "mock",
                "model": "mock-model",
                "temperature": 0.0,
                "max_tokens": 1000,
                "retries": 3,
                "enable_cache": True,
                "caches_dir": str(temp_cache_dir),
                "template_dir": str(temp_template_dir),
            }
        }
    )


class MockLLMProvider:
    """Mock LLM provider for testing."""

    def __init__(self, model: str = "mock-model"):
        self.model = model
        self.call_count = 0

    def generate(
        self,
        messages: list[dict[str, Any]],
        temperature: float = 0.0,
        max_tokens: int = 1000,
    ) -> str:
        """Generate mock output."""
        self.call_count += 1
        # Return valid JSON output
        return '{"name": ["John Smith"], "location": ["Warsaw"]}'


class TestSimpleLLMEngineIntegration:
    """Integration tests for SimpleLLMEngine with mock provider."""

    def test_engine_initialization(self) -> None:
        """Test that _engine initializes correctly."""
        provider = MockLLMProvider()
        engine = SimpleLLMEngine(provider=provider, retries=3)

        assert engine.provider is not None
        assert engine.retries == 3

    def test_generate_with_text_messages(self) -> None:
        """Test generation with text-only messages."""
        provider = MockLLMProvider()
        engine = SimpleLLMEngine(provider=provider, retries=3)

        messages = [
            {"role": "system", "text": "You are helpful."},
            {"role": "user", "text": "Extract entities."},
        ]

        response = engine.generate(messages, temperature=0.0, max_tokens=500)

        assert isinstance(response, str)
        assert len(response) > 0
        assert provider.call_count == 1

    def test_generate_with_multimodal_messages(self, test_image: Image.Image) -> None:
        """Test generation with image text."""
        provider = MockLLMProvider()
        engine = SimpleLLMEngine(provider=provider, retries=3)

        messages = [
            {"role": "system", "text": "You are helpful."},
            {
                "role": "user",
                "text": [
                    {"type": "image", "image": test_image},
                    {"type": "text", "text": "Describe this image."},
                ],
            },
        ]

        response = engine.generate(messages, temperature=0.0, max_tokens=500)

        assert isinstance(response, str)
        assert len(response) > 0

    def test_retry_mechanism(self) -> None:
        """Test that retry mechanism works on failures."""
        from unittest.mock import MagicMock

        # Mock provider that fails twice then succeeds
        provider = MagicMock()
        provider.model = "mock-model"
        call_count = [0]

        def mock_generate(*args: Any, **kwargs: Any) -> str:
            call_count[0] += 1
            if call_count[0] < 3:
                raise Exception("Temporary failure")
            return '{"name": ["Success"]}'

        provider.generate = mock_generate

        engine = SimpleLLMEngine(provider=provider, retries=5)

        messages = [{"role": "user", "text": "Test"}]
        response = engine.generate(messages)

        assert response == '{"name": ["Success"]}'
        assert call_count[0] == 3  # Failed twice, succeeded third time


class TestLLMCacheIntegration:
    """Integration tests for LLM caching with real cache backend."""

    def test_cache_persistence_across_instances(
        self,
        test_image: Image.Image,
        temp_cache_dir: Path,
    ) -> None:
        """Test that cache persists across repository instances."""
        # First instance - store data
        repo1 = LLMCacheRepository.create(
            model="test-model",
            caches_dir=temp_cache_dir,
        )

        messages = [
            {"role": "system", "text": "System text"},
            {"role": "user", "text": "User text"},
        ]
        key = repo1.generate_key(messages=messages, image=test_image)

        item = LLMCacheItem(
            prediction='{"name": ["Cached Result"]}',
            temperature=0.0,
            max_tokens=1000,
        )
        repo1.set(key, item, schematism="test_schematism")

        # Second instance - retrieve data
        repo2 = LLMCacheRepository.create(
            model="test-model",
            caches_dir=temp_cache_dir,
        )

        retrieved = repo2.get(key)

        assert retrieved is not None
        assert retrieved.prediction == '{"name": ["Cached Result"]}'
        assert retrieved.temperature == 0.0

    def test_cache_invalidation(
        self,
        test_image: Image.Image,
        temp_cache_dir: Path,
    ) -> None:
        """Test cache invalidation removes entries."""
        repo = LLMCacheRepository.create(
            model="test-model",
            caches_dir=temp_cache_dir,
        )

        messages = [{"role": "user", "text": "Test"}]
        key = repo.generate_key(messages=messages, image=test_image)

        item = LLMCacheItem(
            prediction='{"test": "data"}',
            temperature=0.0,
            max_tokens=1000,
        )
        repo.set(key, item)

        # Verify exists
        assert repo.get(key) is not None

        # Invalidate
        repo.invalidate(key)

        # Verify removed
        assert repo.get(key) is None

    def test_different_keys_for_different_messages(
        self,
        temp_cache_dir: Path,
    ) -> None:
        """Test that different messages produce different cache keys."""
        repo = LLMCacheRepository.create(
            model="test-model",
            caches_dir=temp_cache_dir,
        )

        messages1 = [{"role": "user", "text": "Prompt 1"}]
        messages2 = [{"role": "user", "text": "Prompt 2"}]

        key1 = repo.generate_key(messages=messages1)
        key2 = repo.generate_key(messages=messages2)

        assert key1 != key2


class TestPromptConstructionServiceIntegration:
    """Integration tests for text construction with real templates."""

    def test_construct_messages_text_only(
        self,
        temp_template_dir: Path,
    ) -> None:
        """Test constructing messages with text only."""
        renderer = Jinja2PromptRenderer(template_dir=temp_template_dir)
        service = PromptConstructionService(prompt_renderer=renderer)

        messages = service.construct_messages(
            text="Sample OCR text",
            context={},
            system_template="system.j2",
            user_template="user.j2",
        )

        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        assert "Sample OCR text" in messages[1]["text"]

    def test_construct_messages_with_hints(
        self,
        temp_template_dir: Path,
    ) -> None:
        """Test constructing messages with context hints."""
        renderer = Jinja2PromptRenderer(template_dir=temp_template_dir)
        service = PromptConstructionService(prompt_renderer=renderer)

        hints = {"name": ["John"], "location": ["Warsaw"]}

        messages = service.construct_messages(
            text="Sample text",
            context={"hints": hints},
            system_template="system.j2",
            user_template="user.j2",
        )

        assert len(messages) == 2
        user_content = messages[1]["text"]
        assert "Sample text" in user_content
        assert "hints" in user_content.lower() or str(hints) in user_content

    def test_construct_messages_multimodal(
        self,
        temp_template_dir: Path,
        test_image: Image.Image,
    ) -> None:
        """Test constructing multimodal messages with image."""
        renderer = Jinja2PromptRenderer(template_dir=temp_template_dir)
        service = PromptConstructionService(prompt_renderer=renderer)

        messages = service.construct_messages(
            image=test_image,
            text="Sample text",
            context={},
            system_template="system.j2",
            user_template="user.j2",
        )

        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"

        # User message should have multimodal text
        user_content = messages[1]["text"]
        assert isinstance(user_content, list)
        # Should have image and text parts
        has_image = any(
            isinstance(part, dict) and part.get("type") == "image"
            for part in user_content
        )
        assert has_image


class TestGenerateLLMPredictionIntegration:
    """Integration tests for GenerateLLMPrediction use case."""

    def test_full_prediction_flow_without_cache(
        self,
        test_image: Image.Image,
        temp_template_dir: Path,
    ) -> None:
        """Test complete prediction flow without caching."""
        provider = MockLLMProvider()
        engine = SimpleLLMEngine(provider=provider, retries=3)

        renderer = Jinja2PromptRenderer(template_dir=temp_template_dir)
        prompt_service = PromptConstructionService(prompt_renderer=renderer)

        use_case = GenerateLLMPrediction(
            engine=engine,
            cache_repository=None,
            prompt_service=prompt_service,
        )

        request = GenerateLLMPredictionRequest(
            text="Sample OCR text",
            context={},
            system_template="system.j2",
            user_template="user.j2",
        )

        response = use_case.execute(request)

        assert response.cache_hit is False
        assert isinstance(response.prediction, dict)
        assert provider.call_count == 1

    def test_full_prediction_flow_with_cache(
        self,
        test_image: Image.Image,
        temp_template_dir: Path,
        temp_cache_dir: Path,
    ) -> None:
        """Test complete prediction flow with caching."""
        provider = MockLLMProvider()
        engine = SimpleLLMEngine(provider=provider, retries=3)

        renderer = Jinja2PromptRenderer(template_dir=temp_template_dir)
        prompt_service = PromptConstructionService(prompt_renderer=renderer)

        cache_repo = LLMCacheRepository.create(
            model="mock-model",
            caches_dir=temp_cache_dir,
        )

        use_case = GenerateLLMPrediction(
            engine=engine,
            cache_repository=cache_repo,
            prompt_service=prompt_service,
        )

        # First request - cache miss
        request1 = GenerateLLMPredictionRequest(
            text="Sample OCR text",
            context={},
            system_template="system.j2",
            user_template="user.j2",
        )
        response1 = use_case.execute(request1)

        assert response1.cache_hit is False
        assert provider.call_count == 1

        # Second request - cache hit
        request2 = GenerateLLMPredictionRequest(
            text="Sample OCR text",
            context={},
            system_template="system.j2",
            user_template="user.j2",
        )
        response2 = use_case.execute(request2)

        assert response2.cache_hit is True
        assert provider.call_count == 1  # Should not increase

        # Predictions should match
        assert response1.prediction == response2.prediction

    def test_multimodal_prediction(
        self,
        test_image: Image.Image,
        temp_template_dir: Path,
    ) -> None:
        """Test prediction with both image and text."""
        provider = MockLLMProvider()
        engine = SimpleLLMEngine(provider=provider, retries=3)

        renderer = Jinja2PromptRenderer(template_dir=temp_template_dir)
        prompt_service = PromptConstructionService(prompt_renderer=renderer)

        use_case = GenerateLLMPrediction(
            engine=engine,
            cache_repository=None,
            prompt_service=prompt_service,
        )

        request = GenerateLLMPredictionRequest(
            image=test_image,
            text="Sample OCR text",
            context={},
            system_template="system.j2",
            user_template="user.j2",
        )

        response = use_case.execute(request)

        assert isinstance(response.prediction, dict)

    def test_prediction_with_hints(
        self,
        temp_template_dir: Path,
    ) -> None:
        """Test prediction with LMv3 hints in context."""
        provider = MockLLMProvider()
        engine = SimpleLLMEngine(provider=provider, retries=3)

        renderer = Jinja2PromptRenderer(template_dir=temp_template_dir)
        prompt_service = PromptConstructionService(prompt_renderer=renderer)

        use_case = GenerateLLMPrediction(
            engine=engine,
            cache_repository=None,
            prompt_service=prompt_service,
        )

        hints = {"name": ["John Smith"], "location": ["Warsaw"]}

        request = GenerateLLMPredictionRequest(
            text="Sample text",
            context={"hints": hints},
            system_template="system.j2",
            user_template="user.j2",
        )

        response = use_case.execute(request)

        assert isinstance(response.prediction, dict)


class TestLLMEngineAdapterIntegration:
    """Integration tests for LLMEngineAdapter using real components."""

    @pytest.fixture
    def mock_llm_adapter(
        self,
        temp_cache_dir: Path,
        temp_template_dir: Path,
    ) -> LLMEngineAdapter:
        """Create LLM adapter with mocked provider."""
        from unittest.mock import patch

        # Mock the provider factory
        with patch(
            "notarius.infrastructure.llm.engine_adapter.llm_provider_factory"
        ) as mock_factory:
            from notarius.infrastructure.cache.llm_cache import LLMCache

            provider = MockLLMProvider()
            cache = LLMCache(model="mock-model", caches_dir=temp_cache_dir)
            mock_factory.return_value = (provider, cache)

            config = OmegaConf.create(
                {
                    "provider": {
                        "provider": "mock",
                        "model": "mock-model",
                        "temperature": 0.0,
                        "max_tokens": 1000,
                        "retries": 3,
                        "enable_cache": True,
                        "template_dir": str(temp_template_dir),
                    }
                }
            )

            adapter = LLMEngineAdapter(config=config, enable_cache=True)
            adapter._mock_provider = provider

            return adapter

    def test_adapter_predict_text_only(
        self,
        mock_llm_adapter: LLMEngineAdapter,
    ) -> None:
        """Test adapter predict with text only."""
        result = mock_llm_adapter.predict(text="Sample OCR text")

        assert isinstance(result, dict)

    def test_adapter_predict_multimodal(
        self,
        mock_llm_adapter: LLMEngineAdapter,
        test_image: Image.Image,
    ) -> None:
        """Test adapter predict with image and text."""
        result = mock_llm_adapter.predict(
            image=test_image,
            text="Sample OCR text",
        )

        assert isinstance(result, dict)

    def test_adapter_predict_with_hints(
        self,
        mock_llm_adapter: LLMEngineAdapter,
    ) -> None:
        """Test adapter predict with context hints."""
        hints = {"name": ["John"], "location": ["Warsaw"]}

        result = mock_llm_adapter.predict(
            text="Sample text",
            context={"hints": hints},
        )

        assert isinstance(result, dict)

    def test_adapter_caching_behavior(
        self,
        mock_llm_adapter: LLMEngineAdapter,
    ) -> None:
        """Test that adapter properly caches results."""
        # First call
        result1 = mock_llm_adapter.predict(text="Sample text")

        # Second call - should use cache
        result2 = mock_llm_adapter.predict(text="Sample text")

        # Results should be identical
        assert result1 == result2

        # Provider should only be called once
        assert mock_llm_adapter._mock_provider.call_count == 1

    def test_adapter_invalidate_cache(
        self,
        mock_llm_adapter: LLMEngineAdapter,
    ) -> None:
        """Test adapter invalidate_cache parameter."""
        # First call
        mock_llm_adapter.predict(text="Sample text")
        assert mock_llm_adapter._mock_provider.call_count == 1

        # Second call with invalidate - should call provider again
        mock_llm_adapter.predict(text="Sample text", invalidate_cache=True)
        assert mock_llm_adapter._mock_provider.call_count == 2


class TestLLMEndToEndIntegration:
    """End-to-end integration tests spanning all LLM components."""

    def test_complete_pipeline_text_only(
        self,
        temp_cache_dir: Path,
        temp_template_dir: Path,
    ) -> None:
        """Test complete pipeline with text-only input."""
        from unittest.mock import patch

        with patch(
            "notarius.infrastructure.llm.engine_adapter.llm_provider_factory"
        ) as mock_factory:
            from notarius.infrastructure.cache.llm_cache import LLMCache

            provider = MockLLMProvider()
            cache = LLMCache(model="mock-model", caches_dir=temp_cache_dir)
            mock_factory.return_value = (provider, cache)

            config = OmegaConf.create(
                {
                    "provider": {
                        "provider": "mock",
                        "model": "mock-model",
                        "retries": 3,
                        "enable_cache": True,
                        "template_dir": str(temp_template_dir),
                    }
                }
            )

            adapter = LLMEngineAdapter.from_config(config)

            # Execute prediction
            result = adapter.predict(text="Extract entities from this text.")

            assert isinstance(result, dict)
            assert "name" in result or "location" in result

    def test_complete_pipeline_multimodal_with_hints(
        self,
        test_image: Image.Image,
        temp_cache_dir: Path,
        temp_template_dir: Path,
    ) -> None:
        """Test complete pipeline with image, text, and hints."""
        from unittest.mock import patch

        with patch(
            "notarius.infrastructure.llm.engine_adapter.llm_provider_factory"
        ) as mock_factory:
            from notarius.infrastructure.cache.llm_cache import LLMCache

            provider = MockLLMProvider()
            cache = LLMCache(model="mock-model", caches_dir=temp_cache_dir)
            mock_factory.return_value = (provider, cache)

            config = OmegaConf.create(
                {
                    "provider": {
                        "provider": "mock",
                        "model": "mock-model",
                        "retries": 3,
                        "enable_cache": True,
                        "template_dir": str(temp_template_dir),
                    }
                }
            )

            adapter = LLMEngineAdapter.from_config(config)

            # Execute with all features
            hints = {"name": ["Hint Name"], "location": ["Hint Location"]}
            result = adapter.predict(
                image=test_image,
                text="OCR text text",
                context={"hints": hints},
            )

            assert isinstance(result, dict)

    def test_adapter_integrates_all_components(
        self,
        temp_cache_dir: Path,
        temp_template_dir: Path,
    ) -> None:
        """Test that adapter properly integrates all components."""
        from unittest.mock import patch

        with patch(
            "notarius.infrastructure.llm.engine_adapter.llm_provider_factory"
        ) as mock_factory:
            from notarius.infrastructure.cache.llm_cache import LLMCache

            provider = MockLLMProvider()
            cache = LLMCache(model="mock-model", caches_dir=temp_cache_dir)
            mock_factory.return_value = (provider, cache)

            config = OmegaConf.create(
                {
                    "provider": {
                        "provider": "mock",
                        "model": "mock-model",
                        "retries": 3,
                        "enable_cache": True,
                        "template_dir": str(temp_template_dir),
                    }
                }
            )

            adapter = LLMEngineAdapter.from_config(config)

            # Verify all components exist
            assert adapter.llm_engine is not None
            assert isinstance(adapter.llm_engine, SimpleLLMEngine)
            assert adapter.cache_repository is not None
            assert isinstance(adapter.cache_repository, LLMCacheRepository)
            assert adapter.prompt_service is not None
            assert isinstance(adapter.prompt_service, PromptConstructionService)
            assert adapter.use_case is not None
            assert isinstance(adapter.use_case, GenerateLLMPrediction)
