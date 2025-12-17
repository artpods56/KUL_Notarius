"""Tests for LLM cache backend and key generator."""

from dataclasses import dataclass
from pathlib import Path

import pytest
from pydantic import BaseModel

from notarius.domain.entities.completions import BaseProviderResponse
from notarius.domain.entities.messages import ChatMessage, TextContent
from notarius.infrastructure.cache.adapters.llm import LLMCache
from notarius.infrastructure.cache.backends.llm import (
    LLMCacheBackend,
    LLMCacheKeyGenerator,
    create_llm_cache_backend,
)
from notarius.infrastructure.llm.conversation import Conversation
from notarius.infrastructure.llm.engine_adapter import (
    CompletionRequest,
    CompletionResult,
)


# Test fixtures


class SampleSchema(BaseModel):
    """Sample Pydantic schema for testing."""

    name: str
    value: int


@dataclass(frozen=True)
class MockProviderResponse(BaseProviderResponse[SampleSchema]):
    """Mock provider structured_response."""

    def to_string(self) -> str:
        return (
            self.structured_response.model_dump_json()
            if self.structured_response
            else ""
        )


@pytest.fixture
def tmp_cache_dir(tmp_path: Path) -> Path:
    """Create a temporary cache directory."""
    return tmp_path / "test_caches"


@pytest.fixture
def llm_cache(tmp_cache_dir: Path) -> LLMCache:
    """Create an LLMCache instance."""
    return LLMCache(model_name="test-model", caches_dir=tmp_cache_dir)


@pytest.fixture
def key_generator() -> LLMCacheKeyGenerator:
    """Create a key generator."""
    return LLMCacheKeyGenerator()


@pytest.fixture
def cache_backend(
    llm_cache: LLMCache, key_generator: LLMCacheKeyGenerator
) -> LLMCacheBackend:
    """Create a cache backend."""
    return LLMCacheBackend(cache=llm_cache, key_generator=key_generator)


@pytest.fixture
def sample_conversation() -> Conversation:
    """Create a sample conversation."""
    messages = [
        ChatMessage(role="system", content=[TextContent(text="You are helpful")]),
        ChatMessage(role="user", content=[TextContent(text="Hello")]),
    ]
    return Conversation.from_messages(messages)


@pytest.fixture
def sample_request(
    sample_conversation: Conversation,
) -> CompletionRequest[SampleSchema]:
    """Create a sample completion request."""
    return CompletionRequest[SampleSchema](
        input=sample_conversation,
        structured_output=SampleSchema,
    )


@pytest.fixture
def sample_result(sample_conversation: Conversation) -> CompletionResult[SampleSchema]:
    """Create a sample completion result."""
    schema = SampleSchema(name="test", value=42)
    response = MockProviderResponse(structured_response=schema, text_response=None)

    # Add assistant structured_response
    conv = sample_conversation.add(
        ChatMessage(role="assistant", content=[TextContent(text=response.to_string())])
    )

    return CompletionResult[SampleSchema](output=response, conversation=conv)


# Test cases


class TestLLMCacheKeyGenerator:
    """Test LLM cache key generator."""

    def test_generate_key_is_deterministic(
        self,
        key_generator: LLMCacheKeyGenerator,
        sample_request: CompletionRequest,
    ) -> None:
        """Test that same request generates same key."""
        key1 = key_generator.generate_key(sample_request)
        key2 = key_generator.generate_key(sample_request)

        assert key1 == key2

    def test_generate_key_is_sha256(
        self,
        key_generator: LLMCacheKeyGenerator,
        sample_request: CompletionRequest,
    ) -> None:
        """Test that generated key is a SHA-256 hash."""
        key = key_generator.generate_key(sample_request)

        # SHA-256 produces 64 character hex string
        assert len(key) == 64
        assert all(c in "0123456789abcdef" for c in key)

    def test_different_messages_generate_different_keys(
        self,
        key_generator: LLMCacheKeyGenerator,
    ) -> None:
        """Test that different messages generate different keys."""
        conv1 = Conversation.from_messages(
            [ChatMessage(role="user", content=[TextContent(text="Hello")])]
        )
        conv2 = Conversation.from_messages(
            [ChatMessage(role="user", content=[TextContent(text="Goodbye")])]
        )

        req1 = CompletionRequest(input=conv1, structured_output=None)
        req2 = CompletionRequest(input=conv2, structured_output=None)

        key1 = key_generator.generate_key(req1)
        key2 = key_generator.generate_key(req2)

        assert key1 != key2

    def test_structured_output_affects_key(
        self,
        key_generator: LLMCacheKeyGenerator,
        sample_conversation: Conversation,
    ) -> None:
        """Test that structured output flag affects the key."""
        req_without = CompletionRequest(
            input=sample_conversation, structured_output=None
        )
        req_with = CompletionRequest(
            input=sample_conversation, structured_output=SampleSchema
        )

        key_without = key_generator.generate_key(req_without)
        key_with = key_generator.generate_key(req_with)

        assert key_without != key_with

    def test_message_order_affects_key(
        self,
        key_generator: LLMCacheKeyGenerator,
    ) -> None:
        """Test that message order affects the key."""
        msg1 = ChatMessage(role="user", content=[TextContent(text="First")])
        msg2 = ChatMessage(role="assistant", content=[TextContent(text="Second")])

        conv1 = Conversation.from_messages([msg1, msg2])
        conv2 = Conversation.from_messages([msg2, msg1])

        req1 = CompletionRequest(input=conv1, structured_output=None)
        req2 = CompletionRequest(input=conv2, structured_output=None)

        key1 = key_generator.generate_key(req1)
        key2 = key_generator.generate_key(req2)

        assert key1 != key2


class TestLLMCacheBackend:
    """Test LLM cache backend."""

    def test_get_from_empty_cache_returns_none(
        self,
        cache_backend: LLMCacheBackend,
    ) -> None:
        """Test that getting from empty cache returns None."""
        result = cache_backend.get("nonexistent_key")
        assert result is None

    def test_set_and_get_completion_result(
        self,
        cache_backend: LLMCacheBackend,
        sample_result: CompletionResult,
    ) -> None:
        """Test setting and getting a completion result."""
        key = "test_key"

        # Set
        success = cache_backend.set(key, sample_result)
        assert success is True

        # Get
        cached = cache_backend.get(key)
        assert cached is not None
        assert len(cached.conversation.messages) == 3
        assert cached.output.structured_response.name == "test"
        assert cached.output.structured_response.value == 42

    def test_cache_preserves_conversation(
        self,
        cache_backend: LLMCacheBackend,
        sample_result: CompletionResult,
    ) -> None:
        """Test that full conversation is preserved."""
        key = "conversation_test"

        cache_backend.set(key, sample_result)
        cached = cache_backend.get(key)

        assert cached is not None
        messages = cached.conversation.messages

        # Should have system, user, and assistant messages
        assert len(messages) == 3
        assert messages[0].role == "system"
        assert messages[1].role == "user"
        assert messages[2].role == "assistant"

    def test_cache_preserves_structured_output(
        self,
        cache_backend: LLMCacheBackend,
        sample_result: CompletionResult,
    ) -> None:
        """Test that structured output is preserved."""
        key = "structured_test"

        cache_backend.set(key, sample_result)
        cached = cache_backend.get(key)

        assert cached is not None
        assert isinstance(cached.output.structured_response, SampleSchema)
        assert cached.output.structured_response.name == "test"
        assert cached.output.structured_response.value == 42

    def test_overwrite_existing_entry(
        self,
        cache_backend: LLMCacheBackend,
        sample_result: CompletionResult,
    ) -> None:
        """Test overwriting an existing cache entry."""
        key = "overwrite_test"

        # First entry
        cache_backend.set(key, sample_result)

        # Second entry with different data
        new_schema = SampleSchema(name="updated", value=99)
        new_response = MockProviderResponse(
            structured_response=new_schema, text_response=None
        )
        new_result = CompletionResult[SampleSchema](
            output=new_response,
            conversation=sample_result.conversation,
        )

        cache_backend.set(key, new_result)
        cached = cache_backend.get(key)

        assert cached is not None
        assert cached.output.structured_response.name == "updated"
        assert cached.output.structured_response.value == 99


class TestLLMCacheBackendIntegration:
    """Test LLM cache backend integration with key generator."""

    def test_end_to_end_caching_workflow(
        self,
        cache_backend: LLMCacheBackend,
        key_generator: LLMCacheKeyGenerator,
        sample_request: CompletionRequest,
        sample_result: CompletionResult,
    ) -> None:
        """Test complete workflow: generate key -> cache -> retrieve."""
        # Generate key from request
        key = key_generator.generate_key(sample_request)

        # Cache the result
        cache_backend.set(key, sample_result)

        # Retrieve using same key
        cached = cache_backend.get(key)

        assert cached is not None
        assert cached.output.structured_response.name == "test"

    def test_same_request_retrieves_same_result(
        self,
        cache_backend: LLMCacheBackend,
        key_generator: LLMCacheKeyGenerator,
        sample_request: CompletionRequest,
        sample_result: CompletionResult,
    ) -> None:
        """Test that same request generates same key and retrieves same result."""
        # Generate key and cache
        key1 = key_generator.generate_key(sample_request)
        cache_backend.set(key1, sample_result)

        # Generate key again from same request
        key2 = key_generator.generate_key(sample_request)

        # Keys should match
        assert key1 == key2

        # Should retrieve the cached result
        cached = cache_backend.get(key2)
        assert cached is not None


class TestCreateLLMCacheBackend:
    """Test factory function for creating cache backend."""

    def test_creates_backend_and_keygen(self, tmp_cache_dir: Path) -> None:
        """Test that factory creates both backend and key generator."""
        backend, keygen = create_llm_cache_backend("test-model")

        assert isinstance(backend, LLMCacheBackend)
        assert isinstance(keygen, LLMCacheKeyGenerator)

    def test_factory_created_backend_works(
        self,
        sample_result: CompletionResult,
    ) -> None:
        """Test that factory-created backend is functional."""
        backend, keygen = create_llm_cache_backend("test-model")

        key = "factory_test"
        backend.set(key, sample_result)
        cached = backend.get(key)

        assert cached is not None
        assert cached.output.structured_response.name == "test"

    def test_different_model_names_create_separate_caches(
        self,
        sample_result: CompletionResult,
    ) -> None:
        """Test that different model names use separate caches."""
        backend1, _ = create_llm_cache_backend("model-1")
        backend2, _ = create_llm_cache_backend("model-2")

        key = "same_key"

        # Cache in model-1
        backend1.set(key, sample_result)

        # Should not exist in model-2 cache
        assert backend2.get(key) is None

        # Should exist in model-1 cache
        assert backend1.get(key) is not None
