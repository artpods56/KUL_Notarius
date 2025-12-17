"""Tests for LLM cache adapter with pickle serialization."""

import pickle
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import patch

import pytest
from pydantic import BaseModel

from notarius.domain.entities.completions import BaseProviderResponse
from notarius.domain.entities.messages import ChatMessage, TextContent
from notarius.infrastructure.cache.adapters.llm import LLMCache
from notarius.infrastructure.llm.conversation import Conversation
from notarius.infrastructure.llm.engine_adapter import CompletionResult


# Test fixtures


class SampleSchema(BaseModel):
    """Sample Pydantic model for structured output testing."""

    name: str
    age: int
    email: str | None = None


@dataclass(frozen=True)
class MockProviderResponse(BaseProviderResponse[SampleSchema]):
    """Mock provider structured_response for testing."""

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
    """Create an LLMCache instance for testing."""
    return LLMCache(model_name="gpt-4", caches_dir=tmp_cache_dir)


@pytest.fixture
def sample_conversation() -> Conversation:
    """Create a sample conversation."""
    messages = [
        ChatMessage(
            role="system", content=[TextContent(text="You are a helpful assistant.")]
        ),
        ChatMessage(role="user", content=[TextContent(text="What is 2+2?")]),
    ]
    return Conversation.from_messages(messages)


@pytest.fixture
def sample_completion_result(
    sample_conversation: Conversation,
) -> CompletionResult[SampleSchema]:
    """Create a sample CompletionResult with structured output."""
    schema = SampleSchema(name="John Doe", age=30, email="john@example.com")
    response = MockProviderResponse(structured_response=schema, text_response=None)

    # Add assistant structured_response to conversation
    updated_conversation = sample_conversation.add(
        ChatMessage(role="assistant", content=[TextContent(text=response.to_string())])
    )

    return CompletionResult[SampleSchema](
        output=response,
        conversation=updated_conversation,
    )


@dataclass(frozen=True)
class TextOnlyResponse(BaseProviderResponse[None]):
    """Mock text-only structured_response (no structured output)."""

    def to_string(self) -> str:
        return self.text_response or ""


@pytest.fixture
def text_only_completion_result(
    sample_conversation: Conversation,
) -> CompletionResult[BaseModel]:
    """Create a CompletionResult with text-only output (no structured schema)."""
    response = TextOnlyResponse(
        structured_response=None, text_response="The answer is 4"
    )
    updated_conversation = sample_conversation.add(
        ChatMessage(role="assistant", content=[TextContent(text=response.to_string())])
    )

    return CompletionResult[BaseModel](
        output=response,
        conversation=updated_conversation,
    )


# Test cases


class TestLLMCacheInitialization:
    """Test cache initialization."""

    def test_init_creates_cache_directory(self, tmp_cache_dir: Path) -> None:
        """Test that cache directory is created on initialization."""
        cache = LLMCache(model_name="gpt-4", caches_dir=tmp_cache_dir)
        expected_path = tmp_cache_dir / "LLMCache" / "gpt-4"

        assert expected_path.exists()
        assert expected_path.is_dir()

    def test_init_with_simple_model_name(self, tmp_cache_dir: Path) -> None:
        """Test initialization with simple model name."""
        cache = LLMCache(model_name="claude-3", caches_dir=tmp_cache_dir)
        assert cache.cache_name == "claude-3"
        assert cache.cache_type == "LLMCache"

    def test_init_with_path_model_name(self, tmp_cache_dir: Path) -> None:
        """Test initialization with path-like model name."""
        cache = LLMCache(
            model_name="/models/gemma-3-27b-it-Q4_K_M.gguf",
            caches_dir=tmp_cache_dir,
        )
        # parse_model_name should extract just the model name
        assert cache.cache_name == "gemma-3-27b-it-Q4_K_M"

    def test_init_with_slash_in_model_name(self, tmp_cache_dir: Path) -> None:
        """Test initialization with slashes in model name (e.g., openai/gpt-4)."""
        cache = LLMCache(model_name="openai/gpt-4-turbo", caches_dir=tmp_cache_dir)
        # Slashes should be replaced with underscores
        assert cache.cache_name == "openai_gpt-4-turbo"

    def test_cache_type_property(self, llm_cache: LLMCache) -> None:
        """Test that cache_type returns correct value."""
        assert llm_cache.cache_type == "LLMCache"


class TestLLMCacheSetAndGet:
    """Test basic cache set and get operations."""

    def test_set_and_get_completion_result(
        self,
        llm_cache: LLMCache,
        sample_completion_result: CompletionResult[SampleSchema],
    ) -> None:
        """Test caching and retrieving a CompletionResult."""
        key = "test_key_1"

        # Cache the result
        success = llm_cache.set(key, sample_completion_result)
        assert success is True

        # Retrieve the result
        cached_result = llm_cache.get(key)
        assert cached_result is not None

        # Verify the data is intact
        assert (
            cached_result.output.to_string()
            == sample_completion_result.output.to_string()
        )
        assert len(cached_result.conversation.messages) == len(
            sample_completion_result.conversation.messages
        )

    def test_get_nonexistent_key_returns_none(self, llm_cache: LLMCache) -> None:
        """Test that getting a nonexistent key returns None."""
        result = llm_cache.get("nonexistent_key")
        assert result is None

    def test_cache_overwrites_existing_entry(
        self,
        llm_cache: LLMCache,
        sample_completion_result: CompletionResult[SampleSchema],
    ) -> None:
        """Test that setting the same key twice overwrites the first value."""
        key = "overwrite_test"

        # Set initial value
        llm_cache.set(key, sample_completion_result)
        initial = llm_cache.get(key)

        # Create a modified result
        new_conversation = Conversation.from_messages(
            [ChatMessage(role="user", content=[TextContent(text="Different message")])]
        )
        new_result = CompletionResult[SampleSchema](
            output=sample_completion_result.output,
            conversation=new_conversation,
        )

        # Overwrite
        llm_cache.set(key, new_result)
        updated = llm_cache.get(key)

        assert updated is not None
        assert len(updated.conversation.messages) != len(initial.conversation.messages)

    def test_delete_cache_entry(
        self,
        llm_cache: LLMCache,
        sample_completion_result: CompletionResult[SampleSchema],
    ) -> None:
        """Test deleting a cache entry."""
        key = "delete_test"

        llm_cache.set(key, sample_completion_result)
        assert llm_cache.get(key) is not None

        llm_cache.delete(key)
        assert llm_cache.get(key) is None


class TestLLMCacheStructuredOutput:
    """Test caching with structured Pydantic models."""

    def test_cache_preserves_structured_output(
        self,
        llm_cache: LLMCache,
        sample_completion_result: CompletionResult[SampleSchema],
    ) -> None:
        """Test that structured output is preserved through caching."""
        key = "structured_test"

        llm_cache.set(key, sample_completion_result)
        cached = llm_cache.get(key)

        assert cached is not None
        # Verify the structured data
        original_output = sample_completion_result.output.structured_response
        cached_output = cached.output.structured_response

        assert cached_output.name == original_output.name
        assert cached_output.age == original_output.age
        assert cached_output.email == original_output.email

    def test_cache_text_only_output(
        self,
        llm_cache: LLMCache,
        text_only_completion_result: CompletionResult[BaseModel],
    ) -> None:
        """Test caching sample without structured output."""
        key = "text_only_test"

        llm_cache.set(key, text_only_completion_result)
        cached = llm_cache.get(key)

        assert cached is not None
        assert cached.output.text_response == "The answer is 4"


class TestLLMCacheConversation:
    """Test conversation preservation in cache."""

    def test_cache_preserves_conversation_history(
        self,
        llm_cache: LLMCache,
        sample_completion_result: CompletionResult[SampleSchema],
    ) -> None:
        """Test that full conversation history is preserved."""
        key = "conversation_test"

        llm_cache.set(key, sample_completion_result)
        cached = llm_cache.get(key)

        assert cached is not None

        # Check message count
        original_messages = sample_completion_result.conversation.messages
        cached_messages = cached.conversation.messages
        assert len(cached_messages) == len(original_messages)

        # Check message content
        for orig, cached_msg in zip(original_messages, cached_messages):
            assert cached_msg.role == orig.role
            assert len(cached_msg.content) == len(orig.content)

    def test_cache_preserves_multimodal_messages(
        self,
        llm_cache: LLMCache,
        tmp_cache_dir: Path,
    ) -> None:
        """Test caching messages with images (multimodal)."""
        from notarius.domain.entities.messages import ImageContent

        # Create a multimodal conversation
        messages = [
            ChatMessage(
                role="user",
                content=[
                    TextContent(text="What's in this image?"),
                    ImageContent(
                        image_url="data:image/png;base64,iVBORw0KG...", detail="high"
                    ),
                ],
            )
        ]
        conversation = Conversation.from_messages(messages)

        schema = SampleSchema(name="Test", age=25)
        response = MockProviderResponse(structured_response=schema, text_response=None)
        result = CompletionResult[SampleSchema](
            output=response,
            conversation=conversation,
        )

        key = "multimodal_test"
        llm_cache.set(key, result)
        cached = llm_cache.get(key)

        assert cached is not None
        assert len(cached.conversation.messages[0].content) == 2
        assert isinstance(cached.conversation.messages[0].content[1], ImageContent)


class TestLLMCachePersistence:
    """Test cache persistence across instances."""

    def test_cache_persists_across_instances(
        self,
        tmp_cache_dir: Path,
        sample_completion_result: CompletionResult[SampleSchema],
    ) -> None:
        """Test that cached data persists when creating new cache instances."""
        key = "persistence_test"

        # Cache with first instance
        cache1 = LLMCache(model_name="gpt-4", caches_dir=tmp_cache_dir)
        cache1.set(key, sample_completion_result)

        # Retrieve with second instance
        cache2 = LLMCache(model_name="gpt-4", caches_dir=tmp_cache_dir)
        cached = cache2.get(key)

        assert cached is not None
        assert cached.output.to_string() == sample_completion_result.output.to_string()

    def test_different_models_use_separate_caches(
        self,
        tmp_cache_dir: Path,
        sample_completion_result: CompletionResult[SampleSchema],
    ) -> None:
        """Test that different model names create isolated caches."""
        key = "same_key"

        cache_gpt4 = LLMCache(model_name="gpt-4", caches_dir=tmp_cache_dir)
        cache_claude = LLMCache(model_name="claude-3", caches_dir=tmp_cache_dir)

        # Cache in gpt-4
        cache_gpt4.set(key, sample_completion_result)

        # Should not exist in claude cache
        assert cache_claude.get(key) is None

        # Should exist in gpt-4 cache
        assert cache_gpt4.get(key) is not None


class TestLLMCacheErrorHandling:
    """Test error handling and edge cases."""

    def test_pickle_error_during_serialization(
        self,
        llm_cache: LLMCache,
    ) -> None:
        """Test handling of pickle errors during serialization."""
        # Create an unpicklable object
        unpicklable = lambda x: x  # Functions defined in local scope can't be pickled

        with patch.object(
            llm_cache.cache, "set", side_effect=pickle.PickleError("Mock error")
        ):
            result = llm_cache.set("test_key", unpicklable)
            assert result is False

    def test_pickle_error_during_deserialization(
        self,
        llm_cache: LLMCache,
    ) -> None:
        """Test handling of pickle errors during deserialization."""
        with patch.object(
            llm_cache.cache,
            "get",
            side_effect=pickle.PickleError("Corrupt data"),
        ):
            result = llm_cache.get("test_key")
            assert result is None

    def test_attribute_error_during_deserialization(
        self,
        llm_cache: LLMCache,
    ) -> None:
        """Test handling of AttributeError (e.g., class definition changed)."""
        with patch.object(
            llm_cache.cache,
            "get",
            side_effect=AttributeError("Class not found"),
        ):
            result = llm_cache.get("test_key")
            assert result is None

    def test_cache_length(
        self,
        llm_cache: LLMCache,
        sample_completion_result: CompletionResult[SampleSchema],
    ) -> None:
        """Test that __len__ returns correct cache size."""
        initial_len = len(llm_cache)

        llm_cache.set("key1", sample_completion_result)
        assert len(llm_cache) == initial_len + 1

        llm_cache.set("key2", sample_completion_result)
        assert len(llm_cache) == initial_len + 2


class TestLLMCacheHelperMethods:
    """Test helper methods like set_completion and get_completion."""

    def test_set_completion_method(
        self,
        llm_cache: LLMCache,
        sample_completion_result: CompletionResult[SampleSchema],
    ) -> None:
        """Test the set_completion convenience method."""
        key = "completion_test"

        # Use set directly (which is what set_completion does)
        success = llm_cache.set(key, sample_completion_result)
        assert success is True

        retrieved = llm_cache.get(key)
        assert retrieved is not None


class TestLLMCacheRealWorldScenarios:
    """Test real-world usage scenarios."""

    def test_multi_turn_conversation_caching(
        self,
        llm_cache: LLMCache,
    ) -> None:
        """Test caching a multi-turn conversation."""
        # Build a multi-turn conversation
        conv = Conversation()
        conv = conv.add(ChatMessage(role="user", content=[TextContent(text="Hello")]))
        conv = conv.add(
            ChatMessage(role="assistant", content=[TextContent(text="Hi!")])
        )
        conv = conv.add(
            ChatMessage(role="user", content=[TextContent(text="How are you?")])
        )

        schema = SampleSchema(name="Assistant", age=1)
        response = MockProviderResponse(structured_response=schema, text_response=None)
        result = CompletionResult[SampleSchema](output=response, conversation=conv)

        key = "multi_turn_test"
        llm_cache.set(key, result)
        cached = llm_cache.get(key)

        assert cached is not None
        assert len(cached.conversation.messages) == 3

    def test_cache_with_large_conversation(
        self,
        llm_cache: LLMCache,
    ) -> None:
        """Test caching a conversation with many messages."""
        # Create a large conversation
        conv = Conversation()
        for i in range(100):
            conv = conv.add(
                ChatMessage(role="user", content=[TextContent(text=f"Message {i}")])
            )
            conv = conv.add(
                ChatMessage(
                    role="assistant", content=[TextContent(text=f"Response {i}")]
                )
            )

        schema = SampleSchema(name="Test", age=1)
        response = MockProviderResponse(structured_response=schema, text_response=None)
        result = CompletionResult[SampleSchema](output=response, conversation=conv)

        key = "large_conv_test"
        llm_cache.set(key, result)
        cached = llm_cache.get(key)

        assert cached is not None
        assert len(cached.conversation.messages) == 200
