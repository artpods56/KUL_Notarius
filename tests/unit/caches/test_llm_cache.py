import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from notarius.infrastructure.cache.llm_adapter import LLMCache
from notarius.infrastructure.cache.utils import get_image_hash, get_text_hash
from typing import Generator, Dict, Any, cast
from PIL.Image import Image as PILImage


class TestLLMCache:
    """Test suite for LLMCache class."""

    @pytest.fixture
    def llm_cache(self, tmp_path: Path) -> LLMCache:
        """Create an LLMCache instance for testing."""
        return LLMCache(model_name="gpt-4", caches_dir=tmp_path)

    @pytest.fixture
    def sample_image(self) -> PILImage:
        """Create a sample PIL image for testing."""
        img = Image.new("RGB", (100, 100), color="blue")
        return img

    def test_init_with_simple_model_name(self, tmp_path: Path) -> None:
        """Test initialization with simple model name."""
        cache = LLMCache(model_name="gpt-4", caches_dir=tmp_path)
        assert cache.model_name == "gpt-4"
        assert cache._cache_loaded

    def test_init_with_path_model_name(self, tmp_path: Path) -> None:
        """Test initialization with absolute path model name."""
        cache = LLMCache(
            model_name="/models/gemma-3-27b-it-Q4_K_M.gguf",
            caches_dir=tmp_path,
        )
        assert cache.model_name == "gemma-3-27b-it-Q4_K_M"

    def test_init_with_slash_in_model_name(self, tmp_path: Path) -> None:
        """Test initialization with slashes in model name."""
        cache = LLMCache(model_name="openai/gpt-4-turbo", caches_dir=tmp_path)
        assert cache.model_name == "openai_gpt-4-turbo"

    def test_parse_model_name_absolute_path(self, llm_cache: LLMCache) -> None:
        """Test parsing of absolute path model names."""
        parsed = llm_cache._parse_model_name("/models/llama-2-7b.gguf")
        assert parsed == "llama-2-7b"

    def test_parse_model_name_with_slashes(self, llm_cache: LLMCache) -> None:
        """Test parsing of model names with slashes."""
        parsed = llm_cache._parse_model_name("anthropic/claude-3")
        assert parsed == "anthropic_claude-3"

    def test_parse_model_name_simple(self, llm_cache: LLMCache) -> None:
        """Test parsing of simple model names."""
        parsed = llm_cache._parse_model_name("gpt-4")
        assert parsed == "gpt-4"

    def test_cache_directory_created(self, tmp_path: Path) -> None:
        """Test that cache directory is created on initialization."""
        cache = LLMCache(model_name="gpt-4", caches_dir=tmp_path)
        expected_dir = tmp_path / "LLMCache" / "gpt-4"
        assert expected_dir.exists()
        assert expected_dir.is_dir()

    def test_normalize_kwargs_all_params(self, llm_cache: LLMCache) -> None:
        """Test normalize_kwargs with all parameters."""
        result = llm_cache.normalize_kwargs(
            image_hash="img123",
            text_hash="txt456",
            messages_hash="msg789",
            hints={"key": "value"},
            extra_param="ignored",
        )
        assert result == {
            "image_hash": "img123",
            "text_hash": "txt456",
            "messages_hash": "msg789",
            "hints": json.dumps({"key": "value"}, ensure_ascii=False),
        }

    def test_normalize_kwargs_partial_params(self, llm_cache: LLMCache) -> None:
        """Test normalize_kwargs with partial parameters."""
        result = llm_cache.normalize_kwargs(image_hash="img123", hints={"foo": "bar"})
        assert result == {
            "image_hash": "img123",
            "text_hash": None,
            "messages_hash": None,
            "hints": json.dumps({"foo": "bar"}, ensure_ascii=False),
        }

    def test_normalize_kwargs_missing_params(self, llm_cache: LLMCache) -> None:
        """Test normalize_kwargs with missing parameters."""
        result = llm_cache.normalize_kwargs()
        assert result == {
            "image_hash": None,
            "text_hash": None,
            "messages_hash": None,
            "hints": json.dumps(None, ensure_ascii=False),
        }

    def test_normalize_kwargs_non_ascii_hints(self, llm_cache: LLMCache) -> None:
        """Test normalize_kwargs with non-ASCII characters in hints."""
        hints = {"language": "русский", "text": "テスト"}
        result = cast(dict[str, str], llm_cache.normalize_kwargs(hints=hints))

        # Ensure non-ASCII is preserved
        assert "русский" in result["hints"]
        assert "テスト" in result["hints"]

    def test_set_and_get_cache_with_image(
        self, llm_cache: LLMCache, sample_image: PILImage
    ) -> None:
        """Test setting and getting cache entries with image hash."""
        image_hash = get_image_hash(sample_image)
        cache_key = llm_cache.generate_hash(image_hash=image_hash)

        test_value = {"output": "This is a test output", "tokens": 150}

        llm_cache.set(cache_key, test_value)
        retrieved = llm_cache.get(cache_key)

        assert retrieved == test_value

    def test_set_and_get_cache_with_text(self, llm_cache: LLMCache) -> None:
        """Test setting and getting cache entries with text hash."""
        text = "What is the capital of France?"
        text_hash = get_text_hash(text)
        cache_key = llm_cache.generate_hash(text_hash=text_hash)

        test_value = {"output": "Paris", "confidence": 0.99}

        llm_cache.set(cache_key, test_value)
        retrieved = llm_cache.get(cache_key)

        assert retrieved == test_value

    def test_set_and_get_cache_with_messages(self, llm_cache: LLMCache) -> None:
        """Test setting and getting cache entries with messages hash."""
        messages = [
            {"role": "user", "text": "Hello"},
            {"role": "assistant", "text": "Hi there!"},
        ]
        messages_str = json.dumps(messages)
        messages_hash = get_text_hash(messages_str)
        cache_key = llm_cache.generate_hash(messages_hash=messages_hash)

        test_value = {"output": "Response to messages", "model": "gpt-4"}

        llm_cache.set(cache_key, test_value)
        retrieved = llm_cache.get(cache_key)

        assert retrieved == test_value

    def test_set_cache_with_tags(self, llm_cache: LLMCache) -> None:
        """Test setting cache with schematism and filename tags."""
        cache_key = llm_cache.generate_hash(text_hash="test_hash")
        test_value = {"output": "Tagged LLM output"}

        llm_cache.set(
            cache_key, test_value, schematism="document_2024", filename="query_001.txt"
        )

        retrieved = llm_cache.get(cache_key)
        assert retrieved == test_value

    def test_delete_cache_entry(self, llm_cache: LLMCache) -> None:
        """Test deleting cache entries."""
        cache_key = llm_cache.generate_hash(text_hash="delete_test")
        test_value = {"output": "To be deleted"}

        llm_cache.set(cache_key, test_value)
        assert llm_cache.get(cache_key) == test_value

        llm_cache.delete(cache_key)
        assert llm_cache.get(cache_key) is None

    def test_cache_length(self, llm_cache: LLMCache) -> None:
        """Test __len__ returns correct cache size."""
        initial_length = len(llm_cache)

        cache_key = llm_cache.generate_hash(text_hash="test_length")
        llm_cache.set(cache_key, {"output": "Entry 1"})

        assert len(llm_cache) == initial_length + 1

    def test_cache_miss_returns_none(self, llm_cache: LLMCache) -> None:
        """Test that cache miss returns None."""
        nonexistent_key = "nonexistent_hash_456"
        result = llm_cache.get(nonexistent_key)
        assert result is None

    def test_multiple_models_separate_caches(self, tmp_path: Path) -> None:
        """Test that different models create separate cache directories."""
        cache_gpt4 = LLMCache(model_name="gpt-4", caches_dir=tmp_path)
        cache_claude = LLMCache(model_name="claude-3", caches_dir=tmp_path)

        gpt4_dir = tmp_path / "LLMCache" / "gpt-4"
        claude_dir = tmp_path / "LLMCache" / "claude-3"

        assert gpt4_dir.exists()
        assert claude_dir.exists()
        assert gpt4_dir != claude_dir

    def test_cache_persists_across_instances(self, tmp_path: Path) -> None:
        """Test that cache persists when creating new instances."""
        cache1 = LLMCache(model_name="gpt-4", caches_dir=tmp_path)
        cache_key = cache1.generate_hash(text_hash="persistent_test")
        test_value = {"output": "Persistent LLM data"}
        cache1.set(cache_key, test_value)

        # Create second instance and retrieve entry
        cache2 = LLMCache(model_name="gpt-4", caches_dir=tmp_path)
        retrieved = cache2.get(cache_key)

        assert retrieved == test_value

    def test_hash_generation_deterministic(self, llm_cache: LLMCache) -> None:
        """Test that hash generation is deterministic."""
        hash1 = llm_cache.generate_hash(text_hash="test", hints={"key": "value"})
        hash2 = llm_cache.generate_hash(text_hash="test", hints={"key": "value"})
        assert hash1 == hash2

    def test_hash_generation_different_for_different_inputs(
        self, llm_cache: LLMCache
    ) -> None:
        """Test that different inputs generate different hashes."""
        hash1 = llm_cache.generate_hash(text_hash="test1")
        hash2 = llm_cache.generate_hash(text_hash="test2")
        assert hash1 != hash2

    def test_cache_overwrite_existing_entry(self, llm_cache: LLMCache) -> None:
        """Test overwriting an existing cache entry."""
        cache_key = llm_cache.generate_hash(text_hash="overwrite_test")

        # Set initial value
        initial_value = {"output": "Initial output"}
        llm_cache.set(cache_key, initial_value)
        assert llm_cache.get(cache_key) == initial_value

        # Overwrite with new value
        new_value = {"output": "Updated output"}
        llm_cache.set(cache_key, new_value)
        assert llm_cache.get(cache_key) == new_value

    def test_complex_hints_serialization(self, llm_cache: LLMCache) -> None:
        """Test complex hints with nested structures."""
        complex_hints = {
            "nested": {"level1": {"level2": "value"}},
            "list": [1, 2, 3],
            "unicode": "🎉",
        }

        cache_key = llm_cache.generate_hash(
            text_hash="complex_test", hints=complex_hints
        )
        test_value = {"output": "Complex hints output"}

        llm_cache.set(cache_key, test_value)
        retrieved = llm_cache.get(cache_key)
        assert retrieved == test_value


class TestCacheUtilsText:
    """Test suite for text hash utility functions."""

    def test_get_text_hash_sha256(self) -> None:
        """Test that get_text_hash returns SHA-256 hash."""
        text = "Hello, World!"
        hash_result = get_text_hash(text)

        # SHA-256 hash should be 64 characters
        assert hash_result is not None
        assert len(hash_result) == 64
        assert isinstance(hash_result, str)

    def test_get_text_hash_deterministic(self) -> None:
        """Test that same text produces same hash."""
        text = "Consistent text"
        hash1 = get_text_hash(text)
        hash2 = get_text_hash(text)
        assert hash1 == hash2

    def test_get_text_hash_different_for_different_texts(self) -> None:
        """Test that different texts produce different hashes."""
        text1 = "First text"
        text2 = "Second text"

        hash1 = get_text_hash(text1)
        hash2 = get_text_hash(text2)
        assert hash1 != hash2

    def test_get_text_hash_with_none(self) -> None:
        """Test that None input returns None."""
        result = get_text_hash(None)
        assert result is None

    def test_get_text_hash_with_unicode(self) -> None:
        """Test hash generation with Unicode characters."""
        text = "Привет мир 🌍"
        hash_result = get_text_hash(text)

        assert hash_result is not None
        assert len(hash_result) == 64
        assert isinstance(hash_result, str)

    def test_get_text_hash_empty_string(self) -> None:
        """Test hash generation with empty string."""
        text = ""
        hash_result = get_text_hash(text)

        # Empty string should still produce a valid hash
        assert hash_result is not None
        assert len(hash_result) == 64
        assert isinstance(hash_result, str)

    def test_get_text_hash_expected_value(self) -> None:
        """Test that hash matches expected SHA-256 value."""
        text = "test"
        expected_hash = hashlib.sha256(text.encode()).hexdigest()

        actual_hash = get_text_hash(text)
        assert actual_hash == expected_hash
