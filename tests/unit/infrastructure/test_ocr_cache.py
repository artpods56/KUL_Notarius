"""Tests for OCR cache adapter with pickle serialization."""

import pickle
from pathlib import Path
from unittest.mock import patch

import pytest
from PIL import Image

from notarius.infrastructure.cache.adapters.ocr import PyTesseractCache
from notarius.infrastructure.ocr.engine_adapter import OCRResponse
from notarius.infrastructure.ocr.types import SimpleOCRResult, StructuredOCRResult
from notarius.schemas.data.structs import BBox


# Test fixtures


@pytest.fixture
def tmp_cache_dir(tmp_path: Path) -> Path:
    """Create a temporary cache directory."""
    return tmp_path / "test_caches"


@pytest.fixture
def ocr_cache(tmp_cache_dir: Path) -> PyTesseractCache:
    """Create a PyTesseractCache instance for testing."""
    return PyTesseractCache(language="eng", caches_dir=tmp_cache_dir)


@pytest.fixture
def sample_image() -> Image.Image:
    """Create a sample PIL image."""
    return Image.new("RGB", (100, 100), color="white")


@pytest.fixture
def simple_ocr_response() -> OCRResponse:
    """Create a simple OCR structured_response (text only)."""
    result = SimpleOCRResult(text="Hello World")
    return OCRResponse(output=result)


@pytest.fixture
def structured_ocr_response() -> OCRResponse:
    """Create a structured OCR structured_response with words and bboxes."""
    words = ["Hello", "World", "Test"]
    bboxes: list[BBox] = [
        (100, 200, 300, 250),
        (320, 200, 450, 250),
        (470, 200, 550, 250),
    ]
    result = StructuredOCRResult(words=words, bboxes=bboxes)
    return OCRResponse(output=result)


# Test cases


class TestOCRCacheInitialization:
    """Test OCR cache initialization."""

    def test_init_creates_cache_directory(self, tmp_cache_dir: Path) -> None:
        """Test that cache directory is created on initialization."""
        cache = PyTesseractCache(language="eng", caches_dir=tmp_cache_dir)
        expected_path = tmp_cache_dir / "PyTesseractCache" / "eng"

        assert expected_path.exists()
        assert expected_path.is_dir()

    def test_init_with_default_language(self, tmp_cache_dir: Path) -> None:
        """Test initialization with default language."""
        cache = PyTesseractCache(caches_dir=tmp_cache_dir)
        assert cache.language == "lat+pol+rus"
        assert cache.cache_type == "PyTesseractCache"

    def test_init_with_custom_language(self, tmp_cache_dir: Path) -> None:
        """Test initialization with custom language."""
        cache = PyTesseractCache(language="fra", caches_dir=tmp_cache_dir)
        assert cache.language == "fra"
        assert cache.cache_name == "fra"

    def test_cache_type_property(self, ocr_cache: PyTesseractCache) -> None:
        """Test that cache_type returns correct value."""
        assert ocr_cache.cache_type == "PyTesseractCache"

    def test_different_languages_use_separate_caches(self, tmp_cache_dir: Path) -> None:
        """Test that different languages create separate cache directories."""
        cache_eng = PyTesseractCache(language="eng", caches_dir=tmp_cache_dir)
        cache_fra = PyTesseractCache(language="fra", caches_dir=tmp_cache_dir)

        eng_dir = tmp_cache_dir / "PyTesseractCache" / "eng"
        fra_dir = tmp_cache_dir / "PyTesseractCache" / "fra"

        assert eng_dir.exists()
        assert fra_dir.exists()
        assert eng_dir != fra_dir


class TestOCRCacheSetAndGet:
    """Test basic cache set and get operations."""

    def test_set_and_get_simple_response(
        self,
        ocr_cache: PyTesseractCache,
        simple_ocr_response: OCRResponse,
    ) -> None:
        """Test caching and retrieving a simple OCR structured_response (text only)."""
        key = "test_key_1"

        # Cache the result
        success = ocr_cache.set(key, simple_ocr_response)
        assert success is True

        # Retrieve the result
        cached_result = ocr_cache.get(key)
        assert cached_result is not None
        assert isinstance(cached_result.output, SimpleOCRResult)
        assert cached_result.output.text == "Hello World"

    def test_set_and_get_structured_response(
        self,
        ocr_cache: PyTesseractCache,
        structured_ocr_response: OCRResponse,
    ) -> None:
        """Test caching and retrieving a structured OCR structured_response."""
        key = "test_key_2"

        # Cache the result
        success = ocr_cache.set(key, structured_ocr_response)
        assert success is True

        # Retrieve the result
        cached_result = ocr_cache.get(key)
        assert cached_result is not None
        assert isinstance(cached_result.output, StructuredOCRResult)
        assert cached_result.output.words == ["Hello", "World", "Test"]
        assert len(cached_result.output.bboxes) == 3

    def test_get_nonexistent_key_returns_none(
        self, ocr_cache: PyTesseractCache
    ) -> None:
        """Test that getting a nonexistent key returns None."""
        result = ocr_cache.get("nonexistent_key")
        assert result is None

    def test_cache_overwrites_existing_entry(
        self,
        ocr_cache: PyTesseractCache,
        simple_ocr_response: OCRResponse,
    ) -> None:
        """Test that setting the same key twice overwrites the first value."""
        key = "overwrite_test"

        # Set initial value
        ocr_cache.set(key, simple_ocr_response)
        initial = ocr_cache.get(key)
        assert initial is not None
        assert initial.output.text == "Hello World"

        # Overwrite with new value
        new_result = SimpleOCRResult(text="Different text")
        new_response = OCRResponse(output=new_result)
        ocr_cache.set(key, new_response)
        updated = ocr_cache.get(key)

        assert updated is not None
        assert updated.output.text == "Different text"

    def test_delete_cache_entry(
        self,
        ocr_cache: PyTesseractCache,
        simple_ocr_response: OCRResponse,
    ) -> None:
        """Test deleting a cache entry."""
        key = "delete_test"

        ocr_cache.set(key, simple_ocr_response)
        assert ocr_cache.get(key) is not None

        ocr_cache.delete(key)
        assert ocr_cache.get(key) is None


class TestOCRCacheStructuredData:
    """Test caching with structured OCR data (words + bboxes)."""

    def test_cache_preserves_words_and_bboxes(
        self,
        ocr_cache: PyTesseractCache,
        structured_ocr_response: OCRResponse,
    ) -> None:
        """Test that words and bounding boxes are preserved through caching."""
        key = "structured_test"

        ocr_cache.set(key, structured_ocr_response)
        cached = ocr_cache.get(key)

        assert cached is not None
        assert isinstance(cached.output, StructuredOCRResult)

        # Verify words
        assert cached.output.words == ["Hello", "World", "Test"]

        # Verify bboxes
        expected_bboxes: list[BBox] = [
            (100, 200, 300, 250),
            (320, 200, 450, 250),
            (470, 200, 550, 250),
        ]
        assert cached.output.bboxes == expected_bboxes

    def test_cache_with_empty_words(
        self,
        ocr_cache: PyTesseractCache,
    ) -> None:
        """Test caching OCR result with no detected words."""
        result = StructuredOCRResult(words=[], bboxes=[])
        response = OCRResponse(output=result)

        key = "empty_words_test"
        ocr_cache.set(key, response)
        cached = ocr_cache.get(key)

        assert cached is not None
        assert cached.output.words == []
        assert cached.output.bboxes == []

    def test_cache_with_many_words(
        self,
        ocr_cache: PyTesseractCache,
    ) -> None:
        """Test caching OCR result with many words."""
        words = [f"word_{i}" for i in range(100)]
        bboxes: list[BBox] = [(i * 10, 0, i * 10 + 50, 20) for i in range(100)]
        result = StructuredOCRResult(words=words, bboxes=bboxes)
        response = OCRResponse(output=result)

        key = "many_words_test"
        ocr_cache.set(key, response)
        cached = ocr_cache.get(key)

        assert cached is not None
        assert len(cached.output.words) == 100
        assert len(cached.output.bboxes) == 100


class TestOCRCachePersistence:
    """Test cache persistence across instances."""

    def test_cache_persists_across_instances(
        self,
        tmp_cache_dir: Path,
        simple_ocr_response: OCRResponse,
    ) -> None:
        """Test that cached data persists when creating new cache instances."""
        key = "persistence_test"

        # Cache with first instance
        cache1 = PyTesseractCache(language="eng", caches_dir=tmp_cache_dir)
        cache1.set(key, simple_ocr_response)

        # Retrieve with second instance
        cache2 = PyTesseractCache(language="eng", caches_dir=tmp_cache_dir)
        cached = cache2.get(key)

        assert cached is not None
        assert cached.output.text == "Hello World"

    def test_different_languages_use_separate_caches(
        self,
        tmp_cache_dir: Path,
        simple_ocr_response: OCRResponse,
    ) -> None:
        """Test that different language caches are isolated."""
        key = "same_key"

        cache_eng = PyTesseractCache(language="eng", caches_dir=tmp_cache_dir)
        cache_fra = PyTesseractCache(language="fra", caches_dir=tmp_cache_dir)

        # Cache in English
        cache_eng.set(key, simple_ocr_response)

        # Should not exist in French cache
        assert cache_fra.get(key) is None

        # Should exist in English cache
        assert cache_eng.get(key) is not None


class TestOCRCacheErrorHandling:
    """Test error handling and edge cases."""

    def test_pickle_error_during_serialization(
        self,
        ocr_cache: PyTesseractCache,
    ) -> None:
        """Test handling of pickle errors during serialization."""
        with patch.object(
            ocr_cache.cache, "set", side_effect=pickle.PickleError("Mock error")
        ):
            result = ocr_cache.set("test_key", lambda: None)  # type: ignore
            assert result is False

    def test_pickle_error_during_deserialization(
        self,
        ocr_cache: PyTesseractCache,
    ) -> None:
        """Test handling of pickle errors during deserialization."""
        with patch.object(
            ocr_cache.cache,
            "get",
            side_effect=pickle.PickleError("Corrupt data"),
        ):
            result = ocr_cache.get("test_key")
            assert result is None

    def test_attribute_error_during_deserialization(
        self,
        ocr_cache: PyTesseractCache,
    ) -> None:
        """Test handling of AttributeError (e.g., class definition changed)."""
        with patch.object(
            ocr_cache.cache,
            "get",
            side_effect=AttributeError("Class not found"),
        ):
            result = ocr_cache.get("test_key")
            assert result is None

    def test_cache_length(
        self,
        ocr_cache: PyTesseractCache,
        simple_ocr_response: OCRResponse,
    ) -> None:
        """Test that __len__ returns correct cache size."""
        initial_len = len(ocr_cache)

        ocr_cache.set("key1", simple_ocr_response)
        assert len(ocr_cache) == initial_len + 1

        ocr_cache.set("key2", simple_ocr_response)
        assert len(ocr_cache) == initial_len + 2


class TestOCRCacheRealWorldScenarios:
    """Test real-world usage scenarios."""

    def test_cache_multilingual_results(
        self,
        tmp_cache_dir: Path,
    ) -> None:
        """Test caching OCR sample from different languages."""
        # English cache
        cache_eng = PyTesseractCache(language="eng", caches_dir=tmp_cache_dir)
        eng_result = SimpleOCRResult(text="Hello World")
        cache_eng.set("doc1", OCRResponse(output=eng_result))

        # French cache
        cache_fra = PyTesseractCache(language="fra", caches_dir=tmp_cache_dir)
        fra_result = SimpleOCRResult(text="Bonjour le monde")
        cache_fra.set("doc1", OCRResponse(output=fra_result))

        # Verify isolation
        cached_eng = cache_eng.get("doc1")
        cached_fra = cache_fra.get("doc1")

        assert cached_eng is not None
        assert cached_fra is not None
        assert cached_eng.output.text == "Hello World"
        assert cached_fra.output.text == "Bonjour le monde"

    def test_cache_with_normalized_bboxes(
        self,
        ocr_cache: PyTesseractCache,
    ) -> None:
        """Test caching OCR sample with normalized bounding boxes (0-1000 range)."""
        words = ["Test"]
        bboxes: list[BBox] = [(100, 200, 300, 400)]  # Normalized coordinates
        result = StructuredOCRResult(words=words, bboxes=bboxes)
        response = OCRResponse(output=result)

        key = "normalized_bbox_test"
        ocr_cache.set(key, response)
        cached = ocr_cache.get(key)

        assert cached is not None
        assert cached.output.bboxes[0] == (100, 200, 300, 400)

    def test_cache_mixed_response_types(
        self,
        ocr_cache: PyTesseractCache,
        simple_ocr_response: OCRResponse,
        structured_ocr_response: OCRResponse,
    ) -> None:
        """Test caching both simple and structured responses in same cache."""
        # Cache simple structured_response
        ocr_cache.set("simple", simple_ocr_response)

        # Cache structured structured_response
        ocr_cache.set("structured", structured_ocr_response)

        # Retrieve both
        cached_simple = ocr_cache.get("simple")
        cached_structured = ocr_cache.get("structured")

        assert cached_simple is not None
        assert isinstance(cached_simple.output, SimpleOCRResult)

        assert cached_structured is not None
        assert isinstance(cached_structured.output, StructuredOCRResult)
