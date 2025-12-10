"""Integration tests for OCR _engine components.

Tests the complete OCR pipeline:
- SimpleOCREngine
- OCRCacheRepository
- GenerateOCRExtraction use case
- OCREngineAdapter

These tests use real components (not mocks) to verify end-to-end functionality.
"""

import tempfile
from pathlib import Path

import pytest
from PIL import Image

from notarius.infrastructure.ocr.engine import SimpleOCREngine
from notarius.infrastructure.ocr.engine_adapter import OCREngineAdapter
from notarius.infrastructure.persistence.ocr_cache_repository import OCRCacheRepository
from notarius.application.use_cases.generate_ocr_extraction import (
    GenerateOCRExtraction,
    GenerateOCRExtractionRequest,
)
from omegaconf import DictConfig, OmegaConf


@pytest.fixture
def temp_cache_dir() -> Path:
    """Create temporary directory for cache."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def test_image() -> Image.Image:
    """Create a simple test image with text."""
    # Create a white image with some black text-like regions
    img = Image.new("RGB", (200, 100), color="white")
    # Note: This is a simple image for testing. Real OCR requires actual text.
    return img


@pytest.fixture
def ocr_config(temp_cache_dir: Path) -> DictConfig:
    """Create OCR configuration."""
    return OmegaConf.create(
        {
            "provider": {
                "language": "eng",
                "psm_mode": 6,
                "oem_mode": 3,
                "enable_cache": True,
                "min_confidence": 0.0,
                "caches_dir": str(temp_cache_dir),
            }
        }
    )


class TestSimpleOCREngineIntegration:
    """Integration tests for SimpleOCREngine with real PyTesseract."""

    def test_extract_text_returns_string(self, test_image: Image.Image) -> None:
        """Test that text extraction returns a string."""
        engine = SimpleOCREngine(language="eng", psm_mode=6, oem_mode=3)

        text = engine.extract_text(test_image)

        assert isinstance(text, str)
        # Empty image may return empty or whitespace
        assert text is not None

    def test_extract_words_and_boxes_returns_lists(
        self, test_image: Image.Image
    ) -> None:
        """Test that word/bbox extraction returns lists."""
        engine = SimpleOCREngine(language="eng", psm_mode=6, oem_mode=3)

        words, bboxes = engine.extract_words_and_boxes(test_image)

        assert isinstance(words, list)
        assert isinstance(bboxes, list)
        assert len(words) == len(bboxes)

        # Verify bbox format (normalized 0-1000)
        for bbox in bboxes:
            assert isinstance(bbox, list)
            assert len(bbox) == 4
            for coord in bbox:
                assert 0 <= coord <= 1000

    def test_different_psm_modes(self, test_image: Image.Image) -> None:
        """Test that different PSM modes can be configured."""
        engine_psm6 = SimpleOCREngine(language="eng", psm_mode=6)
        engine_psm3 = SimpleOCREngine(language="eng", psm_mode=3)

        # Both should work without errors
        text1 = engine_psm6.extract_text(test_image)
        text2 = engine_psm3.extract_text(test_image)

        assert isinstance(text1, str)
        assert isinstance(text2, str)


class TestOCRCacheIntegration:
    """Integration tests for OCR caching with real cache backend."""

    def test_cache_persistence_across_instances(
        self,
        test_image: Image.Image,
        temp_cache_dir: Path,
    ) -> None:
        """Test that cache persists across repository instances."""
        # First instance - store data
        repo1 = OCRCacheRepository.create(language="eng", caches_dir=temp_cache_dir)
        key = repo1.generate_key(test_image)

        from notarius.schemas.data.cache import PyTesseractCacheItem

        item = PyTesseractCacheItem(
            text="Cached text",
            words=["Cached", "text"],
            bbox=[(0, 0, 50, 20), (50, 0, 100, 20)],
        )
        repo1.set(key, item, schematism="test_schematism")

        # Second instance - retrieve data
        repo2 = OCRCacheRepository.create(language="eng", caches_dir=temp_cache_dir)
        retrieved = repo2.get(key)

        assert retrieved is not None
        assert retrieved.text == "Cached text"
        assert retrieved.words == ["Cached", "text"]
        assert len(retrieved.bbox) == 2

    def test_cache_invalidation_removes_entry(
        self,
        test_image: Image.Image,
        temp_cache_dir: Path,
    ) -> None:
        """Test that invalidating cache removes the entry."""
        repo = OCRCacheRepository.create(language="eng", caches_dir=temp_cache_dir)
        key = repo.generate_key(test_image)

        from notarius.schemas.data.cache import PyTesseractCacheItem

        item = PyTesseractCacheItem(
            text="To be removed",
            words=["To", "be", "removed"],
            bbox=[(0, 0, 30, 20), (30, 0, 60, 20), (60, 0, 100, 20)],
        )
        repo.set(key, item)

        # Verify it exists
        assert repo.get(key) is not None

        # Delete
        repo.delete(key)

        # Verify it's gone
        assert repo.get(key) is None


class TestGenerateOCRExtractionIntegration:
    """Integration tests for GenerateOCRExtraction use case."""

    def test_full_extraction_flow_without_cache(self, test_image: Image.Image) -> None:
        """Test complete extraction flow without caching."""
        engine = SimpleOCREngine(language="eng", psm_mode=6)
        use_case = GenerateOCRExtraction(engine=engine, cache_repository=None)

        request = GenerateOCRExtractionRequest(image=test_image, text_only=False)
        response = use_case.execute(request)

        assert response.from_cache is False
        words, bboxes = response.result
        assert isinstance(words, list)
        assert isinstance(bboxes, list)

    def test_full_extraction_flow_with_cache(
        self,
        test_image: Image.Image,
        temp_cache_dir: Path,
    ) -> None:
        """Test complete extraction flow with caching."""
        engine = SimpleOCREngine(language="eng", psm_mode=6)
        cache_repo = OCRCacheRepository.create(
            language="eng", caches_dir=temp_cache_dir
        )
        use_case = GenerateOCRExtraction(engine=engine, cache_repository=cache_repo)

        # First request - cache miss
        request1 = GenerateOCRExtractionRequest(image=test_image, text_only=False)
        response1 = use_case.execute(request1)

        assert response1.from_cache is False
        words1, bboxes1 = response1.result

        # Second request - cache hit
        request2 = GenerateOCRExtractionRequest(image=test_image, text_only=False)
        response2 = use_case.execute(request2)

        assert response2.from_cache is True
        words2, bboxes2 = response2.result

        # Results should match
        assert words1 == words2
        assert len(bboxes1) == len(bboxes2)

    def test_text_only_mode_returns_string(
        self,
        test_image: Image.Image,
        temp_cache_dir: Path,
    ) -> None:
        """Test that text_only mode returns just the text string."""
        engine = SimpleOCREngine(language="eng", psm_mode=6)
        cache_repo = OCRCacheRepository.create(
            language="eng", caches_dir=temp_cache_dir
        )
        use_case = GenerateOCRExtraction(engine=engine, cache_repository=cache_repo)

        request = GenerateOCRExtractionRequest(image=test_image, text_only=True)
        response = use_case.execute(request)

        assert isinstance(response.result, str)

    def test_invalidate_cache_forces_reextraction(
        self,
        test_image: Image.Image,
        temp_cache_dir: Path,
    ) -> None:
        """Test that invalidate_cache forces re-extraction."""
        engine = SimpleOCREngine(language="eng", psm_mode=6)
        cache_repo = OCRCacheRepository.create(
            language="eng", caches_dir=temp_cache_dir
        )
        use_case = GenerateOCRExtraction(engine=engine, cache_repository=cache_repo)

        # First request - populates cache
        request1 = GenerateOCRExtractionRequest(image=test_image, text_only=False)
        response1 = use_case.execute(request1)
        assert response1.from_cache is False

        # Second request with invalidate - should be cache miss
        request2 = GenerateOCRExtractionRequest(
            image=test_image,
            text_only=False,
            invalidate_cache=True,
        )
        response2 = use_case.execute(request2)
        assert response2.from_cache is False


class TestOCREngineAdapterIntegration:
    """Integration tests for OCREngineAdapter using real components."""

    def test_adapter_predict_text_only(
        self,
        test_image: Image.Image,
        ocr_config: DictConfig,
    ) -> None:
        """Test adapter predict in text_only mode."""
        adapter = OCREngineAdapter.from_config(ocr_config)

        result = adapter.predict(test_image, text_only=True)

        assert isinstance(result, str)

    def test_adapter_predict_with_boxes(
        self,
        test_image: Image.Image,
        ocr_config: DictConfig,
    ) -> None:
        """Test adapter predict returning words and boxes."""
        adapter = OCREngineAdapter.from_config(ocr_config)

        result = adapter.predict(test_image, text_only=False)

        assert isinstance(result, tuple)
        words, bboxes = result
        assert isinstance(words, list)
        assert isinstance(bboxes, list)

    def test_adapter_caching_behavior(
        self,
        test_image: Image.Image,
        ocr_config: DictConfig,
    ) -> None:
        """Test that adapter properly caches results."""
        adapter = OCREngineAdapter.from_config(ocr_config)

        # First call - should populate cache
        result1 = adapter.predict(test_image, text_only=False)

        # Second call - should use cache
        result2 = adapter.predict(test_image, text_only=False)

        # Results should be identical
        words1, bboxes1 = result1
        words2, bboxes2 = result2
        assert words1 == words2
        assert len(bboxes1) == len(bboxes2)

    def test_adapter_with_cache_disabled(
        self,
        test_image: Image.Image,
        ocr_config: DictConfig,
    ) -> None:
        """Test adapter works with caching disabled."""
        # Disable cache
        config_no_cache = OmegaConf.create(ocr_config)
        config_no_cache.predictor.enable_cache = False

        adapter = OCREngineAdapter.from_config(config_no_cache)

        # Should work without cache
        result = adapter.predict(test_image, text_only=True)
        assert isinstance(result, str)

    def test_adapter_invalidate_cache(
        self,
        test_image: Image.Image,
        ocr_config: DictConfig,
    ) -> None:
        """Test adapter invalidate_cache parameter."""
        adapter = OCREngineAdapter.from_config(ocr_config)

        # First call
        adapter.predict(test_image, text_only=False)

        # Second call with invalidate
        result = adapter.predict(
            test_image,
            text_only=False,
            invalidate_cache=True,
        )

        assert isinstance(result, tuple)


class TestOCREndToEndIntegration:
    """End-to-end integration tests spanning all OCR components."""

    def test_complete_pipeline_with_metadata(
        self,
        test_image: Image.Image,
        temp_cache_dir: Path,
    ) -> None:
        """Test complete pipeline including metadata tagging."""
        # Build complete pipeline
        engine = SimpleOCREngine(language="eng", psm_mode=6)
        cache_repo = OCRCacheRepository.create(
            language="eng", caches_dir=temp_cache_dir
        )
        use_case = GenerateOCRExtraction(engine=engine, cache_repository=cache_repo)

        # Execute with metadata
        request = GenerateOCRExtractionRequest(
            image=test_image,
            text_only=False,
            schematism="TestSchematism1900",
            filename="page_001.jpg",
        )
        response = use_case.execute(request)

        assert response.from_cache is False
        words, bboxes = response.result
        assert isinstance(words, list)
        assert isinstance(bboxes, list)

        # Verify cache entry exists with metadata
        key = cache_repo.generate_key(test_image)
        cached_item = cache_repo.get(key)
        assert cached_item is not None

    def test_adapter_integrates_with_all_components(
        self,
        test_image: Image.Image,
        ocr_config: DictConfig,
    ) -> None:
        """Test that adapter properly integrates _engine, cache, and use case."""
        adapter = OCREngineAdapter.from_config(ocr_config)

        # Verify all components initialized
        assert adapter.ocr_engine is not None
        assert isinstance(adapter.ocr_engine, SimpleOCREngine)
        assert adapter.cache_repository is not None
        assert isinstance(adapter.cache_repository, OCRCacheRepository)
        assert adapter.use_case is not None
        assert isinstance(adapter.use_case, GenerateOCRExtraction)

        # Verify prediction works through all layers
        result = adapter.predict(test_image, text_only=True)
        assert isinstance(result, str)
