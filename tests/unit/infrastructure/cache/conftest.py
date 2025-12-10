"""Common fixtures for cache tests."""

import tempfile
from pathlib import Path

import pytest
from PIL import Image

from notarius.domain.entities.schematism import SchematismPage, SchematismEntry
from notarius.schemas.data.cache import (
    PyTesseractContent,
    PyTesseractCacheItem,
    LMv3Content,
    LMv3CacheItem,
    LLMContent,
    LLMCacheItem,
)


@pytest.fixture
def temp_cache_dir():
    """Create a temporary directory for cache testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_image():
    """Create a simple test image."""
    return Image.new("RGB", (100, 100), color="white")


@pytest.fixture
def sample_schematism_page() -> SchematismPage:
    """Create a sample SchematismPage for testing."""
    return SchematismPage(
        page_number="42",
        entries=[
            SchematismEntry(
                parish="Test Parish",
                deanery="Test Deanery",
                dedication="St. Test",
                building_material="mur.",
            )
        ],
    )


@pytest.fixture
def sample_ocr_content() -> PyTesseractContent:
    """Create sample OCR content."""
    return PyTesseractContent(
        text="Sample OCR text",
        bbox=[(0, 0, 100, 20), (0, 25, 150, 45)],
        words=["Sample", "OCR", "text"],
        language="eng",
    )


@pytest.fixture
def sample_ocr_cache_item(sample_ocr_content) -> PyTesseractCacheItem:
    """Create sample OCR cache item."""
    return PyTesseractCacheItem(content=sample_ocr_content)


@pytest.fixture
def sample_lmv3_content(sample_schematism_page) -> LMv3Content:
    """Create sample LMv3 content."""
    return LMv3Content(
        raw_predictions=(
            [(0, 0, 100, 20)],  # bboxes
            [1, 2, 3],  # prediction_ids
            ["Test", "Parish"],  # words
        ),
        structured_predictions=sample_schematism_page,
    )


@pytest.fixture
def sample_lmv3_cache_item(sample_lmv3_content) -> LMv3CacheItem:
    """Create sample LMv3 cache item."""
    return LMv3CacheItem(content=sample_lmv3_content)


@pytest.fixture
def sample_llm_content() -> LLMContent:
    """Create sample LLM content."""
    return LLMContent(
        response={
            "page_number": "42",
            "entries": [
                {
                    "parish": "Test Parish",
                    "deanery": "Test Deanery",
                    "dedication": "St. Test",
                    "building_material": "mur.",
                }
            ],
        },
        hints={"context": "test"},
    )


@pytest.fixture
def sample_llm_cache_item(sample_llm_content) -> LLMCacheItem:
    """Create sample LLM cache item."""
    return LLMCacheItem(content=sample_llm_content)
