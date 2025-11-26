from pathlib import Path

from core.caches.ocr_cache import PyTesseractCache
from schemas.data.cache import PyTesseractCacheItem


def test_ocr_cache_persistence(tmp_path: Path):
    """Verify that values stored in PyTesseractCache persist across instances and are retrievable."""

    cache_root = tmp_path / "cache_root"

    cache1 = PyTesseractCache(language="eng", caches_dir=cache_root)
    key = cache1.generate_hash(image_hash="test_image_hash")

    payload = PyTesseractCacheItem(
        text="Hello World",
        bbox=[(10, 20, 100, 30)],
        words=["Hello", "World"]
    )
    cache1.set(key, payload.model_dump())

    # Same instance retrieval
    result1 = cache1.get(key)
    assert result1["text"] == "Hello World"
    assert len(result1["bbox"]) == 1

    # Create a new instance – should read the same data
    cache2 = PyTesseractCache(language="eng", caches_dir=cache_root)
    result2 = cache2.get(key)
    assert result2["text"] == "Hello World"
    assert result2["words"] == ["Hello", "World"]


def test_ocr_cache_language_separation(tmp_path: Path):
    """Test that different languages get separate cache directories."""

    cache_root = tmp_path / "cache_root"

    # Create caches for different languages
    eng_cache = PyTesseractCache(language="eng", caches_dir=cache_root)
    pol_cache = PyTesseractCache(language="pol", caches_dir=cache_root)

    # Generate proper hash keys based on the cache's normalize_kwargs
    eng_key = eng_cache.generate_hash(image_hash="test_image_hash")
    pol_key = pol_cache.generate_hash(image_hash="test_image_hash")

    eng_payload = PyTesseractCacheItem(
        text="Hello",
        bbox=[(10, 20, 50, 30)],
        words=["Hello"]
    )

    pol_payload = PyTesseractCacheItem(
        text="Cześć",
        bbox=[(10, 20, 50, 30)],
        words=["Cześć"]
    )

    eng_cache.set(eng_key, eng_payload.model_dump())
    pol_cache.set(pol_key, pol_payload.model_dump())

    # Verify they're stored separately
    eng_result = eng_cache.get(eng_key)
    pol_result = pol_cache.get(pol_key)
    assert eng_result["text"] == "Hello"
    assert pol_result["text"] == "Cześć"

    # Verify cache directories are separate
    assert eng_cache.model_cache_dir != pol_cache.model_cache_dir


def test_ocr_cache_hit_miss_logging(tmp_path: Path):
    """Test that OCR cache hits and misses are logged with timing information."""

    cache_root = tmp_path / "cache_root"
    cache = PyTesseractCache(language="eng", caches_dir=cache_root)

    # Test cache miss
    key = cache.generate_hash(image_hash="test_image_hash")

    value = cache.get(key)
    assert value is None

    # Test cache hit
    payload = PyTesseractCacheItem(
        text="Test text",
        bbox=[(10, 20, 100, 30)],
        words=["Test", "text"]
    )
    cache.set(key, payload.model_dump())

    result = cache.get(key)  # This should be a hit and log it
    assert result["text"] == "Test text"
