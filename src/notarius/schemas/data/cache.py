from pydantic import BaseModel
from typing import Any

from notarius.application.ports.outbound.cache import BaseCacheItem
from notarius.domain.entities.schematism import SchematismPage
from notarius.schemas.data.structs import BBox


# LLM Cache Content Models (kept for backwards compatibility with backends)
class LLMContent(BaseModel):
    """Content model for LLM responses.

    Note: The new LLMCache uses pickle directly and doesn't need this wrapper.
    This is kept for backwards compatibility with the backends layer.
    """

    text: str
    structured_output: dict[str, Any] | None = None


# LLM Cache Items
class LLMCacheItem(BaseCacheItem[LLMContent]):
    """Cache item model for LLM responses.

    Note: The new LLMCache uses pickle directly and doesn't need this wrapper.
    This is kept for backwards compatibility with the backends layer.
    """

    pass


# OCR Cache Content Models
class PyTesseractContent(BaseModel):
    """Content model for PyTesseract OCR results."""

    text: str
    bbox: list[BBox] | None
    words: list[str] | None
    language: str


# OCR Cache Items
class PyTesseractCacheItem(BaseCacheItem[PyTesseractContent]):
    """Cache item model for PyTesseract OCR ml_models."""

    pass


# LMv3 Cache Content Models
class LMv3Content(BaseModel):
    """Content model for LayoutLMv3 predictions."""

    raw_predictions: tuple[list[Any], list[Any], list[Any]] | None
    structured_predictions: SchematismPage


# LMv3 Cache Items
class LMv3CacheItem(BaseCacheItem[LMv3Content]):
    """Cache item model for LMv3 ml_models."""

    pass
