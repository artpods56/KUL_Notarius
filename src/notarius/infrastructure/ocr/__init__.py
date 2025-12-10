"""OCR infrastructure module."""

from .engine_adapter import OCREngine, OCRRequest, OCRResponse, OCRMode
from .types import StructuredOCRResult, PytesseractOCRResultDict

__all__ = [
    "OCRMode",
    "OCREngine",
    "OCRRequest",
    "OCRResponse",
    "StructuredOCRResult",
    "PytesseractOCRResultDict",
]
