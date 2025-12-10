"""Type definitions for OCR _engine."""

from dataclasses import dataclass
from typing import TypedDict

from notarius.schemas.data.structs import BBox


PytesseractOCRResultDict = TypedDict(
    "PytesseractOCRResultDict",
    {
        "level": list[int],
        "page_num": list[int],
        "block_num": list[int],
        "par_num": list[int],
        "line_num": list[int],
        "word_num": list[int],
        "left": list[int],
        "top": list[int],
        "width": list[int],
        "height": list[int],
        "conf": list[int],
        "text": list[str],
    },
)


@dataclass(frozen=True)
class StructuredOCRResult:
    """Result from OCR processing containing words and bounding boxes.

    Attributes:
        words: List of extracted words from the image
        bboxes: List of normalized bounding boxes (0-1000 range) for each word
    """

    words: list[str]
    bboxes: list[BBox]

@dataclass(frozen=True)
class SimpleOCRResult:
    text: str

OCRResult = StructuredOCRResult | SimpleOCRResult