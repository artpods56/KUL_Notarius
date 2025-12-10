"""Domain service for extracting structured data from schematism pages using LLM.

This service contains pure business logic for extraction, independent of
infrastructure concerns like caching, prompt rendering, or specific LLM providers.
"""

from dataclasses import dataclass
from typing import Any

from PIL.Image import Image as PILImage
from structlog import get_logger

from notarius.domain.entities.schematism import SchematismPage

logger = get_logger(__name__)


@dataclass(frozen=True)
class ExtractionInput:
    """Input data for extraction service."""

    image: PILImage | None = None
    text: str | None = None
    hints: dict[str, Any] | None = None
    metadata: dict[str, Any] | None = None

    def __post_init__(self):
        """Validate input."""
        if self.image is None and self.text is None:
            raise ValueError("At least one of 'image' or 'text' must be provided")


@dataclass(frozen=True)
class ExtractionResult:
    """Result of extraction service."""

    page: SchematismPage
    confidence: float
    from_cache: bool
    raw_response: dict[str, Any] | None = None


class LLMExtractionService:
    """Domain service for extracting structured schematism data.

    This service encapsulates the business logic for:
    - Validating extraction inputs
    - Post-processing extraction results
    - Applying business rules to extracted data
    - Computing confidence scores

    It does NOT:
    - Handle caching (infrastructure concern)
    - Render prompts (infrastructure concern)
    - Make API calls (infrastructure concern)
    """

    def validate_input(self, extraction_input: ExtractionInput) -> None:
        """Validate extraction input according to business rules.

        Args:
            extraction_input: The input to validate

        Raises:
            ValueError: If input violates business rules
        """
        if extraction_input.image is None and extraction_input.text is None:
            raise ValueError("Cannot extract from empty input")

        # Additional business validation
        if extraction_input.text and len(extraction_input.text.strip()) < 10:
            logger.warning(
                "Text input is very short, extraction may be unreliable",
                text_length=len(extraction_input.text),
            )

    def post_process_extraction(
        self, raw_result: dict[str, Any], extraction_input: ExtractionInput
    ) -> SchematismPage:
        """Apply post-processing and business rules to raw extraction result.

        Args:
            raw_result: Raw dictionary from LLM
            extraction_input: Original input (for context)

        Returns:
            Validated and post-processed SchematismPage
        """
        # Parse into domain model (this validates structure)
        page = SchematismPage(**raw_result)

        # Apply business rules
        page = self._apply_business_rules(page, extraction_input)

        return page

    def _apply_business_rules(
        self, page: SchematismPage, extraction_input: ExtractionInput
    ) -> SchematismPage:
        """Apply domain-specific business rules to extracted data.

        Examples:
        - Normalize dedication names
        - Standardize deanery names
        - Fill missing fields from hints
        - Apply domain constraints
        """
        # If hints provided, use them to fill missing data
        if extraction_input.hints:
            page = self._merge_hints(page, extraction_input.hints)

        # Additional business logic here
        # e.g., normalize names, validate hierarchies, etc.

        return page

    def _merge_hints(
        self, page: SchematismPage, hints: dict[str, Any]
    ) -> SchematismPage:
        """Intelligently merge hints into extracted page data.

        Args:
            page: Extracted page
            hints: Hints from previous models (e.g., LMv3)

        Returns:
            Page with hints merged where appropriate
        """
        # Example: If LLM didn't extract deanery but hint has it, use hint
        page_dict = page.model_dump()

        # Only fill fields that are None/empty
        for key, hint_value in hints.items():
            if key in page_dict and not page_dict[key]:
                page_dict[key] = hint_value
                logger.debug("Filled missing field from hint", field=key)

        return SchematismPage(**page_dict)

    def compute_confidence(
        self,
        page: SchematismPage,
        extraction_input: ExtractionInput,
        from_cache: bool,
    ) -> float:
        """Compute confidence score for extraction.

        Args:
            page: Extracted page
            extraction_input: Original input
            from_cache: Whether result came from cache

        Returns:
            Confidence score between 0 and 1
        """
        confidence = 1.0

        # Reduce confidence for cached results (may be stale)
        if from_cache:
            confidence *= 0.95

        # Reduce confidence if many fields are empty
        page_dict = page.model_dump()
        non_empty_fields = sum(1 for v in page_dict.values() if v)
        total_fields = len(page_dict)
        field_completion_rate = non_empty_fields / total_fields if total_fields > 0 else 0
        confidence *= field_completion_rate

        # Reduce confidence if input was low quality
        if extraction_input.text and len(extraction_input.text.strip()) < 50:
            confidence *= 0.8

        return max(0.0, min(1.0, confidence))