"""Service for BIO label processing and structured output building."""

from structlog import get_logger

from notarius.domain.entities.schematism import SchematismPage
from notarius.infrastructure.persistence.bio_parser import (
    repair_bio_labels,
    build_page_json,
)

logger = get_logger(__name__)


class BIOProcessingService:
    """Service for processing BIO-tagged predictions into structured output.

    This service encapsulates domain logic for:
    - Repairing BIO label sequences
    - Building SchematismPage entities from predictions
    """

    def __init__(self):
        """Initialize the BIO processing service."""
        self.logger = logger

    def repair_labels(
        self,
        words: list[str],
        bboxes: list[list[int]],
        predictions: list[str],
    ) -> list[str]:
        """Repair BIO label sequences.

        Fixes invalid BIO sequences such as consecutive B- tags
        of the same type.

        Args:
            words: List of OCR words
            bboxes: Bounding boxes for each word
            predictions: Raw BIO predictions from model

        Returns:
            Repaired BIO predictions
        """
        repaired = repair_bio_labels(predictions)
        self.logger.debug(
            "Repaired BIO labels",
            original_count=len(predictions),
            repaired_count=len(repaired),
        )
        return repaired

    def build_structured_page(
        self,
        words: list[str],
        bboxes: list[list[int]],
        predictions: list[str],
    ) -> SchematismPage:
        """Build SchematismPage from BIO predictions.

        Converts BIO-tagged predictions into a structured
        SchematismPage entity with entries.

        Args:
            words: List of OCR words
            bboxes: Bounding boxes for each word
            predictions: Repaired BIO predictions

        Returns:
            Structured SchematismPage entity
        """
        # Build JSON representation
        page_json = build_page_json(words, bboxes, predictions)

        # Convert to domain entity
        page = SchematismPage(**page_json)

        self.logger.debug(
            "Built structured page",
            num_words=len(words),
            num_entries=len(page.entries),
        )

        return page

    def process(
        self,
        words: list[str],
        bboxes: list[list[int]],
        predictions: list[str],
    ) -> SchematismPage:
        """Full processing pipeline: repair + build.

        This is the primary method that should be used for
        complete BIO processing.

        Args:
            words: List of OCR words
            bboxes: Bounding boxes for each word
            predictions: Raw BIO predictions from model

        Returns:
            Structured SchematismPage entity
        """
        # Step 1: Repair labels
        repaired_predictions = self.repair_labels(words, bboxes, predictions)

        # Step 2: Build structured output
        page = self.build_structured_page(words, bboxes, repaired_predictions)

        self.logger.info(
            "Processed BIO predictions",
            num_words=len(words),
            num_entries=len(page.entries),
        )

        return page
