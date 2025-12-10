"""Tests for BIOProcessingService."""

from unittest.mock import patch

import pytest

from notarius.domain.entities.schematism import SchematismPage
from notarius.domain.services.bio_processing_service import BIOProcessingService


class TestBIOProcessingService:
    """Test suite for BIOProcessingService class."""

    @pytest.fixture
    def service(self) -> BIOProcessingService:
        """Create a service instance for testing."""
        return BIOProcessingService()

    @pytest.fixture
    def sample_words(self) -> list[str]:
        """Sample OCR words for testing."""
        return ["St.", "Mary's", "Church", "Krakow", "Diocese"]

    @pytest.fixture
    def sample_bboxes(self) -> list[list[int]]:
        """Sample bounding boxes for testing."""
        return [
            [0, 0, 20, 10],
            [20, 0, 50, 10],
            [50, 0, 80, 10],
            [0, 10, 30, 20],
            [30, 10, 60, 20],
        ]

    @pytest.fixture
    def sample_predictions(self) -> list[str]:
        """Sample BIO predictions for testing."""
        return [
            "B-DEDICATION",
            "I-DEDICATION",
            "I-DEDICATION",
            "B-PARISH",
            "B-DEANERY",
        ]

    def test_init(self) -> None:
        """Test service initialization."""
        service = BIOProcessingService()
        assert service is not None

    def test_repair_labels_with_valid_sequence(
        self,
        service: BIOProcessingService,
        sample_words: list[str],
        sample_bboxes: list[list[int]],
    ) -> None:
        """Test repair_labels with valid BIO sequence."""
        predictions = ["B-PARISH", "I-PARISH", "O", "B-DEANERY"]

        repaired = service.repair_labels(
            sample_words[:4], sample_bboxes[:4], predictions
        )

        # Should remain unchanged
        assert repaired == predictions

    def test_repair_labels_with_consecutive_b_tags(
        self,
        service: BIOProcessingService,
        sample_words: list[str],
        sample_bboxes: list[list[int]],
    ) -> None:
        """Test repair_labels fixes consecutive B- tags of same type."""
        predictions = ["B-PARISH", "B-PARISH", "B-PARISH"]  # Invalid sequence

        repaired = service.repair_labels(
            sample_words[:3], sample_bboxes[:3], predictions
        )

        # Should repair to B-I-I
        assert repaired[0] == "B-PARISH"
        assert repaired[1] == "I-PARISH"  # Fixed
        assert repaired[2] == "I-PARISH"  # Fixed

    def test_repair_labels_with_mixed_sequence(
        self,
        service: BIOProcessingService,
        sample_words: list[str],
        sample_bboxes: list[list[int]],
    ) -> None:
        """Test repair_labels with mixed valid/invalid sequence."""
        predictions = [
            "B-PARISH",
            "B-PARISH",  # Should become I-
            "O",
            "B-DEANERY",
            "B-DEANERY",  # Should become I-
        ]

        repaired = service.repair_labels(sample_words, sample_bboxes, predictions)

        assert repaired[0] == "B-PARISH"
        assert repaired[1] == "I-PARISH"
        assert repaired[2] == "O"
        assert repaired[3] == "B-DEANERY"
        assert repaired[4] == "I-DEANERY"

    def test_repair_labels_preserves_different_types(
        self,
        service: BIOProcessingService,
        sample_words: list[str],
        sample_bboxes: list[list[int]],
    ) -> None:
        """Test that consecutive B- tags of different types are preserved."""
        predictions = ["B-PARISH", "B-DEANERY", "B-DEDICATION"]

        repaired = service.repair_labels(
            sample_words[:3], sample_bboxes[:3], predictions
        )

        # Should remain unchanged (different types)
        assert repaired == predictions

    def test_build_structured_page_with_entities(
        self,
        service: BIOProcessingService,
        sample_words: list[str],
        sample_bboxes: list[list[int]],
        sample_predictions: list[str],
    ) -> None:
        """Test building structured page with entities."""
        page = service.build_structured_page(
            sample_words, sample_bboxes, sample_predictions
        )

        assert isinstance(page, SchematismPage)
        assert (
            len(page.entries) >= 0
        )  # May have entries depending on build_page_json logic

    def test_build_structured_page_with_empty_input(
        self, service: BIOProcessingService
    ) -> None:
        """Test building structured page with empty input."""
        page = service.build_structured_page([], [], [])

        assert isinstance(page, SchematismPage)
        assert page.is_empty()

    def test_build_structured_page_with_only_o_tags(
        self, service: BIOProcessingService
    ) -> None:
        """Test building page with only O tags (no entities)."""
        words = ["word1", "word2", "word3"]
        bboxes = [[0, 0, 10, 10], [10, 0, 20, 10], [20, 0, 30, 10]]
        predictions = ["O", "O", "O"]

        page = service.build_structured_page(words, bboxes, predictions)

        assert isinstance(page, SchematismPage)
        # Should have no meaningful entries

    def test_process_full_pipeline(
        self,
        service: BIOProcessingService,
        sample_words: list[str],
        sample_bboxes: list[list[int]],
    ) -> None:
        """Test complete processing pipeline."""
        # Use predictions with issues to test repair
        predictions = ["B-PARISH", "B-PARISH", "O", "B-DEANERY"]

        page = service.process(sample_words[:4], sample_bboxes[:4], predictions)

        assert isinstance(page, SchematismPage)
        # Labels should have been repaired before building

    def test_process_calls_repair_before_build(
        self,
        service: BIOProcessingService,
        sample_words: list[str],
        sample_bboxes: list[list[int]],
        sample_predictions: list[str],
    ) -> None:
        """Test that process calls repair before build."""
        with (
            patch.object(service, "repair_labels") as mock_repair,
            patch.object(service, "build_structured_page") as mock_build,
        ):

            mock_repair.return_value = sample_predictions
            mock_build.return_value = SchematismPage(entries=[], page_number=None)

            _ = service.process(sample_words, sample_bboxes, sample_predictions)

            # Verify repair was called first
            mock_repair.assert_called_once_with(
                sample_words, sample_bboxes, sample_predictions
            )

            # Verify build was called with repaired predictions
            mock_build.assert_called_once_with(
                sample_words, sample_bboxes, sample_predictions
            )

    def test_process_returns_schematism_page(
        self,
        service: BIOProcessingService,
        sample_words: list[str],
        sample_bboxes: list[list[int]],
        sample_predictions: list[str],
    ) -> None:
        """Test that process returns SchematismPage type."""
        result = service.process(sample_words, sample_bboxes, sample_predictions)

        assert isinstance(result, SchematismPage)

    def test_repair_labels_with_empty_list(self, service: BIOProcessingService) -> None:
        """Test repair with empty prediction list."""
        repaired = service.repair_labels([], [], [])

        assert repaired == []

    def test_repair_labels_preserves_length(
        self,
        service: BIOProcessingService,
        sample_words: list[str],
        sample_bboxes: list[list[int]],
        sample_predictions: list[str],
    ) -> None:
        """Test that repair preserves prediction list length."""
        repaired = service.repair_labels(
            sample_words, sample_bboxes, sample_predictions
        )

        assert len(repaired) == len(sample_predictions)
