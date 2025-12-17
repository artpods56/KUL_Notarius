"""Tests for the aligner module."""

import pytest

from notarius.domain.services.aligner import FlatHungarianAligner, JSONAligner


class TestFlatHungarianAligner:
    """Tests for FlatHungarianAligner."""

    @pytest.fixture
    def default_weights(self) -> dict[str, float]:
        return {
            "deanery": 1.0,
            "parish": 2.0,
            "dedication": 1.5,
            "building_material": 0.5,
        }

    @pytest.fixture
    def aligner(self, default_weights: dict[str, float]) -> FlatHungarianAligner:
        return FlatHungarianAligner(weights_mapping=default_weights, threshold=0.5)

    @pytest.fixture
    def real_gt_entries(self) -> list[dict[str, str]]:
        """Real ground truth data sample."""
        return [
            {
                "deanery": "Decanatus Neo-Radomscensis",
                "parish": "Kruszyna",
                "dedication": "S. Mathias Ap. Patr. S. Joseph. Spons. BV.",
                "building_material": "mur.",
            },
            {
                "deanery": "Decanatus Neo-Radomscensis",
                "parish": "Lgota",
                "dedication": "S. Clemens PM.",
                "building_material": "lig.",
            },
            {
                "deanery": "Decanatus Neo-Radomscensis",
                "parish": "Makowiska",
                "dedication": "Patroc. S. Joseph. Patr. S. Barthol. Ap.",
                "building_material": "mur.",
            },
        ]

    def test_empty_lists(self, aligner: FlatHungarianAligner) -> None:
        """Test alignment of two empty lists."""
        result1, result2 = aligner.align_entries([], [])
        assert result1 == []
        assert result2 == []

    def test_empty_first_list(self, aligner: FlatHungarianAligner) -> None:
        """Test alignment when first list is empty."""
        list2 = [{"deanery": "A", "parish": "B", "dedication": "C", "building_material": "D"}]
        result1, result2 = aligner.align_entries([], list2)

        assert len(result1) == 1
        assert len(result2) == 1
        assert result2 == list2
        # result1 should contain empty SchematismEntry (fields are None)
        assert result1[0]["parish"] is None

    def test_empty_second_list(self, aligner: FlatHungarianAligner) -> None:
        """Test alignment when second list is empty."""
        list1 = [{"deanery": "A", "parish": "B", "dedication": "C", "building_material": "D"}]
        result1, result2 = aligner.align_entries(list1, [])

        assert len(result1) == 1
        assert len(result2) == 1
        assert result1 == list1
        # result2 should contain empty SchematismEntry (fields are None)
        assert result2[0]["parish"] is None

    def test_perfect_match(self, aligner: FlatHungarianAligner) -> None:
        """Test alignment with identical entries."""
        entries = [
            {"deanery": "Decanatus A", "parish": "Parish1", "dedication": "Ded1", "building_material": "mur."},
            {"deanery": "Decanatus A", "parish": "Parish2", "dedication": "Ded2", "building_material": "lig."},
        ]

        result1, result2 = aligner.align_entries(entries.copy(), entries.copy())

        assert len(result1) == len(result2) == 2
        # Perfect matches should align
        for r1, r2 in zip(result1, result2):
            assert r1["parish"] == r2["parish"]

    def test_real_data_perfect_match(
        self, aligner: FlatHungarianAligner, real_gt_entries: list[dict[str, str]]
    ) -> None:
        """Test alignment with real data - perfect match scenario."""
        # Same entries should align perfectly
        result1, result2 = aligner.align_entries(
            real_gt_entries.copy(), real_gt_entries.copy()
        )

        assert len(result1) == len(result2) == 3
        for r1, r2 in zip(result1, result2):
            assert r1["parish"] == r2["parish"]
            assert r1["deanery"] == r2["deanery"]

    def test_real_data_with_ocr_errors(
        self, aligner: FlatHungarianAligner, real_gt_entries: list[dict[str, str]]
    ) -> None:
        """Test alignment with simulated OCR errors in predictions."""
        pred_entries = [
            {
                "deanery": "Decanatus Neo-Radomscensis",
                "parish": "Kruszyna",  # Exact match
                "dedication": "S. Mathias Ap. Patr. S. Joseph Spons. BV",  # Missing dot
                "building_material": "mur",  # Missing dot
            },
            {
                "deanery": "Decanatus Neo-Radomscensis",
                "parish": "Lgotta",  # Typo: extra 't'
                "dedication": "S. Clemens PM",
                "building_material": "lig.",
            },
            {
                "deanery": "Decanatus Neo-Radomscensis",
                "parish": "Makowiska",
                "dedication": "Patroc. S. Joseph Patr. S. Barthol. Ap.",
                "building_material": "mur.",
            },
        ]

        result1, result2 = aligner.align_entries(real_gt_entries, pred_entries)

        assert len(result1) == len(result2) == 3
        # Entries should be matched correctly despite small errors
        parishes_gt = {r["parish"] for r in result1}
        parishes_pred = {r["parish"] for r in result2}

        assert "Kruszyna" in parishes_gt
        assert "Kruszyna" in parishes_pred
        assert "Makowiska" in parishes_gt
        assert "Makowiska" in parishes_pred

    def test_real_data_different_order(
        self, aligner: FlatHungarianAligner, real_gt_entries: list[dict[str, str]]
    ) -> None:
        """Test alignment when predictions are in different order."""
        # Reverse the order of predictions
        pred_entries = list(reversed(real_gt_entries))

        result1, result2 = aligner.align_entries(real_gt_entries, pred_entries)

        assert len(result1) == len(result2) == 3
        # Hungarian algorithm should find optimal matching regardless of order
        for r1, r2 in zip(result1, result2):
            assert r1["parish"] == r2["parish"]

    def test_real_data_missing_prediction(
        self, aligner: FlatHungarianAligner, real_gt_entries: list[dict[str, str]]
    ) -> None:
        """Test alignment when prediction is missing an entry."""
        # Only first two predictions
        pred_entries = real_gt_entries[:2].copy()

        result1, result2 = aligner.align_entries(real_gt_entries, pred_entries)

        assert len(result1) == len(result2) == 3
        # Two should match, one should have empty placeholder (None)
        empty_count = sum(1 for r in result2 if r["parish"] is None)
        assert empty_count == 1

    def test_real_data_extra_prediction(
        self, aligner: FlatHungarianAligner, real_gt_entries: list[dict[str, str]]
    ) -> None:
        """Test alignment when prediction has extra entry."""
        pred_entries = real_gt_entries.copy() + [
            {
                "deanery": "Decanatus Neo-Radomscensis",
                "parish": "ExtraParish",
                "dedication": "Extra Dedication",
                "building_material": "mur.",
            }
        ]

        result1, result2 = aligner.align_entries(real_gt_entries, pred_entries)

        assert len(result1) == len(result2) == 4
        # Three should match, one GT should be empty placeholder (None)
        empty_gt_count = sum(1 for r in result1 if r["parish"] is None)
        assert empty_gt_count == 1

    def test_threshold_filtering(self, default_weights: dict[str, float]) -> None:
        """Test that entries below threshold are not matched."""
        aligner = FlatHungarianAligner(weights_mapping=default_weights, threshold=0.9)

        list1 = [{"deanery": "A", "parish": "Parish1", "dedication": "Ded1", "building_material": "mur."}]
        list2 = [{"deanery": "B", "parish": "Different", "dedication": "Other", "building_material": "lig."}]

        result1, result2 = aligner.align_entries(list1, list2)

        # With high threshold, dissimilar entries should not match
        # Both should appear with empty counterparts
        assert len(result1) == len(result2) == 2

    def test_similarity_scoring(self, aligner: FlatHungarianAligner) -> None:
        """Test internal similarity scoring."""
        assert aligner._similarity("test", "test") == 1.0
        assert aligner._similarity("test", "tset") < 1.0
        assert aligner._similarity("test", "tset") > 0.5
        assert aligner._similarity("abc", "xyz") < 0.5

    def test_weighted_scoring(self, aligner: FlatHungarianAligner) -> None:
        """Test that weights affect scoring properly."""
        # Parish has weight 2.0, building_material has 0.5
        d1 = {"deanery": "A", "parish": "Same", "dedication": "X", "building_material": "diff1"}
        d2 = {"deanery": "A", "parish": "Same", "dedication": "X", "building_material": "diff2"}

        d3 = {"deanery": "A", "parish": "Different", "dedication": "X", "building_material": "same"}
        d4 = {"deanery": "A", "parish": "VeryDiff", "dedication": "X", "building_material": "same"}

        # Same parish should score higher than same building_material
        score_same_parish = aligner._score(d1, d2)
        score_same_building = aligner._score(d3, d4)

        assert score_same_parish > score_same_building


class TestJSONAligner:
    """Tests for JSONAligner for comparison."""

    @pytest.fixture
    def default_weights(self) -> dict[str, float]:
        return {
            "deanery": 1.0,
            "parish": 2.0,
            "dedication": 1.5,
            "building_material": 0.5,
        }

    @pytest.fixture
    def aligner(self, default_weights: dict[str, float]) -> JSONAligner:
        return JSONAligner(weights_mapping=default_weights, threshold=0.5)

    @pytest.fixture
    def real_gt_entries(self) -> list[dict[str, str]]:
        """Real ground truth data sample."""
        return [
            {
                "deanery": "Decanatus Neo-Radomscensis",
                "parish": "Kruszyna",
                "dedication": "S. Mathias Ap. Patr. S. Joseph. Spons. BV.",
                "building_material": "mur.",
            },
            {
                "deanery": "Decanatus Neo-Radomscensis",
                "parish": "Lgota",
                "dedication": "S. Clemens PM.",
                "building_material": "lig.",
            },
            {
                "deanery": "Decanatus Neo-Radomscensis",
                "parish": "Makowiska",
                "dedication": "Patroc. S. Joseph. Patr. S. Barthol. Ap.",
                "building_material": "mur.",
            },
        ]

    def test_real_data_perfect_match(
        self, aligner: JSONAligner, real_gt_entries: list[dict[str, str]]
    ) -> None:
        """Test JSONAligner with real data - perfect match scenario."""
        result1, result2 = aligner.align_entries(
            {"entries": real_gt_entries.copy()},
            {"entries": real_gt_entries.copy()},
        )

        assert len(result1) == len(result2) == 3
        for r1, r2 in zip(result1, result2):
            assert r1["parish"] == r2["parish"]

    def test_real_data_different_order(
        self, aligner: JSONAligner, real_gt_entries: list[dict[str, str]]
    ) -> None:
        """Test JSONAligner with real data in different order."""
        pred_entries = list(reversed(real_gt_entries))

        result1, result2 = aligner.align_entries(
            {"entries": real_gt_entries},
            {"entries": pred_entries},
        )

        assert len(result1) == len(result2) == 3
        # Greedy aligner may not find optimal matching with different order
        # Just verify it produces valid output
        assert all("parish" in r for r in result1)
        assert all("parish" in r for r in result2)


class TestAlignerComparison:
    """Compare both aligners on the same data."""

    @pytest.fixture
    def default_weights(self) -> dict[str, float]:
        return {
            "deanery": 1.0,
            "parish": 2.0,
            "dedication": 1.5,
            "building_material": 0.5,
        }

    @pytest.fixture
    def real_gt_entries(self) -> list[dict[str, str]]:
        return [
            {
                "deanery": "Decanatus Neo-Radomscensis",
                "parish": "Kruszyna",
                "dedication": "S. Mathias Ap. Patr. S. Joseph. Spons. BV.",
                "building_material": "mur.",
            },
            {
                "deanery": "Decanatus Neo-Radomscensis",
                "parish": "Lgota",
                "dedication": "S. Clemens PM.",
                "building_material": "lig.",
            },
            {
                "deanery": "Decanatus Neo-Radomscensis",
                "parish": "Makowiska",
                "dedication": "Patroc. S. Joseph. Patr. S. Barthol. Ap.",
                "building_material": "mur.",
            },
        ]

    def test_both_aligners_handle_shuffled_order(
        self, default_weights: dict[str, float], real_gt_entries: list[dict[str, str]]
    ) -> None:
        """Compare how both aligners handle shuffled predictions."""
        hungarian = FlatHungarianAligner(weights_mapping=default_weights, threshold=0.5)
        greedy = JSONAligner(weights_mapping=default_weights, threshold=0.5)

        # Shuffle: [2, 0, 1] order
        pred_entries = [real_gt_entries[2], real_gt_entries[0], real_gt_entries[1]]

        h_result1, h_result2 = hungarian.align_entries(real_gt_entries, pred_entries)
        g_result1, g_result2 = greedy.align_entries(
            {"entries": real_gt_entries}, {"entries": pred_entries}
        )

        # Hungarian should find optimal matching
        hungarian_matches = sum(
            1 for r1, r2 in zip(h_result1, h_result2) if r1["parish"] == r2["parish"]
        )

        # Greedy may or may not find all matches
        greedy_matches = sum(
            1 for r1, r2 in zip(g_result1, g_result2) if r1["parish"] == r2["parish"]
        )

        # Hungarian should be at least as good as greedy
        assert hungarian_matches >= greedy_matches
        # With this specific data, Hungarian should find all 3 matches
        assert hungarian_matches == 3
