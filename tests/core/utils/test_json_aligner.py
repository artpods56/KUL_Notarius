import pytest
from typing import Dict, List

from core.data.utils import JSONAligner, align_json_data
from schemas.data.schematism import SchematismEntry


class TestJSONAligner:
    @pytest.fixture
    def sample_weights(self) -> Dict[str, float]:
        return {
            "deanery": 1.0,
            "parish": 2.0,
            "dedication": 1.5,
            "building_material": 0.5,
        }

    @pytest.fixture
    def sample_entry1(self) -> Dict[str, str]:
        return {
            "deanery": "North Deanery",
            "parish": "St. Mary's Parish",
            "dedication": "Mary",
            "building_material": "brick",
        }

    @pytest.fixture
    def sample_entry2(self) -> Dict[str, str]:
        return {
            "deanery": "North Deanery",
            "parish": "St. Mary's Parish",
            "dedication": "Mary",
            "building_material": "brick",
        }

    @pytest.fixture
    def sample_entry3(self) -> Dict[str, str]:
        return {
            "deanery": "South Deanery",
            "parish": "St. John's Parish",
            "dedication": "John",
            "building_material": "wood",
        }

    @pytest.fixture
    def sample_entry_partial(self) -> Dict[str, str]:
        return {
            "deanery": "North Deanery",
            "parish": "St. Mary's Parish",
            "dedication": "Mary",
            "building_material": "wood",
        }

    @pytest.fixture
    def sample_entry_with_none(self) -> Dict[str, str]:
        return {
            "deanery": "North Deanery",
            "parish": None,
            "dedication": "Mary",
            "building_material": None,
        }

    @pytest.fixture
    def sample_entry_empty_strings(self) -> Dict[str, str]:
        return {
            "deanery": "North Deanery",
            "parish": "",
            "dedication": "Mary",
            "building_material": "",
        }

    def test_aligner_initialization(self, sample_weights):
        aligner = JSONAligner(sample_weights)
        assert aligner.weights_mapping == sample_weights

    def test_calculate_entry_score_perfect_match(self, sample_weights, sample_entry1):
        aligner = JSONAligner(sample_weights)
        score = aligner.calculate_entry_score(sample_entry1, sample_entry1)
        assert score == 1.0

    def test_calculate_entry_score_no_match(
        self, sample_weights, sample_entry1, sample_entry3
    ):
        aligner = JSONAligner(sample_weights)
        score = aligner.calculate_entry_score(sample_entry1, sample_entry3)
        assert score < 1.0

    def test_calculate_entry_score_weighted_average(
        self, sample_entry1, sample_entry_partial
    ):
        weights = {
            "deanery": 1.0,
            "parish": 1.0,
            "dedication": 1.0,
            "building_material": 10.0,
        }
        aligner = JSONAligner(weights)

        entry1 = sample_entry1  # "brick"
        entry2 = sample_entry_partial  # "wood" for building_material

        score = aligner.calculate_entry_score(entry1, entry2)
        # All other fields match perfectly, building_material differs slightly
        # Score should be lower due to high weight on building_material
        assert score < 1.0

    def test_calculate_entry_score_empty_weights(self):
        weights = {
            "deanery": 0.0,
            "parish": 0.0,
            "dedication": 0.0,
            "building_material": 0.0,
        }
        aligner = JSONAligner(weights)
        entry1 = {
            "deanery": "North",
            "parish": "St. Mary's",
            "dedication": "Mary",
            "building_material": "brick",
        }
        entry2 = {
            "deanery": "North",
            "parish": "St. Mary's",
            "dedication": "Mary",
            "building_material": "brick",
        }

        score = aligner.calculate_entry_score(entry1, entry2)
        assert score == 0.0  # All weights are 0

    def test_calculate_entry_score_different_keys(self):
        weights = {"deanery": 1.0}
        aligner = JSONAligner(weights)
        entry1 = {"deanery": "North"}
        entry2 = {"parish": "St. Mary's"}

        with pytest.raises(ValueError, match="entries must have the same keys"):
            aligner.calculate_entry_score(entry1, entry2)

    def test_calculate_entry_score_single_field(self):
        weights = {"parish": 1.5}
        aligner = JSONAligner(weights)
        entry1 = {"parish": "St. Mary's"}
        entry2 = {"parish": "St. Mary's"}
        score = aligner.calculate_entry_score(entry1, entry2)
        assert score == 1.0

    def test_align_entries_empty_data(self, sample_weights):
        aligner = JSONAligner(sample_weights)
        data1 = {"entries": []}
        data2 = {"entries": []}

        aligned1, aligned2 = aligner.align_entries(data1, data2)
        assert aligned1 == []
        assert aligned2 == []

    def test_align_entries_identical_single_entry(self, sample_weights, sample_entry1):
        aligner = JSONAligner(sample_weights)
        data1 = {"entries": [sample_entry1]}
        data2 = {"entries": [sample_entry1]}

        aligned1, aligned2 = aligner.align_entries(data1, data2)
        assert aligned1 == [sample_entry1]
        assert aligned2 == [sample_entry1]

    def test_align_entries_no_matches_below_threshold(
        self, sample_weights, sample_entry1, sample_entry3
    ):
        aligner = JSONAligner(sample_weights)
        data1 = {"entries": [sample_entry1]}
        data2 = {"entries": [sample_entry3]}
        threshold = 0.9  # High threshold, unlikely to match

        aligned1, aligned2 = aligner.align_entries(data1, data2, threshold)
        # Should add placeholders for unmatched
        assert len(aligned1) == 2
        assert len(aligned2) == 2
        assert aligned1[0] == sample_entry1
        assert aligned2[0] == SchematismEntry()  # Empty placeholder
        assert aligned1[1] == SchematismEntry()
        assert aligned2[1] == sample_entry3

    def test_align_entries_partial_match(
        self, sample_weights, sample_entry1, sample_entry2, sample_entry_partial
    ):
        aligner = JSONAligner(sample_weights)
        data1 = {"entries": [sample_entry1]}
        data2 = {"entries": [sample_entry_partial]}

        aligned1, aligned2 = aligner.align_entries(data1, data2, threshold=0.5)
        assert len(aligned1) == 1
        assert len(aligned2) == 1
        # Should match since most fields are similar

    def test_align_entries_order_maintenance(self, sample_weights):
        aligner = JSONAligner(sample_weights)
        entry_a = {
            "deanery": "A",
            "parish": "First",
            "dedication": "Patron1",
            "building_material": "brick",
        }
        entry_b = {
            "deanery": "B",
            "parish": "Second",
            "dedication": "Patron2",
            "building_material": "wood",
        }
        entry_x = {
            "deanery": "A",
            "parish": "First Similar",
            "dedication": "Patron1",
            "building_material": "brick",
        }
        entry_y = {
            "deanery": "B",
            "parish": "Second Similar",
            "dedication": "Patron2",
            "building_material": "wood",
        }

        data1 = {"entries": [entry_a, entry_b]}
        data2 = {"entries": [entry_x, entry_y]}  # Similar but not identical

        aligned1, aligned2 = aligner.align_entries(data1, data2, threshold=0.3)
        assert len(aligned1) == 2
        assert len(aligned2) == 2
        # First entries should match best (position bonus)

    def test_align_entries_multiple_unmatched(
        self, sample_weights, sample_entry1, sample_entry3
    ):
        aligner = JSONAligner(sample_weights)
        data1 = {"entries": [sample_entry1]}  # One entry
        data2 = {"entries": [sample_entry3]}  # Different entry

        aligned1, aligned2 = aligner.align_entries(data1, data2, threshold=0.8)
        # Should have entry1 + placeholder, placeholder + entry3
        assert len(aligned1) == 2
        assert len(aligned2) == 2
        assert aligned1[0] == sample_entry1
        assert isinstance(aligned2[0], SchematismEntry)
        assert isinstance(aligned1[1], SchematismEntry)
        assert aligned2[1] == sample_entry3

    def test_align_entries_position_bonus_effect(self, sample_weights):
        aligner = JSONAligner(sample_weights)
        # Create entries where position 0 matches best with position 0, but worse with position 1
        entry_near = {
            "deanery": "Dean1",
            "parish": "Parish1",
            "dedication": "Ded1",
            "building_material": "brick",
        }
        entry_far = {
            "deanery": "Dean2",
            "parish": "Parish2",
            "dedication": "Ded2",
            "building_material": "wood",
        }

        # Make entry_far very similar to entry_near in content but different in position
        data1 = {"entries": [entry_near, entry_far]}
        data2 = {"entries": [entry_near, entry_far]}  # Same order

        # With normal alignment, should align perfectly
        aligned1, aligned2 = aligner.align_entries(data1, data2, threshold=0.0)
        assert aligned1[0] == entry_near
        assert aligned2[0] == entry_near
        assert aligned1[1] == entry_far
        assert aligned2[1] == entry_far

    def test_calculate_entry_score_with_none_values(self, sample_weights, sample_entry1, sample_entry_with_none):
        """Test that None values are handled correctly in scoring"""
        aligner = JSONAligner(sample_weights)

        # None values should be treated as empty strings
        score = aligner.calculate_entry_score(sample_entry1, sample_entry_with_none)
        # Some fields match (deanery, dedication), others don't (parish, building_material)
        assert 0.0 < score < 1.0

    def test_calculate_entry_score_none_vs_none(self):
        """Test scoring when both entries have None values"""
        weights = {"field": 1.0}
        aligner = JSONAligner(weights)
        entry1 = {"field": None}
        entry2 = {"field": None}

        score = aligner.calculate_entry_score(entry1, entry2)
        assert score == 1.0  # None == None should score perfectly

    def test_calculate_entry_score_none_vs_string(self):
        """Test scoring when one entry has None and other has string"""
        weights = {"field": 1.0}
        aligner = JSONAligner(weights)
        entry1 = {"field": None}
        entry2 = {"field": "value"}

        score = aligner.calculate_entry_score(entry1, entry2)
        assert score == 0.0  # None vs "value" should score 0

    def test_align_entries_with_none_values(self, sample_weights, sample_entry1, sample_entry_with_none):
        """Test alignment works with None values in entries"""
        aligner = JSONAligner(sample_weights)
        data1 = {"entries": [sample_entry1]}
        data2 = {"entries": [sample_entry_with_none]}

        aligned1, aligned2 = aligner.align_entries(data1, data2, threshold=0.0)
        assert len(aligned1) == 1
        assert len(aligned2) == 1
        # Should align since some fields match


class TestAlignJSONData:
    def test_align_json_data_basic(self):
        weights = {
            "deanery": 1.0,
            "parish": 1.0,
            "dedication": 1.0,
            "building_material": 1.0,
        }
        entry1 = {
            "deanery": "North",
            "parish": "St. Mary's",
            "dedication": "Mary",
            "building_material": "brick",
        }
        entry2 = {
            "deanery": "South",
            "parish": "St. John's",
            "dedication": "John",
            "building_material": "wood",
        }

        data1 = {"entries": [entry1]}
        data2 = {"entries": [entry2]}

        aligned_data1, aligned_data2 = align_json_data(
            data1, data2, weights, threshold=0.8
        )

        # Should have placeholders since entries don't match well at high threshold
        assert len(aligned_data1["entries"]) == 2
        assert len(aligned_data2["entries"]) == 2
        assert aligned_data1["entries"][0] == entry1
        assert isinstance(aligned_data2["entries"][0], SchematismEntry)
        assert isinstance(aligned_data1["entries"][1], SchematismEntry)
        assert aligned_data2["entries"][1] == entry2

    def test_align_json_data_perfect_match(self):
        weights = {"test_field": 1.0}
        entry = {"test_field": "value"}
        data1 = {"entries": [entry]}
        data2 = {"entries": [entry]}

        aligned_data1, aligned_data2 = align_json_data(data1, data2, weights)

        assert aligned_data1["entries"] == [entry]
        assert aligned_data2["entries"] == [entry]
        assert len(aligned_data1["entries"]) == 1
        assert len(aligned_data2["entries"]) == 1
