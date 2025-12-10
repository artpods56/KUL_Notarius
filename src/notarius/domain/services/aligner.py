from difflib import SequenceMatcher
from typing import final, Dict, Tuple, List

from scipy import optimize

from notarius.domain.entities.schematism import SchematismEntry


@final
class FlatHungarianAligner:
    def __init__(self, weights_mapping: dict[str, float], threshold: float) -> None:
        self.weights_mapping = weights_mapping
        self.threshold = threshold

    def _similarity(self, val1: str, val2: str) -> float:
        if val1 == val2:
            return 1.0
        return SequenceMatcher(None, val1, val2).ratio()

    def _score(self, d1: dict[str, str], d2: dict[str, str]) -> float:
        total_weight = 0.0
        weighted_score = 0.0

        for field, weight in self.weights_mapping.items():
            v1 = d1.get(field, "")
            v2 = d2.get(field, "")

            similarity = self._similarity(v1, v2)
            weighted_score += similarity * weight
            total_weight += weight

        return weighted_score / total_weight if total_weight else 0.0

    def _get_score_matrix(
        self, list1: list[dict[str, str]], list2: list[dict[str, str]]
    ) -> list[list[float]]:
        return [[self._score(d1, d2) for d2 in list2] for d1 in list1]

    def _get_cost_matrix(
        self, score_matrix: list[list[float]], n1: int, n2: int
    ) -> list[list[float]]:
        size = max(n1, n2)
        cost_matrix: list[list[float]] = [[0.0] * size for _ in range(size)]

        for i in range(n1):
            for j in range(n2):
                cost_matrix[i][j] = -score_matrix[i][j]
        return cost_matrix

    def _reconstruct(
        self,
        list1: list[dict[str, str]],
        list2: list[dict[str, str]],
        row_indices: list[int],
        col_indices: list[int],
        score_matrix: list[list[float]],
    ):
        n1, n2 = len(list1), len(list2)

        result1: list[dict[str, str]] = []
        result2: list[dict[str, str]] = []
        matched_from_1: set[int] = set()
        matched_from_2: set[int] = set()

        for row, col in zip(row_indices, col_indices):
            if row < n1 and col < n2:
                if score_matrix[row][col] >= self.threshold:
                    result1.append(list1[row])
                    result2.append(list2[col])
                    matched_from_1.add(row)
                    matched_from_2.add(col)

        for i, item in enumerate(list1):
            if i not in matched_from_1:
                result1.append(item)
                result2.append(SchematismEntry().model_dump())

        for j, item in enumerate(list2):
            if j not in matched_from_2:
                result1.append(SchematismEntry().model_dump())
                result2.append(item)

        return result1, result2

    def align_entries(self, list1: list[dict[str, str]], list2: list[dict[str, str]]):
        """Align two lists using Hungarian algorithm. Order-independent."""
        n1, n2 = len(list1), len(list2)

        if n1 == 0 and n2 == 0:
            return [], []
        if n1 == 0:
            return [SchematismEntry().model_dump()] * n2, list2
        if n2 == 0:
            return list1.copy(), [SchematismEntry().model_dump()] * n1

        score_matrix = self._get_score_matrix(list1, list2)
        cost_matrix = self._get_cost_matrix(score_matrix, n1, n2)

        row_indices, col_indices = optimize.linear_sum_assignment(cost_matrix)

        return self._reconstruct(
            list1, list2, row_indices.tolist(), col_indices.tolist(), score_matrix
        )


class JSONAligner:
    """Aligns entries from two JSONs, maintaining order with empty placeholders."""

    def __init__(self, weights_mapping: dict[str, float], threshold: float) -> None:
        """
        Initialize aligner.

        Args:
            use_fuzzy: If True, uses thefuzz library (needs pip install thefuzz).
                      If False, uses built-in difflib.
        """

        self.weights_mapping = weights_mapping
        self.threshold = threshold

    def calculate_entry_score(self, entry1: Dict, entry2: Dict) -> float:
        """
        Calculate matching score between two entries.

        Returns a normalized score (0-1).
        """
        scores = []
        weights = []

        if entry1.keys() != entry2.keys():
            raise ValueError("entries must have the same keys")

        keys = entry1.keys()

        for key in keys:
            scores.append(
                SequenceMatcher(
                    None, str(entry1.get(key) or ""), str(entry2.get(key) or "")
                ).ratio()
            )
            weights.append(self.weights_mapping[key])

        if not scores:
            return 0.0

        # Calculate weighted average
        total_weight = sum(weights)
        weighted_sum = sum(s * w for s, w in zip(scores, weights))

        return weighted_sum / total_weight if total_weight > 0 else 0.0

    def align_entries(self, data1, data2) -> Tuple[List[Dict], List[Dict]]:
        """
        Align entries from two JSONs.

        Args:
            data1: First JSON data as dictionary
            data2: Second JSON data as dictionary
            threshold: Minimum score to consider a match (0-1)

        Returns:
            Tuple of (aligned_entries1, aligned_entries2) with same length
        """
        entries1 = data1.get("entries", [])
        entries2 = data2.get("entries", [])

        if not entries1 and not entries2:
            return [], []

        aligned1 = []
        aligned2 = []
        used_indices2 = set()

        for i, entry1 in enumerate(entries1):
            best_match = None
            best_score = 0
            best_j = -1

            for j, entry2 in enumerate(entries2):
                if j in used_indices2:
                    continue

                score = self.calculate_entry_score(entry1, entry2)

                if len(entries1) > 1 and len(entries2) > 1:
                    position_diff = abs(i / len(entries1) - j / len(entries2))
                    position_bonus = (1 - position_diff) * 0.05
                    score += position_bonus

                if score > best_score and score >= self.threshold:
                    best_score = score
                    best_match = entry2
                    best_j = j

            if best_match:
                # Found a match
                aligned1.append(entry1)
                aligned2.append(best_match)
                used_indices2.add(best_j)
            else:
                aligned1.append(entry1)
                aligned2.append(SchematismEntry().model_dump())

        for j, entry2 in enumerate(entries2):
            if j not in used_indices2:
                aligned1.append(SchematismEntry().model_dump())
                aligned2.append(entry2)

        return aligned1, aligned2
