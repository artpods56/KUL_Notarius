import json
from typing import Any, Callable

from orchestration.resources import OpRegistry


# Helper functions
def negate_op(
        op: Callable[[dict[str, Any]], bool]
) -> Callable[[dict[str, Any]], bool]:
    """Negate a filter operation."""
    def _f(x: dict[str, Any]) -> bool:
        return not op(x)
    return _f


def merge_filters(filters: list[Callable]) -> Callable:
    """
    Merge multiple filter functions into a single function.
    Each filter function should return True for examples to keep.
    """
    def merged_filter(example):
        return all(f(example) for f in filters)

    return merged_filter


# Register operations
@OpRegistry.register(op_type="filter", name="filter_schematisms")
def filter_schematisms(to_filter: str | list[str]) -> Callable[[str], bool]:
    """
    Factory function that creates a filter for schematisms.

    Args:
        to_filter: Single schematism or list of schematisms to match

    Returns:
        Filter function that returns True if schematism matches
    """
    def _filter_fn(schematism_name: str) -> bool:
        if isinstance(to_filter, str):
            return schematism_name == to_filter
        return schematism_name in to_filter

    return _filter_fn


@OpRegistry.register(op_type="map", name="filter_empty")
def filter_empty_samples(results: str | dict[str, list[Any]]) -> bool:
    """
    Filter out empty examples (i.e. with empty labels).

    Args:
        results: JSON string or dict containing entries

    Returns:
        True if entries exist, False otherwise

    Example:
        results = '{"page_number": null, "entries": []}'  # Returns False
        results = '{"entries": [{"id": 1}]}'  # Returns True
    """
    if isinstance(results, str):
        try:
            parsed: dict[str, Any] = json.loads(results)
        except json.JSONDecodeError:
            return False
    else:
        parsed = results

    entries = parsed.get("entries", [])
    return bool(entries)


# Factory function to get the singleton instance
def get_op_registry() -> OpRegistry:
    """Get the OpRegistry instance for use in Dagster resources."""
    return OpRegistry()