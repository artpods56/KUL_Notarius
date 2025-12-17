"""Utility functions for dataset operations.

This module provides composable utilities for map/filter operations on datasets.
"""

from typing import Callable, Any


def merge_maps(mappers: list[Callable]) -> Callable:
    """Merge multiple map functions into a single callable.

    Each function should accept a single example and return a modified example.

    Args:
        mappers: List of map functions to compose.

    Returns:
        A single function that applies all mappers in sequence.
    """

    def merged_map(example):
        for fn in mappers:
            example = fn(example)
        return example

    return merged_map


def merge_filters(filters: list[Callable]) -> Callable:
    """Merge multiple filter functions into a single function.

    Each filter function should return True for examples to keep.

    Args:
        filters: List of filter functions to compose.

    Returns:
        A single function that returns True only if all filters pass.
    """

    def merged_filter(example):
        return all(f(example) for f in filters)

    return merged_filter


def negate_op(op: Callable[[dict[str, Any]], bool]) -> Callable[[dict[str, Any]], bool]:
    """Negate a filter operation.

    Args:
        op: Filter function to negate.

    Returns:
        Negated filter function.
    """

    def _f(x: dict[str, Any]) -> bool:
        return not op(x)

    return _f


def map_labels(classes_to_remove: set[str]) -> Callable:
    """Create a map function that removes specified label classes.

    Args:
        classes_to_remove: Set of class names to replace with 'O'.

    Returns:
        Map function that modifies the 'labels' field.
    """

    def _map_fn(example):
        example["labels"] = [
            label if label not in classes_to_remove else "O"
            for label in example["labels"]
        ]
        return example

    return _map_fn
