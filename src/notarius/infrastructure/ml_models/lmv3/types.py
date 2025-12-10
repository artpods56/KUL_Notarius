"""Type definitions for LayoutLMv3 model interfaces."""

from typing import Any, Protocol, TypedDict, overload

import torch


class EncodingData(TypedDict, total=False):
    """Type definition for encoding dictionary returned by LayoutLMv3Processor.

    This mirrors the structure of BatchEncoding from transformers,
    but provides better type hints for static analysis.
    """
    pixel_values: torch.Tensor  # Shape: (n_windows, 3, H, W)
    bbox: torch.Tensor          # Shape: (n_windows, seq_len, 4)
    input_ids: torch.Tensor     # Shape: (n_windows, seq_len)
    attention_mask: torch.Tensor  # Shape: (n_windows, seq_len)
    offset_mapping: torch.Tensor  # Shape: (n_windows, seq_len, 2)
    overflow_to_sample_mapping: torch.Tensor  # Shape: (n_windows,)


class EncodingProtocol(Protocol):
    """Protocol for encoding objects from LayoutLMv3Processor.

    This protocol defines the interface we need from BatchEncoding,
    helping static analyzers understand the type structure without
    relying on incomplete stubs from the transformers library.
    """

    # Attribute access
    bbox: torch.Tensor

    # Dictionary-style access
    def __getitem__(self, key: str) -> Any: ...

    def __setitem__(self, key: str, value: Any) -> None: ...

    @overload
    def pop(self, key: str) -> Any: ...

    @overload
    def pop(self, key: str, default: Any) -> Any: ...

    def items(self) -> Any: ...

    def keys(self) -> Any: ...

    def values(self) -> Any: ...
