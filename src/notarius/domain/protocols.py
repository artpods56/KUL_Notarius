import abc
from dataclasses import dataclass
from typing import runtime_checkable, Protocol


@dataclass(frozen=True)
class BaseRequest[InputT](abc.ABC):
    """Base request with any input type."""

    input: InputT


@dataclass(frozen=True)
class BaseResponse[OutputT](abc.ABC):
    """Base response with any output type."""

    output: OutputT


@runtime_checkable
class FileStreamProtocol(Protocol):
    def read(self, size: int = -1, /) -> bytes: ...
    def seek(self, offset: int, whence: int = 0, /) -> int: ...
    def tell(self) -> int: ...
