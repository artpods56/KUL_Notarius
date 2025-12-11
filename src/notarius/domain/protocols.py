import abc
from dataclasses import dataclass


@dataclass(frozen=True)
class BaseRequest[InputT](abc.ABC):
    """Base request with any input type."""

    input: InputT


@dataclass(frozen=True)
class BaseResponse[OutputT](abc.ABC):
    """Base response with any output type."""

    output: OutputT
