import abc
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class BaseRequest[T](abc.ABC):
    input: T


@dataclass(frozen=True)
class BaseResponse[T](abc.ABC):
    output: T
