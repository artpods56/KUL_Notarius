"""
Base classes for use cases in the application layer.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class BaseRequest(ABC): ...


@dataclass
class BaseResponse(ABC): ...


class BaseUseCase[TRequest: BaseRequest, TResponse: BaseResponse](ABC):
    """
    Base class for all use cases following the Command Handler pattern.

    Use cases orchestrate domain services and infrastructure components
    to implement business workflows.
    """

    @abstractmethod
    def execute(self, request: TRequest) -> TResponse:
        """Execute the use case with the given request."""
        pass
