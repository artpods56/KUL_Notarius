from abc import ABC, abstractmethod
from typing import Self, Any

from pydantic import BaseModel

from notarius.domain.protocols import BaseRequest, BaseResponse


class ConfigurableEngine[
    ConfigT: BaseModel,
    RequestT: BaseRequest[Any],
    ResponseT: BaseResponse[Any],
](ABC):
    """Base _engine interface generic over config, request, and structured_response types.

    Type Parameters:
        ConfigT: The configuration type for this _engine
        RequestT: The request type this _engine accepts
        ResponseT: The structured_response type this _engine returns
    """

    @classmethod
    @abstractmethod
    def from_config(cls, config: ConfigT) -> Self:
        raise NotImplementedError

    @abstractmethod
    def process(self, request: RequestT) -> ResponseT:
        raise NotImplementedError
