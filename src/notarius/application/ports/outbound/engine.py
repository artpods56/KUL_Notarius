from abc import ABC, abstractmethod
from typing import Self

from pydantic import BaseModel

from notarius.domain.protocols import BaseRequest, BaseResponse


class ConfigurableEngine[
    ConfigT: BaseModel,
    RequestT: BaseRequest,
    ResponseT: BaseResponse,
](ABC):
    """Base _engine interface generic over config, request, and response types.

    Type Parameters:
        ConfigT: The configuration type for this _engine
        RequestT: The request type this _engine accepts
        ResponseT: The response type this _engine returns
    """

    @classmethod
    @abstractmethod
    def from_config(cls, config: ConfigT, *args, **kwargs) -> Self:
        raise NotImplementedError

    @abstractmethod
    def process(self, request: RequestT) -> ResponseT:
        raise NotImplementedError
