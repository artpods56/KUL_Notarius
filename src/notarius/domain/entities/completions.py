import abc
from dataclasses import dataclass

from notarius.domain.entities.messages import ChatMessage, TextContent


@dataclass(frozen=True)
class BaseProviderResponse[T](abc.ABC):
    response: T

    @abc.abstractmethod
    def to_string(self) -> str:
        raise NotImplementedError

    def to_message(self) -> ChatMessage:
        """Convert the output to a ChatMessage for input history.

        Returns:
            ChatMessage with role="assistant" and text content
        """
        return ChatMessage(
            role="assistant",
            content=[TextContent(text=self.to_string())],
        )
