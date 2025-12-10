from dataclasses import dataclass
from typing import override, cast

from pydantic import BaseModel

from notarius.domain.entities.completions import BaseProviderResponse
from openai.types.responses import ParsedResponse, Response

from notarius.domain.entities.messages import ChatMessage, TextContent


@dataclass(frozen=True)
class OpenAIResponse[T: BaseModel](BaseProviderResponse[T | Response]):
    response: ParsedResponse[T] | Response

    @override
    def to_string(self) -> str:
        if isinstance(self.response, ParsedResponse):
            parsed = self.response.output_parsed
            if parsed is not None:
                return cast(T, parsed).model_dump_json()
            else:
                return self.response.output_text
        else:
            return self.response.output_text

    @override
    def to_message(self) -> ChatMessage:
        """Convert the output to a ChatMessage for input history.

        Returns:
            ChatMessage with role="assistant" and text content
        """
        return ChatMessage(
            role="assistant",
            content=[TextContent(text=self.to_string())],
        )
