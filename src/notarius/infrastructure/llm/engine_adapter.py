"""Clean LLM Engine adapter using refactored components."""

from dataclasses import dataclass
from typing import Self, final, override
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
)
from pydantic import BaseModel

from notarius.application.ports.outbound.engine import ConfigurableEngine
from notarius.domain.entities.completions import BaseProviderResponse
from notarius.domain.protocols import BaseRequest, BaseResponse

from notarius.infrastructure.llm.conversation import (
    Conversation,
)
from notarius.infrastructure.llm.providers.factory import llm_provider_factory
from notarius.schemas.configs import LLMEngineConfig
from notarius.shared.constants import MAX_LLM_RETRIES


@dataclass(frozen=True)
class CompletionRequest[T: BaseModel](BaseRequest[Conversation]):
    """Configuration for a single LLM request."""

    input: Conversation
    structured_output: type[T] | None = None


@dataclass(frozen=True)
class CompletionResult[T: BaseModel](BaseResponse[BaseProviderResponse[T]]):
    """Result of an LLM completion request.

    The input automatically includes the assistant's output.
    """

    output: BaseProviderResponse[T]
    conversation: Conversation

    @property
    def updated_conversation(self) -> Conversation:
        """Get the input with the assistant's output added.

        This is useful for multi-turn conversations where you want to
        maintain the full history including the assistant's replies.
        """
        return self.conversation.add(self.output.to_message())


@final
class LLMEngine(
    ConfigurableEngine[
        LLMEngineConfig,
        CompletionRequest[BaseModel],
        CompletionResult[BaseModel],
    ]
):
    """Engine for interacting with LLM providers using domain types."""

    def __init__(self, config: LLMEngineConfig):
        self.config = config
        self.provider = llm_provider_factory(config)

    @property
    def used_model(self) -> str:
        backend = self.config.backend.type
        return self.config.clients.get(
            backend
        ).model  # pyright: ignore[reportOptionalMemberAccess]

    @classmethod
    @override
    def from_config(cls, config: LLMEngineConfig) -> Self:
        return cls(config=config)

    @override
    @retry(
        stop=stop_after_attempt(MAX_LLM_RETRIES),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
    )
    def process[T: BaseModel](
        self,
        request: CompletionRequest[T],
    ) -> CompletionResult[T]:
        response = self.provider.generate_response(
            request.input.messages, text_format=request.structured_output
        )

        conversation = request.input.add(response.to_message())

        return CompletionResult[T](
            output=response,
            conversation=conversation,
        )
