from dataclasses import dataclass
from pydantic import BaseModel

from notarius.domain.entities.completions import BaseProviderResponse

@dataclass(frozen=True)
class OpenAIResponse[T: BaseModel](BaseProviderResponse[T]):
    structured_response: T | None
    text_response: str | None