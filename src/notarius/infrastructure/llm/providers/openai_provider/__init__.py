"""OpenAI-compatible provider implementation.

This module provides an adapter for OpenAI and OpenAI-compatible APIs,
translating between domain types and provider-specific formats.
"""

from notarius.infrastructure.llm.providers.openai_provider.adapter import (
    OpenAICompatibleProvider,
)
from notarius.infrastructure.llm.providers.openai_provider.completions import (
    OpenAIResponse,
)
from notarius.infrastructure.llm.providers.openai_provider.translator import (
    messages_to_openai,
)

__all__ = [
    "OpenAICompatibleProvider",
    "OpenAIResponse",
    "messages_to_openai",
]
