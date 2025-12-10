"""Factory for creating LLM provider instances."""

from typing import Any

from notarius.application.ports.outbound.llm_provider import LLMProvider
from notarius.infrastructure.llm.providers.openai_provider.adapter import (
    OpenAICompatibleProvider,
)
from notarius.schemas.configs import LLMEngineConfig
from notarius.schemas.configs.llm_model_config import BackendType

# Map of backend types to provider implementations
PROVIDER_MAP: dict[BackendType, type[LLMProvider[Any]]] = {
    "openai": OpenAICompatibleProvider,
    "lm_studio": OpenAICompatibleProvider,
    "openrouter": OpenAICompatibleProvider,
    "llama": OpenAICompatibleProvider,
}


def llm_provider_factory(config: LLMEngineConfig) -> LLMProvider[Any]:
    """Create an LLM provider instance based on configuration.

    Args:
        config: LLM _engine configuration containing backend and client settings

    Returns:
        Configured LLM provider instance

    Raises:
        ValueError: If backend type is not configured or unsupported
    """
    backend_type = config.backend.type

    client_config = config.clients.get(backend_type)
    if client_config is None:
        raise ValueError(
            f"No provider configuration found for backend type: {backend_type}"
        )

    provider_class = PROVIDER_MAP.get(backend_type)
    if not provider_class:
        raise ValueError(f"Unsupported backend type: {backend_type}")

    return provider_class(client_config)
