from typing import Literal

from pydantic import BaseModel, Field, ConfigDict

from notarius.infrastructure.config.constants import ConfigType, ModelsConfigSubtype
from notarius.infrastructure.config.registry import register_config


class GenerationParams(BaseModel):
    max_tokens: int = Field(
        default=4096, description="The maximum number of tokens to generate"
    )
    temperature: float = Field(
        default=0.1, ge=0.0, le=2.0, description="Controls randomness in the output"
    )
    top_p: float = Field(
        default=0.9,
        ge=0.0,
        le=1.0,
        description="Controls diversity via nucleus sampling",
    )


BackendType = Literal["llama", "lm_studio", "openai", "mistral", "openrouter"]


class ClientConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    backend: BackendType = Field(description="Which backend this client uses")

    model: str = Field(default="gpt-3.5-turbo", description="Model description")

    base_url: str = Field(
        default="https://api.openai.com/v1", description="Base URL for the client"
    )
    api_key_env_var: str = Field(
        default="OPENAI_API_KEY",
        description="Environment variable that specifies the API key to use",
    )
    structured_output: bool = Field(
        default=False, description="Whether to use structured output"
    )
    template_dir: str = Field(
        default="prompts",
        description="Path to template directory that is passed to the PromptRenderer",
    )
    params: GenerationParams = Field(
        default_factory=GenerationParams, description="Model API kwargs"
    )


def default_clients() -> dict[str, ClientConfig]:
    return {
        "llama": ClientConfig(
            backend="llama",
            model="your-default-model-name-here",
            base_url="https://api.llama.example/v1",
            api_key_env_var="LLAMA_API_KEY",
            structured_output=False,
            template_dir="prompts",
            params=GenerationParams(
                max_tokens=4096,
                temperature=0.1,
                top_p=0.9,
            ),
        ),
        "lm_studio": ClientConfig(
            backend="lm_studio",
            model="your-lmstudio-model",
            base_url="http://localhost:1234/v1",
            api_key_env_var="LM_STUDIO_KEY",  # or None if not needed
            structured_output=False,
            template_dir="prompts",
            params=GenerationParams(
                max_tokens=4096,
                temperature=0.1,
                top_p=0.9,
            ),
        ),
        "openai": ClientConfig(
            backend="openai",
            model="gpt-4.1-mini",  # or whatever makes your heart flutter
            base_url="https://api.openai.com/v1",
            api_key_env_var="OPENAI_API_KEY",
            structured_output=False,
            template_dir="prompts",
            params=GenerationParams(
                max_tokens=4096,
                temperature=0.1,
                top_p=0.9,
            ),
        ),
        "openrouter": ClientConfig(
            backend="openrouter",
            model="google/gemini-2.5-flash-preview",
            base_url="https://openrouter.ai/api/v1",
            api_key_env_var="OPENROUTER_API_KEY",
            structured_output=False,
            template_dir="prompts",
            params=GenerationParams(
                max_tokens=4096,
                temperature=0.1,
                top_p=0.9,
            ),
        ),
        "mistral": ClientConfig(
            backend="mistral",
            model="mistral-large-latest",
            base_url="https://api.mistral.ai/v1",
            api_key_env_var="MISTRAL_API_KEY",
            structured_output=False,
            template_dir="prompts",
            params=GenerationParams(
                max_tokens=4096,
                temperature=0.1,
                top_p=0.9,
            ),
        ),
    }


class BackendSelection(BaseModel):
    type: BackendType = Field(
        default="openrouter", description="Which backend to use at runtime"
    )
    max_retries: int = Field(default=5, description="Retry attempts on failure")


@register_config(ConfigType.MODELS, ModelsConfigSubtype.LLM)
class LLMEngineConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    backend: BackendSelection = Field(
        default_factory=BackendSelection,
        description="Which backend is currently active",
    )
    clients: dict[str, ClientConfig] = Field(
        default_factory=default_clients,
        description="Registry of available backend client configurations",
    )
