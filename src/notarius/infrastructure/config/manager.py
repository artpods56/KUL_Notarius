from __future__ import annotations

import inspect
from functools import wraps
from pathlib import Path
from typing import Any, Dict, List, cast, final

from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf
from pydantic import BaseModel

from notarius.infrastructure.config.constants import ConfigType, ConfigSubTypes
from notarius.infrastructure.config.registry import (
    get_config_schema,
    get_default_config,
    list_registered_configs,
    validate_config_with_schema,
)
from notarius.infrastructure.config.utils import (
    validate_config_arguments,
    discover_config_files,
)
from notarius.infrastructure.config.constants import ConfigTypeMapping
from notarius.shared.constants import CONFIGS_DIR
from notarius.shared.logger import Logger
from structlog import get_logger

logger: Logger = get_logger(__name__)


def config_to_dict(config: DictConfig) -> dict[str, Any]:
    """Convert OmegaConf config_manager to regular dict."""
    return cast(dict[str, Any], OmegaConf.to_container(config, resolve=True))


@final
class ConfigManager:
    """Centralized configuration management with automatic registry."""

    def __init__(self, configs_dir: Path):
        self.configs_dir = configs_dir
        self._configs = {}
        self._available_configs = None

    def _get_output_file_path(
        self, config_type: ConfigType, config_subtype: ConfigSubTypes, config_name: str
    ) -> Path:
        """Get the output file path for a configuration file."""
        output_dir = self.configs_dir / config_type.value
        if config_subtype.value != "default":
            output_dir = output_dir / config_subtype.value
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir / f"{config_name}.yaml"

    def _get_cache_key(
        self, config_type: ConfigType, config_subtype: ConfigSubTypes, config_name: str
    ) -> str:
        """Generate a consistent cache key for a configuration."""
        return f"{config_type.value}.{config_subtype.value}.{config_name}"

    @property
    def available_configs(self) -> dict[str, dict[str, Any]]:
        """Get available configs by discovering files and checking registry."""
        if self._available_configs is None:
            self._available_configs = discover_config_files(self.configs_dir)
        return self._available_configs

    @property
    def registered_configs(self) -> dict[str, Any]:
        """Get all registered config_manager schemas."""
        return list_registered_configs()

    @validate_config_arguments
    def validate_config(
        self,
        config_type: ConfigType,
        config_subtype: ConfigSubTypes,
        config_dict: dict[str, Any],
    ) -> dict[str, Any]:
        """Validate configuration against appropriate Pydantic model."""
        model_class = get_config_schema(config_type, config_subtype)
        validated_config = validate_config_with_schema(config_dict, model_class)
        return validated_config.model_dump()

    @validate_config_arguments
    def load_config(
        self, config_name: str, config_type: ConfigType, config_subtype: ConfigSubTypes
    ) -> DictConfig:
        """Load configuration from file and validate if applicable."""

        subtype_enum = ConfigTypeMapping.get_subtype_enum(config_type)

        config_dir = self.configs_dir / Path(str(config_type.value))
        # don't nest if its default subtype
        if config_subtype.value != "default":
            config_dir = config_dir / Path(str(config_subtype.value))

        # Check if config_manager file exists
        config_file = config_dir / f"{config_name}.yaml"
        if not config_file.exists():
            logger.error("Config file not found", config_file=str(config_file))
            raise FileNotFoundError(f"Config file not found: {config_file}")

        with initialize_config_dir(config_dir=str(config_dir), version_base="1.1"):
            config = compose(config_name=config_name)
            config_dict = config_to_dict(config)

            validated_dict = self.validate_config(
                config_type, config_subtype, config_dict
            )
            validated_config: DictConfig = OmegaConf.create(validated_dict)

            cache_key = self._get_cache_key(config_type, config_subtype, config_name)
            self._configs[cache_key] = validated_config
            return validated_config

    @validate_config_arguments
    def load_config_as_model(
        self, config_name: str, config_type: ConfigType, config_subtype: ConfigSubTypes
    ) -> BaseModel:
        """Load configuration from file and return as Pydantic model instance.

        This method provides type-safe access to configurations by returning the
        actual Pydantic model instance instead of a DictConfig.

        Args:
            config_name: Name of the configuration file (without .yaml extension)
            config_type: Type of configuration (e.g., ConfigType.MODELS)
            config_subtype: Subtype of configuration (e.g., ModelsConfigSubtype.LLM)

        Returns:
            Validated Pydantic model instance for the configuration

        Example:
            >>> from notarius.infrastructure.config.constants import ConfigType, ModelsConfigSubtype
            >>> manager = get_config_manager()
            >>> llm_config = manager.load_config_as_model(
            ...     "default", ConfigType.MODELS, ModelsConfigSubtype.LLM
            ... )
            >>> # Now llm_config is a typed Pydantic model with full IDE support
        """
        config_dir = self.configs_dir / Path(str(config_type.value))
        # don't nest if its default subtype
        if config_subtype.value != "default":
            config_dir = config_dir / Path(str(config_subtype.value))

        # Check if config_manager file exists
        config_file = config_dir / f"{config_name}.yaml"
        if not config_file.exists():
            logger.error("Config file not found", config_file=str(config_file))
            raise FileNotFoundError(f"Config file not found: {config_file}")

        with initialize_config_dir(config_dir=str(config_dir), version_base="1.1"):
            config = compose(config_name=config_name)
            config_dict = config_to_dict(config)

            # Get the schema class and validate, returning the Pydantic model
            model_class = get_config_schema(config_type, config_subtype)
            validated_model = validate_config_with_schema(config_dict, model_class)

            return validated_model

    def load_config_from_string(
        self, config_name: str, config_type_name: str, config_subtype_name: str
    ):
        type_enum = ConfigType(config_type_name)
        subtype = ConfigTypeMapping.get_subtype_enum(type_enum)
        subtype_enum = subtype(config_subtype_name)

        return self.load_config(config_name, type_enum, subtype_enum)

    @validate_config_arguments
    def _generate_default_config(
        self, config_type: ConfigType, config_subtype: ConfigSubTypes
    ) -> DictConfig:
        """Generate default configuration for a given config_manager type and subtype."""
        config_dict = get_default_config(config_type, config_subtype)
        config = OmegaConf.create(config_dict)
        return config

    def _save_config(
        self,
        config_name: str,
        config_type: ConfigType,
        config_subtype: ConfigSubTypes,
        config: DictConfig,
    ):
        """Save configuration to a YAML file."""
        output_file = self._get_output_file_path(
            config_type, config_subtype, config_name
        )
        with output_file.open("w") as f:
            OmegaConf.save(config=config, f=f)

        logger.info(
            "Saved config_manager to file",
            config_name=config_name,
            config_type=config_type,
            config_subtype=config_subtype,
            path=str(output_file),
        )

    def generate_default_configs(self, overwrite: bool = False, save: bool = True):
        """Generate default configurations for all registered schemas."""
        from notarius.infrastructure.config.registry import CONFIG_REGISTRY

        for (config_type, config_subtype), model_class in CONFIG_REGISTRY.items():
            logger.info(
                "Generating default config_manager.",
                config_type=config_type.value,
                config_subtype=config_subtype,
            )
            default_config = self._generate_default_config(config_type, config_subtype)

            if save:
                output_file = self._get_output_file_path(
                    config_type, config_subtype, "default"
                )
                if output_file.exists() and not overwrite:
                    raise ValueError(
                        f"Config already exists for {config_type}.{config_subtype}. Use overwrite=True."
                    )

                self._save_config(
                    config_name="default",
                    config_type=config_type,
                    config_subtype=config_subtype,
                    config=default_config,
                )
        self.refresh_available_configs()

    @validate_config_arguments
    def generate_default_config(
        self,
        config_type: ConfigType,
        config_subtype: ConfigSubTypes,
        config_name: str = "default",
        save: bool = True,
    ) -> DictConfig:
        """Generate and optionally save a default configuration for specific type/subtype."""
        default_config = self._generate_default_config(config_type, config_subtype)
        if save:
            self._save_config(config_name, config_type, config_subtype, default_config)
        return default_config

    def list_available_configs(self) -> Dict[str, Dict[str, List[str]]]:
        """List all available configuration files."""
        return self.available_configs

    def refresh_available_configs(self):
        """Refresh the cache of available configuration files."""
        self._available_configs = None


def get_config_manager() -> ConfigManager:
    """Get or create a singleton instance of ConfigManager."""
    config_manager_instance = ConfigManager(CONFIGS_DIR)
    return config_manager_instance


config_manager = ConfigManager(CONFIGS_DIR)


def with_configs(**config_args):
    """Decorator to inject configurations into function arguments.

    Args:
        **config_args: Mapping of parameter description to (config_name, config_type, config_subtype) tuple

    Example:
        @with_configs(
            model_config=("default", ConfigType.MODELS, ModelsConfigSubtype.LLM),
            dataset_config=("default", ConfigType.DATASET, DatasetConfigSubtype.TRAINING)
        )
        def train_model(model_config, dataset_config):
            # Both configs are automatically loaded and injected
            pass
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            from notarius.infrastructure.config.manager import get_config_manager

            config_manager = get_config_manager()
            injected_kwargs = {}

            for param_name, (
                config_name,
                config_type,
                config_subtype,
            ) in config_args.items():
                config = config_manager.load_config(
                    config_name, config_type, config_subtype
                )
                injected_kwargs[param_name] = config

            # Respect original kwargs and let user override if needed
            full_kwargs = {**injected_kwargs, **kwargs}

            # Check if the function supports the right args
            sig = inspect.signature(func)
            missing = [name for name in full_kwargs if name not in sig.parameters]
            if missing:
                raise TypeError(
                    f"Injected unexpected config_manager arguments: {missing}"
                )

            return func(*args, **full_kwargs)

        return wrapper

    return decorator
