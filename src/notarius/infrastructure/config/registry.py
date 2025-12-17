"""Config registry for automatic model discovery and validation."""

from typing import Dict, Optional, Any
from notarius.infrastructure.config.exceptions import ConfigNotRegisteredError
from pydantic import BaseModel, ValidationError

from notarius.infrastructure.config.constants import ConfigType, ConfigSubTypes
from notarius.infrastructure.config.utils import validate_config_arguments

from structlog import get_logger

logger = get_logger(__name__)

CONFIG_REGISTRY: dict[tuple[ConfigType, ConfigSubTypes], type[BaseModel]] = {}


def register_config(config_type: ConfigType, config_subtype: ConfigSubTypes):
    """Decorator to register a config_manager schema.

    Args:
        config_type: Enumerate of the main config_manager category (e.g., 'ConfigType.MODELS', 'ConfigType.DATASET')
        config_subtype: The config_manager subtype enumerate (e.g., 'ModelsConfigSubtype.DEFAULT', 'DatasetConfigSubtype.DEFAULT')
    """

    def decorator[SchemaT: BaseModel](cls: type[SchemaT]) -> type[SchemaT]:
        key = (config_type, config_subtype)
        if key in CONFIG_REGISTRY:
            logger.warning(f"Config schema already registered for {key}, overwriting")
        CONFIG_REGISTRY[key] = cls
        return cls

    return decorator


@validate_config_arguments
def get_config_schema(
    config_type: ConfigType, config_subtype: ConfigSubTypes
) -> type[BaseModel]:
    config_schema = CONFIG_REGISTRY.get((config_type, config_subtype), None)

    if not config_schema:
        raise ConfigNotRegisteredError(
            f"Config '{config_subtype}' for {config_type} is not registered.",
            f"Please register a schema using @register_config('{config_type.value}', '{config_subtype.value}')."
            f"Currently registered configs: {CONFIG_REGISTRY}",
        )

    return config_schema


def list_registered_configs() -> dict[str, list[str]]:
    """List all registered config_manager types and subtypes."""
    result: dict[str, list[str]] = {}
    for (config_type, config_subtype), schema in CONFIG_REGISTRY.items():
        if config_type.value not in result:
            result[config_type.value] = []
        result[config_type.value].append(config_subtype.value)
    return result


def validate_config_with_schema[SchemaT: BaseModel](
    config_data: dict[str, Any], schema_class: type[SchemaT]
) -> SchemaT:
    """Validate config_manager data against a Pydantic schema.

    Args:
        config_data: Configuration data as a dictionary
        schema_class: Pydantic model class to validate against

    Returns:
        Validated Pydantic model instance

    Raises:
        ValidationError: If validation fails
    """
    try:
        return schema_class(**config_data)
    except ValidationError as e:
        logger.error(
            "config_validation_failed",
            schema_class=schema_class.__name__,
            error=str(e),
            config_data=config_data,
        )
        raise


@validate_config_arguments
def get_default_config(
    config_type: ConfigType, config_subtype: ConfigSubTypes
) -> Optional[Dict[str, Any]]:
    """Generate default configuration for a given type and subtype."""
    schema_class = get_config_schema(config_type, config_subtype)
    if not schema_class:
        raise ConfigNotRegisteredError(config_type, config_subtype, CONFIG_REGISTRY)
    default_instance = schema_class()
    return default_instance.model_dump()
