from __future__ import annotations

from pathlib import Path

from dagster import ConfigurableResource
from omegaconf import DictConfig

from core.config.manager import ConfigManager
from core.utils.shared import PDF_SOURCE_DIR

from core.config.constants import ConfigType, ConfigSubTypes
from core.config.manager import get_config_manager


class PdfFilesResource(ConfigurableResource):
    pdf_dir: str = str(PDF_SOURCE_DIR)

    def get_pdf_paths(self) -> list[Path]:
        return list(Path(self.pdf_dir).glob("**/*.pdf"))


# config_manager: ConfigManager = get_config_manager()


class ConfigManagerResource(ConfigurableResource):
    config_manager: ClassVar[ConfigManager] = get_config_manager()

    def load_config(
        self, config_name: str, config_type: ConfigType, config_subtype: ConfigSubTypes
    ) -> DictConfig:
        return self.config_manager.load_config(config_name, config_type, config_subtype)

    def load_config_from_string(self, config_name: str, config_type_name: str, config_subtype_name: str) -> DictConfig:
        return self.config_manager.load_config_from_string(config_name, config_type_name, config_subtype_name)


from dagster import ConfigurableResource
from typing import Any, Callable, Literal, ClassVar

OpType = Literal["filter", "map"]
OpRegistryType = dict[tuple[OpType, str], Callable[[Any], Any]]


class OpRegistry(ConfigurableResource):
    """
    Registry for operations using class-level storage.

    Note: _ops_registry is a ClassVar, not a Pydantic field,
    so it's shared across all instances and not serialized.
    """

    # ✅ Use ClassVar to tell Pydantic this is NOT a model field
    _ops_registry: ClassVar[OpRegistryType] = {}

    @classmethod
    def register(cls, op_type: OpType, name: str):
        """
        Decorator to register an operation.

        Args:
            op_type: Type of operation ('filter' or 'map')
            name: Unique name for the operation

        Raises:
            RuntimeError: If operation already registered
        """
        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            op_key = (op_type, name)

            if op_key in cls._ops_registry:
                raise RuntimeError(
                    f"Operation '{name}' of type '{op_type}' already registered"
                )

            cls._ops_registry[op_key] = func

            # Return the original function unchanged
            return func

        return decorator

    def get(self, op_type: OpType, name: str) -> Callable | None:
        """Retrieve a registered operation (instance method for Dagster compatibility)."""
        return self._ops_registry.get((op_type, name), None)

    @classmethod
    def get_op(cls, op_type: OpType, name: str) -> Callable | None:
        """Retrieve a registered operation (class method alternative)."""
        return cls._ops_registry.get((op_type, name))

    @classmethod
    def list_operations(cls, op_type: OpType | None = None) -> list[tuple[OpType, str]]:
        """List all registered operations."""
        if op_type is None:
            return list(cls._ops_registry.keys())
        return [key for key in cls._ops_registry.keys() if key[0] == op_type]

    @classmethod
    def clear(cls) -> None:
        """Clear all registered operations. Useful for testing."""
        cls._ops_registry.clear()


class ParserResource(ConfigurableResource):
    """Resource for the translation parser.

    This resource wraps the Parser class to make it available
    as a Dagster resource that can be injected into assets.
    """

    fuzzy_threshold: int = 80

    def get_parser(self):
        """Get a parser instance with the configured threshold."""
        from core.data.translation_parser import Parser
        return Parser(fuzzy_threshold=self.fuzzy_threshold)
