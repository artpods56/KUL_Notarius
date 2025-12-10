from typing import Type, overload
from pydantic import BaseModel, Field, create_model
import dagster as dg


class AssetKeyHelper:
    @staticmethod
    def build_key(*parts: str) -> str:
        """Build asset key from parts"""
        return "__".join(parts)

    @staticmethod
    def build_prefixed_key(layer: str, source: str, *parts: str) -> str:
        """Build asset key with layer and source prefix"""
        return f"{layer}__{source}__{AssetKeyHelper.build_key(*parts)}"


def dagster_config_from_pydantic[T: BaseModel](
    pydantic_cls: type[T],
) -> type[T]:
    """Create a Dagster Config class from a Pydantic model."""
    field_definitions = {}
    annotations = {}

    for name, field_info in pydantic_cls.model_fields.items():
        annotations[name] = field_info.annotation
        field_definitions[name] = Field(
            default=field_info.default if field_info.default is not None else ...,
            description=field_info.description,
        )

    new_cls = type(
        pydantic_cls.__name__,
        (dg.Config,),
        {"__annotations__": annotations, **field_definitions},
    )
    return new_cls  # pyright: ignore[reportReturnType]


def make_dagster_config[T: BaseModel](original_model: type[T]):
    """
    Creates a new class that inherits from both the original model
    and dagster.Config.
    """
    return create_model(
        original_model.__name__,  # Keep the original name (e.g. 'DatabaseConfig')
        # We inherit from the Original Model (to get fields)
        # AND dagster.Config (to make Dagster happy)
        __base__=(original_model, dg.Config),
    )
