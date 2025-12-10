import pytest
import dagster as dg
from pydantic import BaseModel, Field

from notarius.orchestration.utils import dagster_config_from_pydantic, AssetKeyHelper


class TestDagsterConfigFromPydantic:
    """Tests for dagster_config_from_pydantic utility function."""

    def test_simple_model_conversion(self):
        """Convert a simple Pydantic model with basic fields."""

        class SimpleModel(BaseModel):
            name: str
            age: int
            active: bool

        DagsterConfig = dagster_config_from_pydantic(SimpleModel)

        assert issubclass(DagsterConfig, dg.Config)
        assert DagsterConfig.__name__ == "SimpleModel"
        assert "name" in DagsterConfig.model_fields
        assert "age" in DagsterConfig.model_fields
        assert "active" in DagsterConfig.model_fields

    def test_model_with_defaults(self):
        """Convert a Pydantic model with default values."""

        class ModelWithDefaults(BaseModel):
            name: str = "default_name"
            count: int = 42
            enabled: bool = True

        DagsterConfig = dagster_config_from_pydantic(ModelWithDefaults)

        instance = DagsterConfig()
        assert instance.name == "default_name"
        assert instance.count == 42
        assert instance.enabled is True

    def test_model_with_field_descriptions(self):
        """Convert a Pydantic model with field descriptions."""

        class ModelWithDescriptions(BaseModel):
            name: str = Field(description="The name of the entity")
            count: int = Field(default=10, description="Number of items")

        DagsterConfig = dagster_config_from_pydantic(ModelWithDescriptions)

        name_field = DagsterConfig.model_fields["name"]
        count_field = DagsterConfig.model_fields["count"]

        assert name_field.description == "The name of the entity"
        assert count_field.description == "Number of items"
        assert count_field.default == 10

    def test_model_with_optional_fields(self):
        """Convert a Pydantic model with optional fields."""

        class ModelWithOptionals(BaseModel):
            required_field: str
            optional_field: str | None = None
            optional_with_default: int = 5

        DagsterConfig = dagster_config_from_pydantic(ModelWithOptionals)

        instance = DagsterConfig(required_field="test")
        assert instance.required_field == "test"
        assert instance.optional_field is None
        assert instance.optional_with_default == 5

    def test_model_with_complex_types(self):
        """Convert a Pydantic model with complex field types."""

        class ComplexModel(BaseModel):
            string_list: list[str]
            int_dict: dict[str, int]
            float_value: float

        DagsterConfig = dagster_config_from_pydantic(ComplexModel)

        instance = DagsterConfig(
            string_list=["a", "b", "c"],
            int_dict={"one": 1, "two": 2},
            float_value=3.14,
        )

        assert instance.string_list == ["a", "b", "c"]
        assert instance.int_dict == {"one": 1, "two": 2}
        assert instance.float_value == 3.14

    def test_preserves_field_annotations(self):
        """Ensure field type annotations are preserved in conversion."""

        class TypedModel(BaseModel):
            name: str
            count: int
            ratio: float
            active: bool

        DagsterConfig = dagster_config_from_pydantic(TypedModel)

        annotations = DagsterConfig.__annotations__
        assert annotations["name"] == str
        assert annotations["count"] == int
        assert annotations["ratio"] == float
        assert annotations["active"] == bool

    def test_model_with_field_constraints(self):
        """Convert a Pydantic model with field constraints.

        Note: Field constraints like gt, min_length are not transferred
        to the Dagster Config class, only defaults and descriptions.
        """

        class ValidatedModel(BaseModel):
            positive_number: int = Field(default=5, gt=0, description="Must be positive")
            name: str = Field(default="test", min_length=1, description="Name field")

        DagsterConfig = dagster_config_from_pydantic(ValidatedModel)

        # Default values and descriptions should be preserved
        instance = DagsterConfig()
        assert instance.positive_number == 5
        assert instance.name == "test"

        # Descriptions should be preserved
        assert DagsterConfig.model_fields["positive_number"].description == "Must be positive"
        assert DagsterConfig.model_fields["name"].description == "Name field"

    def test_empty_model(self):
        """Convert an empty Pydantic model with no fields."""

        class EmptyModel(BaseModel):
            pass

        DagsterConfig = dagster_config_from_pydantic(EmptyModel)

        assert issubclass(DagsterConfig, dg.Config)
        assert DagsterConfig.__name__ == "EmptyModel"
        assert len(DagsterConfig.model_fields) == 0

    def test_model_with_nested_pydantic_models(self):
        """Convert a Pydantic model that contains nested Pydantic models."""

        class NestedModel(BaseModel):
            value: str

        class ParentModel(BaseModel):
            nested: NestedModel
            name: str

        DagsterConfig = dagster_config_from_pydantic(ParentModel)

        nested_instance = NestedModel(value="test")
        instance = DagsterConfig(nested=nested_instance, name="parent")

        assert instance.name == "parent"
        assert instance.nested.value == "test"

    def test_preserves_original_model_name(self):
        """Ensure the converted class preserves the original Pydantic model name."""

        class MyCustomModel(BaseModel):
            field: str

        DagsterConfig = dagster_config_from_pydantic(MyCustomModel)

        assert DagsterConfig.__name__ == "MyCustomModel"

    def test_multiple_conversions_are_independent(self):
        """Ensure converting multiple models doesn't cause interference."""

        class ModelA(BaseModel):
            field_a: str

        class ModelB(BaseModel):
            field_b: int

        DagsterConfigA = dagster_config_from_pydantic(ModelA)
        DagsterConfigB = dagster_config_from_pydantic(ModelB)

        assert "field_a" in DagsterConfigA.model_fields
        assert "field_a" not in DagsterConfigB.model_fields
        assert "field_b" in DagsterConfigB.model_fields
        assert "field_b" not in DagsterConfigA.model_fields


class TestAssetKeyHelper:
    """Tests for AssetKeyHelper utility class."""

    def test_build_key_single_part(self):
        """Build asset key from a single part."""
        key = AssetKeyHelper.build_key("part1")
        assert key == "part1"

    def test_build_key_multiple_parts(self):
        """Build asset key from multiple parts."""
        key = AssetKeyHelper.build_key("part1", "part2", "part3")
        assert key == "part1__part2__part3"

    def test_build_key_empty_parts(self):
        """Build asset key with no parts returns empty string."""
        key = AssetKeyHelper.build_key()
        assert key == ""

    def test_build_prefixed_key(self):
        """Build asset key with layer and source prefix."""
        key = AssetKeyHelper.build_prefixed_key("raw", "source1", "table1")
        assert key == "raw__source1__table1"

    def test_build_prefixed_key_multiple_parts(self):
        """Build prefixed asset key with multiple additional parts."""
        key = AssetKeyHelper.build_prefixed_key(
            "processed", "source2", "part1", "part2", "part3"
        )
        assert key == "processed__source2__part1__part2__part3"

    def test_build_prefixed_key_no_additional_parts(self):
        """Build prefixed asset key with only layer and source."""
        key = AssetKeyHelper.build_prefixed_key("layer", "source")
        assert key == "layer__source__"

    def test_build_key_with_special_characters(self):
        """Build asset key with parts containing special characters."""
        key = AssetKeyHelper.build_key("part-1", "part_2", "part.3")
        assert key == "part-1__part_2__part.3"

    def test_build_prefixed_key_with_special_characters(self):
        """Build prefixed asset key with special characters in parts."""
        key = AssetKeyHelper.build_prefixed_key("my-layer", "my_source", "my.part")
        assert key == "my-layer__my_source__my.part"
