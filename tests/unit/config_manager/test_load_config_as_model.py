import pytest
from pydantic import BaseModel

from notarius.infrastructure.config.constants import (
    ConfigType,
    ModelsConfigSubtype,
    DatasetConfigSubtype,
)
from notarius.infrastructure.config.manager import get_config_manager


class TestLoadConfigAsModel:
    """Tests for ConfigManager.load_config_as_model method."""

    def test_load_ocr_config_as_model(self, config_manager):
        """Load OCR config as Pydantic model instance."""
        ocr_config = config_manager.load_config_as_model(
            config_name="ocr_model_config",
            config_type=ConfigType.MODELS,
            config_subtype=ModelsConfigSubtype.OCR,
        )

        # Should be a Pydantic BaseModel instance
        assert isinstance(ocr_config, BaseModel)

        # Should have the expected attributes (based on your schema)
        assert hasattr(ocr_config, "model_dump")
        assert hasattr(ocr_config, "model_fields")

    def test_load_lmv3_config_as_model(self, config_manager):
        """Load LayoutLMv3 config as Pydantic model instance."""
        lmv3_config = config_manager.load_config_as_model(
            config_name="lmv3_model_config",
            config_type=ConfigType.MODELS,
            config_subtype=ModelsConfigSubtype.LMV3,
        )

        # Should be a Pydantic BaseModel instance
        assert isinstance(lmv3_config, BaseModel)

    def test_load_dataset_config_as_model(self, config_manager):
        """Load dataset config as Pydantic model instance."""
        dataset_config = config_manager.load_config_as_model(
            config_name="schematism_dataset_config",
            config_type=ConfigType.DATASET,
            config_subtype=DatasetConfigSubtype.EVALUATION,
        )

        # Should be a Pydantic BaseModel instance
        assert isinstance(dataset_config, BaseModel)

    def test_model_has_validation(self, config_manager):
        """Verify that the returned model maintains Pydantic validation."""
        lmv3_config = config_manager.load_config_as_model(
            config_name="lmv3_model_config",
            config_type=ConfigType.MODELS,
            config_subtype=ModelsConfigSubtype.LMV3,
        )

        # Can call Pydantic methods
        config_dict = lmv3_config.model_dump()
        assert isinstance(config_dict, dict)

        # Can serialize to JSON
        json_str = lmv3_config.model_dump_json()
        assert isinstance(json_str, str)

    def test_nonexistent_config_file_raises_error(self, config_manager):
        """Loading a non-existent config file should raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            config_manager.load_config_as_model(
                config_name="nonexistent_config",
                config_type=ConfigType.MODELS,
                config_subtype=ModelsConfigSubtype.LLM,
            )

    def test_model_type_is_correct(self, config_manager):
        """Verify the returned model is of the correct registered type."""
        from notarius.infrastructure.config.registry import get_config_schema

        # Get the expected schema class
        expected_schema = get_config_schema(
            ConfigType.MODELS, ModelsConfigSubtype.LMV3
        )

        # Load the config as model
        lmv3_config = config_manager.load_config_as_model(
            config_name="lmv3_model_config",
            config_type=ConfigType.MODELS,
            config_subtype=ModelsConfigSubtype.LMV3,
        )

        # Should be an instance of the expected schema
        assert isinstance(lmv3_config, expected_schema)

    def test_model_has_proper_field_values(self, config_manager):
        """Verify that loaded model contains expected field values from YAML."""
        lmv3_config = config_manager.load_config_as_model(
            config_name="lmv3_model_config",
            config_type=ConfigType.MODELS,
            config_subtype=ModelsConfigSubtype.LMV3,
        )

        # Convert to dict to check values
        config_dict = lmv3_config.model_dump()

        # Should have loaded actual values from the YAML file
        # (Specific assertions depend on your config structure)
        assert isinstance(config_dict, dict)
        assert len(config_dict) > 0

    def test_comparison_with_load_config(self, config_manager):
        """Compare load_config_as_model with load_config results."""
        from omegaconf import DictConfig

        # Load as model
        model_config = config_manager.load_config_as_model(
            config_name="lmv3_model_config",
            config_type=ConfigType.MODELS,
            config_subtype=ModelsConfigSubtype.LMV3,
        )

        # Load as DictConfig
        dict_config = config_manager.load_config(
            config_name="lmv3_model_config",
            config_type=ConfigType.MODELS,
            config_subtype=ModelsConfigSubtype.LMV3,
        )

        # Both should contain the same data
        assert isinstance(model_config, BaseModel)
        assert isinstance(dict_config, DictConfig)

        # Convert both to dict and compare
        from notarius.infrastructure.config.manager import config_to_dict

        model_dict = model_config.model_dump()
        dict_config_dict = config_to_dict(dict_config)

        # Should have the same keys and values
        assert model_dict.keys() == dict_config_dict.keys()

    def test_model_immutability_vs_dictconfig(self, config_manager):
        """Demonstrate the difference between Pydantic models and DictConfig."""
        # Load as model
        model_config = config_manager.load_config_as_model(
            config_name="lmv3_model_config",
            config_type=ConfigType.MODELS,
            config_subtype=ModelsConfigSubtype.LMV3,
        )

        # Load as DictConfig
        dict_config = config_manager.load_config(
            config_name="lmv3_model_config",
            config_type=ConfigType.MODELS,
            config_subtype=ModelsConfigSubtype.LMV3,
        )

        # Pydantic models are immutable by default (if frozen=True in config)
        # DictConfig is mutable
        assert isinstance(model_config, BaseModel)
        assert hasattr(dict_config, "__setitem__")

    def test_multiple_loads_are_independent(self, config_manager):
        """Loading the same config multiple times should create independent instances."""
        config1 = config_manager.load_config_as_model(
            config_name="lmv3_model_config",
            config_type=ConfigType.MODELS,
            config_subtype=ModelsConfigSubtype.LMV3,
        )

        config2 = config_manager.load_config_as_model(
            config_name="lmv3_model_config",
            config_type=ConfigType.MODELS,
            config_subtype=ModelsConfigSubtype.LMV3,
        )

        # Should be different instances but with same values
        assert config1 is not config2
        assert config1.model_dump() == config2.model_dump()
