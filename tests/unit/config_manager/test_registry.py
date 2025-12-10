import pytest
from pathlib import Path
from tempfile import TemporaryDirectory

from notarius.infrastructure.config.registry import (
    get_config_schema,
    list_registered_configs,
    validate_config_with_schema,
    get_default_config,
)
from notarius.infrastructure.config.constants import ConfigType, TestsConfigSubtype
from notarius.infrastructure.config.utils import discover_config_files
from notarius.schemas.configs.tests_config import BaseTestsConfig

from structlog import get_logger

logger = get_logger(__name__)


@pytest.fixture
def tests_config_type():
    return ConfigType.TESTS.value


@pytest.fixture
def tests_config_subtype():
    return TestsConfigSubtype.DEFAULT.value


def test_register_config():
    """Test that configs are registered correctly."""
    schema = get_config_schema(ConfigType.TESTS, TestsConfigSubtype.DEFAULT)
    assert schema is not None
    assert schema == BaseTestsConfig


def test_list_registered_configs(tests_config_type, tests_config_subtype):
    """Test listing registered configs."""
    registered = list_registered_configs()
    assert tests_config_type in registered
    assert tests_config_subtype in registered[tests_config_type]


def test_validate_config_with_schema():
    """Test config_manager validation with schema."""
    config_data = {"default_field": "sample_value"}
    validated = validate_config_with_schema(config_data, BaseTestsConfig)

    assert validated.default_field == "sample_value"


def test_validate_config_with_schema_failure():
    """Test config_manager validation failure."""
    config_data = {"default_field": 0}
    with pytest.raises(Exception):
        validate_config_with_schema(config_data, BaseTestsConfig)


def test_get_default_config():
    """Test getting default config_manager."""
    default = get_default_config(ConfigType.TESTS, TestsConfigSubtype.DEFAULT)
    assert default is not None
    assert default["default_field"] == "default"


def test_discover_config_files():
    """Test discovering config_manager files."""
    with TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create nested structure
        config_dir = temp_path / "ml_models" / "llm"
        config_dir.mkdir(parents=True)

        # Create a test config_manager file
        config_file = config_dir / "test.yaml"
        config_file.write_text("test: config_manager")

        # Test discovery
        discovered = discover_config_files(temp_path)
        assert "ml_models" in discovered
        assert "llm" in discovered["ml_models"]
        assert "test" in discovered["ml_models"]["llm"]


def test_default_config_generation(config_manager, monkeypatch):
    """Test discovering config_manager files in flat structure."""
    with TemporaryDirectory() as temp_dir:
        monkeypatch.setattr(config_manager, "configs_dir", Path(temp_dir))

        config_manager.generate_default_configs()
        available_configs = config_manager.list_available_configs()

        for config_type in ConfigType:
            assert config_type.value in available_configs


def test_get_config_schema_nonexistent():
    """Test getting non-existent config_manager schema."""
    with pytest.raises(AttributeError) as exc_info:
        schema = get_config_schema(
            ConfigType.TESTS,
            TestsConfigSubtype.NONEXISTENT,  # pyright: ignore[reportAttributeAccessIssue]
        )


def test_get_default_config_nonexistent():
    """Test getting default config_manager for non-existent schema."""

    with pytest.raises(AttributeError) as exc_info:
        default = get_default_config(
            ConfigType.TESTS,
            TestsConfigSubtype.NONEXISTENT,  # pyright: ignore[reportAttributeAccessIssue]
        )


def test_discover_mixed_config_structure():
    """Test discovering mixed config_manager structures (both direct files and subdirectories)."""
    with TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create mixed structure:
        # ml_models/
        #   ├── config_manager.yaml        (default subtype)
        #   ├── global.yaml        (default subtype)
        #   ├── llm/
        #   │   └── openai.yaml
        #   └── lmv3/
        #       └── base.yaml

        models_dir = temp_path / "ml_models"
        models_dir.mkdir()

        # Create direct YAML files
        (models_dir / "config_manager.yaml").write_text("config_manager: true")
        (models_dir / "global.yaml").write_text("global: true")

        # Create subdirectories
        llm_dir = models_dir / "llm"
        llm_dir.mkdir()
        (llm_dir / "openai.yaml").write_text("backend: openai")

        lmv3_dir = models_dir / "lmv3"
        lmv3_dir.mkdir()
        (lmv3_dir / "base.yaml").write_text("model: base")

        # Test discovery
        discovered = discover_config_files(temp_path)

        # Verify mixed structure is handled correctly
        assert "ml_models" in discovered
        assert "default" in discovered["ml_models"]
        assert "llm" in discovered["ml_models"]
        assert "lmv3" in discovered["ml_models"]

        assert set(discovered["ml_models"]["default"]) == {"config_manager", "global"}
        assert discovered["ml_models"]["llm"] == ["openai"]
        assert discovered["ml_models"]["lmv3"] == ["base"]
