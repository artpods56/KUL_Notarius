import pytest

from notarius.infrastructure.config.constants import (
    ConfigType,
    DatasetConfigSubtype,
    ModelsConfigSubtype,
)
from notarius.infrastructure.config.exceptions import InvalidConfigSubtype
from notarius.infrastructure.config.utils import validate_config_arguments


@validate_config_arguments
def _standalone(config_type, config_subtype):  # noqa: ANN001  (type checked by decorator)
    """Minimal stand-alone target used only for decorator tests."""
    return config_type, config_subtype


class _Dummy:
    """Class with a decorated instance method to ensure *cls* is ignored."""

    @validate_config_arguments
    def method(self, config_type, config_subtype):  # noqa: ANN001
        return config_type, config_subtype


def test_standalone_valid():
    """Decorator passes through correct config_manager parameters on a standalone function."""

    result = _standalone(ConfigType.DATASET, DatasetConfigSubtype.DEFAULT)
    assert result == (ConfigType.DATASET, DatasetConfigSubtype.DEFAULT)


def test_method_valid():
    """Decorator correctly handles *cls* and validates instance methods."""

    obj = _Dummy()
    result = obj.method(ConfigType.MODELS, ModelsConfigSubtype.LLM)
    assert result == (ConfigType.MODELS, ModelsConfigSubtype.LLM)


def test_missing_arguments():
    """Calling without the required args should raise ``ValueError``."""

    with pytest.raises(ValueError):
        _standalone()  # pyright: ignore[reportCallIssue]


def test_invalid_subtype():
    """Mismatched subtype (MODEL subtype for DATASET type) should fail."""

    with pytest.raises(InvalidConfigSubtype):
        _standalone(ConfigType.DATASET, ModelsConfigSubtype.LLM)
