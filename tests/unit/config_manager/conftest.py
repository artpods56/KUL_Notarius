import pytest

from notarius.infrastructure.config.manager import ConfigManager
from notarius.shared.constants import CONFIGS_DIR


@pytest.fixture(scope="session")
def config_manager() -> ConfigManager:
    """One ConfigManager shared by the whole test session."""
    return ConfigManager(CONFIGS_DIR)
