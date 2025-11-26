"""
This module initializes and generates default configuration files.

The module sets up logging, loads environment variables, and uses a configuration
manager to generate default configuration files. It ensures that the necessary
initial setup is performed before generating configurations.
"""
from dotenv import load_dotenv
from structlog import get_logger

from core.config.manager import ConfigManager
from core.utils.logging import setup_logging
from core.utils.shared import CONFIGS_DIR

setup_logging()
load_dotenv()

logger = get_logger(__name__)

def main():
    logger.info("Generating default configs...")
    logger.info(f"Using config directory: {CONFIGS_DIR}")
    config_manager = ConfigManager(CONFIGS_DIR)
    logger.info("Starting generation of default configs...")
    config_manager.generate_default_configs(overwrite=True)
    logger.info("Default configs generation completed successfully!")

if __name__ == "__main__":
    main()
