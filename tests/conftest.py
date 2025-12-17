import json

import pytest
from PIL import Image
from dotenv import load_dotenv
from omegaconf import DictConfig

from notarius.infrastructure.config.constants import (
    DatasetConfigSubtype,
    ModelsConfigSubtype,
)
from notarius.infrastructure.config.manager import ConfigType
from notarius.domain.entities.schematism import SchematismPage



@pytest.fixture(scope="session", autouse=True)
def _load_dotenv_once_for_everybody() -> None:
    """Load .env file once per test session."""
    load_dotenv()


@pytest.fixture(scope="session")
def dataset_config(config_manager) -> DictConfig:
    return config_manager.load_config(
        config_name="schematism_dataset_config",
        config_type=ConfigType.DATASET,
        config_subtype=DatasetConfigSubtype.EVALUATION,
    )


@pytest.fixture(scope="session")
def llm_model_config(config_manager) -> DictConfig:
    """Loads the configuration for LLM model."""
    return config_manager.load_config(
        config_name="tests_llm_config",
        config_type=ConfigType.MODELS,
        config_subtype=ModelsConfigSubtype.LLM,
    )


@pytest.fixture(scope="session")
def lmv3_model_config(config_manager) -> DictConfig:
    """Loads the configuration for LayoutLMv3 model."""
    return config_manager.load_config(
        config_name="lmv3_model_config",
        config_type=ConfigType.MODELS,
        config_subtype=ModelsConfigSubtype.LMV3,
    )


@pytest.fixture(scope="session")
def ocr_model_config(config_manager) -> DictConfig:
    return config_manager.load_config(
        config_name="ocr_model_config",
        config_type=ConfigType.MODELS,
        config_subtype=ModelsConfigSubtype.OCR,
    )


@pytest.fixture
def sample_structured_response() -> str:
    return json.dumps(
        {
            "page_number": "56",
            "entries": [
                {
                    "parish": "sample",
                    "deanery": None,
                    "dedication": "sample",
                    "building_material": "sample",
                }
            ],
        }
    )


@pytest.fixture
def sample_pil_image():
    """Generate a synthetic PIL image for testing instead of loading from file."""
    # Create a synthetic image with some text-like patterns
    pil_image = Image.new("RGB", (800, 600), color="white")
    return pil_image


@pytest.fixture
def sample_page_data(sample_structured_response) -> SchematismPage:
    return SchematismPage(**json.loads(sample_structured_response))


@pytest.fixture
def large_sample_image():
    """Large sample image for performance testing"""
    return Image.new(mode="RGB", size=(2000, 2000))


@pytest.fixture
def malformed_json_response():
    """Sample malformed JSON output"""
    return '{"page_number": "56", "entries": [{'
