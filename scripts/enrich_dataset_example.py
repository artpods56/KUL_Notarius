"""Example script for enriching a dataset with contextual LLM predictions.

This demonstrates how to use the EnrichDatasetWithContextualLLM use case
to process a dataset of schematism pages while maintaining input context.
"""

from pathlib import Path

from dotenv import load_dotenv
from omegaconf import DictConfig, OmegaConf
import structlog

from notarius.application.use_cases.gen.enrich_dataset_with_contextual_llm import (
    EnrichDatasetWithContextualLLM,
    EnrichDatasetRequest,
)
from notarius.infrastructure.config.constants import ConfigType, ModelsConfigSubtype
from notarius.infrastructure.config.manager import with_configs
from notarius.infrastructure.llm.engine_adapter import LLMEngine
from notarius.infrastructure.llm.prompt_manager import Jinja2PromptRenderer
from notarius.schemas.configs import LLMEngineConfig
from notarius.schemas.data.pipeline import BaseDataset, PredictionDataItem

logger = structlog.get_logger(__name__)

# Load environment variables
load_dotenv()


def create_sample_dataset() -> BaseDataset[PredictionDataItem]:
    """Create a sample dataset for demonstration.

    In production, you would load this from your data source.
    Replace this with actual dataset loading logic.
    """
    # Example: Create dataset from a directory of images
    data_dir = Path("data/schematism_pages")

    items = []
    if data_dir.exists():
        for idx, image_path in enumerate(sorted(data_dir.glob("*.jpg"))):
            item = PredictionDataItem(
                image_path=str(image_path),
                text=None,  # OCR text would be loaded here if available
                metadata={"sample_id": idx, "source": "example"},
            )
            items.append(item)

    if not items:
        logger.warning(
            "No images found - using mock data",
            data_dir=str(data_dir),
        )
        # Fallback to mock data for demonstration
        items = [
            PredictionDataItem(
                image_path="path/to/page1.jpg",
                text="Mock OCR text for page 1",
                metadata={"page": 1},
            ),
            PredictionDataItem(
                image_path="path/to/page2.jpg",
                text="Mock OCR text for page 2",
                metadata={"page": 2},
            ),
        ]

    return BaseDataset[PredictionDataItem](items=items)


@with_configs(
    llm_model_config=("llm_model_config", ConfigType.MODELS, ModelsConfigSubtype.LLM),
)
def main(llm_model_config: DictConfig):
    """Main execution function.

    Args:
        llm_model_config: LLM configuration loaded from configs
    """
    logger.info("Starting contextual dataset enrichment example")

    # Convert OmegaConf to Pydantic config
    config_dict = OmegaConf.to_container(llm_model_config)
    config = LLMEngineConfig(**config_dict)  # type: ignore

    # Initialize components
    engine = LLMEngine(config=config)
    prompt_renderer = Jinja2PromptRenderer()

    # Create use case
    use_case = EnrichDatasetWithContextualLLM(
        engine=engine,
        prompt_renderer=prompt_renderer,
    )

    # Load or create dataset
    dataset = create_sample_dataset()
    logger.info("Dataset loaded", item_count=len(dataset.items))

    # Create enrichment request
    request = EnrichDatasetRequest(
        dataset=dataset,
        system_template="system.j2",  # Your detailed extraction instructions
        user_template="user.j2",  # Template for OCR/hints
        include_ocr=True,
        include_hints=False,  # Set to True if you have model hints
    )

    # Execute enrichment with contextual input
    logger.info("Starting enrichment process")
    response = use_case.execute(request)

    # Display sample
    logger.info(
        "Enrichment completed",
        successful=response.successful_predictions,
        failed=response.failed_predictions,
        total=len(dataset.items),
        conversation_length=len(response.conversation.messages),
    )

    # Example: Access enriched data
    for idx, item in enumerate(response.enriched_dataset.items):
        if item.predictions:
            logger.info(
                "Item prediction",
                index=idx,
                page_number=item.predictions.page_number,
                entries_count=len(item.predictions.entries),
                valid_entries=item.predictions.count_valid_entries(),
            )

            # Example: Print first entry of each page
            if item.predictions.entries:
                first_entry = item.predictions.entries[0]
                logger.info(
                    "First entry on page",
                    page=item.predictions.page_number,
                    parish=first_entry.parish,
                    deanery=first_entry.deanery,
                    dedication=first_entry.dedication,
                )

    # Optional: Save enriched dataset
    # output.enriched_dataset.save_to_file("enriched_dataset.json")

    logger.info("Example completed successfully")


if __name__ == "__main__":
    main()  # type: ignore
