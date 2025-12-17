from dotenv import load_dotenv
from omegaconf import DictConfig

from notarius.orchestration.operations import filter_schematisms, filter_empty_samples
from notarius.infrastructure.config.constants import (
    ConfigType,
    DatasetConfigSubtype,
    ModelsConfigSubtype,
)
from notarius.infrastructure.config.helpers import with_configs
from notarius.application.types import EngineConfigMap
from notarius.infrastructure.ml_models.lmv3.engine_adapter import LMv3Engine
from notarius.infrastructure.ocr.adapter import OcrEngine
from notarius.infrastructure.llm.llm_engine import LLMEngine
from notarius.pipeline.pipeline import (
    IngestionPhase,
    Pipeline,
    DatasetProcessingPhase,
)
from notarius.pipeline.steps.export import (
    DownloadSamplesStep,
)
from notarius.pipeline.steps.ingestion import (
    HuggingFaceIngestionStep,
    FilterMapSpec,
)
from notarius.pipeline.steps.postprocessing import ReplaceValuesStep
from notarius.pipeline.steps.wrappers import (
    HuggingFaceToPipelineDataStep,
)
from notarius.misc.logging import setup_logging
from notarius.shared.constants import TMP_DIR

setup_logging()

import structlog

logger = structlog.get_logger(__name__)

envs = load_dotenv()
if not envs:
    logger.warning("No environment variables loaded.")


@with_configs(
    dataset_config=(
        "schematism_dataset_config",
        ConfigType.DATASET,
        DatasetConfigSubtype.EVALUATION,
    ),
    llm_model_config=("llm_model_config", ConfigType.MODELS, ModelsConfigSubtype.LLM),
    lmv3_model_config=(
        "lmv3_model_config",
        ConfigType.MODELS,
        ModelsConfigSubtype.LMV3,
    ),
    ocr_model_config=("ocr_model_config", ConfigType.MODELS, ModelsConfigSubtype.OCR),
)
def main(
    dataset_config: DictConfig,
    llm_model_config: DictConfig,
    lmv3_model_config: DictConfig,
    ocr_model_config: DictConfig,
):
    model_configs: EngineConfigMap = {
        LLMEngine: llm_model_config,
        LMv3Engine: lmv3_model_config,
        OcrEngine: ocr_model_config,
    }

    pipeline = Pipeline(model_configs=model_configs, batched=True, batch_size=9)
    schematisms_to_filter = list(
        dataset_config.full_schematisms + dataset_config.partial_schematisms
    )
    languages = list(dataset_config.get("languages", []))
    column_map = dict(dataset_config.get("column_map", {}))

    ingestion_phase = IngestionPhase(
        name="ingestion",
        steps=[
            HuggingFaceToPipelineDataStep(
                wrapped_step=HuggingFaceIngestionStep(
                    dataset_config=dataset_config,
                    filter_map_specs=[
                        FilterMapSpec(
                            operations=[
                                filter_schematisms(schematisms_to_filter),
                            ],
                            input_columns=["schematism_name"],
                            operation_type="filter",
                        ),
                        FilterMapSpec(
                            operations=[
                                filter_empty_samples,
                            ],
                            input_columns=["sample"],
                            operation_type="filter",
                        ),
                    ],
                ),
                column_map=column_map,
            ),
        ],
        description="Dataset ingestion from local files.",
    )

    export_phase = DatasetProcessingPhase(
        name="export",
        steps=[
            ReplaceValuesStep(
                source="ground_truth",
                old_value="[brak_informacji]",
                new_value=None,
            ),
            DownloadSamplesStep(directory=TMP_DIR / "input", shuffle=False),
        ],
        description="Exporting the data to file and database.",
        depends_on=ingestion_phase,
    )
    pipeline.add_phases(
        [
            ingestion_phase,
            export_phase,
        ]
    )

    pipeline.run()


if __name__ == "__main__":
    main()
