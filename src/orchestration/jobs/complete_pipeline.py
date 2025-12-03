import dagster as dg

from orchestration.configs.exporting_config import (
    EVAL__EXCEL_EXPORT_PARSED_DATAFRAME__PANDAS,
    EVAL__EXCEL_EXPORT_SOURCE_DATAFRAME__PANDAS,
    EVAL__WANDB_EXPORT_DATAFRAME__PANDAS,
)
from orchestration.configs.postprocessing_config import (
    PRED__DEANERY_FILLED_DATASET__PYDANTIC__OP_CONFIG,
    PRED__PARSED_DATASET__PYDANTIC__OP_CONFIG,
    GT__ALIGNED_SOURCE_DATASET__PYDANTIC__OP_CONFIG,
    GT__ALIGNED_PARSED_DATASET__PYDANTIC__OP_CONFIG,
)
from orchestration.configs.prediction_config import (
    PRED__OCR_ENRICHED_DATASET__PYDANTIC__OP_CONFIG,
    PRED__LMV3_ENRICHED_DATASET__PYDANTIC__OP_CONFIG,
    PRED__LLM_ENRICHED_DATASET__PYDANTIC__OP_CONFIG,
)
from orchestration.configs.references_config import (
    RES_HF_DATASET_CONFIG_OP_CONFIG,
    RES_OCR_MODEL_CONFIG_OP_CONFIG,
    RES_LLM_MODEL_CONFIG_OP_CONFIG,
    RES_LMV3_MODEL_CONFIG_OP_CONFIG,
)
from orchestration.configs.transformation_config import (
    FILTERED__HF__DATASET_OP_CONFIG,
    GT_SOURCE_DATASET_PYDANTIC_OP_CONFIG,
    GT_PARSED_DATASET_PYDANTIC_OP_CONFIG,
    BASE__DATASET__PYDANTIC_OP_CONFIG,
    EVAL__ALIGNED_SOURCE_DATAFRAME__PANDAS__OP_CONFIG,
)
from orchestration.jobs.exporting import exporting_assets
from orchestration.jobs.ingestion import dataset_config_assets, ingestion_assets
from orchestration.jobs.postprocessing import postprocessing_assets
from orchestration.jobs.prediction import (
    models_config_assets,
    models_assets,
    prediction_assets,
)

# Combine all assets - Dagster will automatically resolve dependencies and execute in order
all_pipeline_assets = [
    # Step 1: Configs
    *dataset_config_assets,
    *models_config_assets,
    # Step 2: Ingestion
    *ingestion_assets,
    # Step 3: Models (loaded once, used throughout)
    *models_assets,
    # Step 4: Prediction
    *prediction_assets,
    # Step 5: Postprocessing
    *postprocessing_assets,
    # Step 6: Export
    *exporting_assets,
]

# Complete pipeline job - materializes all assets from start to finish
complete_pipeline_job = dg.define_asset_job(
    name="complete_pipeline",
    description="Full end-to-end pipeline: ingestion -> prediction -> postprocessing -> export",
    selection=dg.AssetSelection.assets(*all_pipeline_assets),
    config={
        "ops": {
            # Ingestion configs
            **RES_HF_DATASET_CONFIG_OP_CONFIG,
            **FILTERED__HF__DATASET_OP_CONFIG,
            **GT_SOURCE_DATASET_PYDANTIC_OP_CONFIG,
            **GT_PARSED_DATASET_PYDANTIC_OP_CONFIG,
            **BASE__DATASET__PYDANTIC_OP_CONFIG,
            # Model configs
            **RES_OCR_MODEL_CONFIG_OP_CONFIG,
            **RES_LLM_MODEL_CONFIG_OP_CONFIG,
            **RES_LMV3_MODEL_CONFIG_OP_CONFIG,
            # Prediction configs
            **PRED__OCR_ENRICHED_DATASET__PYDANTIC__OP_CONFIG,
            **PRED__LMV3_ENRICHED_DATASET__PYDANTIC__OP_CONFIG,
            **PRED__LLM_ENRICHED_DATASET__PYDANTIC__OP_CONFIG,
            # Postprocessing configs
            **PRED__DEANERY_FILLED_DATASET__PYDANTIC__OP_CONFIG,
            **PRED__PARSED_DATASET__PYDANTIC__OP_CONFIG,
            **GT__ALIGNED_SOURCE_DATASET__PYDANTIC__OP_CONFIG,
            **GT__ALIGNED_PARSED_DATASET__PYDANTIC__OP_CONFIG,
            **EVAL__ALIGNED_SOURCE_DATAFRAME__PANDAS__OP_CONFIG,
            # Export configs
            **EVAL__EXCEL_EXPORT_PARSED_DATAFRAME__PANDAS,
            **EVAL__EXCEL_EXPORT_SOURCE_DATAFRAME__PANDAS,
            **EVAL__WANDB_EXPORT_DATAFRAME__PANDAS,
        }
    },
)
