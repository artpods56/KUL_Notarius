import dagster as dg
from dagster import ConfigMapping

from orchestration.configs.ingestion_config import (
    RAW_HF_DATASET_OP_CONFIG,
)
from orchestration.configs.references_config import RES_OCR_MODEL_CONFIG_OP_CONFIG
from orchestration.configs.transformation_config import (
    FILTERED_HF_DATASET_OP_CONFIG,
    GT_DATASET_PYDANTIC_OP_CONFIG,
)
from orchestration.configs.models_config import RES_OCR_MODEL_OP_CONFIG

from orchestration.assets.ingest import (
    raw__hf__dataset,
    raw__pdf__dataset,
)

from orchestration.assets.transform import (
    filtered__hf__dataset,
    gt__dataset__pydantic,
)

from orchestration.assets.configs import ocr_model__config

from orchestration.assets.models import ocr_model

import core.data.filters  # type: ignore
import schemas.configs  # type: ignore

from orchestration.resources import OpRegistry
from orchestration.resources import PdfFilesResource, ConfigManagerResource


ingestion_assets = [raw__hf__dataset, filtered__hf__dataset, gt__dataset__pydantic]

ingestion_job = dg.define_asset_job(
    name="ingestion_pipeline",
    selection=dg.AssetSelection.assets(*ingestion_assets),
    config={
        "ops": {
            **RAW_HF_DATASET_OP_CONFIG,
            **FILTERED_HF_DATASET_OP_CONFIG,
            **GT_DATASET_PYDANTIC_OP_CONFIG,
        }
    },
)

configs_assets = [ocr_model__config]

models_assets = [ocr_model]

prediction_assets = []

prediction_job = dg.define_asset_job(
    name="prediction_pipeline",
    selection=dg.AssetSelection.assets(
        *configs_assets, *models_assets, *prediction_assets
    ),
    config={"ops": {**RES_OCR_MODEL_CONFIG_OP_CONFIG, **RES_OCR_MODEL_OP_CONFIG}},
)


defs = dg.Definitions(
    assets=[
        # ingestion
        *ingestion_assets,
        # configs
        *configs_assets,
        # models
        *models_assets,
        # prediction
        *prediction_assets,
    ],
    jobs=[ingestion_job, prediction_job],
    resources={
        # "pdf_files": PdfFilesResource(),
        "config_manager": ConfigManagerResource(),
        "op_registry": OpRegistry,
    },
)
