import dagster as dg
from dagster import ConfigMapping

from orchestration.configs.ingestion_config import HUGGINGFACE_DATASET_INGESTION_OP_CONFIG
from orchestration.configs.transformation_config import HUGGINGFACE_DATASET_TRANSFORMATION_OP_CONFIG, \
    HUGGINGFACE_DATASET_CONVERSION_CONFIG

from orchestration.assets.ingest import (
    huggingface_dataset,
    pdf_to_dataset,
    DatasetConfig,
)

from core.data import filters
import schemas.configs # type: ignore
from orchestration.assets.transform import (
    filtered_huggingface_dataset,
    hf_ground_truth_dataset,
)
from orchestration.resources import OpRegistry
from orchestration.resources import PdfFilesResource, ConfigManagerResource

# op_registry = get_op_registry()

evaluation_job = dg.define_asset_job(
    name="ingestion_pipeline",
    selection=dg.AssetSelection.assets(
        huggingface_dataset, filtered_huggingface_dataset, hf_ground_truth_dataset
    ),
    config={
        "ops":{
            **HUGGINGFACE_DATASET_INGESTION_OP_CONFIG,
            **HUGGINGFACE_DATASET_TRANSFORMATION_OP_CONFIG,
            **HUGGINGFACE_DATASET_CONVERSION_CONFIG
            }
    }
)

defs = dg.Definitions(
    assets=[huggingface_dataset, filtered_huggingface_dataset, hf_ground_truth_dataset],
    jobs=[evaluation_job],
    resources={
        # "pdf_files": PdfFilesResource(),
        "config_manager": ConfigManagerResource(),
        "op_registry":  OpRegistry

    },
)