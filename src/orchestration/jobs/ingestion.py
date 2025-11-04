import dagster as dg

from orchestration.assets.configs import hf_dataset__config
from orchestration.assets.extract.ingest import raw__hf__dataset
from orchestration.assets.transform.preprocess import filtered__hf__dataset
from orchestration.assets.transform.transform import (
    gt__source_dataset__pydantic,
    gt__parsed_dataset__pydantic,
    base__dataset__pydantic,
)
from orchestration.configs.references_config import RES_HF_DATASET_CONFIG_OP_CONFIG
from orchestration.configs.transformation_config import (
    FILTERED__HF__DATASET_OP_CONFIG,
    GT_SOURCE_DATASET_PYDANTIC_OP_CONFIG,
    GT_PARSED_DATASET_PYDANTIC_OP_CONFIG,
    BASE__DATASET__PYDANTIC_OP_CONFIG,
)

dataset_config_assets = [hf_dataset__config]
ingestion_assets = [
    raw__hf__dataset,
    filtered__hf__dataset,
    gt__source_dataset__pydantic,
    gt__parsed_dataset__pydantic,
    base__dataset__pydantic,
]
ingestion_job = dg.define_asset_job(
    name="ingestion_pipeline",
    selection=dg.AssetSelection.assets(*dataset_config_assets, *ingestion_assets),
    config={
        "ops": {
            # configs refs
            **RES_HF_DATASET_CONFIG_OP_CONFIG,
            # asset refs
            # **RAW_HF_DATASET_OP_CONFIG,
            **FILTERED__HF__DATASET_OP_CONFIG,
            **GT_SOURCE_DATASET_PYDANTIC_OP_CONFIG,
            **GT_PARSED_DATASET_PYDANTIC_OP_CONFIG,
            **BASE__DATASET__PYDANTIC_OP_CONFIG,
        }
    },
)
