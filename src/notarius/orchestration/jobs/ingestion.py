import dagster as dg

from notarius.orchestration.assets.configs import hf_dataset__config
from notarius.orchestration.assets.extract.ingest import (
    raw__hf__dataset,
    # test_raw__hf__dataset,
)
from notarius.orchestration.assets.transform.preprocess import filtered__hf__dataset
from notarius.orchestration.assets.transform.transform import (
    gt__source_dataset__pydantic,
    gt__parsed_dataset__pydantic,
    base__dataset__pydantic,
)
from notarius.orchestration.configs.ingestion_config import (
    RAW_HF_DATASET_OP_CONFIG,
)

from notarius.orchestration.configs.transformation_config import (
    FILTERED__HF__DATASET_OP_CONFIG,
)

dataset_config_assets = []
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
            # asset refs
            **RAW_HF_DATASET_OP_CONFIG,
            **FILTERED__HF__DATASET_OP_CONFIG,
        }
    },
)
