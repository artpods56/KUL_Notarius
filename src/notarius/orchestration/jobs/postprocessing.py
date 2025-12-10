import dagster as dg

from notarius.orchestration.assets.transform.postprocess import (
    pred__deanery_filled_dataset__pydantic,
    pred__parsed_dataset__pydantic,
    gt__aligned_parsed_dataset__pydantic,
    gt__aligned_source_dataset__pydantic,
)
from notarius.orchestration.assets.transform.transform import (
    eval__aligned_source_dataframe__pandas,
    eval__aligned_parsed_dataframe__pandas,
)
from notarius.orchestration.configs.postprocessing_config import (
    PRED__DEANERY_FILLED_DATASET__PYDANTIC__OP_CONFIG,
    PRED__PARSED_DATASET__PYDANTIC__OP_CONFIG,
    GT__ALIGNED_SOURCE_DATASET__PYDANTIC__OP_CONFIG,
    GT__ALIGNED_PARSED_DATASET__PYDANTIC__OP_CONFIG,
)
from notarius.orchestration.configs.transformation_config import (
    EVAL__ALIGNED_SOURCE_DATAFRAME__PANDAS__OP_CONFIG,
)

postprocessing_assets = [
    pred__deanery_filled_dataset__pydantic,
    pred__parsed_dataset__pydantic,
    gt__aligned_parsed_dataset__pydantic,
    gt__aligned_source_dataset__pydantic,
    eval__aligned_source_dataframe__pandas,
    eval__aligned_parsed_dataframe__pandas,
]
postprocessing_job = dg.define_asset_job(
    name="postprocessing_pipeline",
    selection=dg.AssetSelection.assets(*postprocessing_assets),
    config={
        "ops": {
            # asset refs
            **PRED__DEANERY_FILLED_DATASET__PYDANTIC__OP_CONFIG,
            **PRED__PARSED_DATASET__PYDANTIC__OP_CONFIG,
            **EVAL__ALIGNED_SOURCE_DATAFRAME__PANDAS__OP_CONFIG,
            **GT__ALIGNED_SOURCE_DATASET__PYDANTIC__OP_CONFIG,
            **GT__ALIGNED_PARSED_DATASET__PYDANTIC__OP_CONFIG,
        }
    },
)
