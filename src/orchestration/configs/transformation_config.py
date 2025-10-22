from orchestration.assets.transform import OpConfig, DatasetMappingConfig
from orchestration.constants import DataSource, AssetLayer
from orchestration.utils import AssetKeyHelper

"""
Asset: [[transform.py#filtered__hf__dataset]]
Defined in: src/orchestration/assets/ingest.py
"""
FILTERED_HF_DATASET_OP_CONFIG = {
    AssetKeyHelper.build_prefixed_key(
        AssetLayer.INT, DataSource.HUGGINGFACE, "filtered", "hf", "dataset"
    ): {
        "config": OpConfig(
            op_type="filter",
            op_name="filter_schematisms",
            input_columns=["schematism_name"],
            kwargs={"to_filter": ["wloclawek_1872"]},
        ).model_dump()
    }
}

"""
Asset: [[transform.py#gt__dataset__pydantic]]
Defined in: src/orchestration/assets/transform.py
"""
GT_DATASET_PYDANTIC_OP_CONFIG = {
    AssetKeyHelper.build_prefixed_key(
        AssetLayer.INT, DataSource.HUGGINGFACE, "gt", "dataset", "pydantic"
    ): {"config": DatasetMappingConfig().model_dump()}
}
