from orchestration.assets.transform import (
    OpConfig,
    DatasetMappingConfig,
    PandasDataFrameConfig,
)
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
            kwargs={
                "to_filter": [
                    "wloclawek_1872",
                    "wloclawek_1873",
                    "tarnow_1870",
                    "chelmno_1871",
                ]
            },
        ).model_dump()
    }
}


"""
Asset: [[transform.py#base__dataset__pydantic]]
Defined in: src/orchestration/assets/transform.py
"""
BASE_DATASET_PYDANTIC_OP_CONFIG = {
    AssetKeyHelper.build_prefixed_key(
        AssetLayer.INT, DataSource.HUGGINGFACE, "base", "dataset", "pydantic"
    ): {"config": DatasetMappingConfig().model_dump()}
}


"""
Asset: [[transform.py#gt__source_dataset__pydantic]]
Defined in: src/orchestration/assets/transform.py
"""
GT_SOURCE_DATASET_PYDANTIC_OP_CONFIG = {
    AssetKeyHelper.build_prefixed_key(
        AssetLayer.INT, DataSource.HUGGINGFACE, "gt", "source_dataset", "pydantic"
    ): {"config": DatasetMappingConfig(ground_truth_source="source").model_dump()}
}

"""
Asset: [[transform.py#gt__parsed_dataset__pydantic]]
Defined in: src/orchestration/assets/transform.py
"""
GT_PARSED_DATASET_PYDANTIC_OP_CONFIG = {
    AssetKeyHelper.build_prefixed_key(
        AssetLayer.INT, DataSource.HUGGINGFACE, "gt", "parsed_dataset", "pydantic"
    ): {"config": DatasetMappingConfig(ground_truth_source="parsed").model_dump()}
}


"""
Asset: [[transform.py#eval__aligned_dataframe__pandas]]
Defined in: src/orchestration/assets/transform.py
"""
EVAL_ALIGNED_DATAFRAME_PANDAS_OP_CONFIG = {
    AssetKeyHelper.build_prefixed_key(
        AssetLayer.MRT, DataSource.HUGGINGFACE, "eval", "aligned_dataframe", "pandas"
    ): {"config": PandasDataFrameConfig().model_dump()}
}
