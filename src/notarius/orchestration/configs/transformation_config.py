from notarius.orchestration.assets.transform.preprocess import OpConfig
from notarius.orchestration.assets.transform.transform import (
    PandasDataFrameConfig,
)
from notarius.orchestration.constants import DataSource, AssetLayer
from notarius.orchestration.utils import AssetKeyHelper

"""
Asset: [[preprocess.py#filtered__hf__dataset]]
Defined in: [[src/orchestration/assets/transform/preprocess.py]]
Resolves to: int__huggingface__filtered__hf__dataset
"""
FILTERED__HF__DATASET_OP_CONFIG = {
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

#
# """
# Asset: [[transform.py#base__dataset__pydantic]]
# Defined in: [[src/orchestration/assets/transform/transform.py]]
# Resolves to: int__huggingface__base__dataset__pydantic
# """
# BASE__DATASET__PYDANTIC_OP_CONFIG = {
#     AssetKeyHelper.build_prefixed_key(
#         AssetLayer.INT, DataSource.HUGGINGFACE, "base", "dataset", "pydantic"
#     ): {"config": DatasetMappingConfig().model_dump()}
# }
#
#
# """
# Asset: [[transform.py#gt__source_dataset__pydantic]]
# Defined in: [[src/orchestration/assets/transform/transform.py]]
# Resolves to: int__huggingface__gt__source_dataset__pydantic
# """
# GT_SOURCE_DATASET_PYDANTIC_OP_CONFIG = {
#     AssetKeyHelper.build_prefixed_key(
#         AssetLayer.INT, DataSource.HUGGINGFACE, "gt", "source_dataset", "pydantic"
#     ): {"config": DatasetMappingConfig(ground_truth_source="source").model_dump()}
# }
#
# """
# Asset: [[transform.py#gt__parsed_dataset__pydantic]]
# Defined in: [[src/orchestration/assets/transform/transform.py]]
# Resolves to: int__huggingface__gt__parsed_dataset__pydantic
# """
# GT_PARSED_DATASET_PYDANTIC_OP_CONFIG = {
#     AssetKeyHelper.build_prefixed_key(
#         AssetLayer.INT, DataSource.HUGGINGFACE, "gt", "parsed_dataset", "pydantic"
#     ): {"config": DatasetMappingConfig(ground_truth_source="parsed").model_dump()}
# }


"""
Asset: [[transform.py#eval__aligned_source_dataframe__pandas]]
Defined in: [[src/orchestration/assets/transform/transform.py]]
Resolves to: mrt__huggingface__eval__aligned_source_dataframe__pandas
"""
EVAL__ALIGNED_SOURCE_DATAFRAME__PANDAS__OP_CONFIG = {
    AssetKeyHelper.build_prefixed_key(
        AssetLayer.MRT,
        DataSource.HUGGINGFACE,
        "eval",
        "aligned_source_dataframe",
        "pandas",
    ): {"config": PandasDataFrameConfig().model_dump()}
}


"""
Asset: [[transform.py#eval__aligned_parsed_dataframe__pandas]]
Defined in: [[src/orchestration/assets/transform/transform.py]]
Resolves to: mrt__huggingface__eval__aligned_parsed_dataframe__pandas
"""
EVAL__ALIGNED_PARSED_DATAFRAME__PANDAS__OP_CONFIG = {
    AssetKeyHelper.build_prefixed_key(
        AssetLayer.MRT,
        DataSource.HUGGINGFACE,
        "eval",
        "aligned_parsed_dataframe",
        "pandas",
    ): {"config": PandasDataFrameConfig().model_dump()}
}
