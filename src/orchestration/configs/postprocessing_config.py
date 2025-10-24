from orchestration.assets.postprocess import DeaneryFillingConfig, ParsingConfig, JSONAlignmentConfig
from orchestration.configs.shared import ConfigReference
from orchestration.constants import DataSource, AssetLayer
from orchestration.utils import AssetKeyHelper

"""
Asset: [[postprocess.py#deanery_filled__dataset]]
Defined in: src/orchestration/assets/postprocess.py
Resolves to: fct__huggingface__deanery_filled__dataset
"""
DEANERY_FILLED_DATASET_OP_CONFIG = {
    AssetKeyHelper.build_prefixed_key(
        AssetLayer.FCT, DataSource.HUGGINGFACE, "deanery_filled", "dataset"
    ): {
        "config": DeaneryFillingConfig().model_dump()
    }
}

"""
Asset: [[postprocess.py#parsed__dataset]]
Defined in: src/orchestration/assets/postprocess.py
Resolves to: fct__huggingface__parsed__dataset
"""
PARSED_DATASET_OP_CONFIG = {
    AssetKeyHelper.build_prefixed_key(
        AssetLayer.FCT, DataSource.HUGGINGFACE, "parsed", "dataset"
    ): {
        "config": ParsingConfig().model_dump()
    }
}

"""
Asset: [[postprocess.py#gt_aligned__dataset]]
Defined in: src/orchestration/assets/postprocess.py
Resolves to: fct__huggingface__gt_aligned__dataset
"""
GT_ALIGNED_DATASET_OP_CONFIG = {
    AssetKeyHelper.build_prefixed_key(
        AssetLayer.FCT, DataSource.HUGGINGFACE, "gt_aligned", "dataset"
    ): {
        "config": JSONAlignmentConfig().model_dump()
    }
}
