from orchestration.assets.transform.postprocess import (
    DeaneryFillingConfig,
    ParsingConfig,
    JSONAlignmentConfig,
)
from orchestration.constants import DataSource, AssetLayer
from orchestration.utils import AssetKeyHelper

"""
Asset: [[postprocess.py#pred__deanery_filled_dataset__pydantic]]
Defined in: src/orchestration/assets/postprocess.py
Resolves to: fct__huggingface__pred__deanery_filled_dataset__pydantic
"""
PRED__DEANERY_FILLED_DATASET__PYDANTIC__OP_CONFIG = {
    AssetKeyHelper.build_prefixed_key(
        AssetLayer.FCT,
        DataSource.HUGGINGFACE,
        "pred",
        "deanery_filled_dataset",
        "pydantic",
    ): {"config": DeaneryFillingConfig().model_dump()}
}

"""
Asset: [[postprocess.py#pred__parsed_dataset__pydantic]]
Defined in: src/orchestration/assets/postprocess.py
Resolves to: fct__huggingface__pred__parsed_dataset__pydantic
"""
PRED__PARSED_DATASET__PYDANTIC__OP_CONFIG = {
    AssetKeyHelper.build_prefixed_key(
        AssetLayer.FCT, DataSource.HUGGINGFACE, "pred", "parsed_dataset", "pydantic"
    ): {"config": ParsingConfig().model_dump()}
}

"""
Asset: [[postprocess.py#gt_aligned__dataset]]
Defined in: src/orchestration/assets/postprocess.py
Resolves to: fct__huggingface__gt__aligned_source_dataset__pydantic
"""
GT__ALIGNED_SOURCE_DATASET__PYDANTIC__OP_CONFIG = {
    AssetKeyHelper.build_prefixed_key(
        AssetLayer.FCT,
        DataSource.HUGGINGFACE,
        "gt",
        "aligned_source_dataset",
        "pydantic",
    ): {"config": JSONAlignmentConfig().model_dump()}
}

"""
Asset: [[postprocess.py#gt__aligned_parsed_dataset__pydantic]]
Defined in: src/orchestration/assets/postprocess.py
Resolves to: fct__huggingface__gt__aligned_parsed_dataset__pydantic
"""
GT__ALIGNED_PARSED_DATASET__PYDANTIC__OP_CONFIG = {
    AssetKeyHelper.build_prefixed_key(
        AssetLayer.FCT,
        DataSource.HUGGINGFACE,
        "gt",
        "aligned_parsed_dataset",
        "pydantic",
    ): {"config": JSONAlignmentConfig().model_dump()}
}
