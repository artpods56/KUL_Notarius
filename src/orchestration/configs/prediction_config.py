from orchestration.assets.transform.predict import OcrConfig, LMv3Config, LLMConfig
from orchestration.constants import DataSource, AssetLayer
from orchestration.utils import AssetKeyHelper

"""
Asset: [[predict.py#ocr_enriched__dataset]]
Defined in: src/orchestration/assets/predict.py
Resolves to: stg__huggingface__pred__ocr_enriched_dataset_pydantic
"""
PRED__OCR_ENRICHED_DATASET__PYDANTIC__OP_CONFIG = {
    AssetKeyHelper.build_prefixed_key(
        AssetLayer.STG,
        DataSource.HUGGINGFACE,
        "pred",
        "ocr_enriched_dataset",
        "pydantic",
    ): {
        "config": OcrConfig(
            text_only=True,
            overwrite=False,
        ).model_dump()
    }
}

"""
Asset: [[predict.py#pred__lmv3_enriched_dataset__pydantic]]
Defined in: src/orchestration/assets/predict.py
Resolves to: stg__huggingface__pred__lmv3_enriched_dataset__pydantic
"""
PRED__LMV3_ENRICHED_DATASET__PYDANTIC__OP_CONFIG = {
    AssetKeyHelper.build_prefixed_key(
        AssetLayer.STG,
        DataSource.HUGGINGFACE,
        "pred",
        "lmv3_enriched_dataset",
        "pydantic",
    ): {"config": LMv3Config(raw_predictions=False).model_dump()}
}

"""
Asset: [[predict.py#pred__llm_enriched_dataset__pydantic]]
Defined in: src/orchestration/assets/predict.py
Resolves to: stg__huggingface__pred__llm_enriched_dataset__pydantic
"""
PRED__LLM_ENRICHED_DATASET__PYDANTIC__OP_CONFIG = {
    AssetKeyHelper.build_prefixed_key(
        AssetLayer.STG,
        DataSource.HUGGINGFACE,
        "pred",
        "llm_enriched_dataset",
        "pydantic",
    ): {
        "config": LLMConfig(
            system_prompt="system.j2",
            user_prompt="user.j2",
            use_lmv3_hints=True,
        ).model_dump()
    }
}
