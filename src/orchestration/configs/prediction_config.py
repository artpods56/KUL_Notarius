from orchestration.assets.predict import OcrConfig, LMv3Config, LLMConfig
from orchestration.configs.shared import ConfigReference
from orchestration.constants import DataSource, AssetLayer
from orchestration.utils import AssetKeyHelper

"""
Asset: [[predict.py#ocr_enriched__dataset]]
Defined in: src/orchestration/assets/predict.py
Resolves to: stg__huggingface__ocr_enriched__dataset
"""
OCR_ENRICHED_DATASET_OP_CONFIG = {
    AssetKeyHelper.build_prefixed_key(
        AssetLayer.STG, DataSource.HUGGINGFACE, "ocr_enriched", "dataset"
    ): {
        "config": OcrConfig(
            text_only=True,
            overwrite=False,
        ).model_dump()
    }
}

"""
Asset: [[predict.py#lmv3_enriched__dataset]]
Defined in: src/orchestration/assets/predict.py
Resolves to: stg__huggingface__lmv3_enriched_dataset
"""
LMV3_ENRICHED_DATASET_OP_CONFIG = {
    AssetKeyHelper.build_prefixed_key(
        AssetLayer.STG, DataSource.HUGGINGFACE, "lmv3_enriched", "dataset"
    ): {"config": LMv3Config(raw_predictions=False).model_dump()}
}

"""
Asset: [[predict.py#llm_enriched__dataset]]
Defined in: src/orchestration/assets/predict.py
Resolves to: stg__huggingface__llm_enriched__dataset
"""
LLM_ENRICHED_DATASET_OP_CONFIG = {
    AssetKeyHelper.build_prefixed_key(
        AssetLayer.STG, DataSource.HUGGINGFACE, "llm_enriched", "dataset"
    ): {
        "config": LLMConfig(
            system_prompt="system.j2",
            user_prompt="user.j2",
            use_lmv3_hints=True,
        ).model_dump()
    }
}
