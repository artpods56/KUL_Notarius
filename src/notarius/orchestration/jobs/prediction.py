import dagster as dg

from notarius.orchestration.assets.transform.predict import (
    pred__ocr_enriched_dataset__pydantic,
    pred__lmv3_enriched_dataset__pydantic,
    pred__llm_enriched_dataset__pydantic,
    pred__llm_ocr_enriched_dataset__pydantic,
)
from notarius.orchestration.configs.prediction_config import (
    PRED__OCR_ENRICHED_DATASET__PYDANTIC__OP_CONFIG,
    PRED__LMV3_ENRICHED_DATASET__PYDANTIC__OP_CONFIG,
    PRED__LLM_ENRICHED_DATASET__PYDANTIC__OP_CONFIG,
    PRED__LLM_OCR_ENRICHED_DATASET__PYDANTIC__OP_CONFIG,
)


models_config_assets = []
models_assets = []
prediction_assets = [
    pred__ocr_enriched_dataset__pydantic,
    pred__lmv3_enriched_dataset__pydantic,
    pred__llm_enriched_dataset__pydantic,
    pred__llm_ocr_enriched_dataset__pydantic,
]
prediction_job = dg.define_asset_job(
    name="prediction_pipeline",
    selection=dg.AssetSelection.assets(*models_config_assets, *prediction_assets),
    config={
        "ops": {
            # configs refs
            # ml_models refs
            # asset refs
            **PRED__OCR_ENRICHED_DATASET__PYDANTIC__OP_CONFIG,
            **PRED__LMV3_ENRICHED_DATASET__PYDANTIC__OP_CONFIG,
            **PRED__LLM_ENRICHED_DATASET__PYDANTIC__OP_CONFIG,
            **PRED__LLM_OCR_ENRICHED_DATASET__PYDANTIC__OP_CONFIG,
        }
    },
)
