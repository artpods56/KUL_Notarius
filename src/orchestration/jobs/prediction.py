import dagster as dg

from orchestration.assets.configs import (
    ocr_model__config,
    lmv3_model__config,
    llm_model__config,
)
from orchestration.assets.models import ocr_model, lmv3_model, llm_model, parser
from orchestration.assets.transform.predict import (
    pred__ocr_enriched_dataset__pydantic,
    pred__lmv3_enriched_dataset__pydantic,
    pred__llm_enriched_dataset__pydantic,
)
from orchestration.configs.prediction_config import (
    PRED__OCR_ENRICHED_DATASET__PYDANTIC__OP_CONFIG,
    PRED__LMV3_ENRICHED_DATASET__PYDANTIC__OP_CONFIG,
    PRED__LLM_ENRICHED_DATASET__PYDANTIC__OP_CONFIG,
)
from orchestration.configs.references_config import (
    RES_OCR_MODEL_CONFIG_OP_CONFIG,
    RES_LLM_MODEL_CONFIG_OP_CONFIG,
    RES_LMV3_MODEL_CONFIG_OP_CONFIG,
)

models_config_assets = [
    ocr_model__config,
    lmv3_model__config,
    llm_model__config,
]
models_assets = [ocr_model, lmv3_model, llm_model, parser]
prediction_assets = [
    pred__ocr_enriched_dataset__pydantic,
    pred__lmv3_enriched_dataset__pydantic,
    pred__llm_enriched_dataset__pydantic,
]
prediction_job = dg.define_asset_job(
    name="prediction_pipeline",
    selection=dg.AssetSelection.assets(
        *models_config_assets, *models_assets, *prediction_assets
    ),
    config={
        "ops": {
            # configs refs
            **RES_OCR_MODEL_CONFIG_OP_CONFIG,
            **RES_LLM_MODEL_CONFIG_OP_CONFIG,
            **RES_LMV3_MODEL_CONFIG_OP_CONFIG,
            # models refs
            # asset refs
            **PRED__OCR_ENRICHED_DATASET__PYDANTIC__OP_CONFIG,
            **PRED__LMV3_ENRICHED_DATASET__PYDANTIC__OP_CONFIG,
            **PRED__LLM_ENRICHED_DATASET__PYDANTIC__OP_CONFIG,
        }
    },
)
