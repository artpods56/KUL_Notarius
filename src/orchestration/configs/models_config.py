from orchestration.assets.models import OcrModelConfig
from orchestration.configs.shared import ConfigReference
from orchestration.constants import DataSource, AssetLayer
from orchestration.utils import AssetKeyHelper

"""
Asset: [[models.py#ocr_model__model]]
Defined in: src/orchestration/assets/configs.py
Resolves to: res__ocr_model
"""
RES_OCR_MODEL_OP_CONFIG = {
    AssetKeyHelper.build_key(AssetLayer.RES, "ocr_model"): {
        "config": OcrModelConfig(
            enable_cache=True,
        ).model_dump()
    }
}
