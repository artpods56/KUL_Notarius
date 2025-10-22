from orchestration.configs.shared import ConfigReference
from orchestration.constants import DataSource, AssetLayer
from orchestration.utils import AssetKeyHelper

"""
Asset: [[configs.py#ocr_model__config]]
Defined in: src/orchestration/assets/configs.py
Resolves to: res__ocr_model__config
"""
RES_OCR_MODEL_CONFIG_OP_CONFIG = {
    AssetKeyHelper.build_key(AssetLayer.RES, "ocr_model", "config"): {
        "config": ConfigReference(
            config_name="ocr_model_config",
            config_type_name="models",
            config_subtype_name="ocr",
        ).model_dump()
    }
}
