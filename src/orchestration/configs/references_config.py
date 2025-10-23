from orchestration.configs.shared import ConfigReference
from orchestration.constants import DataSource, AssetLayer
from orchestration.utils import AssetKeyHelper


####################
# Datasets Configs #
####################

"""
Asset: [[configs.py#hf_dataset__config]]
Defined in: src/orchestration/assets/configs.py
Resolves to: res__hf_dataset__config
"""
RES_HF_DATASET_CONFIG_OP_CONFIG = {
    AssetKeyHelper.build_key(AssetLayer.RES, "hf_dataset", "config"): {
        "config": ConfigReference(
            config_name="schematism_dataset_config",
            config_type_name="lmv3_dataset",
            config_subtype_name="evaluation",
        ).model_dump()
    }
}


##################
# Models Configs #
##################

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


"""
Asset: [[configs.py#lmv3_model__config]]
Defined in: src/orchestration/assets/configs.py
Resolves to: res__lmv3_model__config
"""
RES_LMV3_MODEL_CONFIG_OP_CONFIG = {
    AssetKeyHelper.build_key(AssetLayer.RES, "lmv3_model", "config"): {
        "config": ConfigReference(
            config_name="lmv3_model_config",
            config_type_name="models",
            config_subtype_name="lmv3",
        ).model_dump()
    }
}


"""
Asset: [[configs.py#llm_model__config]]
Defined in: src/orchestration/assets/configs.py
Resolves to: res__llm_model__config
"""
RES_LLM_MODEL_CONFIG_OP_CONFIG = {
    AssetKeyHelper.build_key(AssetLayer.RES, "llm_model", "config"): {
        "config": ConfigReference(
            config_name="llm_model_config",
            config_type_name="models",
            config_subtype_name="llm",
        ).model_dump()
    }
}
