from typing import cast

from notarius.infrastructure.config.constants import (
    ConfigSubTypes,
    ConfigType,
    DatasetConfigSubtype,
)
from notarius.orchestration.assets.extract.ingest import AssetBaseDatasetConfig
from notarius.infrastructure.config.manager import config_manager
from notarius.orchestration.constants import DataSource, AssetLayer
from notarius.orchestration.utils import AssetKeyHelper

"""
Asset: [[ingest.py#raw__hf__dataset]]
Defined in: [[src/orchestration/assets/extract/ingest.py]]
Resolves to: stg__huggingface__raw__hf__dataset
"""
RAW_HF_DATASET_OP_CONFIG = {
    AssetKeyHelper.build_prefixed_key(
        AssetLayer.STG, DataSource.HUGGINGFACE, "raw", "hf", "dataset"
    ): {
        "config": config_manager.load_config_as_model(
            config_name="base_huggingface_config",
            config_type=ConfigType.DATASET,
            config_subtype=DatasetConfigSubtype.DEFAULT,
        ).model_dump()
    }
}
