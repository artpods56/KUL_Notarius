from orchestration.constants import DataSource, AssetLayer
from orchestration.utils import AssetKeyHelper

"""
Asset: [[ingest.py#raw__hf__dataset]]
Defined in: [[src/orchestration/assets/extract/ingest.py]]
Resolves to: stg__huggingface__raw__hf__dataset
"""
RAW_HF_DATASET_OP_CONFIG = {
    AssetKeyHelper.build_prefixed_key(
        AssetLayer.STG, DataSource.HUGGINGFACE, "raw", "hf", "dataset"
    ): {"config": {}}
}
