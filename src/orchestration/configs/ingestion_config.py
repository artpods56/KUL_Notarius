from orchestration.configs.shared import ConfigReference
from orchestration.constants import DataSource, AssetLayer
from orchestration.utils import AssetKeyHelper

"""
Asset: [[ingest.py#raw__hf__dataset]]
Defined in: src/orchestration/assets/ingest.py
Resolves to: int_huggingface__raw__hf__dataset
"""
# RAW_HF_DATASET_OP_CONFIG = {
#     AssetKeyHelper.build_prefixed_key(
#         AssetLayer.STG, DataSource.HUGGINGFACE, "raw", "hf", "lmv3_dataset"
#     ): {
#         "config": ConfigReference(
#             config_name="schematism_dataset_config",
#             config_type_name="lmv3_dataset",
#             config_subtype_name="evaluation",
#         ).model_dump()
#     }
# }
