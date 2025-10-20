from dagster import ConfigMapping

from orchestration.assets.ingest import DatasetConfig, pdf_to_dataset
from orchestration.assets.transform import OpConfig





HUGGINGFACE_DATASET_INGESTION_OP_CONFIG = {
            "huggingface_dataset": {
                "config": DatasetConfig(
                    config_name="schematism_dataset_config",
                    config_type_name="dataset",
                    config_subtype_name="evaluation",
                ).model_dump()
            }
    }

HUGGINGFACE_DATASET_TRANSFORMATION_OP_CONFIG = {
            "filtered_huggingface_dataset": {
                "config": OpConfig(
                    op_type="filter",
                    op_name="filter_schematisms",
                    input_columns=["schematism_name"],
                    kwargs={
                        "to_filter": ["wloclawek_1872"]
                    }
                ).model_dump()
            }
    }