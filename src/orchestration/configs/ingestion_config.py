from orchestration.assets.ingest import DatasetConfig

HUGGINGFACE_DATASET_INGESTION_OP_CONFIG = {
            "huggingface_dataset": {
                "config": DatasetConfig(
                    config_name="schematism_dataset_config",
                    config_type_name="dataset",
                    config_subtype_name="evaluation",
                ).model_dump()
            }
    }

