from orchestration.assets.transform import OpConfig, DatasetMappingConfig

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

HUGGINGFACE_DATASET_CONVERSION_CONFIG = {
    "hf_ground_truth_dataset": {
        "config": DatasetMappingConfig().model_dump()}
}