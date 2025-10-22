import dagster as dg
from dagster import AssetExecutionContext, MetadataValue
from omegaconf import OmegaConf

from orchestration.configs.shared import ConfigReference
from orchestration.constants import AssetLayer, Kinds, ResourceGroup
from orchestration.resources import ConfigManagerResource


@dg.asset(
    key_prefix=[AssetLayer.RES],
    group_name=ResourceGroup.CONFIG,
    kinds={Kinds.PYTHON, Kinds.YAML},
)
def ocr_model__config(
    context: AssetExecutionContext,
    config: ConfigReference,
    config_manager: ConfigManagerResource,
):

    config_reference = {
        "config_name": config.config_name,
        "config_type_name": config.config_type_name,
        "config_subtype_name": config.config_subtype_name,
    }

    config = config_manager.load_config_from_string(**config_reference)

    context.add_asset_metadata(
        {"config_reference": MetadataValue.json(config_reference)}
    )

    context.add_output_metadata(
        {"config": MetadataValue.json(OmegaConf.to_container(config, resolve=True))}
    )

    return config
