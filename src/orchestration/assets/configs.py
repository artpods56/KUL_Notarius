from typing import Sequence
import dagster as dg
from dagster import AssetExecutionContext, MetadataValue
from omegaconf import OmegaConf

from orchestration.configs.shared import ConfigReference
from orchestration.constants import AssetLayer, Kinds, ResourceGroup
from orchestration.resources import ConfigManagerResource


def asset_factory__config(asset_name: str, key_prefix: Sequence[str] = None):
    """Factory function to create config loading assets."""

    @dg.asset(
        name=asset_name,
        key_prefix=key_prefix or [AssetLayer.RES],
        group_name=ResourceGroup.CONFIG,
        kinds={Kinds.PYTHON, Kinds.YAML},
    )
    def _asset__config(
        context: AssetExecutionContext,
        config: ConfigReference,
        config_manager: ConfigManagerResource,
    ):
        config_reference = {
            "config_name": config.config_name,
            "config_type_name": config.config_type_name,
            "config_subtype_name": config.config_subtype_name,
        }

        loaded_config = config_manager.load_config_from_string(**config_reference)

        context.add_asset_metadata(
            {"config_reference": MetadataValue.json(config_reference)}
        )

        context.add_output_metadata(
            {
                "config": MetadataValue.json(
                    OmegaConf.to_container(loaded_config, resolve=True)
                )
            }
        )

        return loaded_config

    return _asset__config


# --- dataset configs ---
hf_dataset__config = asset_factory__config("hf_dataset__config")

# --- model configs ---
ocr_model__config = asset_factory__config("ocr_model__config")
lmv3_model__config = asset_factory__config("lmv3_model__config")
llm_model__config = asset_factory__config("llm_model__config")
