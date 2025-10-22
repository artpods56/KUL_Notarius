import dagster as dg
from dagster import AssetIn, AssetExecutionContext, MetadataValue
from omegaconf import DictConfig, OmegaConf

from core.models.ocr.model import OcrModel
from orchestration.configs.shared import BaseModelConfig
from orchestration.constants import AssetLayer, ResourceGroup, Kinds


class OcrModelConfig(BaseModelConfig):
    pass


@dg.asset(
    key_prefix=[
        AssetLayer.RES,
    ],
    group_name=ResourceGroup.MODEL,
    kinds={Kinds.PYTHON},
    ins={"model_config": AssetIn(key=[AssetLayer.RES, "ocr_model__config"])},
)
def ocr_model(
    context: AssetExecutionContext, config: OcrModelConfig, model_config: DictConfig
):

    ocr_model = OcrModel(config=model_config, enable_cache=config.enable_cache)

    context.add_asset_metadata(
        {
            "config": MetadataValue.json(
                OmegaConf.to_container(model_config, resolve=True)
            ),
            "enable_cache": MetadataValue.bool(config.enable_cache),
        }
    )

    return ocr_model
