from typing import Any

import dagster as dg
from dagster import AssetIn, Out, AssetExecutionContext, MetadataValue
from omegaconf import DictConfig, OmegaConf

from core.models.base import ConfigurableModel
from core.models.llm.model import LLMModel
from core.models.lmv3.model import LMv3Model
from core.models.ocr.model import OcrModel
from orchestration.configs.shared import BaseModelConfig
from orchestration.constants import AssetLayer, ResourceGroup, Kinds


class OcrModelConfig(BaseModelConfig):
    pass


# @dg.asset(
#     key_prefix=[
#         AssetLayer.RES,
#     ],
#     group_name=ResourceGroup.MODEL,
#     kinds={Kinds.PYTHON},
#     ins={"model_config": AssetIn(key=[AssetLayer.RES, "lmv3_model__config"])},
# )
# def lmv3_model(
#     context: AssetExecutionContext, config: OcrModelConfig, model_config: DictConfig
# ):
#
#     lmv3_model = OcrModel(config=model_config, enable_cache=config.enable_cache)
#
#     context.add_asset_metadata(
#         {
#             "config": MetadataValue.json(
#                 OmegaConf.to_container(model_config, resolve=True)
#             ),
#             "enable_cache": MetadataValue.bool(config.enable_cache),
#         }
#     )
#
#     return lmv3_model
#

# class OcrModelConfig(BaseModelConfig):
#     pass
#
#
# @dg.asset(
#     key_prefix=[
#         AssetLayer.RES,
#     ],
#     group_name=ResourceGroup.MODEL,
#     kinds={Kinds.PYTHON},
#     ins={"model_config": AssetIn(key=[AssetLayer.RES, "ocr_model__config"])},
# )
# def ocr_model(
#     context: AssetExecutionContext, config: OcrModelConfig, model_config: DictConfig
# ):
#
#     ocr_model = OcrModel(config=model_config, enable_cache=config.enable_cache)
#
#     context.add_asset_metadata(
#         {
#             "config": MetadataValue.json(
#                 OmegaConf.to_container(model_config, resolve=True)
#             ),
#             "enable_cache": MetadataValue.bool(config.enable_cache),
#         }
#     )
#
#     return ocr_model


def model_factory[ModelT: ConfigurableModel](
    asset_name: str,
    model_config: str,
    model_class: type[ModelT],
    extra_kwargs: dict[str, Any] | None = None,
    key_prefix: list[str] | None = None,
):

    @dg.asset(
        name=asset_name,
        key_prefix=key_prefix or [AssetLayer.RES],
        group_name=ResourceGroup.MODEL,
        kinds={Kinds.PYTHON},
        ins={"model_config": AssetIn(key=[AssetLayer.RES, model_config])},
        io_manager_key="mem_io_manager",
    )
    def _model_asset(context: AssetExecutionContext, model_config: DictConfig):

        model = model_class(config=model_config, **extra_kwargs)

        context.add_asset_metadata(
            {
                "config": MetadataValue.json(
                    OmegaConf.to_container(model_config, resolve=True)
                ),
                "kwargs": MetadataValue.json(extra_kwargs),
            }
        )

        return model

    return _model_asset


ocr_model = model_factory(
    asset_name="ocr_model",
    model_config="ocr_model__config",
    model_class=OcrModel,
    extra_kwargs={"enable_cache": True},
)

lmv3_model = model_factory(
    asset_name="lmv3_model",
    model_config="lmv3_model__config",
    model_class=LMv3Model,
    extra_kwargs={"enable_cache": True},
)

llm_model = model_factory(
    asset_name="llm_model",
    model_config="llm_model__config",
    model_class=LLMModel,
    extra_kwargs={"enable_cache": True},
)


@dg.asset(
    key_prefix=[AssetLayer.RES],
    group_name=ResourceGroup.MODEL,
    kinds={Kinds.PYTHON},
)
def parser(context: AssetExecutionContext):
    """Create a parser instance for translating/normalizing schematism entries.

    This asset creates a Parser instance that can perform fuzzy matching
    and translation of dedications, building materials, and deaneries.
    """
    from core.data.translation_parser import Parser

    parser_instance = Parser()

    context.add_asset_metadata(
        {
            "fuzzy_threshold": MetadataValue.int(parser_instance.fuzzy_threshold),
            "mapping_keys": MetadataValue.json(list(parser_instance.mappings.keys())),
        }
    )

    return parser_instance
