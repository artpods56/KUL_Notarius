import dagster as dg
from dagster import AssetIn, AssetExecutionContext

from orchestration.configs.shared import ConfigReference
from orchestration.constants import AssetLayer, ResourceGroup, DataSource, Kinds
from orchestration.resources import ConfigManagerResource
from schemas.data.pipeline import BaseDataset, GroundTruthDataItem
from tests.conftest import ocr_model_config


@dg.asset(
    key_prefix=[AssetLayer.STG, DataSource.HUGGINGFACE],
    group_name=ResourceGroup.DATA,
    kinds={Kinds.PYDANTIC},
    ins={"dataset": AssetIn(key="gt__dataset__pydantic")},
)
def ocr_enriched__dataset(
    context: AssetExecutionContext,
    dataset: BaseDataset[GroundTruthDataItem],
    config: ConfigReference,
    config_manager: ConfigManagerResource,
):

    ocr_model_config = config_manager.load_config_from_string(
        config_name=config.config_name,
        config_type_name=config.config_type_name,
        config_subtype_name=config.config_subtype_name,
    )
