"""
Configs for assets defined in this file lives in [[transformation_config.py]]
"""

import random
from typing import Literal, Any, Mapping

import pandas as pd
import dagster as dg
from dagster import (
    AssetExecutionContext,
    MetadataValue,
    AssetIn,
    TableSchema,
)
from datasets import Dataset

from core.data.filters import negate_op
from orchestration.constants import DataSource, AssetLayer, ResourceGroup, Kinds
from orchestration.resources import OpRegistry

import structlog

from schemas.data.dataset import BaseHuggingFaceDatasetSchema, ETLSpecificDatasetFields
from schemas.data.pipeline import (
    GroundTruthDataItem,
    BaseDataset,
    BaseMetaData,
    BaseDataItem,
    GtAlignedPredictionDataItem,
)
from schemas.data.schematism import SchematismPage, SchematismEntry

logger = structlog.get_logger(__name__)


class PandasDataFrameConfig(dg.Config):
    pass


@dg.asset(
    key_prefix=[AssetLayer.MRT, DataSource.HUGGINGFACE],
    kinds={Kinds.PYTHON, Kinds.PANDAS},
    group_name=ResourceGroup.DATA,
    ins={"dataset": AssetIn("gt_aligned__dataset")},
)
def eval__aligned_dataframe__pandas(
    context: AssetExecutionContext,
    dataset: BaseDataset[GtAlignedPredictionDataItem],
    config: PandasDataFrameConfig,
):

    rows = []

    def flatten_aligned_pages(
        aligned_pages: tuple[SchematismPage, SchematismPage],
    ) -> list[dict[str, Any]]:

        flat_entries = []

        page_a, page_b = aligned_pages

        for entry_a, entry_b in zip(page_a.entries, page_b.entries):
            flattened_pages_dict = {}
            for key in SchematismEntry.model_fields.keys():

                val_a = getattr(entry_a, key)
                val_b = getattr(entry_b, key)

                flattened_pages_dict.update({f"{key}_a": val_a, f"{key}_b": val_b})
            flat_entries.append(flattened_pages_dict)

        return flat_entries

    for item in dataset.items:

        for flat_record in flatten_aligned_pages(item.aligned_schematism_pages):

            constructed_row = {
                "sample_id": item.metadata.sample_id,
                "filename": item.metadata.filename,
                "schematism_name": item.metadata.schematism_name,
                **flat_record,
            }

            rows.append(constructed_row)

    df = pd.DataFrame(rows)

    column_schema = TableSchema.from_name_type_dict(df.dtypes.astype(str).to_dict())

    return dg.MaterializeResult(
        value=df,
        metadata={
            "dagster/table_name": "table",
            "dagster/column_schema": column_schema,
            "dagster/row_count": len(df),
            "preview": MetadataValue.md(df.head(30).to_markdown()),
        },
    )


class OpConfig(dg.Config):
    op_type: Literal["map", "filter"]
    op_name: str
    input_columns: list[str]
    negate: bool = False
    kwargs: dict[str, Any] = {}


@dg.asset(
    key_prefix=[AssetLayer.INT, DataSource.HUGGINGFACE],
    group_name=ResourceGroup.DATA,
    kinds={Kinds.PYTHON, Kinds.HUGGINGFACE},
    ins={"dataset": AssetIn(key="raw__hf__dataset")},
)
def filtered__hf__dataset(
    context: AssetExecutionContext,
    dataset: Dataset,
    config: OpConfig,
    op_registry: OpRegistry,
):

    if config.op_type == "map" and config.negate:
        raise NotImplementedError("Negated map operations not implemented.")

    op_func = op_registry.get_op(op_type=config.op_type, name=config.op_name)

    if op_func is None:
        raise RuntimeError(f"No op registered for {config.op_type, config.op_name}")

    op = op_func(**config.kwargs)

    if config.op_type == "filter":
        filtered_dataset = dataset.filter(
            op if not config.negate else negate_op(op),
            input_columns=config.input_columns,
        )
    elif config.op_type == "map":
        filtered_dataset = dataset.map(op, input_columns=config.input_columns)
    else:
        raise RuntimeError(f"Unknown op_type {config.op_type}")

    logger.info(op_registry.list_operations())

    context.add_asset_metadata(
        {
            "op_type": MetadataValue.text(config.op_type),
            "op_name": MetadataValue.text(config.op_name),
            "input_columns": MetadataValue.json(config.input_columns),
            "negate": MetadataValue.bool(config.negate),
            "op_kwargs": MetadataValue.json(config.kwargs),
        }
    )

    no_image_dataset = filtered_dataset.remove_columns("image")

    context.add_output_metadata(
        {
            "num_rows": MetadataValue.int(len(filtered_dataset)),
            "num_columns": MetadataValue.int(len(filtered_dataset.column_names)),
            "column_names": MetadataValue.json(filtered_dataset.column_names),
            "random_sample": MetadataValue.json(
                {
                    k: v
                    for k, v in no_image_dataset[
                        random.randrange(0, len(filtered_dataset))
                    ].items()
                }
            ),
        }
    )

    return filtered_dataset


class DatasetMappingConfig(
    dg.Config, BaseHuggingFaceDatasetSchema, ETLSpecificDatasetFields
):
    """This config specifies the name of"""

    pass


def pydantic_dataset_factory[ModelT: BaseDataItem](
    asset_name: str, ins: Mapping[str, AssetIn], pydantic_model: type[ModelT]
):

    @dg.asset(
        name=asset_name,
        key_prefix=[AssetLayer.INT, DataSource.HUGGINGFACE],
        group_name=ResourceGroup.DATA,
        kinds={Kinds.PYTHON, Kinds.PYDANTIC},
        ins=ins,
    )
    def _pydantic_dataset_asset(
        context: AssetExecutionContext,
        dataset: Dataset,
        config: DatasetMappingConfig,
    ):
        """Convert HuggingFace dataset to Pydantic BaseDataset with BaseDataItem.

        This asset creates a base dataset containing only image and metadata,
        without ground truth. It serves as the starting point for prediction
        pipelines (OCR and LMv3 enrichment).

        Args:
            context: Dagster execution context for logging and metadata
            dataset: HuggingFace dataset to convert
            config: Configuration for field mappings

        Returns:
            BaseDataset containing BaseDataItem instances
        """

        items: list[BaseDataItem] = []

        for sample in dataset:

            item_structure = {
                "image": sample.get(config.image),
                "metadata": BaseMetaData(
                    sample_id=sample.get(config.sample_id),
                    schematism_name=sample.get(config.schematism_name),
                    filename=sample.get(config.filename),
                ),
            }

            if config.ground_truth_source is not None:
                item_structure["ground_truth"] = sample.get(config.ground_truth_source)

            item = pydantic_model(
                **item_structure,
            )

            items.append(item)

        random_sample = items[random.randint(0, len(items) - 1)]

        context.add_asset_metadata(
            {
                "config": MetadataValue.json(config.model_dump()),
            }
        )

        context.add_output_metadata(
            {
                "num_items": MetadataValue.int(len(items)),
                "random_sample": MetadataValue.json(
                    {
                        k: v
                        for k, v in random_sample.model_dump().items()
                        if k != "image"
                    }
                ),
            }
        )

        if pydantic_model is GroundTruthDataItem:
            return BaseDataset[GroundTruthDataItem](items=items)
        elif pydantic_model is BaseDataItem:
            return BaseDataset[BaseDataItem](items=items)
        else:
            raise RuntimeError(f"Unsupported pydantic model type: {pydantic_model}")

    return _pydantic_dataset_asset


base__dataset__pydantic = pydantic_dataset_factory(
    asset_name="base__dataset__pydantic",
    ins={"dataset": AssetIn(key="filtered__hf__dataset")},
    pydantic_model=BaseDataItem,
)

gt__source_dataset__pydantic = pydantic_dataset_factory(
    asset_name="gt__source_dataset__pydantic",
    ins={"dataset": AssetIn(key="filtered__hf__dataset")},
    pydantic_model=GroundTruthDataItem,
)

gt__parsed_dataset__pydantic = pydantic_dataset_factory(
    asset_name="gt__parsed_dataset__pydantic",
    ins={"dataset": AssetIn(key="filtered__hf__dataset")},
    pydantic_model=GroundTruthDataItem,
)
