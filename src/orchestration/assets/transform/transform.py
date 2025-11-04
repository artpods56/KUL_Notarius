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

from orchestration.constants import DataSource, AssetLayer, ResourceGroup, Kinds

import structlog

from orchestration.resources import ImageStorageResource
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


def asset_factory__eval__aligned_dataframe_pandas(
    asset_name: str, ins: Mapping[str, AssetIn]
):
    @dg.asset(
        name=asset_name,
        key_prefix=[AssetLayer.MRT, DataSource.HUGGINGFACE],
        kinds={Kinds.PYTHON, Kinds.PANDAS},
        group_name=ResourceGroup.DATA,
        ins=ins,  # {"dataset": AssetIn("gt__aligned_dataset")},
    )
    def _asset__eval__aligned_dataframe__pandas(
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

    return _asset__eval__aligned_dataframe__pandas


eval__aligned_source_dataframe__pandas = asset_factory__eval__aligned_dataframe_pandas(
    asset_name="eval__aligned_source_dataframe__pandas",
    ins={"dataset": AssetIn("gt__aligned_source_dataset__pydantic")},
)
eval__aligned_parsed_dataframe__pandas = asset_factory__eval__aligned_dataframe_pandas(
    asset_name="eval__aligned_parsed_dataframe__pandas",
    ins={"dataset": AssetIn("gt__aligned_parsed_dataset__pydantic")},
)


class DatasetMappingConfig(
    dg.Config, BaseHuggingFaceDatasetSchema, ETLSpecificDatasetFields
):
    """This config specifies the name of"""

    pass


def asset_factory__pydantic_dataset[ModelT: BaseDataItem](
    asset_name: str, ins: Mapping[str, AssetIn], pydantic_model: type[ModelT]
):

    @dg.asset(
        name=asset_name,
        key_prefix=[AssetLayer.INT, DataSource.HUGGINGFACE],
        group_name=ResourceGroup.DATA,
        kinds={Kinds.PYTHON, Kinds.PYDANTIC},
        ins=ins,
    )
    def _asset__pydantic_dataset(
        context: AssetExecutionContext,
        dataset: Dataset,
        config: DatasetMappingConfig,
        image_storage: ImageStorageResource
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

            pil_image = sample.get(config.image)

            item_structure = {
                "image_path": image_storage.save_image(pil_image),
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
                "random_sample": MetadataValue.json(random_sample.model_dump()
                ),
            }
        )

        if pydantic_model is GroundTruthDataItem:
            return BaseDataset[GroundTruthDataItem](items=items)
        elif pydantic_model is BaseDataItem:
            return BaseDataset[BaseDataItem](items=items)
        else:
            raise RuntimeError(f"Unsupported pydantic model type: {pydantic_model}")

    return _asset__pydantic_dataset


base__dataset__pydantic = asset_factory__pydantic_dataset(
    asset_name="base__dataset__pydantic",
    ins={"dataset": AssetIn(key="filtered__hf__dataset")},
    pydantic_model=BaseDataItem,
)

gt__source_dataset__pydantic = asset_factory__pydantic_dataset(
    asset_name="gt__source_dataset__pydantic",
    ins={"dataset": AssetIn(key="filtered__hf__dataset")},
    pydantic_model=GroundTruthDataItem,
)

gt__parsed_dataset__pydantic = asset_factory__pydantic_dataset(
    asset_name="gt__parsed_dataset__pydantic",
    ins={"dataset": AssetIn(key="filtered__hf__dataset")},
    pydantic_model=GroundTruthDataItem,
)
