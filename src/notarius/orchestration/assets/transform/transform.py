"""
Configs for assets defined in this file lives in [[transformation_config.py]]
"""

import random
from typing import Any, Mapping, cast, Iterable, Literal, Sequence

from numpy.random import f
import pandas as pd
import dagster as dg
from dagster import (
    AssetExecutionContext,
    MetadataValue,
    AssetIn,
    TableSchema,
)
from datasets import Dataset
from pydantic import Field

from notarius.infrastructure.persistence.storage import ImageRepository
from notarius.orchestration.constants import (
    DataSource,
    AssetLayer,
    ResourceGroup,
    Kinds,
)

import structlog

from notarius.orchestration.resources.base import ImageStorageResource
from notarius.schemas.data.dataset import (
    SchematismsDatasetItem,
)
from notarius.schemas.data.pipeline import (
    GroundTruthDataItem,
    BaseDataset,
    BaseMetaData,
    BaseDataItem,
    GtAlignedPredictionDataItem,
    # Concrete subclasses for pickle compatibility
    BaseItemDataset,
    GroundTruthDataset,
)
from notarius.domain.entities.schematism import SchematismEntry, SchematismPage

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

            # Handle empty pages - emit a single row with all None values
            if not page_a.entries and not page_b.entries:
                empty_row = {}
                for key in SchematismEntry.model_fields.keys():
                    empty_row[f"{key}_a"] = None
                    empty_row[f"{key}_b"] = None
                return [empty_row]

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


class BaseDatasetConfig(dg.Config):
    pass


def asset_factory__base_dataset(
    asset_name: str,
    ins: Mapping[str, AssetIn],
):
    """Factory for creating base dataset assets (without ground truth)."""

    @dg.asset(
        name=asset_name,
        key_prefix=[AssetLayer.INT, DataSource.HUGGINGFACE],
        group_name=ResourceGroup.DATA,
        kinds={Kinds.PYTHON, Kinds.PYDANTIC},
        ins=ins,
    )
    def _asset__base_dataset(
        context: AssetExecutionContext,
        hf_dataset: Dataset,
        config: BaseDatasetConfig,
        images_repository: dg.ResourceParam[ImageRepository],
        pdf_dataset: BaseItemDataset | None = None,
    ) -> BaseItemDataset:
        """Convert HuggingFace dataset to Pydantic BaseDataset with BaseDataItem.

        This asset creates a base dataset containing only image and metadata,
        without ground truth. It serves as the starting point for prediction
        pipelines (OCR and LMv3 enrichment).

        Args:
            context: Dagster execution context for logging and metadata
            hf_dataset: HuggingFace dataset to convert
            config: Configuration for the asset

        Returns:
            BaseDataset containing BaseDataItem instances
        """

        items: list[BaseDataItem] = []

        for i, sample in enumerate(cast(Iterable[SchematismsDatasetItem], hf_dataset)):
            image_name = f"{sample['schematism_name']}_{sample['filename']}"

            # Skip writing if image already exists on disk
            if images_repository.exists(image_name):
                image_path = images_repository.get_path(image_name)
            else:
                image_path = images_repository.add(sample["image"], image_name)

            metadata = BaseMetaData(
                sample_id=sample.get("sample_id", i),
                schematism_name=sample["schematism_name"],
                filename=sample["filename"],
            )
            items.append(BaseDataItem(image_path=str(image_path), metadata=metadata))

        if pdf_dataset:
            items.extend(pdf_dataset.items)

        combined_dataset = BaseItemDataset(items=items)

        context.add_output_metadata(
            {
                "all_items": MetadataValue.int(len(items)),
                "loaded_schematisms": MetadataValue.json(
                    {
                        schematism_name: len(dataset.items)
                        for schematism_name, dataset in combined_dataset.group_by_schematism()
                    }
                ),
                "random_sample": MetadataValue.json(random.choice(items).model_dump()),
            }
        )
        return combined_dataset

    return _asset__base_dataset


class GroundTruthDatasetConfig(dg.Config):
    ground_truth_source: str = Field(
        description="Source field name for ground truth data in the HuggingFace dataset"
    )


def asset_factory__ground_truth_dataset(
    asset_name: str,
    ins: Mapping[str, AssetIn],
):
    """Factory for creating ground truth dataset assets."""

    @dg.asset(
        name=asset_name,
        key_prefix=[AssetLayer.INT, DataSource.HUGGINGFACE],
        group_name=ResourceGroup.DATA,
        kinds={Kinds.PYTHON, Kinds.PYDANTIC},
        ins=ins,
    )
    def _asset__ground_truth_dataset(
        context: AssetExecutionContext,
        hf_dataset: Dataset,
        config: GroundTruthDatasetConfig,
        images_repository: dg.ResourceParam[ImageRepository],
    ):
        """Convert HuggingFace dataset to Pydantic GroundTruthDataset.

        This asset creates a dataset containing image, metadata, and ground truth.
        It is used for evaluation and alignment pipelines.

        Args:
            context: Dagster execution context for logging and metadata
            hf_dataset: HuggingFace dataset to convert
            config: Configuration specifying the ground truth source field

        Returns:
            GroundTruthDataset containing GroundTruthDataItem instances
        """

        items: list[GroundTruthDataItem] = []

        for i, sample in enumerate(cast(Iterable[SchematismsDatasetItem], hf_dataset)):
            image_name = f"{sample['schematism_name']}_{sample['filename']}"

            if images_repository.exists(image_name):
                image_path = images_repository.get_path(image_name)
            else:
                image_path = images_repository.add(sample["image"], image_name)

            metadata = BaseMetaData(
                sample_id=sample.get("sample_id", i),
                schematism_name=sample["schematism_name"],
                filename=sample["filename"],
            )

            ground_truth: SchematismPage | None = sample.get(
                config.ground_truth_source, None
            )

            if ground_truth is None:
                raise ValueError(
                    f"Ground truth field '{config.ground_truth_source}' not found in sample: {sample}"
                )

            items.append(
                GroundTruthDataItem(
                    ground_truth=ground_truth,
                    image_path=str(image_path),
                    metadata=metadata,
                )
            )

        dataset = GroundTruthDataset(items=items)

        context.add_output_metadata(
            {
                "all_items": MetadataValue.int(len(items)),
                "loaded_schematisms": MetadataValue.json(
                    {
                        schematism_name: len(ds.items)
                        for schematism_name, ds in dataset.group_by_schematism()
                    }
                ),
                "random_sample": MetadataValue.json(random.choice(items).model_dump()),
            }
        )
        return dataset

    return _asset__ground_truth_dataset


base__dataset__pydantic = asset_factory__base_dataset(
    asset_name="base__dataset__pydantic",
    ins={
        "hf_dataset": AssetIn(key="preprocessed__hf__dataset"),
        "pdf_dataset": AssetIn(key="raw__pdf__dataset"),
    },
)

gt__source_dataset__pydantic = asset_factory__ground_truth_dataset(
    asset_name="gt__source_dataset__pydantic",
    ins={"hf_dataset": AssetIn(key="preprocessed__hf__dataset")},
)

gt__parsed_dataset__pydantic = asset_factory__ground_truth_dataset(
    asset_name="gt__parsed_dataset__pydantic",
    ins={"hf_dataset": AssetIn(key="preprocessed__hf__dataset")},
)
