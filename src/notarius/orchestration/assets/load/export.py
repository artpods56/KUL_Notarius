from datetime import datetime
from pathlib import Path
from typing import Mapping

import dagster as dg
import pandas as pd
from PIL import Image
from dagster import AssetExecutionContext, AssetIn, MetadataValue
from rapidfuzz import fuzz

from notarius.application.use_cases.export import (
    ExportDataFrameToWandB,
    WandBExportRequest,
)
from notarius.application.use_cases.export.wandb_dataframe_export import (
    DataFrameExportConfig,
)
from notarius.orchestration.constants import (
    AssetLayer,
    DataSource,
    ResourceGroup,
    Kinds,
)
from notarius.orchestration.resources.base import (
    ExcelWriterResource,
    WandBRunResource,
    ImageStorageResource,
)
from notarius.schemas.data.pipeline import BaseDataset, BaseDataItem


class PandasDataFrameExport(dg.Config):
    file_name: str
    group_by_key: str = "schematism_name"
    include_index: bool = True
    include_header: bool = True
    apply_styling: bool = True
    fuzzy_threshold: int = 80


def asset_factory__eval__excel_export_dataframe__pandas(
    asset_name: str, ins: Mapping[str, AssetIn]
):
    @dg.asset(
        name=asset_name,
        key_prefix=[AssetLayer.MRT, DataSource.HUGGINGFACE],
        group_name=ResourceGroup.DATA,
        kinds={Kinds.PYTHON, Kinds.EXCEL},
        ins=ins,
    )
    def _asset__eval__excel_export_dataframe__pandas(
        context: AssetExecutionContext,
        dataframe: pd.DataFrame,
        config: PandasDataFrameExport,
        excel_writer: ExcelWriterResource,
    ):
        def get_fuzzy_score(val_a, val_b) -> float:
            """Calculate fuzzy match score between two values."""
            # Handle NaN/None values
            if pd.isna(val_a) and pd.isna(val_b):
                return 100.0  # Both empty is a match
            if pd.isna(val_a) or pd.isna(val_b):
                return 0.0  # One empty, one not is no match

            # Convert to strings
            str_a = str(val_a).strip()
            str_b = str(val_b).strip()

            if str_a == str_b:
                return 100.0

            return fuzz.ratio(str_a, str_b)

        def style_cell(row, col_name):
            """Style a cell based on fuzzy matching with its pair."""
            # Only style _a and _b columns
            if not (col_name.endswith("_a") or col_name.endswith("_b")):
                return ""

            # Find the pair column
            if col_name.endswith("_a"):
                pair_col = col_name[:-2] + "_b"
                val_a = row[col_name]
                val_b = row[pair_col] if pair_col in row.index else None
            else:  # ends with _b
                pair_col = col_name[:-2] + "_a"
                val_a = row[pair_col] if pair_col in row.index else None
                val_b = row[col_name]

            # If pair column doesn't exist, don't style
            if val_a is None or val_b is None:
                return ""

            # Calculate score
            score = get_fuzzy_score(val_a, val_b)

            # Return CSS style based on score
            if score == 100.0:
                return "background-color: #90EE90"  # Green
            elif score >= config.fuzzy_threshold:
                return "background-color: #FFFFE0"  # Yellow
            else:
                return "background-color: #FFB6C1"  # Red

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        full_file_name = Path(f"{timestamp}_{config.file_name}")

        with excel_writer.get_writer(full_file_name) as writer:
            for key, group in dataframe.groupby(config.group_by_key):
                if config.apply_styling:
                    # Apply styling using pandas Styler
                    styled = group.style.apply(
                        lambda row: [style_cell(row, col) for col in group.columns],
                        axis=1,
                    )
                    styled.to_excel(
                        writer,
                        sheet_name=key,
                        index=config.include_index,
                        engine="xlsxwriter",
                    )
                    context.log.info(f"Applied styling to sheet '{key}'")
                else:
                    # Write without styling
                    group.to_excel(
                        writer,
                        sheet_name=key,
                        index=config.include_index,
                        header=config.include_header,
                    )

    return _asset__eval__excel_export_dataframe__pandas


eval__excel_export_parsed_dataframe__pandas = (
    asset_factory__eval__excel_export_dataframe__pandas(
        asset_name="eval__excel_export_parsed_dataframe__pandas",
        ins={"dataframe": AssetIn(key="eval__aligned_parsed_dataframe__pandas")},
    )
)

eval__excel_export_source_dataframe__pandas = (
    asset_factory__eval__excel_export_dataframe__pandas(
        asset_name="eval__excel_export_source_dataframe__pandas",
        ins={"dataframe": AssetIn(key="eval__aligned_source_dataframe__pandas")},
    )
)


class WandBDataFrameExport(dg.Config):
    parsed_table_name: str = "eval_parsed_dataframe"
    source_table_name: str = "eval_source_dataframe"
    group_by_key: str | None = None
    include_images: bool = True
    sample_id_column: str = "sample_id"


@dg.asset(
    key_prefix=[AssetLayer.MRT, DataSource.HUGGINGFACE],
    group_name=ResourceGroup.DATA,
    kinds={Kinds.PYTHON, Kinds.WANDB},
    ins={
        "parsed_dataframe": AssetIn(key="eval__aligned_parsed_dataframe__pandas"),
        "source_dataframe": AssetIn(key="eval__aligned_source_dataframe__pandas"),
        "pydantic_dataset": AssetIn(key="base__dataset__pydantic"),
    },
)
async def eval__wandb_export_dataframe__pandas(
    context: AssetExecutionContext,
    parsed_dataframe: pd.DataFrame,
    source_dataframe: pd.DataFrame,
    pydantic_dataset: BaseDataset[BaseDataItem],
    config: WandBDataFrameExport,
    wandb_run: WandBRunResource,
    image_storage: ImageStorageResource,
):
    """Export parsed and source dataframes to Weights & Biases as tables."""

    def _build_sample_id_to_image(
        dataset: BaseDataset[BaseDataItem],
    ) -> dict[str, Image.Image]:
        mapping = {}
        for item in dataset.items:
            if item.image_path:
                mapping[item.metadata.sample_id] = image_storage.load_image(
                    item.image_path
                )
        return mapping

    run = wandb_run.get_wandb_run()

    # Build image mapping
    sample_id_to_image = _build_sample_id_to_image(pydantic_dataset)

    # Create use case
    use_case = ExportDataFrameToWandB(wandb_run=run)

    # Configure exports for both dataframes
    request = WandBExportRequest(
        exports=[
            DataFrameExportConfig(
                dataframe=parsed_dataframe,
                table_name=config.parsed_table_name,
                group_by_key=config.group_by_key,
                include_images=config.include_images,
                sample_id_column=config.sample_id_column,
            ),
            DataFrameExportConfig(
                dataframe=source_dataframe,
                table_name=config.source_table_name,
                group_by_key=config.group_by_key,
                include_images=config.include_images,
                sample_id_column=config.sample_id_column,
            ),
        ],
        sample_id_to_image=sample_id_to_image,
    )

    # Execute use case
    response = await use_case.execute(request)

    # Add metadata
    context.add_output_metadata(
        {
            "tables_logged": MetadataValue.json(response.tables_logged),
            "total_rows": MetadataValue.int(response.total_rows),
        }
    )

    context.log.info(
        f"Logged {len(response.tables_logged)} tables to W&B with {response.total_rows} total rows"
    )
