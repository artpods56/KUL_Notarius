from datetime import datetime
from pathlib import Path
from typing import Mapping

import dagster as dg
import pandas as pd
import wandb
from PIL import Image
from dagster import AssetExecutionContext, AssetIn
from rapidfuzz import fuzz

from notarius.orchestration.constants import (
    AssetLayer,
    DataSource,
    ResourceGroup,
    Kinds,
)
from notarius.orchestration.resources import (
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
    table_name: str = "eval_aligned_dataframe"
    group_by_key: str | None = None
    include_images: bool = True
    sample_id_column: str = "sample_id"


@dg.asset(
    key_prefix=[AssetLayer.MRT, DataSource.HUGGINGFACE],
    group_name=ResourceGroup.DATA,
    kinds={Kinds.PYTHON, Kinds.WANDB},
    ins={
        "dataframe": AssetIn(key="eval__aligned_parsed_dataframe__pandas"),
        "pydantic_dataset": AssetIn(key="base__dataset__pydantic"),
    },
)
def eval__wandb_export_dataframe__pandas(
    context: AssetExecutionContext,
    dataframe: pd.DataFrame,
    pydantic_dataset: BaseDataset[BaseDataItem],
    config: WandBDataFrameExport,
    wandb_run: WandBRunResource,
    image_storage: ImageStorageResource,
):
    """Export dataframe to Weights & Biases as a table."""

    def _build_sample_id2mapping(
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

    sample_id2image_mapping: dict[str, Image.Image] = _build_sample_id2mapping(
        pydantic_dataset
    )

    if config.group_by_key:
        # Log each group as a separate table
        for key, group in dataframe.groupby(config.group_by_key):
            if config.include_images:
                columns = ["image"] + list(group.columns)
                data = []

                for sample_id, grouped_samples in group.groupby(
                    config.sample_id_column
                ):
                    # Get image for this sample_id
                    pil_image = sample_id2image_mapping[sample_id]
                    wandb_image = wandb.Image(pil_image.resize((400, 600)))

                    # Add image only to the first row, None for the rest
                    for i, (idx, row) in enumerate(grouped_samples.iterrows()):
                        if i == 0:
                            data.append([wandb_image] + row.tolist())
                        else:
                            data.append([None] + row.tolist())
            else:
                columns = list(group.columns)
                data = group.values.tolist()

            table = wandb.Table(columns=columns, data=data)
            run.log({f"{config.table_name}_{key}": table})
            context.log.info(f"Logged table for group '{key}' to W&B")
    else:
        # Log the entire dataframe as one table
        if config.include_images:
            columns = ["image"] + list(dataframe.columns)
            data = []

            for sample_id, grouped_samples in dataframe.groupby(
                config.sample_id_column
            ):
                # Get image for this sample_id
                pil_image = sample_id2image_mapping[sample_id]
                wandb_image = wandb.Image(pil_image.resize((400, 600)))

                # Add image only to the first row, None for the rest
                for i, (idx, row) in enumerate(grouped_samples.iterrows()):
                    if i == 0:
                        data.append([wandb_image] + row.tolist())
                    else:
                        data.append([None] + row.tolist())
        else:
            columns = list(dataframe.columns)
            data = dataframe.values.tolist()

        table = wandb.Table(columns=columns, data=data)
        run.log({config.table_name: table})
        context.log.info(f"Logged table '{config.table_name}' to W&B")
