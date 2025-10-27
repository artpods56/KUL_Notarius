from datetime import datetime
from pathlib import Path

import dagster as dg
import pandas as pd
import wandb
from PIL import Image
from dagster import AssetExecutionContext, AssetIn

from orchestration.constants import AssetLayer, DataSource, ResourceGroup, Kinds
from orchestration.resources import ExcelWriterResource, WandBRunResource
from schemas.data.pipeline import BaseDataset, BaseDataItem


class PandasDataFrameExport(dg.Config):
    file_name: str
    group_by_key: str = "schematism_name"
    include_index: bool = True
    include_header: bool = True


@dg.asset(
    key_prefix=[AssetLayer.MRT, DataSource.HUGGINGFACE],
    group_name=ResourceGroup.DATA,
    kinds={Kinds.PYTHON, Kinds.EXCEL},
    ins={"dataframe": AssetIn(key="eval__aligned_dataframe__pandas")},
)
def eval__export_dataframe__pandas(
    context: AssetExecutionContext,
    dataframe: pd.DataFrame,
    config: PandasDataFrameExport,
    excel_writer: ExcelWriterResource,
):

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    full_file_name = Path(f"{timestamp}_{config.file_name}")

    with excel_writer.get_writer(full_file_name) as writer:
        for key, group in dataframe.groupby(config.group_by_key):
            group.to_excel(
                writer,
                sheet_name=key,
                index=config.include_index,
                header=config.include_header,
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
        "dataframe": AssetIn(key="eval__aligned_dataframe__pandas"),
        "pydantic_dataset": AssetIn(key="base__dataset__pydantic"),
    },
)
def eval__wandb_export_dataframe__pandas(
    context: AssetExecutionContext,
    dataframe: pd.DataFrame,
    pydantic_dataset: BaseDataset[BaseDataItem],
    config: WandBDataFrameExport,
    wandb_run: WandBRunResource,
):
    """Export dataframe to Weights & Biases as a table."""

    def _build_sample_id2mapping(
        dataset: BaseDataset[BaseDataItem],
    ) -> dict[str, Image.Image]:
        mapping = {}
        for item in dataset.items:
            mapping[item.metadata.sample_id] = item.image

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
