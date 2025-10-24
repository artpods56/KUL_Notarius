from datetime import datetime
from pathlib import Path

import dagster as dg
import pandas as pd
from dagster import AssetExecutionContext, AssetIn

from orchestration.constants import AssetLayer, DataSource, ResourceGroup, Kinds
from orchestration.resources import ExcelWriterResource


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
