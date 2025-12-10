"""
Configs for assets defined in this file lives in [[ingestion_config.py]]
"""

import random
from typing import TYPE_CHECKING, Any, cast

from dagster import AssetExecutionContext, MetadataValue, AssetIn
from datasets import Dataset, IterableDataset
from datasets.fingerprint import generate_fingerprint
from numpy import isin
from omegaconf import DictConfig
from pymupdf import pymupdf

from notarius.infrastructure.persistence import dataset_repository
from notarius.orchestration.utils import (
    dagster_config_from_pydantic,
    make_dagster_config,
)
from notarius.schemas.configs.dataset_config import BaseDatasetConfig
from notarius.schemas.data.pipeline import BaseDataItem
import dagster as dg

from notarius.orchestration.resources import (
    PdfFilesResource,
)
from notarius.orchestration.constants import (
    DataSource,
    AssetLayer,
    ResourceGroup,
    Kinds,
)

from structlog import get_logger

logger = get_logger(__name__)


class PdfToDatasetConfig(dg.Config):
    page_range: list[int] | None = None
    modes: list[str] = ["image", "text"]


@dg.asset(
    key_prefix=[AssetLayer.STG, DataSource.FILE],
    group_name=ResourceGroup.DATA,
    kinds={Kinds.PYTHON},
)
def raw__pdf__dataset(
    config: PdfToDatasetConfig, pdf_files: PdfFilesResource
) -> list[BaseDataItem]:
    page_range = config.page_range
    modes = config.modes

    items: list[BaseDataItem] = []

    pdf_paths = pdf_files.get_pdf_paths()

    for pdf_path in pdf_paths:
        with pymupdf.Document(pdf_path) as pdf:
            if page_range:
                pages_range = pdf.pages(*page_range)
            else:
                pages_range = pdf.pages()

            for page in pages_range:
                item_data: dict[str, Any] = {}

                if "text" in modes:
                    text = page.get_text()
                    item_data["text"] = text

                if "image" in modes:
                    pix = page.get_pixmap()
                    image = pix.pil_image()

                    item_data["image"] = image.convert("L").convert("RGB")
                else:
                    raise ValueError(f"Provided modes: '{modes}' are not supported")

                items.append(BaseDataItem(**item_data))

    return items


# class RawHuggingFaceDatasetConfig(dg.Config):
#     streaming: bool = False
#
#
# @dg.asset(
#     key_prefix=[AssetLayer.STG, DataSource.HUGGINGFACE],
#     group_name=ResourceGroup.DATA,
#     kinds={Kinds.PYTHON, Kinds.HUGGINGFACE},
#     ins={"dataset_config": AssetIn(key=[AssetLayer.RES, "hf_dataset__config"])},
# )
# def raw__hf__dataset(
#     context: AssetExecutionContext,
#     dataset_config: BaseDatasetConfig,
#     config: RawHuggingFaceDatasetConfig,
# ):
#     dataset = dataset_repository.load_huggingface_dataset(
#         config=dataset_config,
#         streaming=config.streaming,
#     )
#     return dataset


if TYPE_CHECKING:
    AssetBaseDatasetConfig = BaseDatasetConfig
else:
    AssetBaseDatasetConfig = make_dagster_config(BaseDatasetConfig)


@dg.asset(
    key_prefix=[AssetLayer.STG, DataSource.HUGGINGFACE],
    group_name=ResourceGroup.DATA,
    kinds={Kinds.PYTHON, Kinds.HUGGINGFACE},
)
def raw__hf__dataset(context: AssetExecutionContext, config: AssetBaseDatasetConfig):
    dataset = dataset_repository.load_huggingface_dataset(
        config=config,
        streaming=config.streaming,
    )

    if isinstance(dataset, IterableDataset):
        raise NotImplementedError("The pipeline doesn't support streamed datasets.")

    x = dataset

    dataset = dataset.add_column(
        name="sample_id",
        column=list(range(len(dataset))),
        new_fingerprint=generate_fingerprint(dataset),
    )

    return dataset
