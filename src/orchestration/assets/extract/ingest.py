"""
Configs for assets defined in this file lives in [[ingestion_config.py]]
"""

import random
from typing import Any, cast

from dagster import AssetExecutionContext, MetadataValue, AssetIn
from datasets import Dataset
from omegaconf import OmegaConf, DictConfig
from pymupdf import pymupdf

from core.data.utils import get_dataset
from orchestration.configs.shared import ConfigReference
from schemas.data.pipeline import BaseDataItem
import dagster as dg

from orchestration.resources import (
    PdfFilesResource,
    ConfigManagerResource,
    ImageStorageResource,
)
from orchestration.constants import DataSource, AssetLayer, ResourceGroup, Kinds

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


@dg.asset(
    key_prefix=[AssetLayer.STG, DataSource.HUGGINGFACE],
    group_name=ResourceGroup.DATA,
    kinds={Kinds.PYTHON, Kinds.HUGGINGFACE},
    ins={"dataset_config": AssetIn(key=[AssetLayer.RES, "hf_dataset__config"])},
)
def raw__hf__dataset(
    context: AssetExecutionContext, dataset_config: DictConfig
) -> Dataset:

    dataset = get_dataset(dataset_config)

    dataset = dataset.add_column("sample_id", range(len(dataset)))

    random_sample = cast(Dataset, dataset)[random.randint(0, len(dataset) - 1)]

    context.add_output_metadata(
        {
            "len": MetadataValue.int(len(dataset)),
            "random_sample": MetadataValue.json(
                {k: v for k, v in random_sample.items() if k != "image"}
            ),
        }
    )

    return dataset
