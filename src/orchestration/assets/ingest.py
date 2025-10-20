import random
from typing import Any

from dagster import AssetExecutionContext, MetadataValue
from datasets import Dataset
from omegaconf import OmegaConf
from pymupdf import pymupdf

from core.config.constants import (
    ConfigType,
    ConfigSubTypes,
    DatasetConfigSubtype,
    ConfigTypeMapping,
)
from core.data.utils import get_dataset
from core.schemas.data.pipeline import GroundTruthDataItem, BaseDataItem
import dagster as dg

from orchestration.resources import PdfFilesResource, ConfigManagerResource


from structlog import get_logger

logger = get_logger(__name__)


class PdfToDatasetConfig(dg.Config):
    page_range: list[int] | None = None
    modes: list[str] = ["image", "text"]


@dg.asset()
def pdf_to_dataset(config: PdfToDatasetConfig, pdf_files: PdfFilesResource) -> list[BaseDataItem]:

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

                elif "image" in modes:

                    pix = page.get_pixmap()
                    image = pix.pil_image()

                    item_data["image"] = image.convert("L").convert("RGB")
                else:
                    raise ValueError(f"Provided modes: '{modes}' are not supported")

                items.append(BaseDataItem(**item_data))

    return items


class DatasetConfig(dg.Config):
    config_name: str
    config_type_name: str
    config_subtype_name: str


@dg.asset(group_name="data", compute_kind="python")
def huggingface_dataset(context: AssetExecutionContext, config: DatasetConfig, config_manager: ConfigManagerResource) -> Dataset:

    type_enum = ConfigType(config.config_type_name)
    subtype = ConfigTypeMapping.get_subtype_enum(type_enum)
    subtype_enum = subtype(config.config_subtype_name)

    dataset_config = config_manager.load_config(
        config_name=config.config_name,
        config_type=type_enum,
        config_subtype=subtype_enum,
    )

    dataset = get_dataset(dataset_config)

    context.add_asset_metadata({
        "config_name": MetadataValue.text(config.config_name),
        "config_type_name": MetadataValue.text(config.config_type_name),
        "config_subtype_name": MetadataValue.text(config.config_subtype_name),
        "dataset_config": MetadataValue.json(OmegaConf.to_container(dataset_config, resolve=True)),
    })


    dataset = dataset.remove_columns(dataset_config.column_map.get("image_column", "image"))

    context.add_output_metadata({
        "len": MetadataValue.int(len(dataset)),
        "random_sample": MetadataValue.json(dataset[random.randrange(0, len(dataset))]),
    })

    return dataset



