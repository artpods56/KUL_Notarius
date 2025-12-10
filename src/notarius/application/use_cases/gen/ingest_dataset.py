"""Use cases for ingesting datasets from various sources."""

from dataclasses import dataclass
from typing import Any
from unittest.mock import Base

import pymupdf
from datasets import Dataset
from PIL import Image

from notarius.application.use_cases.base import BaseRequest, BaseResponse, BaseUseCase
from notarius.infrastructure.persistence import dataset_repository
from notarius.schemas.configs.dataset_config import BaseDatasetConfig
from notarius.schemas.data.pipeline import BaseDataItem, BaseDataset
from notarius.orchestration.resources import PdfFilesResource
from notarius.infrastructure.ml_models.lmv3.dataset_utils import get_dataset
from omegaconf import DictConfig
from structlog import get_logger

from notarius.shared.logger import Logger

logger: Logger = get_logger(__name__)


@dataclass
class IngestHuggingFaceRequest(BaseRequest):
    """Request to ingest HuggingFace dataset."""

    dataset_config: BaseDatasetConfig
    streaming: bool


@dataclass
class IngestHuggingFaceResponse(BaseResponse):
    """Response containing ingested HuggingFace dataset."""

    dataset: Dataset
    dataset_length: int


class IngestHuggingFaceDataset(
    BaseUseCase[IngestHuggingFaceRequest, IngestHuggingFaceResponse]
):
    """
    Use case for ingesting HuggingFace datasets.

    This use case loads a dataset from HuggingFace and adds sample IDs
    for tracking purposes.
    """

    async def execute(
        self, request: IngestHuggingFaceRequest
    ) -> IngestHuggingFaceResponse:
        """
        Execute the HuggingFace dataset ingestion workflow.

        Args:
            request: Request containing HuggingFace dataset configuration

        Returns:
            Response with loaded dataset
        """
        dataset = dataset_repository.load_huggingface_dataset(
            config=request.dataset_config, streaming=request.streaming
        )
        logger.info(
            "HuggingFace dataset ingestion completed", dataset_length=len(dataset)
        )

        return IngestHuggingFaceResponse(dataset=dataset, dataset_length=len(dataset))
