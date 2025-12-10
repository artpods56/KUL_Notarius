from dataclasses import dataclass
from typing import final, override, cast

from PIL.Image import Image as PILImage
from structlog import get_logger

from notarius.application.use_cases.base import BaseRequest, BaseResponse, BaseUseCase
from notarius.infrastructure.ocr import OCREngine, OCRRequest, OCRMode
from notarius.infrastructure.ocr.engine_adapter import OCRResponse
from notarius.infrastructure.ocr.types import (
    SimpleOCRResult,
    StructuredOCRResult,
    OCRResult,
)
from notarius.infrastructure.persistence.ocr_cache_repository import OCRCacheRepository
from notarius.orchestration.resources import ImageStorageResource
from notarius.schemas.data.cache import PyTesseractCacheItem
from notarius.schemas.data.pipeline import BaseDataset, BaseDataItem

from notarius.shared.logger import Logger

logger: Logger = get_logger(__name__)


@dataclass
class EnrichWithOCRRequest(BaseRequest):
    """Request to enrich dataset with OCR predictions."""

    dataset: BaseDataset[BaseDataItem]
    mode: OCRMode = "text"
    overwrite: bool = False
    use_cache: bool = True


@dataclass
class EnrichWithOCRResponse(BaseResponse):
    """Response containing OCR-enriched dataset."""

    dataset: BaseDataset[BaseDataItem]
    ocr_executions: int
    cache_hits: int


@final
class EnrichDatasetWithOCR(BaseUseCase[EnrichWithOCRRequest, EnrichWithOCRResponse]):
    """
    Use case for enriching a dataset with OCR text predictions.

    This use case takes a dataset of items with images and enriches them with OCR text.
    It supports caching to avoid redundant OCR processing and can optionally overwrite existing text.
    """

    def __init__(
        self,
        ocr_engine: OCREngine,
        image_storage: ImageStorageResource,
        cache_repository: OCRCacheRepository,
    ):
        """
        Initialize the use case.

        Args:
            ocr_engine: Engine for performing OCR predictions
            image_storage: Resource for loading images from storage
            cache_repository: Repository for caching OCR results
        """
        self.ocr_engine = ocr_engine
        self.image_storage = image_storage
        self.cache_repository = cache_repository

    def return_cached_or_none(
        self, image: PILImage, cache_key: str
    ) -> OCRResponse | None:
        """Return cached OCR text if available, None otherwise.

        Args:
            image: PIL Image to lookup
            cache_key: Cache key for the image

        Returns:
            Cached text if found, None otherwise
        """
        cached_item = self.cache_repository.get(cache_key)
        if cached_item:
            return OCRResponse(output=SimpleOCRResult(text=cached_item.content.text))
        else:
            return None

    def return_processed(self, image: PILImage, mode: OCRMode) -> OCRResponse:
        """Perform OCR on image and return OCR result.

        Args:
            image: PIL Image to process
            mode: OCR mode (text or structured)

        Returns:
            OCR result SimpleOCRResult
        """
        ocr_request = OCRRequest(input=image, mode=mode)
        return self.ocr_engine.process(ocr_request)

    @override
    async def execute(self, request: EnrichWithOCRRequest) -> EnrichWithOCRResponse:
        """
        Execute the OCR enrichment workflow with caching support.

        Args:
            request: Request containing dataset and enrichment parameters

        Returns:
            Response with enriched dataset and execution statistics
        """
        ocr_executions = 0
        cache_hits = 0

        dataset_len = len(request.dataset.items)

        new_dataset_items: list[BaseDataItem] = []

        for i, item in enumerate(request.dataset.items):
            if not item.image_path:
                logger.debug(f"Skipping item {i}/{dataset_len} - no image_path")
                continue

            image = self.image_storage.load_image(item.image_path).convert("RGB")
            logger.info(f"Processing {i+1}/{dataset_len} sample with OCR.")

            cache_key = self.cache_repository.generate_key(image)

            if request.use_cache:
                response = self.return_cached_or_none(image, cache_key)
                if not response:
                    response = self.return_processed(image, request.mode)
                    if isinstance(response.output, SimpleOCRResult):
                        _ = self.cache_repository.set(
                            key=cache_key,
                            text=response.output.text,
                            language=self.ocr_engine.config.language,
                        )
                    ocr_executions += 1
                else:
                    cache_hits += 1
            else:
                response = self.return_processed(image, request.mode)
                ocr_executions += 1

            new_dataset_items.append(
                BaseDataItem(
                    image_path=item.image_path,
                    text=cast(SimpleOCRResult, response.output).text,
                    metadata=item.metadata,
                )
            )

        logger.info(
            "OCR enrichment completed",
            total_items=dataset_len,
            ocr_executions=ocr_executions,
            cache_hits=cache_hits,
            cache_enabled=request.use_cache,
        )

        return EnrichWithOCRResponse(
            dataset=BaseDataset[BaseDataItem](items=new_dataset_items),
            ocr_executions=ocr_executions,
            cache_hits=cache_hits,
        )
