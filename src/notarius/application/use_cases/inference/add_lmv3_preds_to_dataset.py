from dataclasses import dataclass
from typing import final, override

from PIL.Image import Image
from structlog import get_logger

from notarius.application.use_cases.base import BaseRequest, BaseResponse, BaseUseCase
from notarius.infrastructure.ml_models.lmv3.engine_adapter import (
    LMv3Engine,
    LMv3Request,
    LMv3Response,
)
from notarius.infrastructure.persistence.lmv3_cache_repository import (
    LMv3CacheRepository,
)
from notarius.orchestration.resources import ImageStorageResource
from notarius.schemas.data.cache import LMv3CacheItem
from notarius.schemas.data.pipeline import BaseDataItem, BaseDataset, PredictionDataItem

from notarius.shared.logger import Logger

logger: Logger = get_logger(__name__)


@dataclass
class EnrichWithLMv3Request(BaseRequest):
    """Request to enrich dataset with LayoutLMv3 predictions."""

    dataset: BaseDataset[BaseDataItem]
    overwrite: bool = False
    use_cache: bool = True


@dataclass
class EnrichWithLMv3Response(BaseResponse):
    """Response containing LayoutLMv3-enriched dataset."""

    dataset: BaseDataset[PredictionDataItem]
    lmv3_executions: int
    cache_hits: int


@final
class EnrichDatasetWithLMv3(BaseUseCase[EnrichWithLMv3Request, EnrichWithLMv3Response]):
    """
    Use case for enriching a dataset with LayoutLMv3 predictions.

    This use case takes a dataset of items with images and enriches them with
    structured predictions from the LayoutLMv3 model. Supports caching to avoid
    redundant inference.
    """

    def __init__(
        self,
        lmv3_engine: LMv3Engine,
        image_storage: ImageStorageResource,
        cache_repository: LMv3CacheRepository,
    ):
        """
        Initialize the use case.

        Args:
            lmv3_engine: Engine for performing LayoutLMv3 predictions
            image_storage: Resource for loading images from storage
            cache_repository: Optional repository for caching LMv3 results
        """
        self.lmv3_engine = lmv3_engine
        self.image_storage = image_storage
        self.cache_repository = cache_repository

    def return_cached_or_none(
        self, image: Image, cache_key: str
    ) -> LMv3Response | None:

        cached_item = self.cache_repository.get(cache_key)
        if cached_item:
            return LMv3Response(output=cached_item.content.structured_predictions)
        else:
            return None

    def return_processed(self, image: Image) -> LMv3Response:
        lmv3_request = LMv3Request(input=image)
        response = self.lmv3_engine.process(lmv3_request)

        return LMv3Response(output=response.output)

    @override
    async def execute(self, request: EnrichWithLMv3Request) -> EnrichWithLMv3Response:
        """
        Execute the LayoutLMv3 enrichment workflow with caching support.

        Args:
            request: Request containing dataset and enrichment parameters

        Returns:
            Response with enriched dataset and execution statistics
        """
        lmv3_executions = 0
        cache_hits = 0

        dataset_len = len(request.dataset.items)
        new_dataset_items: list[PredictionDataItem] = []

        for i, item in enumerate(request.dataset.items):
            if not item.image_path:
                logger.debug(f"Skipping item {i}/{dataset_len} - no image_path")
                continue

            image = self.image_storage.load_image(item.image_path).convert("RGB")
            logger.info(f"Processing {i+1}/{dataset_len} sample with LMv3.")

            cache_key = self.cache_repository.generate_key(image)

            if request.use_cache:
                response = self.return_cached_or_none(image, cache_key)
                if not response:
                    response = self.return_processed(image)
                    _ = self.cache_repository.set(
                        key=cache_key, structured_predictions=response.output
                    )
                    lmv3_executions += 1
                else:
                    cache_hits += 1
            else:
                response = self.return_processed(image)
                lmv3_executions += 1

            new_dataset_items.append(
                PredictionDataItem(
                    image_path=item.image_path,
                    text=item.text,
                    predictions=response.output,
                    metadata=item.metadata,
                )
            )

        logger.info(
            "LayoutLMv3 enrichment completed",
            total_items=dataset_len,
            lmv3_executions=lmv3_executions,
            cache_hits=cache_hits,
            cache_enabled=request.use_cache,
        )

        return EnrichWithLMv3Response(
            dataset=BaseDataset(items=new_dataset_items),
            lmv3_executions=lmv3_executions,
            cache_hits=cache_hits,
        )
