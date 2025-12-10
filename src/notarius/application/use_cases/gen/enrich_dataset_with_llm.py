"""Use case for enriching entire dataset with LLM extractions.

This is the orchestration-level use case that processes multiple items in batch.
"""

from dataclasses import dataclass

from structlog import get_logger

from notarius.application.use_cases.base import BaseRequest, BaseResponse, BaseUseCase
from notarius.application.use_cases.gen.extract_schematism_page import (
    ExtractSchematismPage,
    ExtractSchematismPageRequest,
)
from notarius.schemas.configs.llm_model_config import PromptTemplates
from notarius.schemas.data.pipeline import (
    BaseDataset,
    BaseDataItem,
    PredictionDataItem,
)
from notarius.orchestration.resources import ImageStorageResource

logger = get_logger(__name__)


@dataclass(frozen=True)
class EnrichDatasetWithLLMRequest(BaseRequest):
    """Request to enrich dataset with LLM extractions."""

    lmv3_dataset: BaseDataset[PredictionDataItem]
    ocr_dataset: BaseDataset[BaseDataItem]
    use_lmv3_hints: bool = True
    templates: PromptTemplates | None = None
    invalidate_cache: bool = False


@dataclass(frozen=True)
class EnrichDatasetWithLLMResponse(BaseResponse):
    """Response containing enriched dataset."""

    dataset: BaseDataset[PredictionDataItem]
    llm_executions: int
    cached_executions: int
    failed_executions: int
    success_rate: float
    average_confidence: float


class EnrichDatasetWithLLM(
    BaseUseCase[EnrichDatasetWithLLMRequest, EnrichDatasetWithLLMResponse]
):
    """Use case for enriching an entire dataset with LLM extractions.

    This is a higher-level orchestration use case that:
    - Iterates over dataset items
    - Delegates single-item extraction to ExtractSchematismPage
    - Aggregates results and statistics
    - Handles errors gracefully

    Separation of concerns:
    - ExtractSchematismPage: Single item extraction logic
    - EnrichDatasetWithLLM: Dataset-level orchestration
    """

    def __init__(
        self,
        extract_page_use_case: ExtractSchematismPage,
        image_storage: ImageStorageResource,
    ):
        """Initialize the use case.

        Args:
            extract_page_use_case: Use case for single-page extraction
            image_storage: Resource for loading images
        """
        self.extract_page = extract_page_use_case
        self.image_storage = image_storage

    async def execute(
        self, request: EnrichDatasetWithLLMRequest
    ) -> EnrichDatasetWithLLMResponse:
        """Execute dataset enrichment.

        Args:
            request: Enrichment request with datasets and configuration

        Returns:
            Response with enriched dataset and execution statistics
        """
        enriched_items: list[PredictionDataItem] = []
        llm_executions = 0
        cached_executions = 0
        failed_executions = 0
        total_confidence = 0.0

        # Process each item in parallel-ready fashion
        for lmv3_item, ocr_item in zip(
            request.lmv3_dataset.items, request.ocr_dataset.items
        ):
            try:
                enriched_item = await self._process_single_item(
                    lmv3_item=lmv3_item,
                    ocr_item=ocr_item,
                    use_lmv3_hints=request.use_lmv3_hints,
                    templates=request.templates,
                    invalidate_cache=request.invalidate_cache,
                )

                enriched_items.append(enriched_item.item)

                # Track statistics
                if enriched_item.from_cache:
                    cached_executions += 1
                else:
                    llm_executions += 1

                total_confidence += enriched_item.confidence

            except Exception as e:
                logger.error(
                    "Failed to process item, using fallback",
                    error=str(e),
                    sample_id=lmv3_item.metadata.get("sample_id"),
                )
                failed_executions += 1
                enriched_items.append(lmv3_item)  # Fallback to LMv3 predictions

        # Calculate statistics
        total_items = len(request.lmv3_dataset.items)
        successful_items = llm_executions + cached_executions
        success_rate = successful_items / total_items if total_items > 0 else 0.0
        average_confidence = (
            total_confidence / successful_items if successful_items > 0 else 0.0
        )

        logger.info(
            "Dataset enrichment completed",
            total_items=total_items,
            llm_executions=llm_executions,
            cached_executions=cached_executions,
            failed_executions=failed_executions,
            success_rate=success_rate,
            average_confidence=average_confidence,
        )

        return EnrichDatasetWithLLMResponse(
            dataset=BaseDataset[PredictionDataItem](items=enriched_items),
            llm_executions=llm_executions,
            cached_executions=cached_executions,
            failed_executions=failed_executions,
            success_rate=success_rate,
            average_confidence=average_confidence,
        )

    async def _process_single_item(
        self,
        lmv3_item: PredictionDataItem,
        ocr_item: BaseDataItem,
        use_lmv3_hints: bool,
        templates: PromptTemplates | None,
        invalidate_cache: bool,
    ) -> "_EnrichedItem":
        """Process a single dataset item.

        Args:
            lmv3_item: Item with LMv3 predictions
            ocr_item: Item with OCR text
            use_lmv3_hints: Whether to use LMv3 predictions as hints
            templates: Prompt templates to use
            invalidate_cache: Whether to invalidate cache

        Returns:
            Enriched item with metadata
        """
        # Load image if available
        image = None
        if lmv3_item.image_path:
            image = self.image_storage.load_image(lmv3_item.image_path)

        # Build hints from LMv3 if enabled
        hints = None
        if use_lmv3_hints and lmv3_item.predictions:
            hints = lmv3_item.predictions.model_dump()

        # Build metadata for logging/caching
        metadata = {
            "sample_id": lmv3_item.metadata.get("sample_id"),
            "source": lmv3_item.metadata.get("source"),
        }

        # Create extraction request
        extraction_request = ExtractSchematismPageRequest(
            image=image,
            text=ocr_item.text,
            hints=hints,
            templates=templates,
            invalidate_cache=invalidate_cache,
            metadata=metadata,
        )

        # Execute extraction
        extraction_response = await self.extract_page.execute(extraction_request)

        # Build enriched item
        enriched_item = PredictionDataItem(
            image_path=lmv3_item.image_path,
            text=ocr_item.text,
            metadata=lmv3_item.metadata,
            predictions=extraction_response.result.page,
        )

        return _EnrichedItem(
            item=enriched_item,
            confidence=extraction_response.result.confidence,
            from_cache=extraction_response.result.from_cache,
        )


@dataclass(frozen=True)
class _EnrichedItem:
    """Internal type for tracking enriched items with metadata."""

    item: PredictionDataItem
    confidence: float
    from_cache: bool
