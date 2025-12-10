"""Use case for predicting with LLM model."""

from dataclasses import dataclass

from notarius.application.use_cases.base import BaseRequest, BaseResponse, BaseUseCase
from notarius.application.ports.outbound.engine import ConfigurableEngine
from notarius.schemas.data.pipeline import (
    BaseDataset,
    BaseDataItem,
    PredictionDataItem,
)
from notarius.orchestration.resources import ImageStorageResource
from notarius.domain.entities.schematism import SchematismPage
from structlog import get_logger

logger = get_logger(__name__)


@dataclass
class PredictWithLLMRequest(BaseRequest):
    """Request to predict with LLM model."""

    lmv3_dataset: BaseDataset[PredictionDataItem]
    ocr_dataset: BaseDataset[BaseDataItem]
    system_prompt: str = "system.j2"
    user_prompt: str = "user.j2"
    use_lmv3_hints: bool = True


@dataclass
class PredictWithLLMResponse(BaseResponse):
    """Response containing LLM predictions."""

    dataset: BaseDataset[PredictionDataItem]
    llm_executions: int
    success_rate: float


class PredictDatasetWithLLM(BaseUseCase[PredictWithLLMRequest, PredictWithLLMResponse]):
    """
    Use case for predicting schematism entities using LLM.

    This use case takes datasets with LMv3 predictions and OCR text, and uses
    an LLM to generate improved predictions. It can optionally use LMv3 predictions
    as hints to guide the LLM.
    """

    def __init__(
        self,
        llm_engine: ConfigurableEngine,
        image_storage: ImageStorageResource,
    ):
        """Initialize the use case.

        Args:
            llm_engine: Engine for performing LLM predictions
            image_storage: Resource for loading images from storage
        """
        self.llm_engine = llm_engine
        self.image_storage = image_storage

    async def execute(self, request: PredictWithLLMRequest) -> PredictWithLLMResponse:
        """
        Execute the LLM prediction workflow.

        Args:
            request: Request containing datasets and prediction parameters

        Returns:
            Response with predicted dataset and execution statistics
        """
        llm_executions = 0
        items: list[PredictionDataItem] = []

        for lmv3_item, ocr_item in zip(
            request.lmv3_dataset.items, request.ocr_dataset.items
        ):
            if lmv3_item.image_path is None and lmv3_item.text is None:
                logger.warning(
                    "Skipping item without image_path or text",
                    sample_id=lmv3_item.metadata,
                )
                continue

            # Load image from storage if available
            image = (
                self.image_storage.load_image(lmv3_item.image_path)
                if lmv3_item.image_path
                else None
            )

            # Build context for LLM
            llm_context = {}
            if request.use_lmv3_hints and lmv3_item.predictions:
                llm_context["hints"] = lmv3_item.predictions.model_dump()

            try:
                # Generate predictions using LLM _engine (clean interface)
                response = self.llm_engine.predict(
                    image=image,
                    text=ocr_item.text,
                    context=llm_context,
                    system_template=request.system_prompt,
                    user_template=request.user_prompt,
                )
                llm_executions += 1

                # Convert output to SchematismPage
                predictions = SchematismPage(**response)

                # Create prediction item
                produced_item = PredictionDataItem(
                    image_path=lmv3_item.image_path,
                    text=ocr_item.text,
                    metadata=lmv3_item.metadata,
                    predictions=predictions,
                )

                items.append(produced_item)

            except Exception as e:
                logger.error(
                    "Failed to generate LLM prediction",
                    error=str(e),
                    sample_id=lmv3_item.metadata,
                )
                # Fall back to LMv3 predictions on error
                items.append(lmv3_item)

        success_rate = (
            llm_executions / len(request.lmv3_dataset.items)
            if request.lmv3_dataset.items
            else 0.0
        )

        logger.info(
            "LLM prediction completed",
            total_items=len(request.lmv3_dataset.items),
            llm_executions=llm_executions,
            success_rate=success_rate,
        )

        return PredictWithLLMResponse(
            dataset=BaseDataset[PredictionDataItem](items=items),
            llm_executions=llm_executions,
            success_rate=success_rate,
        )
