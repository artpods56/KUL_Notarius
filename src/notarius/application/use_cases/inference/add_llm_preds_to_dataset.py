"""Use case for predicting with LLM model."""

from dataclasses import dataclass
from typing import final, override

from structlog import get_logger

from notarius.application.ports.outbound.cached_engine import CachedEngine
from notarius.application.use_cases.base import BaseRequest, BaseResponse, BaseUseCase
from notarius.infrastructure.cache.backends.llm import create_llm_cache_backend
from notarius.infrastructure.llm.conversation import Conversation
from notarius.infrastructure.llm.engine_adapter import LLMEngine, CompletionRequest
from notarius.infrastructure.llm.prompt_manager import Jinja2PromptRenderer
from notarius.infrastructure.llm.utils import (
    construct_text_message,
    construct_image_message,
)
from notarius.schemas.data.pipeline import (
    BaseDataset,
    BaseDataItem,
    PredictionDataItem,
)
from notarius.orchestration.resources import ImageStorageResource
from notarius.domain.entities.schematism import SchematismPage

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
    cache_hits: int
    success_rate: float


@final
class PredictDatasetWithLLM(BaseUseCase[PredictWithLLMRequest, PredictWithLLMResponse]):
    """
    Use case for predicting schematism entities using LLM.

    This use case takes datasets with LMv3 predictions and OCR text, and uses
    an LLM to generate improved predictions. It can optionally use LMv3 predictions
    as hints to guide the LLM. Uses CachedEngine for automatic caching!
    """

    def __init__(
        self,
        llm_engine: LLMEngine,
        image_storage: ImageStorageResource,
        model_name: str,
        prompt_renderer: Jinja2PromptRenderer | None = None,
        enable_cache: bool = True,
    ):
        """Initialize the use case.

        Args:
            llm_engine: Engine for performing LLM predictions
            image_storage: Resource for loading images from storage
            model_name: Model name for cache namespacing
            prompt_renderer: Optional Jinja2 prompt renderer
            enable_cache: Whether to enable caching (default: True)
        """
        self.image_storage = image_storage
        self.prompt_renderer = prompt_renderer or Jinja2PromptRenderer()

        # Wrap engine with caching
        if enable_cache:
            backend, keygen = create_llm_cache_backend(model_name)
            self.llm_engine = CachedEngine(
                engine=llm_engine,
                cache_backend=backend,
                key_generator=keygen,
                enabled=True,
            )
        else:
            self.llm_engine = llm_engine

    @override
    async def execute(self, request: PredictWithLLMRequest) -> PredictWithLLMResponse:
        """
        Execute the LLM prediction workflow with automatic caching.

        Args:
            request: Request containing datasets and prediction parameters

        Returns:
            Response with predicted dataset and execution statistics
        """
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

            # Build context for prompts
            llm_context = {}
            if request.use_lmv3_hints and lmv3_item.predictions:
                hints = lmv3_item.predictions.model_dump()
                llm_context["hints"] = hints

            llm_context["ocr_text"] = ocr_item.text

            # Render prompts
            system_prompt = self.prompt_renderer.render_prompt(
                template_name=request.system_prompt, context=llm_context
            )
            user_prompt = self.prompt_renderer.render_prompt(
                template_name=request.user_prompt, context=llm_context
            )

            if not lmv3_item.image_path:
                continue

            image = self.image_storage.load_image(lmv3_item.image_path)

            # Build conversation
            system_message = construct_text_message(text=system_prompt, role="system")
            user_message = construct_image_message(
                pil_image=image, text=user_prompt, role="user"
            )
            messages = (system_message, user_message)
            conversation = Conversation(messages=messages)

            # Process with cached engine - caching happens automatically!
            llm_request = CompletionRequest(
                input=conversation,
                structured_output=SchematismPage,
            )
            response = self.llm_engine.process(llm_request)

            # Create prediction item
            produced_item = PredictionDataItem(
                image_path=lmv3_item.image_path,
                text=ocr_item.text,
                metadata=lmv3_item.metadata,
                predictions=response.output.response,
            )

            items.append(produced_item)

        # Get stats from cached engine
        if isinstance(self.llm_engine, CachedEngine):
            stats = self.llm_engine.stats
            llm_executions = stats["misses"]
            cache_hits = stats["hits"]
        else:
            llm_executions = len(items)
            cache_hits = 0

        success_rate = (
            len(items) / len(request.lmv3_dataset.items)
            if request.lmv3_dataset.items
            else 0.0
        )

        logger.info(
            "LLM prediction completed",
            total_items=len(request.lmv3_dataset.items),
            llm_executions=llm_executions,
            cache_hits=cache_hits,
            success_rate=success_rate,
            cache_enabled=isinstance(self.llm_engine, CachedEngine),
        )

        return PredictWithLLMResponse(
            dataset=BaseDataset[PredictionDataItem](items=items),
            llm_executions=llm_executions,
            cache_hits=cache_hits,
            success_rate=success_rate,
        )
