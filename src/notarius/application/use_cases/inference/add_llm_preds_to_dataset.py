"""Use case for predicting with LLM model."""

from dataclasses import dataclass
from typing import final, override, cast
import json

from openai.types.responses import ParsedResponse

from notarius.application.use_cases.base import BaseRequest, BaseResponse, BaseUseCase
from notarius.infrastructure.llm.conversation import Conversation
from notarius.infrastructure.llm.engine_adapter import LLMEngine, CompletionRequest
from notarius.infrastructure.llm.prompt_manager import Jinja2PromptRenderer
from notarius.infrastructure.llm.utils import (
    construct_text_message,
    construct_image_message,
)
from notarius.infrastructure.persistence.llm_cache_repository import LLMCacheRepository
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
    use_cache: bool = False


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
    as hints to guide the LLM.
    """

    def __init__(
        self,
        llm_engine: LLMEngine,
        image_storage: ImageStorageResource,
        cache_repository: LLMCacheRepository,
        prompt_renderer: Jinja2PromptRenderer | None = None,
    ):
        """Initialize the use case.

        Args:
            llm_engine: Engine for performing LLM predictions
            image_storage: Resource for loading images from storage
            cache_repository: Repository for caching LLM results
            prompt_renderer: Optional Jinja2 prompt renderer
        """
        self.llm_engine = llm_engine
        self.image_storage = image_storage
        self.cache_repository = cache_repository
        self.prompt_renderer = prompt_renderer or Jinja2PromptRenderer()

    def return_cached_or_none(self, cache_key: str) -> SchematismPage | None:
        """Return cached LLM prediction if available, None otherwise.

        Args:
            cache_key: Cache key for the request

        Returns:
            Cached SchematismPage if found, None otherwise
        """
        cached_item = self.cache_repository.get(cache_key)
        if cached_item:

            schema = cached_item.content.response

            return SchematismPage(**cached_item.content.response)
        else:
            return None

    def return_processed(self, conversation: Conversation) -> SchematismPage:
        """Perform LLM prediction and return SchematismPage.

        Args:
            conversation: Conversation with messages for LLM

        Returns:
            Predicted SchematismPage
        """
        llm_request = CompletionRequest(
            input=conversation,
            structured_output=SchematismPage,
        )
        response = self.llm_engine.process(llm_request)
        return response.output.response

    @override
    async def execute(self, request: PredictWithLLMRequest) -> PredictWithLLMResponse:
        """
        Execute the LLM prediction workflow with caching support.

        Args:
            request: Request containing datasets and prediction parameters

        Returns:
            Response with predicted dataset and execution statistics
        """
        llm_executions = 0
        cache_hits = 0
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
            hints = None
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

            # Serialize messages for cache key

            payload = {
                "user_prompt": user_prompt,
                "system_prompt": system_prompt,
            }

            messages_str = json.dumps(payload)

            # Generate cache key
            cache_key = self.cache_repository.generate_key(
                image=image,
                messages=messages_str,
                hints=hints,
            )

            # Try cache or process
            if request.use_cache:
                structured_response = self.return_cached_or_none(cache_key)
                if not structured_response:
                    logger.debug("Cache miss", key=cache_key[:16])
                    structured_response = self.return_processed(conversation)

                    # Cache the result
                    _ = self.cache_repository.set(
                        key=cache_key,
                        response=structured_response.model_dump(),
                        hints=hints,
                    )
                    llm_executions += 1
                else:
                    logger.debug("Cache hit", key=cache_key[:16])
                    cache_hits += 1
            else:
                structured_response = cast(
                    ParsedResponse[SchematismPage], self.return_processed(conversation)
                )
                llm_executions += 1

            # Create prediction item
            produced_item = PredictionDataItem(
                image_path=lmv3_item.image_path,
                text=ocr_item.text,
                metadata=lmv3_item.metadata,
                predictions=structured_response.output_parsed,
            )

            items.append(produced_item)

        success_rate = (
            llm_executions / len(request.lmv3_dataset.items)
            if request.lmv3_dataset.items
            else 0.0
        )

        logger.info(
            "LLM prediction completed",
            total_items=len(request.lmv3_dataset.items),
            llm_executions=llm_executions,
            cache_hits=cache_hits,
            success_rate=success_rate,
            cache_enabled=request.use_cache,
        )

        return PredictWithLLMResponse(
            dataset=BaseDataset[PredictionDataItem](items=items),
            llm_executions=llm_executions,
            cache_hits=cache_hits,
            success_rate=success_rate,
        )
