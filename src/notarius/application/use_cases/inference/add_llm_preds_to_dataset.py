"""Use case for predicting with LLM model."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import final, override, Callable, Any


from notarius.application.ports.outbound.cached_engine import CachedEngine
from notarius.application.ports.outbound.engine import CachedEngineStats, EngineStats
from notarius.application.use_cases.base import BaseRequest, BaseResponse, BaseUseCase
from notarius.domain.entities.messages import strip_images_from_message
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
    PredictionDataset,
)
from notarius.orchestration.resources.base import ImageStorageResource
from notarius.domain.entities.schematism import PageContext, SchematismPage
from notarius.shared.logger import get_logger

logger = get_logger(__name__)


@dataclass
class PredictWithLLMRequest(BaseRequest):
    """Request to predict with LLM model."""

    lmv3_dataset: BaseDataset[PredictionDataItem]
    ocr_dataset: BaseDataset[BaseDataItem]
    system_prompt: str = "system.j2"
    user_prompt: str = "user.j2"

    group_by_schematism_name: bool = True  # Group processing by schematism


@dataclass
class PredictWithLLMResponse(BaseResponse):
    """Response containing LLM predictions."""

    dataset: BaseDataset[PredictionDataItem]
    execution_stats: CachedEngineStats | EngineStats


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
        use_lmv3_hints: bool = True,
        accumulate_context: bool = False,
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
        self.use_lmv3_hints = use_lmv3_hints
        self.accumulate_context = accumulate_context

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

    def _prepare_context(
        self,
        item: PredictionDataItem,
        next_item: PredictionDataItem | None,
        previous_context: PageContext | None,
    ) -> dict[str, Any]:
        return {
            "ocr_text": item.text,
            "next_page_ocr_text": next_item.text if next_item else None,
            "hints": (
                item.predictions.model_dump()
                if item.predictions and self.use_lmv3_hints
                else {}
            ),
            "previous_context": (
                previous_context.model_dump() if previous_context else {}
            ),
        }

    def _process_dataset_items(
        self,
        items: Sequence[PredictionDataItem],
        system_prompt: str,
        user_prompt: str,
        schematism_name: str = "unknown",
    ) -> Sequence[PredictionDataItem]:
        rendered_system_prompt = self.prompt_renderer.render_prompt(
            template_name=system_prompt, context={}
        )
        system_message = construct_text_message(
            text=rendered_system_prompt, role="system"
        )

        conversation = Conversation().add(system_message)
        previous_context: PageContext | None = None
        for i, item in enumerate(items):
            next_item = items[i + 1] if i + 1 < len(items) else None
            if not item.image_path:
                logger.debug(
                    "Skipping sample",
                    sample_index=i,
                    items_num=len(items),
                    schematism_name=schematism_name,
                )
                continue

            logger.info(
                "Processing sample",
                sample_index=i,
                items_num=len(items),
                schematism_name=schematism_name,
            )

            llm_context = self._prepare_context(item, next_item, previous_context)

            try:
                image = self.image_storage.load_image(item.image_path).convert("RGB")

                rendered_user_prompt = self.prompt_renderer.render_prompt(
                    template_name=user_prompt, context=llm_context
                )

                user_message = construct_image_message(
                    pil_image=image, text=rendered_user_prompt, role="user"
                )

                request_conversation = conversation.add(user_message)

                llm_request = CompletionRequest[SchematismPage](
                    input=request_conversation,
                    structured_output=SchematismPage,
                )
                result = self.llm_engine.process(llm_request)

                structured_response = result.output.structured_response

                if isinstance(structured_response, SchematismPage):
                    item.predictions = structured_response

                    if structured_response.context and self.accumulate_context:
                        previous_context = structured_response.context
                        text_only_user_message = strip_images_from_message(user_message)
                        conversation = conversation.add(text_only_user_message)
                        conversation = conversation.add(result.output.to_message())

                else:
                    raise ValueError(
                        f"Unexpected structured response: {structured_response}"
                    )

            except Exception as e:
                logger.error(
                    "LLM OCR extraction failed",
                    error=str(e),
                    image_path=item.image_path,
                    sample_index=i,
                )
        return items

    def _merge_items(
        self,
        lmv3_items: Sequence[PredictionDataItem],
        ocr_items: Sequence[BaseDataItem],
    ) -> list[PredictionDataItem]:

        merger: Callable[[PredictionDataItem, BaseDataItem], PredictionDataItem] = (
            lambda lmv3_item, ocr_item: PredictionDataItem(
                image_path=lmv3_item.image_path,
                text=ocr_item.text,
                metadata=lmv3_item.metadata,
                predictions=lmv3_item.predictions,
            )
        )

        return list(map(merger, lmv3_items, ocr_items))

    @override
    async def execute(self, request: PredictWithLLMRequest) -> PredictWithLLMResponse:
        """
        Execute the LLM prediction workflow with automatic caching.

        Args:
            request: Request containing datasets and prediction parameters

        Returns:
            Response with predicted dataset and execution statistics
        """

        merged_items = self._merge_items(
            request.lmv3_dataset.items, request.ocr_dataset.items
        )

        merged_dataset = PredictionDataset(items=merged_items)

        all_items: list[PredictionDataItem] = []

        if request.group_by_schematism_name:
            for schematism_name, dataset in merged_dataset.group_by_schematism():
                items = self._process_dataset_items(
                    schematism_name=schematism_name,
                    items=dataset.items,
                    system_prompt=request.system_prompt,
                    user_prompt=request.user_prompt,
                )

                all_items.extend(items)

        else:
            items = self._process_dataset_items(
                items=merged_dataset.items,
                system_prompt=request.system_prompt,
                user_prompt=request.user_prompt,
            )

            all_items.extend(items)

        return PredictWithLLMResponse(
            dataset=PredictionDataset(items=all_items),
            execution_stats=self.llm_engine.stats,
        )
