"""Use case for enriching dataset with OCR predictions using LLM via OpenRouter."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Never, final, override

from notarius.application.ports.outbound.cached_engine import (
    CachedEngine,
    CachedEngineStats,
)
from notarius.application.ports.outbound.engine import EngineStats
from notarius.application.use_cases.base import BaseRequest, BaseResponse, BaseUseCase
from notarius.infrastructure.cache.backends.llm import create_llm_cache_backend
from notarius.infrastructure.llm.conversation import Conversation
from notarius.infrastructure.llm.engine_adapter import LLMEngine, CompletionRequest
from notarius.infrastructure.llm.prompt_manager import Jinja2PromptRenderer
from notarius.infrastructure.llm.utils import (
    construct_text_message,
    construct_image_message,
)
from notarius.orchestration.resources.base import ImageStorageResource
from notarius.schemas.data.pipeline import BaseDataset, BaseDataItem, BaseItemDataset
from notarius.shared.logger import get_logger

logger = get_logger(__name__)


@dataclass
class EnrichWithLLMOCRRequest(BaseRequest):
    """Request to enrich dataset with LLM-based OCR predictions."""

    dataset: BaseDataset[BaseDataItem]
    system_prompt: str = "tasks/ocr/system.j2"
    user_prompt: str = "tasks/ocr/user.j2"

    group_by_schematism_name: bool = True


@dataclass
class EnrichWithLLMOCRResponse(BaseResponse):
    """Response containing LLM OCR-enriched dataset."""

    dataset: BaseDataset[BaseDataItem]
    execution_stats: CachedEngineStats | EngineStats


@final
class EnrichDatasetWithLLMOCR(
    BaseUseCase[EnrichWithLLMOCRRequest, EnrichWithLLMOCRResponse]
):
    """
    Use case for enriching a dataset with OCR text using LLM vision capabilities.

    This use case takes a dataset of items with images and enriches them with OCR text
    extracted using an LLM (e.g., via OpenRouter). Uses CachedEngine for automatic caching.

    The LLM-based OCR provides high-fidelity text extraction with Markdown-accurate
    structural reconstruction, preserving layout, hierarchy, and emphasis.
    """

    def __init__(
        self,
        llm_engine: LLMEngine,
        image_storage: ImageStorageResource,
        model_name: str,
        prompt_renderer: Jinja2PromptRenderer | None = None,
        enable_cache: bool = True,
    ):
        """
        Initialize the use case.

        Args:
            llm_engine: Engine for performing LLM-based OCR predictions
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

    def _process_dataset_items(
        self,
        items: Sequence[BaseDataItem],
        system_prompt: str,
        user_prompt: str,
        schematism_name: str | None = "unknown",
    ) -> Sequence[BaseDataItem]:
        rendered_system_prompt = self.prompt_renderer.render_prompt(
            template_name=system_prompt, context={}
        )
        system_message = construct_text_message(
            text=rendered_system_prompt, role="system"
        )

        rendered_user_prompt = self.prompt_renderer.render_prompt(
            template_name=user_prompt, context={}
        )

        for i, item in enumerate(items):
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

            try:
                image = self.image_storage.load_image(item.image_path).convert("RGB")

                user_message = construct_image_message(
                    pil_image=image, text=rendered_user_prompt, role="user"
                )

                conversation = Conversation().add(system_message).add(user_message)

                llm_request = CompletionRequest[Never](
                    input=conversation,
                    structured_output=None,
                )
                result = self.llm_engine.process(llm_request)

                # Extract text from response
                extracted_text = result.output.text_response

                item.text = extracted_text

            except Exception as e:
                logger.error(
                    "LLM OCR extraction failed",
                    error=str(e),
                    image_path=item.image_path,
                    sample_index=i,
                )
        return items

    @override
    async def execute(
        self, request: EnrichWithLLMOCRRequest
    ) -> EnrichWithLLMOCRResponse:
        """
        Execute the LLM OCR enrichment workflow with automatic caching.

        Args:
            request: Request containing dataset and enrichment parameters

        Returns:
            Response with enriched dataset and execution statistics
        """

        if request.group_by_schematism_name:
            for schematism_name, dataset in request.dataset.group_by_schematism():
                dataset.items = self._process_dataset_items(
                    schematism_name=schematism_name,
                    items=dataset.items,
                    system_prompt=request.system_prompt,
                    user_prompt=request.user_prompt,
                )
        else:
            request.dataset.items = self._process_dataset_items(
                items=request.dataset.items,
                system_prompt=request.system_prompt,
                user_prompt=request.user_prompt,
            )

        return EnrichWithLLMOCRResponse(
            dataset=request.dataset, execution_stats=self.llm_engine.stats
        )

        for i, item in enumerate(request.dataset.items):
            if not item.image_path:
                logger.debug(f"Skipping item {i}/{dataset_len} - no image_path")
                new_dataset_items.append(item)
                continue

            logger.info(f"Processing {i + 1}/{dataset_len} sample with LLM OCR.")

            try:
                # Load and prepare image
                image = self.image_storage.load_image(item.image_path).convert("RGB")

                # Render user prompt
                user_prompt = self.prompt_renderer.render_prompt(
                    template_name=request.user_prompt, context={}
                )

                # Build user message with image
                user_message = construct_image_message(
                    pil_image=image, text=user_prompt, role="user"
                )

                # Create conversation with system and user messages
                conversation = Conversation().add(system_message).add(user_message)

                # Process with cached engine - caching happens automatically!
                # Note: No structured output since we want raw markdown text
                llm_request = CompletionRequest(
                    input=conversation,
                    structured_output=None,
                )
                result = self.llm_engine.process(llm_request)

                # Extract text from response
                extracted_text = result.output.text_response

                new_dataset_items.append(
                    BaseDataItem(
                        image_path=item.image_path,
                        text=extracted_text,
                        metadata=item.metadata,
                    )
                )
                successful_extractions += 1

            except Exception as e:
                logger.error(
                    "LLM OCR extraction failed",
                    error=str(e),
                    image_path=item.image_path,
                    sample_index=i,
                )
                # Keep original item on failure
                new_dataset_items.append(item)

        # Get stats from cached engine
        if isinstance(self.llm_engine, CachedEngine):
            stats = self.llm_engine.stats
            llm_executions = stats["misses"]
            cache_hits = stats["hits"]
        else:
            llm_executions = successful_extractions
            cache_hits = 0

        success_rate = successful_extractions / dataset_len if dataset_len > 0 else 0.0

        logger.info(
            "LLM OCR enrichment completed",
            total_items=dataset_len,
            successful_extractions=successful_extractions,
            llm_executions=llm_executions,
            cache_hits=cache_hits,
            success_rate=success_rate,
            cache_enabled=isinstance(self.llm_engine, CachedEngine),
        )

        return EnrichWithLLMOCRResponse(
            dataset=BaseItemDataset(items=new_dataset_items),
            llm_executions=llm_executions,
            cache_hits=cache_hits,
            success_rate=success_rate,
        )
