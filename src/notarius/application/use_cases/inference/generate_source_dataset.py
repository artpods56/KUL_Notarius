"""Use case for generating source (Latin) dataset from parsed (Polish) ground truth."""

from dataclasses import dataclass
from typing import final, override, cast

from structlog import get_logger, BoundLogger

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
from notarius.domain.entities.messages import strip_images_from_message
from notarius.schemas.data.pipeline import (
    BaseDataset,
    BaseDataItem,
    PredictionDataItem,
    GroundTruthDataItem,
    PredictionDataset,
)
from notarius.orchestration.resources.base import ImageStorageResource
from notarius.domain.entities.schematism import SchematismPage

logger = cast(BoundLogger, get_logger(__name__))


@dataclass
class GenerateSourceDatasetRequest(BaseRequest):
    """Request to generate source (Latin) dataset from parsed (Polish) ground truth."""

    # Dataset with parsed (Polish) ground truth entries - the "shopping list"
    parsed_ground_truth_dataset: BaseDataset[GroundTruthDataItem]
    # Dataset with OCR text for each page
    ocr_dataset: BaseDataset[BaseDataItem]
    # Dataset with images (can be same as ocr_dataset or separate)
    image_dataset: BaseDataset[BaseDataItem]
    # Prompt templates
    system_prompt: str = "tasks/source_generation/system.j2"
    user_prompt: str = "tasks/source_generation/user.j2"
    # Whether to accumulate context across pages (for deanery inheritance)
    accumulate_context: bool = True


@dataclass
class GenerateSourceDatasetResponse(BaseResponse):
    """Response containing generated source (Latin) dataset."""

    dataset: BaseDataset[PredictionDataItem]
    llm_executions: int
    cache_hits: int
    success_rate: float


@final
class GenerateSourceDataset(
    BaseUseCase[GenerateSourceDatasetRequest, GenerateSourceDatasetResponse]
):
    """
    Use case for generating source (Latin) dataset from parsed (Polish) ground truth.

    This use case takes parsed ground truth entries (Polish, normalized) and uses an LLM
    to find and extract the corresponding Latin text from page images. The result is a
    source dataset that can be used for evaluation of the extraction pipeline.

    The workflow:
    1. For each page, provide the LLM with:
       - The page image
       - Parsed ground truth entries (Polish) as guidance
       - OCR text (optional, for hints)
       - Previous page context (for deanery inheritance)
    2. LLM locates the Latin text on the page corresponding to each parsed entry
    3. Output is source entries in Latin, ordered by page visual order
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
            llm_engine: Engine for performing LLM completions
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
            self.llm_engine: LLMEngine | CachedEngine = CachedEngine(
                engine=llm_engine,
                cache_backend=backend,
                key_generator=keygen,
                enabled=True,
            )
        else:
            self.llm_engine = llm_engine

    @override
    async def execute(
        self, request: GenerateSourceDatasetRequest
    ) -> GenerateSourceDatasetResponse:
        """
        Execute the source generation workflow.

        Args:
            request: Request containing datasets and generation parameters

        Returns:
            Response with generated source dataset and execution statistics
        """
        items: list[PredictionDataItem] = []

        # Render system prompt
        system_prompt = self.prompt_renderer.render_prompt(
            template_name=request.system_prompt, context={}
        )
        system_message = construct_text_message(text=system_prompt, role="system")

        # Initialize conversation with system message
        conversation = Conversation().add(system_message)

        # Track context from previous pages (for deanery inheritance)
        previous_context: dict | None = None

        # Build lookups by sample_id for efficient access
        ocr_by_sample_id: dict[str, BaseDataItem] = {}
        for ocr_item in request.ocr_dataset.items:
            if ocr_item.metadata:
                ocr_by_sample_id[ocr_item.metadata.sample_id] = ocr_item

        image_by_sample_id: dict[str, BaseDataItem] = {}
        for image_item in request.image_dataset.items:
            if image_item.metadata:
                image_by_sample_id[image_item.metadata.sample_id] = image_item

        i = 0
        dataset_len = len(request.parsed_ground_truth_dataset.items)

        for gt_item in request.parsed_ground_truth_dataset.items:
            if not gt_item.metadata:
                logger.warning("Skipping item without metadata")
                continue

            sample_id = gt_item.metadata.sample_id
            logger.info(
                f"Processing {i + 1}/{dataset_len} sample for source generation.",
                sample_id=sample_id,
            )

            # Get corresponding OCR and image data
            ocr_item = ocr_by_sample_id.get(sample_id)
            image_item = image_by_sample_id.get(sample_id)

            if not image_item or not image_item.image_path:
                logger.warning(
                    "Skipping item without image",
                    sample_id=sample_id,
                )
                continue

            if not gt_item.ground_truth:
                logger.warning(
                    "Skipping item without ground truth",
                    sample_id=sample_id,
                )
                continue

            # Build context for user prompt
            llm_context: dict = {
                "parsed_entries": gt_item.ground_truth.model_dump(),
            }

            if ocr_item and ocr_item.text:
                llm_context["ocr_text"] = ocr_item.text

            # Add previous page context if accumulating
            if request.accumulate_context and previous_context:
                llm_context["previous_context"] = previous_context

            # Render user prompt with context
            user_prompt = self.prompt_renderer.render_prompt(
                template_name=request.user_prompt, context=llm_context
            )

            # Load image
            image = self.image_storage.load_image(image_item.image_path)

            # Build user message with image
            user_message = construct_image_message(
                pil_image=image, text=user_prompt, role="user"
            )

            # Create request conversation: base conversation + current user message
            request_conversation = conversation.add(user_message)

            # Process with LLM
            llm_request = CompletionRequest(
                input=request_conversation,
                structured_output=SchematismPage,
            )
            result = self.llm_engine.process(llm_request)

            structured_response = result.output.structured_response
            if isinstance(structured_response, SchematismPage):
                # Create output item with generated source data
                produced_item = PredictionDataItem(
                    image_path=image_item.image_path,
                    text=ocr_item.text if ocr_item else None,
                    metadata=gt_item.metadata,
                    predictions=structured_response,  # This now contains Latin source data
                )

                # Extract context for next iteration if accumulating
                if request.accumulate_context and structured_response.context:
                    previous_context = {
                        "active_deanery": structured_response.context.active_deanery,
                        "last_page_number": structured_response.context.last_page_number,
                    }
            else:
                raise ValueError(
                    f"Unexpected structured response type: {type(structured_response)}"
                )

            # Update conversation history for context accumulation mode
            if request.accumulate_context:
                # Strip image from user message to avoid payload bloat
                text_only_user_message = strip_images_from_message(user_message)
                # Add text-only user message and assistant response to history
                conversation = conversation.add(text_only_user_message)
                conversation = conversation.add(result.output.to_message())

            items.append(produced_item)
            i += 1

        # Get stats from cached engine
        if isinstance(self.llm_engine, CachedEngine):
            stats = self.llm_engine.stats
            llm_executions = stats["misses"]
            cache_hits = stats["hits"]
        else:
            llm_executions = len(items)
            cache_hits = 0

        success_rate = (
            len(items) / len(request.parsed_ground_truth_dataset.items)
            if request.parsed_ground_truth_dataset.items
            else 0.0
        )

        logger.info(
            "Source generation completed",
            total_items=len(request.parsed_ground_truth_dataset.items),
            generated_items=len(items),
            llm_executions=llm_executions,
            cache_hits=cache_hits,
            success_rate=success_rate,
            cache_enabled=isinstance(self.llm_engine, CachedEngine),
        )

        return GenerateSourceDatasetResponse(
            dataset=PredictionDataset(items=items),
            llm_executions=llm_executions,
            cache_hits=cache_hits,
            success_rate=success_rate,
        )
