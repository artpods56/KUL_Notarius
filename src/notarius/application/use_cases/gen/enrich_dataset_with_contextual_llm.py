"""Use case for enriching prediction dataset with LLM using multi-turn contextual input.

This use case processes a dataset of schematism pages sequentially, maintaining
input context across pages. Each page's extraction benefits from knowing
what was extracted from previous pages (deanery context, formatting patterns, etc.).
"""

from dataclasses import dataclass
from typing import Any

from PIL.Image import Image as PILImage
from pydantic import BaseModel
from structlog import get_logger

from notarius.domain.entities.messages import ChatMessage, TextContent, ImageContent
from notarius.domain.entities.schematism import SchematismPage
from notarius.infrastructure.llm.conversation_builder import (
    Conversation,
    CompletionRequest,
)
from notarius.infrastructure.llm.engine_adapter import LLMEngine
from notarius.infrastructure.llm.prompt_manager import Jinja2PromptRenderer
from notarius.infrastructure.llm.utils import encode_image
from notarius.schemas.data.pipeline import BaseDataset, PredictionDataItem

logger = get_logger(__name__)


@dataclass
class EnrichDatasetRequest:
    """Request for enriching a dataset with contextual LLM predictions.

    Attributes:
        dataset: Dataset to enrich with predictions
        system_template: Jinja2 template name for system prompt
        user_template: Jinja2 template name for user prompt
        include_ocr: Whether to include OCR text in user prompts
        include_hints: Whether to include model hints in user prompts
    """

    dataset: BaseDataset[PredictionDataItem]
    system_template: str = "system.j2"
    user_template: str = "user.j2"
    include_ocr: bool = True
    include_hints: bool = True


@dataclass
class EnrichDatasetResponse:
    """Response from dataset enrichment.

    Attributes:
        enriched_dataset: Dataset with LLM predictions added to each item
        conversation: Final input history (useful for debugging)
        successful_predictions: Number of successfully processed items
        failed_predictions: Number of items that failed processing
    """

    enriched_dataset: BaseDataset[PredictionDataItem]
    conversation: Conversation
    successful_predictions: int
    failed_predictions: int


class EnrichDatasetWithContextualLLM:
    """Enrich prediction dataset with LLM using multi-turn contextual input.

    This use case processes schematism pages sequentially, building up input
    context as it goes. Each page extraction benefits from the context of previous
    pages, which helps with:
    - Maintaining deanery context across pages
    - Learning formatting patterns from earlier pages
    - Detecting inconsistencies or anomalies
    - Better handling of split entries across page boundaries

    Example:
        ```python
        _engine = LLMEngine.from_config(config)
        prompt_renderer = Jinja2PromptRenderer()

        use_case = EnrichDatasetWithContextualLLM(
            _engine=_engine,
            prompt_renderer=prompt_renderer,
        )

        request = EnrichDatasetRequest(
            dataset=my_dataset,
            system_template="system.j2",
            user_template="user.j2",
        )

        output = use_case.execute(request)
        print(f"Processed {output.successful_predictions} pages")
        ```
    """

    def __init__(
        self,
        engine: LLMEngine,
        prompt_renderer: Jinja2PromptRenderer,
    ):
        """Initialize the use case.

        Args:
            engine: LLM _engine for generating predictions
            prompt_renderer: Renderer for Jinja2 prompt templates
        """
        self.engine = engine
        self.prompt_renderer = prompt_renderer

    def _build_system_message(self, template: str) -> ChatMessage:
        """Build system message from template.

        Args:
            template: Jinja2 template name for system prompt

        Returns:
            ChatMessage with system role and rendered prompt
        """
        system_prompt = self.prompt_renderer.render_prompt(template, {})

        return ChatMessage(
            role="system",
            content=[TextContent(text=system_prompt)],
        )

    def _build_user_message(
        self,
        image: PILImage,
        template: str,
        ocr_text: str | None = None,
        hints: dict[str, Any] | None = None,
    ) -> ChatMessage:
        """Build user message with image and optional OCR/hints.

        Args:
            image: PIL Image of schematism page
            template: Jinja2 template name for user prompt
            ocr_text: Optional OCR text to include
            hints: Optional model hints to include

        Returns:
            ChatMessage with user role, image, and rendered prompt
        """
        # Prepare template context
        context: dict[str, Any] = {}
        if ocr_text:
            context["ocr_text"] = ocr_text
        if hints:
            # Convert hints dict to JSON string for template
            import json

            context["hints"] = json.dumps(hints, indent=2)

        # Render user prompt
        user_prompt = self.prompt_renderer.render_prompt(template, context)

        # Build multimodal message content
        content = [
            TextContent(text=user_prompt),
            ImageContent(
                image_url=encode_image(image),
                detail="high",
            ),
        ]

        return ChatMessage(role="user", content=content)

    def execute(self, request: EnrichDatasetRequest) -> EnrichDatasetResponse:
        """Execute the dataset enrichment with contextual input.

        Processes each item in the dataset sequentially, maintaining input
        context across all pages. The input history allows the LLM to:
        - Remember deanery context from previous pages
        - Learn formatting patterns
        - Maintain consistency in extraction style

        Args:
            request: Enrichment request with dataset and configuration

        Returns:
            EnrichDatasetResponse with enriched dataset and statistics

        Raises:
            ValueError: If dataset items are missing required fields
        """
        logger.info(
            "Starting contextual dataset enrichment",
            dataset_size=len(request.dataset.items),
            system_template=request.system_template,
            user_template=request.user_template,
        )

        # Initialize input with system message
        system_message = self._build_system_message(request.system_template)
        conversation = Conversation().add(system_message)

        # Track statistics
        successful = 0
        failed = 0
        enriched_items: list[PredictionDataItem] = []

        # Process each item sequentially with context
        for idx, item in enumerate(request.dataset.items):
            logger.info(
                "Processing item",
                index=idx,
                total=len(request.dataset.items),
                image_path=item.image_path,
            )

            try:
                # Validate item has required fields
                if not item.image_path:
                    raise ValueError(f"Item {idx} missing image_path")

                # Load image
                from PIL import Image

                image = Image.open(item.image_path)

                # Prepare OCR and hints if requested
                ocr_text = item.text if request.include_ocr else None
                hints = None
                if request.include_hints and hasattr(item, "metadata"):
                    hints = item.metadata

                # Build user message for this page
                user_message = self._build_user_message(
                    image=image,
                    template=request.user_template,
                    ocr_text=ocr_text,
                    hints=hints,
                )

                # Add to input
                conversation = conversation.add(user_message)

                # Generate prediction with input context
                completion_request = CompletionRequest(
                    conversation=conversation,
                    structured_output=SchematismPage,
                )

                result = self.engine.process(completion_request)

                # Extract structured prediction
                if isinstance(result.output.structured_response, SchematismPage):
                    prediction = result.output.structured_response
                else:
                    # Handle parsed output
                    import json

                    prediction_json = json.loads(result.output.to_string())
                    prediction = SchematismPage(**prediction_json)

                # Update input with assistant's output
                # This is critical - next page will have context of this extraction
                conversation = result.updated_conversation

                # Create enriched item
                enriched_item = PredictionDataItem(
                    image_path=item.image_path,
                    text=item.text,
                    metadata=item.metadata,
                    predictions=prediction,  # Add LLM prediction
                )

                enriched_items.append(enriched_item)
                successful += 1

                logger.info(
                    "Successfully processed item",
                    index=idx,
                    page_number=prediction.page_number,
                    entries_count=len(prediction.entries),
                )

            except Exception as e:
                logger.error(
                    "Failed to process item",
                    index=idx,
                    error=str(e),
                    error_type=type(e).__name__,
                )
                failed += 1

                # Add item without prediction to maintain dataset integrity
                enriched_items.append(item)

        # Build enriched dataset
        enriched_dataset = BaseDataset[PredictionDataItem](items=enriched_items)

        logger.info(
            "Completed contextual dataset enrichment",
            successful=successful,
            failed=failed,
            total=len(request.dataset.items),
        )

        return EnrichDatasetResponse(
            enriched_dataset=enriched_dataset,
            conversation=conversation,
            successful_predictions=successful,
            failed_predictions=failed,
        )
