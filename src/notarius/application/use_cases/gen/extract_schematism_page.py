"""Use case for extracting a single schematism page using LLM.

This use case orchestrates infrastructure components (cache, prompts, LLM _engine)
and domain services to perform a single extraction operation.
"""

from dataclasses import dataclass
from typing import Any

from PIL.Image import Image as PILImage
from structlog import get_logger

from notarius.application.use_cases.base import BaseRequest, BaseResponse, BaseUseCase
from notarius.domain.services.llm_extraction_service import (
    ExtractionInput,
    ExtractionResult,
    LLMExtractionService,
)
from notarius.domain.services.prompt_service import PromptConstructionService
from notarius.infrastructure.llm.conversation import (
    Conversation,
)
from notarius.infrastructure.llm.engine_adapter import LLMEngine, CompletionRequest
from notarius.infrastructure.persistence.llm_cache_repository import LLMCacheRepository
from notarius.schemas.configs.llm_model_config import PromptTemplates
from notarius.schemas.data.cache import LLMCacheItem
from notarius.domain.entities.schematism import SchematismPage

logger = get_logger(__name__)


@dataclass(frozen=True)
class ExtractSchematismPageRequest(BaseRequest):
    """Request to extract a single schematism page."""

    image: PILImage | None = None
    text: str | None = None
    hints: dict[str, Any] | None = None
    templates: PromptTemplates | None = None
    invalidate_cache: bool = False
    metadata: dict[str, Any] | None = None


@dataclass(frozen=True)
class ExtractSchematismPageResponse(BaseResponse):
    """Response containing extracted schematism page."""

    result: ExtractionResult
    messages: str  # For debugging/logging


class ExtractSchematismPage(
    BaseUseCase[ExtractSchematismPageRequest, ExtractSchematismPageResponse]
):
    """Use case for extracting structured data from a single schematism page.

    Responsibilities:
    - Orchestrate infrastructure (cache, prompt rendering, LLM)
    - Delegate business logic to domain service
    - Handle cross-cutting concerns (logging, caching, retries)
    """

    def __init__(
        self,
        llm_engine: LLMEngine,
        extraction_service: LLMExtractionService,
        prompt_service: PromptConstructionService,
        cache_repository: LLMCacheRepository | None = None,
    ):
        """Initialize the use case.

        Args:
            llm_engine: Engine for LLM completions
            extraction_service: Domain service for extraction logic
            prompt_service: Service for constructing prompts
            cache_repository: Optional cache repository
        """
        self.llm_engine = llm_engine
        self.extraction_service = extraction_service
        self.prompt_service = prompt_service
        self.cache_repository = cache_repository

    async def execute(
        self, request: ExtractSchematismPageRequest
    ) -> ExtractSchematismPageResponse:
        """Execute the extraction workflow.

        Workflow:
        1. Validate input (domain service)
        2. Check cache (infrastructure)
        3. Build prompts (infrastructure)
        4. Call LLM (infrastructure)
        5. Post-process result (domain service)
        6. Store in cache (infrastructure)

        Args:
            request: Extraction request

        Returns:
            Response with extracted page and metadata
        """
        # 1. Create domain input and validate
        extraction_input = ExtractionInput(
            image=request.image,
            text=request.text,
            hints=request.hints,
            metadata=request.metadata,
        )
        self.extraction_service.validate_input(extraction_input)

        # 2. Try cache if enabled
        cache_key = None
        if self.cache_repository:
            cache_key = await self._build_cache_key(request, extraction_input)

            if request.invalidate_cache:
                await self.cache_repository.delete(cache_key)
                logger.debug("Cache invalidated", key=cache_key[:16])

            cached_item = await self.cache_repository.get(cache_key)
            if cached_item:
                logger.info("Returning cached extraction")
                return await self._build_response_from_cache(
                    cached_item, extraction_input
                )

        # 3. Build prompts
        templates = request.templates or PromptTemplates()
        context = self._build_prompt_context(extraction_input)

        messages = self.prompt_service.build_messages(
            image=request.image,
            text=request.text,
            system_template=templates.system_template,
            user_template=templates.user_template,
            context=context,
        )

        # 4. Call LLM
        logger.info("Generating new extraction via LLM")
        conversation = Conversation().add_many(messages)
        completion_request = CompletionRequest(
            input=conversation,
            structured_output=SchematismPage,
        )

        result = await self.llm_engine.process(completion_request)

        # 5. Post-process with domain service
        raw_dict = result.output.structured_output.model_dump()
        processed_page = self.extraction_service.post_process_extraction(
            raw_dict, extraction_input
        )

        # 6. Store in cache
        if self.cache_repository and cache_key:
            await self._store_in_cache(cache_key, processed_page, request)

        # 7. Build output
        confidence = self.extraction_service.compute_confidence(
            processed_page, extraction_input, from_cache=False
        )

        extraction_result = ExtractionResult(
            page=processed_page,
            confidence=confidence,
            from_cache=False,
            raw_response=raw_dict,
        )

        return ExtractSchematismPageResponse(
            result=extraction_result,
            messages=str(messages),
        )

    async def _build_cache_key(
        self, request: ExtractSchematismPageRequest, extraction_input: ExtractionInput
    ) -> str:
        """Build cache key from request."""
        return self.cache_repository.generate_key(
            image=request.image,
            text=request.text,
            hints=request.hints,
            template_name=(
                request.templates.system_template if request.templates else "default"
            ),
        )

    async def _build_response_from_cache(
        self, cached_item: LLMCacheItem, extraction_input: ExtractionInput
    ) -> ExtractSchematismPageResponse:
        """Build output from cached item."""
        page = SchematismPage(**cached_item.response)

        confidence = self.extraction_service.compute_confidence(
            page, extraction_input, from_cache=True
        )

        extraction_result = ExtractionResult(
            page=page,
            confidence=confidence,
            from_cache=True,
        )

        return ExtractSchematismPageResponse(
            result=extraction_result,
            messages="[from cache]",
        )

    async def _store_in_cache(
        self,
        cache_key: str,
        page: SchematismPage,
        request: ExtractSchematismPageRequest,
    ) -> None:
        """Store extraction result in cache."""
        cache_item = LLMCacheItem(
            response=page.model_dump(),
            hints=request.hints,
        )

        await self.cache_repository.set(
            key=cache_key,
            item=cache_item,
            metadata=request.metadata,
        )
        logger.debug("Extraction cached", key=cache_key[:16])

    def _build_prompt_context(
        self, extraction_input: ExtractionInput
    ) -> dict[str, Any]:
        """Build context dictionary for prompt rendering."""
        context: dict[str, Any] = {}

        if extraction_input.hints:
            context["hints"] = extraction_input.hints

        if extraction_input.metadata:
            context["metadata"] = extraction_input.metadata

        return context
