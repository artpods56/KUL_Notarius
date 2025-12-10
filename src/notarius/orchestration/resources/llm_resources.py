"""Dagster resources for LLM enrichment pipeline.

This shows how to properly configure LLM components as Dagster resources,
not assets. Resources are stateless services that assets use.
"""

from dagster import ConfigurableResource

from notarius.application.use_cases.gen.enrich_dataset_with_llm import (
    EnrichDatasetWithLLM,
)
from notarius.application.use_cases.gen.extract_schematism_page import (
    ExtractSchematismPage,
)
from notarius.domain.services.llm_extraction_service import LLMExtractionService
from notarius.domain.services.prompt_service import PromptConstructionService
from notarius.infrastructure.llm.engine_adapter import LLMEngine
from notarius.infrastructure.llm.prompt_manager import Jinja2PromptRenderer
from notarius.infrastructure.persistence.llm_cache_repository import LLMCacheRepository
from notarius.infrastructure.cache.storage import LLMCache
from notarius.schemas.configs.llm_model_config import LLMEngineConfig
from notarius.shared.constants import PROMPTS_DIR


class LLMEngineResource(ConfigurableResource):
    """Resource for LLM _engine.

    This is a proper Dagster resource - it's a stateless service
    that can be shared across multiple assets.
    """

    model_name: str = "gpt-4o"
    temperature: float = 0.0
    max_tokens: int = 4000
    api_key: str  # Should come from env var in production

    def create_engine(self) -> LLMEngine:
        """Create LLM _engine instance.

        Returns:
            Configured LLM _engine
        """
        config = LLMEngineConfig(
            provider={
                "type": "openai",
                "model": self.model_name,
                "api_key": self.api_key,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
            }
        )
        return LLMEngine.from_config(config)


class PromptServiceResource(ConfigurableResource):
    """Resource for prompt construction service."""

    template_dir: str = str(PROMPTS_DIR)

    def create_service(self) -> PromptConstructionService:
        """Create prompt service instance.

        Returns:
            Configured prompt service
        """
        renderer = Jinja2PromptRenderer(template_dir=self.template_dir)
        return PromptConstructionService(prompt_renderer=renderer)


class LLMCacheResource(ConfigurableResource):
    """Resource for LLM cache repository."""

    cache_name: str = "default"
    enable_cache: bool = True

    def create_repository(self) -> LLMCacheRepository | None:
        """Create cache repository instance.

        Returns:
            Cache repository or None if caching disabled
        """
        if not self.enable_cache:
            return None

        cache = LLMCache(cache_name=self.cache_name)
        return LLMCacheRepository(cache=cache)


class LLMExtractionServiceResource(ConfigurableResource):
    """Resource for domain extraction service.

    Note: Domain services are stateless, so they can be resources.
    """

    def create_service(self) -> LLMExtractionService:
        """Create extraction service instance.

        Returns:
            Domain extraction service
        """
        return LLMExtractionService()


class ExtractSchematismPageUseCaseResource(ConfigurableResource):
    """Resource that creates the single-page extraction use case.

    This demonstrates proper dependency injection - the use case
    depends on other resources.
    """

    def create_use_case(
        self,
        llm_engine: LLMEngine,
        prompt_service: PromptConstructionService,
        extraction_service: LLMExtractionService,
        cache_repository: LLMCacheRepository | None,
    ) -> ExtractSchematismPage:
        """Create use case with injected dependencies.

        Args:
            llm_engine: LLM _engine instance
            prompt_service: Prompt service instance
            extraction_service: Domain extraction service
            cache_repository: Optional cache repository

        Returns:
            Configured use case
        """
        return ExtractSchematismPage(
            llm_engine=llm_engine,
            extraction_service=extraction_service,
            prompt_service=prompt_service,
            cache_repository=cache_repository,
        )


class EnrichDatasetWithLLMUseCaseResource(ConfigurableResource):
    """Resource that creates the dataset enrichment use case.

    This is the orchestration-level use case that processes batches.
    """

    def create_use_case(
        self,
        extract_page_use_case: ExtractSchematismPage,
        image_storage: "ImageStorageResource",
    ) -> EnrichDatasetWithLLM:
        """Create use case with injected dependencies.

        Args:
            extract_page_use_case: Single-page extraction use case
            image_storage: Image storage resource

        Returns:
            Configured dataset enrichment use case
        """
        return EnrichDatasetWithLLM(
            extract_page_use_case=extract_page_use_case,
            image_storage=image_storage,
        )


# Example of how to configure these in definitions.py:
"""
# In src/notarius/orchestration/definitions.py

from notarius.orchestration.resources.llm_resources import (
    LLMEngineResource,
    PromptServiceResource,
    LLMCacheResource,
    LLMExtractionServiceResource,
    ExtractSchematismPageUseCaseResource,
    EnrichDatasetWithLLMUseCaseResource,
)

defs = dg.Definitions(
    assets=[...],
    jobs=[...],
    resources={
        # Infrastructure resources
        "llm_engine": LLMEngineResource(
            model_name=EnvVar("LLM_MODEL_NAME"),
            api_key=EnvVar("OPENAI_API_KEY"),
        ),
        "prompt_service": PromptServiceResource(),
        "llm_cache": LLMCacheResource(enable_cache=True),
        "image_storage": ImageStorageResource(...),

        # Domain service resources
        "extraction_service": LLMExtractionServiceResource(),

        # Use case resources (these are factories, not instances)
        # They get instantiated in assets with proper dependency injection
    },
)

# Then in your asset:
@dg.asset
def llm_enriched_dataset(
    base_dataset: BaseDataset,
    llm_engine: LLMEngineResource,
    prompt_service: PromptServiceResource,
    extraction_service: LLMExtractionServiceResource,
    llm_cache: LLMCacheResource,
    image_storage: ImageStorageResource,
) -> BaseDataset[PredictionDataItem]:
    # Build use case with dependency injection
    extract_page = ExtractSchematismPage(
        llm_engine=llm_engine.create_engine(),
        extraction_service=extraction_service.create_service(),
        prompt_service=prompt_service.create_service(),
        cache_repository=llm_cache.create_repository(),
    )

    enrich_dataset = EnrichDatasetWithLLM(
        extract_page_use_case=extract_page,
        image_storage=image_storage,
    )

    # Execute
    request = EnrichDatasetWithLLMRequest(
        lmv3_dataset=base_dataset,
        ocr_dataset=base_dataset,
    )

    output = await enrich_dataset.execute(request)
    return output.dataset
"""
