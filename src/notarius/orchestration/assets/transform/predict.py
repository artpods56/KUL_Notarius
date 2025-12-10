import random
from typing import Any

import dagster as dg
from dagster import AssetIn, AssetExecutionContext, MetadataValue
from pydantic import BaseModel

from notarius.application.use_cases.inference.add_llm_preds_to_dataset import (
    PredictDatasetWithLLM,
    PredictWithLLMRequest,
)
from notarius.infrastructure.cache.storage import (
    LLMCache,
    LMv3Cache,
    PyTesseractCache,
)

from notarius.infrastructure.ocr.engine_adapter import OCRMode
from notarius.infrastructure.persistence.llm_cache_repository import LLMCacheRepository
from notarius.infrastructure.persistence.lmv3_cache_repository import (
    LMv3CacheRepository,
)
from notarius.infrastructure.persistence.ocr_cache_repository import OCRCacheRepository
from notarius.orchestration.constants import (
    AssetLayer,
    ResourceGroup,
    DataSource,
    Kinds,
)
from notarius.orchestration.resources import (
    ImageStorageResource,
    OCREngineResource,
    LMv3EngineResource,
    LLMEngineResource,
)
from notarius.schemas.data.pipeline import (
    BaseDataset,
    PredictionDataItem,
    BaseDataItem,
)
from notarius.application.use_cases.inference.add_ocr_to_dataset import (
    EnrichWithOCRRequest,
    EnrichDatasetWithOCR,
)
from notarius.application.use_cases.inference.add_lmv3_preds_to_dataset import (
    EnrichDatasetWithLMv3,
    EnrichWithLMv3Request,
)


class OcrConfig(dg.Config):
    mode: OCRMode = "text"
    overwrite: bool = False


@dg.asset(
    key_prefix=[AssetLayer.STG, DataSource.HUGGINGFACE],
    group_name=ResourceGroup.DATA,
    kinds={
        Kinds.PYTHON,
        Kinds.PYDANTIC,
    },
    ins={
        "dataset": AssetIn(key="base__dataset__pydantic"),
    },
)
async def pred__ocr_enriched_dataset__pydantic(
    context: AssetExecutionContext,
    dataset: BaseDataset[BaseDataItem],
    config: OcrConfig,
    image_storage: ImageStorageResource,
    ocr_engine: OCREngineResource,
):
    ocr_model = ocr_engine.get_engine()

    use_case = EnrichDatasetWithOCR(
        ocr_engine=ocr_model,
        image_storage=image_storage,
        cache_repository=OCRCacheRepository(PyTesseractCache()),
    )

    request = EnrichWithOCRRequest(
        dataset=dataset,
        mode=config.mode,
        overwrite=config.overwrite,
    )
    response = await use_case.execute(request)

    # Add Dagster metadata
    random_sample = response.dataset.items[
        random.randint(0, len(response.dataset.items) - 1)
    ]

    context.add_output_metadata(
        {
            "len": MetadataValue.int(len(response.dataset.items)),
            "random_sample": MetadataValue.json(
                {k: v for k, v in random_sample.model_dump().items() if k != "image"}
            ),
            "items_with_text": MetadataValue.int(
                len([item for item in response.dataset.items if item.text])
            ),
            "ocr_executions": MetadataValue.int(response.ocr_executions),
        }
    )

    return response.dataset


class LMv3Config(dg.Config):
    raw_predictions: bool = False
    overwrite: bool = False


@dg.asset(
    key_prefix=[AssetLayer.STG, DataSource.HUGGINGFACE],
    group_name=ResourceGroup.DATA,
    kinds={Kinds.PYTHON, Kinds.PYDANTIC},
    ins={
        "dataset": AssetIn(key="base__dataset__pydantic"),
    },
)
async def pred__lmv3_enriched_dataset__pydantic(
    context: AssetExecutionContext,
    dataset: BaseDataset[BaseDataItem],
    config: LMv3Config,
    image_storage: ImageStorageResource,
    lmv3_engine: LMv3EngineResource,
):
    # Get the actual _engine instance from the resource
    lmv3_model = lmv3_engine.get_engine()

    # Create use case
    use_case = EnrichDatasetWithLMv3(
        lmv3_engine=lmv3_model,
        image_storage=image_storage,
        cache_repository=LMv3CacheRepository(
            cache=LMv3Cache(checkpoint="layoutlmv3_focalloss_4000")
        ),
    )

    # Execute use case
    request = EnrichWithLMv3Request(
        dataset=dataset,
        overwrite=config.overwrite,
    )
    response = await use_case.execute(request)

    # Add Dagster metadata
    random_sample = dataset.items[random.randint(0, len(dataset.items) - 1)]

    context.add_asset_metadata(
        {
            "raw_predictions": MetadataValue.bool(config.raw_predictions),
        }
    )

    context.add_output_metadata(
        {
            "len": MetadataValue.int(len(dataset.items)),
            "random_sample": MetadataValue.json(
                {k: v for k, v in random_sample.model_dump().items() if k != "image"}
            ),
            "lmv3_executions": MetadataValue.int(response.lmv3_executions),
        }
    )

    return response.dataset


class LLMConfig(dg.Config):
    system_prompt: str = "system.j2"
    user_prompt: str = "user.j2"
    use_lmv3_hints: bool = True


@dg.asset(
    key_prefix=[AssetLayer.STG, DataSource.HUGGINGFACE],
    group_name=ResourceGroup.DATA,
    kinds={Kinds.PYTHON, Kinds.PYDANTIC},
    ins={
        "lmv3_dataset": AssetIn(key="pred__lmv3_enriched_dataset__pydantic"),
        "ocr_dataset": AssetIn(key="pred__ocr_enriched_dataset__pydantic"),
    },
)
async def pred__llm_enriched_dataset__pydantic(
    context: AssetExecutionContext,
    lmv3_dataset: BaseDataset,
    ocr_dataset: BaseDataset[BaseDataItem],
    config: LLMConfig,
    image_storage: ImageStorageResource,
    llm_engine: LLMEngineResource,
):
    """Generate LLM predictions for each item in the lmv3_dataset.

    This asset takes the LMv3-enriched lmv3_dataset and uses an LLM to generate
    improved predictions, optionally using the LMv3 predictions as hints.
    """

    # Get the actual _engine instance from the resource
    llm_model = llm_engine.get_engine()

    # Create use case
    use_case = PredictDatasetWithLLM(
        llm_engine=llm_model,
        image_storage=image_storage,
        cache_repository=LLMCacheRepository(cache=LLMCache(llm_model.used_model)),
    )

    # Execute use case
    request = PredictWithLLMRequest(
        lmv3_dataset=lmv3_dataset,
        ocr_dataset=ocr_dataset,
        system_prompt=config.system_prompt,
        user_prompt=config.user_prompt,
        use_lmv3_hints=config.use_lmv3_hints,
    )
    response = await use_case.execute(request)

    # Add Dagster metadata
    random_sample = (
        response.dataset.items[random.randint(0, len(response.dataset.items) - 1)]
        if response.dataset.items
        else None
    )

    context.add_asset_metadata(
        {
            "system_prompt": MetadataValue.text(config.system_prompt),
            "user_prompt": MetadataValue.text(config.user_prompt),
            "use_lmv3_hints": MetadataValue.bool(config.use_lmv3_hints),
        }
    )

    context.add_output_metadata(
        {
            "len": MetadataValue.int(len(response.dataset.items)),
            "random_sample": (
                MetadataValue.json(
                    {
                        k: v
                        for k, v in random_sample.model_dump().items()
                        if k != "image"
                    }
                )
                if random_sample
                else None
            ),
            "llm_executions": MetadataValue.int(response.llm_executions),
            "success_rate": MetadataValue.float(response.success_rate),
        }
    )

    return response.dataset
