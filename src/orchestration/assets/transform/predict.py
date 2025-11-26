import random

import dagster as dg
from dagster import AssetIn, AssetExecutionContext, MetadataValue

from core.data.schematism_parser import SchematismPage
from core.models.llm.model import LLMModel
from core.models.lmv3.model import LMv3Model
from core.models.ocr.model import OcrModel
from orchestration.constants import AssetLayer, ResourceGroup, DataSource, Kinds
from orchestration.resources import ImageStorageResource
from schemas.data.pipeline import (
    BaseDataset,
    GroundTruthDataItem,
    PredictionDataItem,
    BaseDataItem,
)


class OcrConfig(dg.Config):
    text_only: bool = True
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
        "ocr_model": AssetIn(key=[AssetLayer.RES, "ocr_model"]),
    },
)
def pred__ocr_enriched_dataset__pydantic(
    context: AssetExecutionContext,
    dataset: BaseDataset[BaseDataItem],
    ocr_model: OcrModel,
    config: OcrConfig,
    image_storage: ImageStorageResource,
) -> BaseDataset[BaseDataItem]:

    ocr_executions = 0

    for item in dataset.items:

        if not item.image_path:
            continue

        image = image_storage.load_image(item.image_path)

        if not item.text:
            item.text = ocr_model.predict(image=image, text_only=config.text_only)
            ocr_executions += 1
        elif config.overwrite:
            item.text = ocr_model.predict(image=image, text_only=config.text_only)
            ocr_executions += 1

    random_sample = dataset.items[random.randint(0, len(dataset.items) - 1)]

    context.add_asset_metadata(
        {
            "text_only": MetadataValue.bool(config.text_only),
            "overwrite": MetadataValue.bool(config.overwrite),
        }
    )

    context.add_output_metadata(
        {
            "len": MetadataValue.int(len(dataset.items)),
            "random_sample": MetadataValue.json(
                {k: v for k, v in random_sample.model_dump().items() if k != "image"}
            ),
            "items_with_text": MetadataValue.int(
                len([item for item in dataset.items if item.text])
            ),
            "ocr_executions": MetadataValue.int(ocr_executions),
        }
    )

    return dataset


class LMv3Config(dg.Config):
    raw_predictions: bool = False


@dg.asset(
    key_prefix=[AssetLayer.STG, DataSource.HUGGINGFACE],
    group_name=ResourceGroup.DATA,
    kinds={Kinds.PYTHON, Kinds.PYDANTIC},
    ins={
        "dataset": AssetIn(key="base__dataset__pydantic"),
        "lmv3_model": AssetIn(key=[AssetLayer.RES, "lmv3_model"]),
    },
)
def pred__lmv3_enriched_dataset__pydantic(
    context: AssetExecutionContext,
    dataset: BaseDataset[BaseDataItem],
    lmv3_model: LMv3Model,
    config: LMv3Config,
    image_storage: ImageStorageResource,
) -> BaseDataset[PredictionDataItem]:

    lmv3_executions = 0

    items: list[PredictionDataItem] = []

    for base_item in dataset.items:

        if base_item.image_path is None:
            continue

        image = image_storage.load_image(base_item.image_path)

        predictions = lmv3_model.predict(
            image,
            raw_predictions=config.raw_predictions,
        )
        lmv3_executions += 1

        prediction_item = PredictionDataItem(
            image_path=base_item.image_path,
            text=base_item.text,
            metadata=base_item.metadata,
            predictions=predictions,
        )

        items.append(prediction_item)

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
            "lmv3_executions": MetadataValue.int(lmv3_executions),
        }
    )

    return BaseDataset[PredictionDataItem](items=items)


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
        "llm_model": AssetIn(key=[AssetLayer.RES, "llm_model"]),
    },
)
def pred__llm_enriched_dataset__pydantic(
    context: AssetExecutionContext,
    lmv3_dataset: BaseDataset[PredictionDataItem],
    ocr_dataset: BaseDataset[BaseDataItem],
    llm_model: LLMModel,
    config: LLMConfig,
    image_storage: ImageStorageResource,
) -> BaseDataset[PredictionDataItem]:
    """Generate LLM predictions for each item in the lmv3_dataset.

    This asset takes the LMv3-enriched lmv3_dataset and uses an LLM to generate
    improved predictions, optionally using the LMv3 predictions as hints.
    """

    llm_executions = 0
    items: list[PredictionDataItem] = []

    for lmv3_item, ocr_item in zip(lmv3_dataset.items, ocr_dataset.items):

        if lmv3_item.image_path is None and lmv3_item.text is None:
            context.log.warning(f"Skipping item without image_path or text")
            continue

        image = image_storage.load_image(lmv3_item.image_path) if lmv3_item.image_path else None

        llm_context = {}
        if config.use_lmv3_hints and lmv3_item.predictions:
            llm_context["hints"] = lmv3_item.predictions.model_dump()

        try:
            response, parsed_messages = llm_model.predict(
                image=image,
                text=ocr_item.text,
                context=llm_context,
                system_prompt=config.system_prompt,
                user_prompt=config.user_prompt,
            )
            llm_executions += 1

            # Update predictions with LLM response
            produced_item = PredictionDataItem(
                image_path=lmv3_item.image_path,
                text=ocr_item.text,
                metadata=lmv3_item.metadata,
                predictions=response,
            )

            items.append(produced_item)

        except Exception as e:
            context.log.error(f"Failed to generate LLM prediction: {e}")
            items.append(lmv3_item)

    random_sample = items[random.randint(0, len(items) - 1)] if items else None

    context.add_asset_metadata(
        {
            "system_prompt": MetadataValue.text(config.system_prompt),
            "user_prompt": MetadataValue.text(config.user_prompt),
            "use_lmv3_hints": MetadataValue.bool(config.use_lmv3_hints),
        }
    )

    context.add_output_metadata(
        {
            "len": MetadataValue.int(len(items)),
            "random_sample": MetadataValue.json(
                {k: v for k, v in random_sample.model_dump().items() if k != "image"}
            ),
            "llm_executions": MetadataValue.int(llm_executions),
            "success_rate": MetadataValue.float(
                llm_executions / len(lmv3_dataset.items) if lmv3_dataset.items else 0.0
            ),
        }
    )

    return BaseDataset[PredictionDataItem](items=items)
