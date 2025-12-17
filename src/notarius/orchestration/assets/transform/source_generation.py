"""Assets for generating source (Latin) dataset from parsed (Polish) ground truth."""

import json
import random
from datetime import datetime
from pathlib import Path

import dagster as dg
from dagster import AssetExecutionContext, AssetIn, MetadataValue

from notarius.application.use_cases.inference import (
    GenerateSourceDataset,
    GenerateSourceDatasetRequest,
)
from notarius.orchestration.constants import (
    AssetLayer,
    DataSource,
    ResourceGroup,
    Kinds,
)
from notarius.orchestration.resources.base import (
    ImageStorageResource,
    LLMEngineResource,
)
from notarius.schemas.data.pipeline import (
    BaseDataset,
    BaseDataItem,
    GroundTruthDataItem,
    PredictionDataItem,
)
from notarius.shared.constants import OUTPUTS_DIR


class SourceGenerationConfig(dg.Config):
    """Configuration for source generation."""

    system_prompt: str = "tasks/source_generation/system.j2"
    user_prompt: str = "tasks/source_generation/user.j2"
    accumulate_context: bool = True
    enable_cache: bool = True


@dg.asset(
    key_prefix=[AssetLayer.FCT, DataSource.HUGGINGFACE],
    group_name=ResourceGroup.DATA,
    kinds={Kinds.PYTHON, Kinds.PYDANTIC},
    ins={
        "parsed_gt_dataset": AssetIn(key="gt__parsed_dataset__pydantic"),
        "ocr_dataset": AssetIn(key="pred__llm_ocr_enriched_dataset__pydantic"),
        "image_dataset": AssetIn(key="base__dataset__pydantic"),
    },
)
async def source__generated_dataset__pydantic(
    context: AssetExecutionContext,
    parsed_gt_dataset: BaseDataset[GroundTruthDataItem],
    ocr_dataset: BaseDataset[BaseDataItem],
    image_dataset: BaseDataset[BaseDataItem],
    config: SourceGenerationConfig,
    llm_engine_resource: LLMEngineResource,
    image_storage: ImageStorageResource,
) -> BaseDataset[PredictionDataItem]:
    """Generate source (Latin) dataset from parsed (Polish) ground truth.

    This asset uses an LLM to find Latin source text on page images
    that corresponds to the parsed Polish ground truth entries.

    Args:
        context: Dagster execution context
        parsed_gt_dataset: Dataset with parsed Polish ground truth
        ocr_dataset: Dataset with OCR text for each page
        image_dataset: Dataset with page images
        config: Source generation configuration
        llm_engine: LLM engine resource
        image_storage: Image storage resource

    Returns:
        Dataset with generated Latin source entries
    """
    engine = llm_engine_resource.get_engine()
    use_case = GenerateSourceDataset(
        llm_engine=engine,
        image_storage=image_storage,
        model_name=engine.used_model,
        enable_cache=config.enable_cache,
    )

    request = GenerateSourceDatasetRequest(
        parsed_ground_truth_dataset=parsed_gt_dataset,
        ocr_dataset=ocr_dataset,
        image_dataset=image_dataset,
        system_prompt=config.system_prompt,
        user_prompt=config.user_prompt,
        accumulate_context=config.accumulate_context,
    )

    response = await use_case.execute(request)

    # Log a random sample for inspection
    if response.dataset.items:
        random_sample = random.choice(response.dataset.items)
        sample_preview = {
            k: v for k, v in random_sample.model_dump().items() if k != "image"
        }
    else:
        sample_preview = {}

    context.add_output_metadata(
        {
            "dataset_size": MetadataValue.int(len(response.dataset.items)),
            "llm_executions": MetadataValue.int(response.llm_executions),
            "cache_hits": MetadataValue.int(response.cache_hits),
            "success_rate": MetadataValue.float(response.success_rate),
            "random_sample": MetadataValue.json(sample_preview),
        }
    )

    context.log.info(
        f"Generated source dataset with {len(response.dataset.items)} items "
        f"(LLM calls: {response.llm_executions}, cache hits: {response.cache_hits})"
    )

    return response.dataset


class SourceExportConfig(dg.Config):
    """Configuration for source dataset JSON export."""

    output_dir: str = str(OUTPUTS_DIR / "source_generation")
    group_by_schematism: bool = True
    pretty_print: bool = True


@dg.asset(
    key_prefix=[AssetLayer.MRT, DataSource.HUGGINGFACE],
    group_name=ResourceGroup.DATA,
    kinds={Kinds.PYTHON, Kinds.JSON},
    ins={
        "source_dataset": AssetIn(key="source__generated_dataset__pydantic"),
    },
)
def source__exported_json(
    context: AssetExecutionContext,
    source_dataset: BaseDataset[PredictionDataItem],
    config: SourceExportConfig,
) -> dict[str, Path]:
    """Export generated source dataset to JSON files for manual review.

    Args:
        context: Dagster execution context
        source_dataset: Dataset with generated Latin source entries
        config: Export configuration

    Returns:
        Dictionary mapping schematism names to output file paths
    """
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_files: dict[str, Path] = {}

    if config.group_by_schematism:
        # Group items by schematism name
        by_schematism: dict[str, list[dict]] = {}

        for item in source_dataset.items:
            if not item.metadata:
                continue

            schematism_name = item.metadata.schematism_name or "unknown"
            if schematism_name not in by_schematism:
                by_schematism[schematism_name] = []

            # Build export record
            record = {
                "sample_id": item.metadata.sample_id,
                "filename": item.metadata.filename,
                "schematism_name": schematism_name,
            }

            if item.predictions:
                record["source"] = item.predictions.model_dump()

            by_schematism[schematism_name].append(record)

        # Write one JSON file per schematism
        for schematism_name, records in by_schematism.items():
            # Sort by sample_id for consistent ordering
            records.sort(key=lambda r: r.get("sample_id", ""))

            output_file = output_dir / f"{timestamp}_{schematism_name}_source.json"
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "schematism_name": schematism_name,
                        "generated_at": timestamp,
                        "total_records": len(records),
                        "records": records,
                    },
                    f,
                    ensure_ascii=False,
                    indent=2 if config.pretty_print else None,
                )

            output_files[schematism_name] = output_file
            context.log.info(
                f"Exported {len(records)} records for '{schematism_name}' to {output_file}"
            )
    else:
        # Single file with all records
        all_records = []
        for item in source_dataset.items:
            if not item.metadata:
                continue

            record = {
                "sample_id": item.metadata.sample_id,
                "filename": item.metadata.filename,
                "schematism_name": item.metadata.schematism_name,
            }

            if item.predictions:
                record["source"] = item.predictions.model_dump()

            all_records.append(record)

        # Sort by schematism_name, then sample_id
        all_records.sort(
            key=lambda r: (r.get("schematism_name", ""), r.get("sample_id", ""))
        )

        output_file = output_dir / f"{timestamp}_all_source.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "generated_at": timestamp,
                    "total_records": len(all_records),
                    "records": all_records,
                },
                f,
                ensure_ascii=False,
                indent=2 if config.pretty_print else None,
            )

        output_files["all"] = output_file
        context.log.info(f"Exported {len(all_records)} records to {output_file}")

    context.add_output_metadata(
        {
            "output_dir": MetadataValue.path(str(output_dir)),
            "files_created": MetadataValue.int(len(output_files)),
            "file_paths": MetadataValue.json(
                {k: str(v) for k, v in output_files.items()}
            ),
            "total_records": MetadataValue.int(len(source_dataset.items)),
        }
    )

    return output_files
