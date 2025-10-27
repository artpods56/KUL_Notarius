from datetime import datetime

import dagster as dg
from dagster import ConfigMapping, mem_io_manager, in_process_executor

from core.utils.shared import OUTPUTS_DIR

#
# from orchestration.configs.ingestion_config import (
#     RAW_HF_DATASET_OP_CONFIG,
# )
from orchestration.configs.prediction_config import (
    OCR_ENRICHED_DATASET_OP_CONFIG,
    LMV3_ENRICHED_DATASET_OP_CONFIG,
    LLM_ENRICHED_DATASET_OP_CONFIG,
)
from orchestration.configs.postprocessing_config import (
    DEANERY_FILLED_DATASET_OP_CONFIG,
    PARSED_DATASET_OP_CONFIG,
    GT_ALIGNED_DATASET_OP_CONFIG,
)
from orchestration.configs.references_config import (
    RES_HF_DATASET_CONFIG_OP_CONFIG,
    RES_OCR_MODEL_CONFIG_OP_CONFIG,
    RES_LLM_MODEL_CONFIG_OP_CONFIG,
    RES_LMV3_MODEL_CONFIG_OP_CONFIG,
)
from orchestration.configs.transformation_config import (
    FILTERED_HF_DATASET_OP_CONFIG,
    GT_SOURCE_DATASET_PYDANTIC_OP_CONFIG,
    BASE_DATASET_PYDANTIC_OP_CONFIG,
    GT_PARSED_DATASET_PYDANTIC_OP_CONFIG,
    EVAL_ALIGNED_DATAFRAME_PANDAS_OP_CONFIG,
)

from orchestration.configs.exporting_config import (
    EVAL_EXPORT_DATAFRAME_PANDAS,
    EVAL_WANDB_EXPORT_DATAFRAME_PANDAS,
)
from orchestration.configs.models_config import RES_OCR_MODEL_OP_CONFIG

from orchestration.assets.configs import (
    hf_dataset__config,
    ocr_model__config,
    lmv3_model__config,
    llm_model__config,
)

from orchestration.assets.ingest import (
    raw__hf__dataset,
    raw__pdf__dataset,
)

from orchestration.assets.transform import (
    filtered__hf__dataset,
    base__dataset__pydantic,
    gt__source_dataset__pydantic,
    gt__parsed_dataset__pydantic,
    eval__aligned_dataframe__pandas,
)

from orchestration.assets.predict import (
    ocr_enriched__dataset,
    lmv3_enriched__dataset,
    llm_enriched__dataset,
)

from orchestration.assets.postprocess import (
    deanery_filled__dataset,
    parsed__dataset,
    gt_aligned__dataset,
)

from orchestration.assets.export import (
    eval__export_dataframe__pandas,
    eval__wandb_export_dataframe__pandas,
)

from orchestration.assets.configs import ocr_model__config

from orchestration.assets.models import ocr_model, lmv3_model, llm_model, parser

import core.data.filters  # type: ignore
import schemas.configs  # type: ignore

from orchestration.resources import OpRegistry, ExcelWriterResource, WandBRunResource
from orchestration.resources import PdfFilesResource, ConfigManagerResource


dataset_config_assets = [hf_dataset__config]
ingestion_assets = [
    raw__hf__dataset,
    filtered__hf__dataset,
    gt__source_dataset__pydantic,
    gt__parsed_dataset__pydantic,
    base__dataset__pydantic,
]
ingestion_job = dg.define_asset_job(
    name="ingestion_pipeline",
    selection=dg.AssetSelection.assets(*dataset_config_assets, *ingestion_assets),
    config={
        "ops": {
            # configs refs
            **RES_HF_DATASET_CONFIG_OP_CONFIG,
            # asset refs
            # **RAW_HF_DATASET_OP_CONFIG,
            **FILTERED_HF_DATASET_OP_CONFIG,
            **GT_SOURCE_DATASET_PYDANTIC_OP_CONFIG,
            **GT_PARSED_DATASET_PYDANTIC_OP_CONFIG,
            **BASE_DATASET_PYDANTIC_OP_CONFIG,
        }
    },
)

models_config_assets = [
    ocr_model__config,
    lmv3_model__config,
    llm_model__config,
]
models_assets = [ocr_model, lmv3_model, llm_model, parser]
prediction_assets = [
    ocr_enriched__dataset,
    lmv3_enriched__dataset,
    llm_enriched__dataset,
]


prediction_job = dg.define_asset_job(
    name="prediction_pipeline",
    selection=dg.AssetSelection.assets(
        *models_config_assets, *models_assets, *prediction_assets
    ),
    config={
        "ops": {
            # configs refs
            **RES_OCR_MODEL_CONFIG_OP_CONFIG,
            **RES_LLM_MODEL_CONFIG_OP_CONFIG,
            **RES_LMV3_MODEL_CONFIG_OP_CONFIG,
            # models refs
            # asset refs
            **OCR_ENRICHED_DATASET_OP_CONFIG,
            **LMV3_ENRICHED_DATASET_OP_CONFIG,
            **LLM_ENRICHED_DATASET_OP_CONFIG,
        }
    },
)

postprocessing_assets = [
    deanery_filled__dataset,
    parsed__dataset,
    gt_aligned__dataset,
    eval__aligned_dataframe__pandas,
]

postprocessing_job = dg.define_asset_job(
    name="postprocessing_pipeline",
    selection=dg.AssetSelection.assets(parser, *postprocessing_assets),
    config={
        "ops": {
            # asset refs
            **DEANERY_FILLED_DATASET_OP_CONFIG,
            **PARSED_DATASET_OP_CONFIG,
            **GT_ALIGNED_DATASET_OP_CONFIG,
            **EVAL_ALIGNED_DATAFRAME_PANDAS_OP_CONFIG,
        }
    },
)


exporting_assets = [eval__export_dataframe__pandas, eval__wandb_export_dataframe__pandas]

exporting_job = dg.define_asset_job(
    name="exporting_pipeline",
    selection=dg.AssetSelection.assets(*exporting_assets),
    config={"ops": {**EVAL_EXPORT_DATAFRAME_PANDAS, **EVAL_WANDB_EXPORT_DATAFRAME_PANDAS}},
)


defs = dg.Definitions(
    assets=[
        # ingestion
        *dataset_config_assets,
        *ingestion_assets,
        # configs
        *models_config_assets,
        # models
        *models_assets,
        # prediction
        *prediction_assets,
        # postprocessing
        *postprocessing_assets,
        # export
        *exporting_assets,
    ],
    jobs=[ingestion_job, prediction_job, postprocessing_job, exporting_job],
    resources={
        # "pdf_files": PdfFilesResource(),
        "config_manager": ConfigManagerResource(),
        "op_registry": OpRegistry,
        "mem_io_manager": mem_io_manager,
        "excel_writer": ExcelWriterResource(writing_path=str(OUTPUTS_DIR)),
        "wandb_run": WandBRunResource(
            project_name="KUL_IDUB_EcclesiaSchematisms",
            run_name=f"dagster_eval_{datetime.now().isoformat()}",
            mode="online",
        ),
    },
    executor=in_process_executor,
)
