from datetime import datetime

import dagster as dg
from dagster import mem_io_manager, in_process_executor

from core.utils.shared import OUTPUTS_DIR

import core.data.filters  # type: ignore
import schemas.configs  # type: ignore

from orchestration.jobs.exporting import exporting_assets, exporting_job
from orchestration.jobs.ingestion import (
    dataset_config_assets,
    ingestion_assets,
    ingestion_job,
)
from orchestration.jobs.postprocessing import postprocessing_assets, postprocessing_job
from orchestration.jobs.prediction import (
    models_config_assets,
    models_assets,
    prediction_assets,
    prediction_job,
)
from orchestration.jobs.complete_pipeline import complete_pipeline_job

from orchestration.resources import OpRegistry, ExcelWriterResource, WandBRunResource
from orchestration.resources import ConfigManagerResource

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
    jobs=[
        ingestion_job,
        prediction_job,
        postprocessing_job,
        exporting_job,
        complete_pipeline_job,
    ],
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
