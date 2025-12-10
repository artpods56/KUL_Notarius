from datetime import datetime

import dagster as dg
from dagster import in_process_executor, mem_io_manager

from notarius.domain.services.parser import Parser
from notarius.infrastructure.config.manager import get_config_manager
from notarius.shared.constants import TMP_DIR, OUTPUTS_DIR
from notarius.orchestration.dill_io_manager import dill_io_manager

import notarius.orchestration.operations  # type: ignore
import notarius.schemas.configs  # type: ignore

from notarius.orchestration.jobs.exporting import exporting_assets, exporting_job
from notarius.orchestration.jobs.ingestion import (
    dataset_config_assets,
    ingestion_assets,
    ingestion_job,
)
from notarius.orchestration.jobs.postprocessing import (
    postprocessing_assets,
    postprocessing_job,
)
from notarius.orchestration.jobs.prediction import (
    models_config_assets,
    prediction_assets,
    prediction_job,
)
from notarius.orchestration.jobs.complete_pipeline import complete_pipeline_job

from notarius.orchestration.resources import (
    OpRegistry,
    ExcelWriterResource,
    WandBRunResource,
    ImageStorageResource,
    OCREngineResource,
    LMv3EngineResource,
    LLMEngineResource,
)

defs = dg.Definitions(
    assets=[
        # ingestion
        *dataset_config_assets,
        *ingestion_assets,
        # configs
        *models_config_assets,
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
        "config_manager": get_config_manager(),
        "parser": Parser(),
        "image_storage": ImageStorageResource(
            image_storage_path=str(TMP_DIR / "dagster_image_storage")
        ),
        "op_registry": OpRegistry,
        "io_manager": mem_io_manager,  # dill_io_manager(base_dir=str(TMP_DIR / "dagster_dill_storage")),
        "excel_writer": ExcelWriterResource(writing_path=str(OUTPUTS_DIR)),
        "wandb_run": WandBRunResource(
            project_name="KUL_IDUB_EcclesiaSchematisms",
            run_name=f"dagster_eval_{datetime.now().isoformat()}",
            mode="online",
        ),
        # ML Engines
        "ocr_engine": OCREngineResource(),
        "lmv3_engine": LMv3EngineResource(ocr_engine=OCREngineResource()),
        "llm_engine": LLMEngineResource(),
    },
    executor=in_process_executor,
)
