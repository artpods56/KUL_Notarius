import dagster as dg
import weave
from dagster import in_process_executor

from notarius.orchestration.resources.storage import (
    LocalStorageResource,
    ImageRepositoryResource,
)
from notarius.orchestration.resources.base import (
    OCREngineResource,
    LMv3EngineResource,
    LLMEngineResource,
    ExcelWriterResource,
    WandBRunResource,
)
from notarius.shared.constants import OUTPUTS_DIR
from datetime import datetime
from notarius.orchestration.jobs.ingestion import ingestion_job, ingestion_assets
from notarius.orchestration.jobs.prediction import prediction_job, prediction_assets
from notarius.orchestration.jobs.exporting import exporting_job, exporting_assets
from notarius.config import app_config

from dotenv import load_dotenv

_ = load_dotenv()

weave.init("KUL_IDUB_EcclesiaSchematisms")


storage = LocalStorageResource(storage_root=str(app_config.storage_root))


defs = dg.Definitions(
    assets=[*ingestion_assets, *prediction_assets],
    jobs=[ingestion_job, prediction_job],
    resources={
        "file_storage": storage,
        "images_repository": ImageRepositoryResource(storage_resource=storage),
        # ML Engines
        "ocr_engine": OCREngineResource(),
        "lmv3_engine": LMv3EngineResource(ocr_engine=OCREngineResource()),
        "llm_engine_resource": LLMEngineResource(),
        # Export resources
        "excel_writer": ExcelWriterResource(writing_path=str(OUTPUTS_DIR)),
        "wandb_run": WandBRunResource(
            project_name="KUL_IDUB_EcclesiaSchematisms",
            run_name=f"dagster_eval_{datetime.now().isoformat()}",
            mode="online",
        ),
    },
    executor=in_process_executor,
)
