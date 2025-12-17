import dagster as dg
import weave
from dagster import in_process_executor

from notarius.orchestration.resources.storage import (
    LocalStorageResource,
    ImageRepositoryResource,
)
from notarius.orchestration.jobs.ingestion import ingestion_job, ingestion_assets
from notarius.config import app_config

from dotenv import load_dotenv

_ = load_dotenv()

weave.init("KUL_IDUB_EcclesiaSchematisms")


storage = LocalStorageResource(storage_root=str(app_config.storage_root))


defs = dg.Definitions(
    assets=[*ingestion_assets],
    jobs=[ingestion_job],
    resources={
        "file_storage": storage,
        "images_repository": ImageRepositoryResource(storage_resource=storage),
    },
    executor=in_process_executor,
)
