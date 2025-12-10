from pathlib import Path
from typing import Literal

from pydantic_settings import SettingsConfigDict, BaseSettings

from notarius.shared.constants import REPOSITORY_ROOT


class AppConfig(BaseSettings):
    model_config = SettingsConfigDict(env_file=REPOSITORY_ROOT / ".env", extra="allow")

    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
    logs_dir: Path


app_config = AppConfig()  # pyright: ignore[reportCallIssue]
