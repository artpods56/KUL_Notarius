import logging
import sys
from typing import TYPE_CHECKING, Any

import structlog

from notarius.config import app_config

DEFAULT_LEVEL = logging.INFO


def setup_logging() -> None:
    logs_dir = app_config.logs_dir
    logs_dir.mkdir(parents=True, exist_ok=True)

    foreign_pre_chain = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.TimeStamper(fmt="%Y-%m-%d %H:%M:%S.%f", utc=True),
        structlog.processors.add_log_level,
        structlog.stdlib.add_logger_name,
    ]

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(
        structlog.stdlib.ProcessorFormatter(
            processor=structlog.dev.ConsoleRenderer(colors=True),
            foreign_pre_chain=foreign_pre_chain,
        )
    )

    file_handler = logging.FileHandler(logs_dir / "app.log", encoding="utf-8")
    file_handler.setFormatter(
        structlog.stdlib.ProcessorFormatter(
            processor=structlog.processors.JSONRenderer(),
            foreign_pre_chain=foreign_pre_chain,
        )
    )

    logging.basicConfig(
        level=app_config.log_level,
        handlers=[console_handler, file_handler],
    )

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.stdlib.filter_by_level,
            structlog.processors.add_log_level,
            structlog.stdlib.add_logger_name,
            structlog.processors.TimeStamper(fmt="%Y-%m-%d %H:%M:%S.%f", utc=True),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(app_config.log_level),
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )


# proxy type to the structlog bound logger that satisfies beartype
type Logger = Any
if TYPE_CHECKING:
    from structlog.stdlib import BoundLogger

    type Logger = BoundLogger
