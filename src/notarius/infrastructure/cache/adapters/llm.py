"""LLM cache adapter using pickle for automatic serialization."""

import pickle
from pathlib import Path
from typing import final, override, cast

from pydantic import BaseModel
from structlog import get_logger

from notarius.application.ports.outbound.cache import BaseCache
from notarius.infrastructure.llm.engine_adapter import CompletionResult
from notarius.infrastructure.llm.utils import parse_model_name
from notarius.shared.logger import Logger

logger: Logger = get_logger(__name__)


@final
class LLMCache(BaseCache[CompletionResult[BaseModel]]):
    """Type-safe cache for LLM responses using pickle serialization.

    Pickle automatically handles the complex nested structure of:
    - CompletionResult (dataclass)
    - BaseProviderResponse (dataclass)
    - Conversation (dataclass)
    - Pydantic models (BaseModel)

    No manual serialization/deserialization needed!
    """

    _item_type = CompletionResult

    def __init__(self, model_name: str, caches_dir: Path | None = None):
        super().__init__(cache_name=parse_model_name(model_name), caches_dir=caches_dir)

    @property
    @override
    def cache_type(self):
        return "LLMCache"

    @override
    def get(self, key: str) -> CompletionResult[BaseModel] | None:
        """Retrieve CompletionResult from cache.

        Args:
            key: Cache key

        Returns:
            CompletionResult if found, None otherwise
        """
        try:
            raw_data = self.cache.get(key)
            if raw_data is None:
                return None

            # Diskcache with Disk backend handles unpickling automatically
            return cast(CompletionResult[BaseModel], raw_data)

        except (pickle.PickleError, AttributeError, ImportError) as e:
            logger.warning(
                "cache_deserialization_failed",
                key=key,
                error=str(e),
                error_type=type(e).__name__,
            )
            return None

    @override
    def set[T: BaseModel](
        self,
        key: str,
        value: CompletionResult[T],
    ) -> bool:
        try:
            return self.cache.set(key, value)
        except (pickle.PickleError, TypeError) as e:
            logger.error(
                "cache_serialization_failed",
                key=key,
                error=str(e),
            )
            return False
