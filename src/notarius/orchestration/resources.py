from __future__ import annotations
from dagster import ConfigurableResource
from typing import Any, Callable, Literal, ClassVar, cast
from contextlib import contextmanager
from pathlib import Path

from typing import Any

import pandas as pd
import wandb
from PIL import Image
from dagster import ConfigurableResource
from omegaconf import DictConfig
from pydantic import PrivateAttr

from notarius.infrastructure.cache.storage import get_image_hash
from notarius.infrastructure.config.manager import config_manager
from notarius.schemas.configs.dataset_config import BaseDatasetConfig
from notarius.schemas.data.pipeline import BaseDataset
from notarius.shared.constants import PDF_SOURCE_DIR

from notarius.infrastructure.config.constants import (
    ConfigType,
    ConfigSubTypes,
    ModelsConfigSubtype,
)
from notarius.infrastructure.config.manager import get_config_manager
from notarius.infrastructure.llm.engine_adapter import LLMEngine
from notarius.infrastructure.ml_models.lmv3.engine_adapter import LMv3Engine
from notarius.infrastructure.ocr.engine_adapter import OCREngine
from notarius.schemas.configs import (
    LLMEngineConfig,
    BaseLMv3ModelConfig,
    PytesseractOCRConfig,
)


class PdfFilesResource(
    ConfigurableResource  # pyright: ignore[reportMissingTypeArgument]
):
    pdf_dir: str = str(PDF_SOURCE_DIR)

    def get_pdf_paths(self) -> list[Path]:
        return list(Path(self.pdf_dir).glob("**/*.pdf"))


OpType = Literal["filter", "map"]
OpRegistryType = dict[tuple[OpType, str], Callable[[Any], Any]]


class OpRegistry(ConfigurableResource):  # pyright: ignore[reportMissingTypeArgument]
    """
    Registry for operations using class-level storage.

    Note: _ops_registry is a ClassVar, not a Pydantic field,
    so it's shared across all instances and not serialized.
    """

    # ✅ Use ClassVar to tell Pydantic this is NOT a model field
    _ops_registry: ClassVar[OpRegistryType] = {}

    @classmethod
    def register(cls, op_type: OpType, name: str):
        """
        Decorator to register an operation.

        Args:
            op_type: Type of operation ('filter' or 'map')
            name: Unique name for the operation

        Raises:
            RuntimeError: If operation already registered
        """

        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            op_key = (op_type, name)

            if op_key in cls._ops_registry:
                raise RuntimeError(
                    f"Operation '{name}' of type '{op_type}' already registered"
                )

            cls._ops_registry[op_key] = func

            # Return the original function unchanged
            return func

        return decorator

    def get(self, op_type: OpType, name: str):
        """Retrieve a registered operation (instance method for Dagster compatibility)."""
        return self._ops_registry.get((op_type, name), None)

    @classmethod
    def get_op(cls, op_type: OpType, name: str):
        """Retrieve a registered operation (class method alternative)."""
        return cls._ops_registry.get((op_type, name))

    @classmethod
    def list_operations(cls, op_type: OpType | None = None) -> list[tuple[OpType, str]]:
        """List all registered operations."""
        if op_type is None:
            return list(cls._ops_registry.keys())
        return [key for key in cls._ops_registry.keys() if key[0] == op_type]

    @classmethod
    def clear(cls) -> None:
        """Clear all registered operations. Useful for testing."""
        cls._ops_registry.clear()


class ExcelWriterResource(
    ConfigurableResource  # pyright: ignore[reportMissingTypeArgument]
):
    writing_path: str

    @contextmanager
    def get_writer(self, file_name: str):
        writer_path = Path(self.writing_path) / Path(file_name)

        writer = pd.ExcelWriter(writer_path)

        try:
            yield writer
        finally:
            writer.close()


class WandBRunResource(
    ConfigurableResource  # pyright: ignore[reportMissingTypeArgument]
):
    run_name: str
    project_name: str
    mode: str = "online"

    _wandb_run: ClassVar[wandb.Run | None] = None

    def get_wandb_run(self) -> wandb.Run:
        if WandBRunResource._wandb_run is None:
            WandBRunResource._wandb_run = wandb.init(
                project=self.project_name, name=self.run_name, mode=self.mode
            )
        return WandBRunResource._wandb_run


class ImageStorageResource(
    ConfigurableResource  # pyright: ignore[reportMissingTypeArgument]
):
    image_storage_path: str

    def save_image(self, image: Image.Image) -> str:
        image_hash = get_image_hash(image)
        file_name = Path(image_hash).with_suffix(".jpeg")

        storage_dir = Path(self.image_storage_path)
        storage_dir.mkdir(parents=True, exist_ok=True)

        file_path = Path(self.image_storage_path) / file_name

        if file_path.exists():
            return str(file_path)
        else:
            image.save(file_path)
            return str(file_path)

    def load_image(self, file_path: str) -> Image.Image:
        if not Path(file_path).exists():
            raise FileNotFoundError(f"File '{file_path}' does not exist")
        return Image.open(file_path)


class OCREngineResource(
    ConfigurableResource  # pyright: ignore[reportMissingTypeArgument]
):
    """OCR _engine resource for text and structured extraction."""

    _engine: OCREngine | None = PrivateAttr(default=None)

    def setup_for_execution(self, context):
        """Initialize the OCR _engine."""
        ocr_config = cast(
            PytesseractOCRConfig,
            config_manager.load_config_as_model(
                config_name="ocr_model_config",
                config_type=ConfigType.MODELS,
                config_subtype=ModelsConfigSubtype.OCR,
            ),
        )
        self._engine = OCREngine.from_config(config=ocr_config)

    def get_engine(self) -> OCREngine:
        """Get the OCR _engine instance."""
        if self._engine is None:
            raise RuntimeError(
                "OCREngine not initialized. Call setup_for_execution first."
            )
        return self._engine


class LMv3EngineResource(
    ConfigurableResource  # pyright: ignore[reportMissingTypeArgument]
):
    """LayoutLMv3 _engine resource for document understanding."""

    ocr_engine: OCREngineResource
    _engine: LMv3Engine | None = PrivateAttr(default=None)

    def setup_for_execution(self, context):
        """Initialize the LMv3 _engine."""
        lmv3_config = cast(
            BaseLMv3ModelConfig,
            config_manager.load_config_as_model(
                config_name="lmv3_model_config",
                config_type=ConfigType.MODELS,
                config_subtype=ModelsConfigSubtype.LMV3,
            ),
        )
        # LMv3 depends on OCR _engine
        ocr_engine_instance = self.ocr_engine.get_engine()
        self._engine = LMv3Engine.from_config(
            config=lmv3_config, ocr_engine=ocr_engine_instance
        )

    def get_engine(self) -> LMv3Engine:
        """Get the LMv3 _engine instance."""
        if self._engine is None:
            raise RuntimeError(
                "LMv3Engine not initialized. Call setup_for_execution first."
            )
        return self._engine


class LLMEngineResource(
    ConfigurableResource  # pyright: ignore[reportMissingTypeArgument]
):
    """LLM _engine resource for language model operations."""

    _engine: LLMEngine | None = PrivateAttr(default=None)

    def setup_for_execution(self, context):
        """Initialize the LLM _engine."""
        llm_config = cast(
            LLMEngineConfig,
            config_manager.load_config_as_model(
                config_name="llm_model_config",
                config_type=ConfigType.MODELS,
                config_subtype=ModelsConfigSubtype.LLM,
            ),
        )
        self._engine = LLMEngine.from_config(config=llm_config)

    def get_engine(self) -> LLMEngine:
        """Get the LLM _engine instance."""
        if self._engine is None:
            raise RuntimeError(
                "LLMEngine not initialized. Call setup_for_execution first."
            )
        return self._engine
