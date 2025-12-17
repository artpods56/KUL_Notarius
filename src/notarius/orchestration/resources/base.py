from __future__ import annotations
from dagster import ConfigurableResource
from typing import ClassVar, cast, override
from contextlib import contextmanager
from pathlib import Path


import pandas as pd
import wandb
import weave
from PIL import Image
from pydantic import PrivateAttr

from notarius.infrastructure.cache.storage import get_image_hash
from notarius.infrastructure.config.manager import config_manager
from notarius.shared.constants import PDF_SOURCE_DIR

from notarius.infrastructure.config.constants import (
    ConfigType,
    ModelsConfigSubtype,
)
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


class WeaveResource(ConfigurableResource):  # pyright: ignore[reportMissingTypeArgument]
    def init_weave(self, run_name: str):
        return weave.init(run_name)


class ImageStorageResource(
    ConfigurableResource  # pyright: ignore[reportMissingTypeArgument]
):
    image_storage_path: str
    storage_root: str

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

    _engine_config: LLMEngineConfig | None = PrivateAttr(default=None)

    @override
    def setup_for_execution(self, context):
        """Initialize the LLM _engine."""
        self._engine_config = cast(
            LLMEngineConfig,
            config_manager.load_config_as_model(
                config_name="llm_model_config",
                config_type=ConfigType.MODELS,
                config_subtype=ModelsConfigSubtype.LLM,
            ),
        )

    def get_engine_config(self) -> LLMEngineConfig:
        """Get the LLM _engine config."""
        if self._engine_config is None:
            raise RuntimeError("LLMEngineConfig not initialized.")
        return self._engine_config

    def get_engine(self) -> LLMEngine:
        """Get the LLM _engine instance."""
        return LLMEngine.from_config(config=self.get_engine_config().model_copy())
