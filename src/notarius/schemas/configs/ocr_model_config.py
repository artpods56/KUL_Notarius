from pydantic import BaseModel, Field

from notarius.infrastructure.config.registry import register_config
from notarius.infrastructure.config.constants import ConfigType, ModelsConfigSubtype


@register_config(ConfigType.MODELS, ModelsConfigSubtype.OCR)
class PytesseractOCRConfig(BaseModel):
    """Configuration for OCR model using PyTesseract."""

    language: str = Field(
        default="lat+pol+rus",
        description="Languages passed to Tesseract (e.g. 'eng+deu', 'lat+pol+rus')",
    )
    enable_cache: bool = Field(default=True, description="Enable caching of OCR sample")
    psm_mode: int = Field(
        default=6, description="Page segmentation mode for Tesseract (--psm parameter)"
    )
    oem_mode: int = Field(
        default=3, description="OCR Engine Mode for Tesseract (--oem parameter)"
    )

    @property
    def tesseract_config(self) -> str:
        return f"--psm {self.psm_mode} --oem {self.oem_mode}"
