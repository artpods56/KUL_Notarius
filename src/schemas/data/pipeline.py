from typing import Any, Dict, Literal, Generic, TypeVar

from PIL import Image
from pydantic import BaseModel, Field

from schemas.data.metrics import PageDataMetrics
from schemas.data.schematism import SchematismPage, SchematismEntry

PageDataSourceField = Literal[
    "ground_truth",
    "source_ground_truth",
    "lmv3_prediction",
    "llm_prediction",
    "parsed_prediction",
]


class BaseMetaData(BaseModel):
    sample_id: int = Field(description="Sample ID")
    schematism_name: str = Field(description="Schematism name")
    filename: str = Field(description="Schematism filename")


class BaseDataItem(BaseModel):
    image_path: str | None = Field(description="Path to the saved image.")
    text: str | None = Field(default=None, description="OCR text extracted from image.")

    metadata: BaseMetaData | None = Field(
        default=None, description="Metadata of the lmv3_dataset item"
    )

    class Config:
        arbitrary_types_allowed = True



class HasGroundTruthMixin(BaseModel):
    ground_truth: SchematismPage = Field(description="Ground truth page")


class HasPredictionsMixin(BaseModel):
    predictions: SchematismPage = Field(description="Prediction page")


class HasAlignedPagesMixin(BaseModel):
    aligned_schematism_pages: tuple[SchematismPage, SchematismPage] = Field(
        description="Tuple of aligned predictions and ground truth"
    )


class GroundTruthDataItem(BaseDataItem, HasGroundTruthMixin):
    pass


class PredictionDataItem(BaseDataItem, HasPredictionsMixin):
    pass


class GtAlignedPredictionDataItem(BaseDataItem, HasAlignedPagesMixin):
    pass


class EvaluationDataItem(BaseDataItem, HasGroundTruthMixin):
    pass


# TypeVar for generic BaseDataset - using older syntax for pickle compatibility
ItemT = TypeVar('ItemT', bound=BaseDataItem)


class BaseDataset(BaseModel, Generic[ItemT]):
    items: list[ItemT] = Field(description="List of items")


class PipelineData(BaseModel):
    """
    Pipeline data with required fields for ingestion and optional fields for pipeline stages
    """

    # Required fields - available at ingestion time
    image: Image.Image | None = Field(default=None, description="Image used for prediction.")
    ground_truth: SchematismPage | None = Field(
        default=None, description="Ground truth data."
    )
    source_ground_truth: SchematismPage | None = Field(
        default=None, description="Source truth data."
    )

    # Optional fields - populated during pipeline processing
    text: str | None = Field(default=None, description="OCR text extracted from image.")
    language: str | None = Field(default=None, description="Detected language code.")
    language_confidence: float | None = Field(
        default=None, description="Confidence score for language detection."
    )

    # Model prediction fields
    lmv3_prediction: SchematismPage | None = Field(
        default=None, description="Predictions from LayoutLMv3 model."
    )
    llm_prediction: SchematismPage | None = Field(
        default=None, description="Predictions from LLM model."
    )
    parsed_prediction: SchematismPage | None = Field(
        default=None, description="Parsed predictions_data from LLM model."
    )
    parsed_messages: str | None = Field(
        default=None, description="Parsed messages from LLM model."
    )

    # Additional metadata
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Metadata for this pipeline."
    )
    evaluation_results: PageDataMetrics | None = Field(
        default=None, description="Results from evaluation step."
    )

    class Config:
        arbitrary_types_allowed = True
