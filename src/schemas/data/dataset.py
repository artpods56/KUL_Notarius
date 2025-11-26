from pydantic import BaseModel, Field


class ETLSpecificDatasetFields(BaseModel):
    sample_id: str = Field(
        default="sample_id", description="Sample id of the lmv3_dataset."
    )
    ground_truth_source: str = Field(
        default="parsed", description="Ground truth source of the dataset."
    )


class BaseHuggingFaceDatasetSchema(BaseModel):
    image: str = Field(default="image", description="Image used for prediction.")
    source: str = Field(default="source", description="Ground truth data.")
    parsed: str = Field(default="parsed", description="Source truth data.")
    schematism_name: str = Field(
        default="schematism_name", description="Name of schematism."
    )
    filename: str = Field(default="filename", description="Name of file.")
