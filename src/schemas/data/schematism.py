from typing import List

from pydantic import BaseModel, Field, ConfigDict


class SchematismEntry(BaseModel):
    """Model for individual parish entry data."""

    model_config = ConfigDict(
        extra="forbid",  # This makes Pydantic add additionalProperties: false
    )

    deanery: str | None = Field(
        default=None, description="Deanery description, null if not on page"
    )
    parish: str | None = Field(default=None, description="Name of the parish")
    dedication: str | None = Field(
        default=None, description="Church dedication/patron saint information"
    )
    building_material: str | None = Field(
        default=None,
        description="Building material (e.g., 'lig.' for wood, 'mur.' for brick/stone)",
    )


class SchematismPage(BaseModel):
    """Model for page data containing parish entries."""

    model_config = ConfigDict(
        extra="forbid",  # This makes Pydantic add additionalProperties: false
    )

    page_number: str | None = Field(None, description="Page number as string")
    entries: List[SchematismEntry] = Field(
        list(), description="List of parish entries on this page"
    )
