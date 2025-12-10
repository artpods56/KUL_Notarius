"""
Inbound Port: Extraction service interface

This defines what the application offers for schematism extraction
"""

from abc import ABC, abstractmethod
from PIL import Image

from notarius.domain.entities.schematism import SchematismPage


class ExtractionService(ABC):
    """
    PRIMARY PORT: Extraction service interface

    This defines what the application offers for schematism extraction
    """

    @abstractmethod
    async def extract_from_image(
        self, image: Image.Image, context: dict | None = None
    ) -> SchematismPage:
        """Extract structured data from a schematism page image"""
        pass
