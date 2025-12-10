"""
Inbound Port: Parsing service interface

This defines what the application offers for data parsing
"""

from abc import ABC, abstractmethod

from notarius.domain.entities.schematism import SchematismPage


class ParsingService(ABC):
    """
    PRIMARY PORT: Parsing service interface

    This defines what the application offers for data parsing
    """

    @abstractmethod
    async def parse_page(self, raw_page: SchematismPage) -> SchematismPage:
        """Parse and normalize schematism page data"""
        pass
