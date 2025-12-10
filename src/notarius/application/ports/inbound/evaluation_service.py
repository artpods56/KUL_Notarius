"""
Inbound Port: Evaluation service interface

This defines what the application offers for quality evaluation
"""

from abc import ABC, abstractmethod

from notarius.domain.entities.schematism import SchematismPage
from notarius.schemas.data.metrics import PageDataMetrics


class EvaluationService(ABC):
    """
    PRIMARY PORT: Evaluation service interface

    This defines what the application offers for quality evaluation
    """

    @abstractmethod
    async def evaluate(
        self, predictions: SchematismPage, ground_truth: SchematismPage
    ) -> PageDataMetrics:
        """Evaluate extraction quality"""
        pass
