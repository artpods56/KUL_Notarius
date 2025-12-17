"""Inference use cases for enriching datasets with model predictions."""

from notarius.application.use_cases.inference.add_ocr_to_dataset import (
    EnrichDatasetWithOCR,
    EnrichWithOCRRequest,
    EnrichWithOCRResponse,
)
from notarius.application.use_cases.inference.add_lmv3_preds_to_dataset import (
    EnrichDatasetWithLMv3,
    EnrichWithLMv3Request,
    EnrichWithLMv3Response,
)
from notarius.application.use_cases.inference.add_llm_preds_to_dataset import (
    PredictDatasetWithLLM,
    PredictWithLLMRequest,
    PredictWithLLMResponse,
)
from notarius.application.use_cases.inference.add_llm_ocr_to_dataset import (
    EnrichDatasetWithLLMOCR,
    EnrichWithLLMOCRRequest,
    EnrichWithLLMOCRResponse,
)
from notarius.application.use_cases.inference.generate_source_dataset import (
    GenerateSourceDataset,
    GenerateSourceDatasetRequest,
    GenerateSourceDatasetResponse,
)

__all__ = [
    "EnrichDatasetWithOCR",
    "EnrichWithOCRRequest",
    "EnrichWithOCRResponse",
    "EnrichDatasetWithLMv3",
    "EnrichWithLMv3Request",
    "EnrichWithLMv3Response",
    "PredictDatasetWithLLM",
    "PredictWithLLMRequest",
    "PredictWithLLMResponse",
    "EnrichDatasetWithLLMOCR",
    "EnrichWithLLMOCRRequest",
    "EnrichWithLLMOCRResponse",
    "GenerateSourceDataset",
    "GenerateSourceDatasetRequest",
    "GenerateSourceDatasetResponse",
]
