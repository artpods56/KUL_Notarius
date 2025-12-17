from dataclasses import dataclass
from typing import override, final, cast

import numpy as np
import torch
from PIL.Image import Image
from structlog import get_logger
from transformers import (
    LayoutLMv3ForTokenClassification,
    LayoutLMv3Processor,
)

from notarius.domain.entities.schematism import SchematismPage
from notarius.domain.protocols import BaseRequest, BaseResponse
from notarius.application.ports.outbound.engine import ConfigurableEngine, track_stats
from notarius.infrastructure.ocr import StructuredOCRResult
from notarius.infrastructure.ocr.engine_adapter import OCREngine, OCRRequest
from notarius.schemas.configs import BaseLMv3ModelConfig
from notarius.infrastructure.ml_models.lmv3.utils import (
    sliding,
    repair_bio_labels,
    build_page_json,
)
from notarius.schemas.data.structs import BBox
from notarius.infrastructure.ml_models.lmv3.types import EncodingProtocol

logger = get_logger(__name__)


@dataclass(frozen=True)
class LMv3Request(BaseRequest):
    input: Image

    def __post_init__(self):
        # check if image is 3 dims
        if np.array(self.input).ndim != 3:
            raise ValueError(
                "Only RGB images are supported. Ensure the input image has 3 dimensions."
            )


@dataclass(frozen=True)
class LMv3Response(BaseResponse):
    output: SchematismPage


@final
class LMv3Engine(ConfigurableEngine[BaseLMv3ModelConfig, LMv3Request, LMv3Response]):
    """LayoutLMv3 model wrapper with unified predict interface and caching."""

    def __init__(self, config: BaseLMv3ModelConfig, ocr_engine: OCREngine):
        self._init_stats()
        self.config = config
        self.processor = LayoutLMv3Processor.from_pretrained(
            config.processor.checkpoint,
            local_files_only=True if config.processor.local_files_only else False,
            apply_ocr=self.config.apply_ocr,
        )
        self.model = LayoutLMv3ForTokenClassification.from_pretrained(
            config.inference.checkpoint,
            local_files_only=True if config.inference.local_files_only else False,
            device_map="auto",
        )
        self.ocr_engine = ocr_engine

    @classmethod
    @override
    def from_config(
        cls, config: BaseLMv3ModelConfig, ocr_engine: OCREngine
    ) -> "LMv3Engine":
        return cls(config=config, ocr_engine=ocr_engine)

    @property
    def id2label(self) -> dict[int, str]:
        mapping = self.model.config.id2label
        if mapping is None:
            raise ValueError("Model doesn't include id2label mapping.")
        return mapping

    @property
    def label2id(self) -> dict[str, int]:
        mapping = self.model.config.label2id
        if mapping is None:
            raise ValueError("Model doesn't include label2id mapping.")
        return mapping

    def _inference(
        self,
        image: Image,
        words: list[str] | None,
        boxes: list[BBox] | None,
    ) -> tuple[list[BBox], list[int], list[str]]:
        """Retrieve predictions for a single image.

        Args:
            image: PIL Image to process

        Returns:
            Tuple of (bboxes, prediction_ids, words)
        """

        width, height = image.size

        # Cast to our Protocol type for better type inference
        encoding = cast(  # pyright: ignore[reportInvalidCast]
            EncodingProtocol,
            self.processor(
                image,  # First positional argument
                text=words,  # Changed from 'words' to 'text'
                boxes=boxes,  # pyright: ignore[reportArgumentType]
                truncation=True,
                stride=128,
                padding="max_length",
                max_length=512,
                return_overflowing_tokens=True,
                return_offsets_mapping=True,
                return_tensors="pt",
            ),
        )

        encoding.pop("offset_mapping")
        encoding.pop("overflow_to_sample_mapping")

        # Process pixel values
        pixel_values = cast(torch.Tensor, encoding["pixel_values"])
        x: list[torch.Tensor] = []
        for i in range(0, len(pixel_values)):
            tensor_pixel_values = pixel_values[i].clone().detach()
            x.append(tensor_pixel_values)

        stacked_pixels = torch.stack(x)
        encoding["pixel_values"] = stacked_pixels

        # Move all input tensors to the same device as the model
        device = next(self.model.parameters()).device
        for k, v in encoding.items():
            if isinstance(v, torch.Tensor):
                encoding[k] = v.clone().detach().to(device)

        with torch.no_grad():
            outputs = self.model(**encoding)

        # Extract predictions from model output
        logits = cast(torch.Tensor, outputs.logits)
        predictions_tensor = logits.argmax(-1).squeeze()
        predictions_raw = predictions_tensor.tolist()

        # Extract bounding boxes
        bbox_tensor = encoding.bbox.squeeze()
        token_boxes_raw = bbox_tensor.tolist()

        # Handle single window case (when no sliding was needed)
        # After squeeze(), we get either:
        #   - Single window: list[int] (for predictions) and list[list[int]] (for token_boxes)
        #   - Multiple windows: list[list[int]] (for predictions) and list[list[list[int]]] (for token_boxes)
        # The check `len(token_boxes_raw) == 512` indicates single window (512 tokens)
        if isinstance(token_boxes_raw, list) and len(token_boxes_raw) == 512:
            # Single window case - wrap both in lists to match expected format
            predictions: list[list[int]] = [cast(list[int], predictions_raw)]
            token_boxes: list[list[list[int]]] = [
                cast(list[list[int]], token_boxes_raw)
            ]
        else:
            # Multiple windows case
            predictions = cast(list[list[int]], predictions_raw)
            token_boxes = cast(list[list[list[int]]], token_boxes_raw)

        bboxes, preds, flattened_words = sliding(
            self.processor,
            token_boxes,
            predictions,
            encoding,
            width,
            height,
            self.id2label,
        )

        return bboxes, preds, flattened_words

    @override
    @track_stats
    def process(self, request: LMv3Request) -> LMv3Response:
        """Process an image and return structured predictions.

        Args:
            request: LMv3Request containing the input image

        Returns:
            LMv3Response with structured SchematismPage output
        """

        ocr_request = OCRRequest(input=request.input, mode="structured")
        ocr_response_output = self.ocr_engine.process(ocr_request).output

        words = None
        bboxes = None
        if isinstance(ocr_response_output, StructuredOCRResult):
            words = ocr_response_output.words
            bboxes = ocr_response_output.bboxes

        bboxes, preds, words = self._inference(
            image=request.input,
            words=words,
            boxes=bboxes,
        )

        # Convert prediction IDs to labels
        predictions = [self.id2label[p] for p in preds]

        # Repair BIO label sequences
        repaired_predictions = repair_bio_labels(predictions)

        # Build structured JSON
        page_json = build_page_json(
            words=words, bboxes=bboxes, labels=repaired_predictions
        )

        # Convert to domain entity
        page = SchematismPage(**page_json)

        return LMv3Response(output=page)
