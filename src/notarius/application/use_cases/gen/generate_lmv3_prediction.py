"""Use case for generating LMv3 predictions with caching."""

from PIL.Image import Image as PILImage
from structlog import get_logger

from notarius.domain.entities.schematism import SchematismPage
from notarius.domain.services.bio_processing_service import BIOProcessingService
from notarius.infrastructure.ml_models.lmv3.engine import SimpleLMv3Engine
from notarius.infrastructure.persistence.lmv3_cache_repository import (
    LMv3CacheRepository,
)
from notarius.schemas.data.cache import LMv3CacheItem

logger = get_logger(__name__)


class GenerateLMv3PredictionRequest:
    """Request object for LMv3 prediction generation."""

    def __init__(
        self,
        image: PILImage,
        words: list[str],
        bboxes: list,
        raw_predictions: bool = False,
        invalidate_cache: bool = False,
        schematism: str | None = None,
        filename: str | None = None,
    ):
        """Initialize request.

        Args:
            image: PIL Image for document understanding
            words: OCR words from image
            bboxes: Bounding boxes corresponding to words
            raw_predictions: Whether to return only raw predictions (no BIO processing)
            invalidate_cache: Whether to invalidate existing cache entry
            schematism: Optional schematism name for cache tagging
            filename: Optional filename for cache tagging
        """
        self.image = image
        self.words = words
        self.bboxes = bboxes
        self.raw_predictions = raw_predictions
        self.invalidate_cache = invalidate_cache
        self.schematism = schematism
        self.filename = filename


class GenerateLMv3PredictionResponse:
    """Response object for LMv3 prediction generation."""

    def __init__(
        self,
        prediction: SchematismPage | tuple[list[str], list, list[str]],
        from_cache: bool = False,
    ):
        """Initialize output.

        Args:
            prediction: Either SchematismPage (structured) or tuple (raw predictions)
            from_cache: Whether result came from cache
        """
        self.prediction = prediction
        self.from_cache = from_cache


class GenerateLMv3Prediction:
    """Use case for generating LMv3 predictions with caching support.

    This use case orchestrates:
    1. Cache lookup (if enabled)
    2. LMv3 prediction (if cache miss)
    3. BIO post-processing (if structured predictions requested)
    4. Cache storage
    """

    def __init__(
        self,
        engine: SimpleLMv3Engine,
        cache_repository: LMv3CacheRepository | None,
        bio_service: BIOProcessingService,
    ):
        """Initialize the use case.

        Args:
            engine: Simplified LMv3 _engine for predictions
            cache_repository: Optional cache repository (None disables caching)
            bio_service: Service for BIO label processing
        """
        self.engine = engine
        self.cache_repository = cache_repository
        self.bio_service = bio_service
        self.logger = logger

    def execute(
        self, request: GenerateLMv3PredictionRequest
    ) -> GenerateLMv3PredictionResponse:
        """Execute the LMv3 prediction use case.

        Args:
            request: Request with prediction parameters

        Returns:
            Response with prediction and metadata
        """
        # Try cache if enabled
        if self.cache_repository is not None:
            cache_key = self.cache_repository.generate_key(
                image=request.image,
                raw_predictions=request.raw_predictions,
            )

            # Invalidate if requested
            if request.invalidate_cache:
                self.cache_repository.delete(cache_key)
                self.logger.debug("Cache invalidated", key=cache_key[:16])

            # Check cache
            cache_item = self.cache_repository.get(cache_key)
            if cache_item is not None:
                self.logger.info("Returning cached prediction")

                # Return appropriate prediction format
                if request.raw_predictions:
                    return GenerateLMv3PredictionResponse(
                        prediction=cache_item.raw_predictions,
                        from_cache=True,
                    )
                else:
                    return GenerateLMv3PredictionResponse(
                        prediction=cache_item.structured_predictions,
                        from_cache=True,
                    )

        # Generate raw predictions
        self.logger.info("Generating new prediction", num_words=len(request.words))
        words, pred_bboxes, predictions = self.engine.predict(
            image=request.image,
            words=request.words,
            bboxes=request.bboxes,
        )

        # Store raw predictions
        raw_predictions = (words, pred_bboxes, predictions)

        # Process into structured output if needed
        if not request.raw_predictions:
            structured_predictions = self.bio_service.process(
                words=words,
                bboxes=pred_bboxes,
                predictions=predictions,
            )
        else:
            # For raw predictions, create empty structured output for cache
            structured_predictions = SchematismPage(entries=[])

        # Store in cache if enabled
        if self.cache_repository is not None:
            cache_item = LMv3CacheItem(
                raw_predictions=raw_predictions,
                structured_predictions=structured_predictions,
            )
            self.cache_repository.set(
                key=cache_key,
                item=cache_item,
                schematism=request.schematism,
                filename=request.filename,
            )
            self.logger.debug("Prediction cached", key=cache_key[:16])

        # Return appropriate format
        if request.raw_predictions:
            return GenerateLMv3PredictionResponse(
                prediction=raw_predictions,
                from_cache=False,
            )
        else:
            return GenerateLMv3PredictionResponse(
                prediction=structured_predictions,
                from_cache=False,
            )
