"""Tests for EnrichDatasetWithLMv3 use case."""

import pytest
from unittest.mock import MagicMock, patch
from PIL import Image

from notarius.application.use_cases.inference.add_lmv3_preds_to_dataset import (
    EnrichDatasetWithLMv3,
    EnrichWithLMv3Request,
    EnrichWithLMv3Response,
)
from notarius.application.ports.outbound.cached_engine import CachedEngine
from notarius.infrastructure.ml_models.lmv3.engine_adapter import (
    LMv3Engine,
    LMv3Response,
)
from notarius.domain.entities.schematism import SchematismPage, SchematismEntry
from notarius.schemas.data.pipeline import (
    BaseDataset,
    BaseDataItem,
    PredictionDataItem,
    BaseMetaData,
)


# Test fixtures


@pytest.fixture
def mock_image() -> Image.Image:
    """Create a mock PIL image."""
    return Image.new("RGB", (100, 100), color="white")


@pytest.fixture
def mock_schematism_page() -> SchematismPage:
    """Create a mock SchematismPage prediction."""
    return SchematismPage(
        entries=[
            SchematismEntry(
                deanery="Test Deanery",
                parish="Test Parish",
            )
        ]
    )


@pytest.fixture
def mock_lmv3_engine(mock_schematism_page: SchematismPage) -> MagicMock:
    """Create a mock LMv3 engine."""
    engine = MagicMock(spec=LMv3Engine)
    engine.process.return_value = LMv3Response(output=mock_schematism_page)
    return engine


@pytest.fixture
def mock_image_storage(mock_image: Image.Image) -> MagicMock:
    """Create a mock image storage resource."""
    storage = MagicMock()
    storage.load_image.return_value = mock_image
    return storage


@pytest.fixture
def sample_metadata() -> BaseMetaData:
    """Create sample metadata."""
    return BaseMetaData(
        sample_id=1,
        schematism_name="test_schematism",
        filename="test_file.jpg",
    )


@pytest.fixture
def sample_dataset(sample_metadata: BaseMetaData) -> BaseDataset[BaseDataItem]:
    """Create a sample dataset with items."""
    items = [
        BaseDataItem(
            image_path="/path/to/image1.jpg",
            text="Sample OCR text 1",
            metadata=sample_metadata,
        ),
        BaseDataItem(
            image_path="/path/to/image2.jpg",
            text="Sample OCR text 2",
            metadata=BaseMetaData(
                sample_id=2,
                schematism_name="test_schematism",
                filename="test_file2.jpg",
            ),
        ),
    ]
    return BaseDataset[BaseDataItem](items=items)


@pytest.fixture
def empty_dataset() -> BaseDataset[BaseDataItem]:
    """Create an empty dataset."""
    return BaseDataset[BaseDataItem](items=[])


@pytest.fixture
def dataset_with_missing_paths(
    sample_metadata: BaseMetaData,
) -> BaseDataset[BaseDataItem]:
    """Create a dataset with items missing image paths."""
    items = [
        BaseDataItem(image_path=None, text="text", metadata=sample_metadata),
        BaseDataItem(
            image_path="/path/to/image.jpg", text="text", metadata=sample_metadata
        ),
    ]
    return BaseDataset[BaseDataItem](items=items)


class TestEnrichWithLMv3Request:
    """Test suite for EnrichWithLMv3Request dataclass."""

    def test_request_creation(self, sample_dataset: BaseDataset[BaseDataItem]) -> None:
        """Test request creation."""
        request = EnrichWithLMv3Request(dataset=sample_dataset)

        assert request.dataset is sample_dataset


class TestEnrichWithLMv3Response:
    """Test suite for EnrichWithLMv3Response dataclass."""

    def test_response_creation(
        self, mock_schematism_page: SchematismPage, sample_metadata: BaseMetaData
    ) -> None:
        """Test response creation with all fields."""
        prediction_dataset = BaseDataset[PredictionDataItem](
            items=[
                PredictionDataItem(
                    image_path="/path/image.jpg",
                    text="text",
                    predictions=mock_schematism_page,
                    metadata=sample_metadata,
                )
            ]
        )
        response = EnrichWithLMv3Response(
            dataset=prediction_dataset,
            lmv3_executions=5,
            cache_hits=3,
        )

        assert response.dataset is prediction_dataset
        assert response.lmv3_executions == 5
        assert response.cache_hits == 3


class TestEnrichDatasetWithLMv3:
    """Test suite for EnrichDatasetWithLMv3 use case."""

    def test_init_with_cache_disabled(
        self,
        mock_lmv3_engine: MagicMock,
        mock_image_storage: MagicMock,
    ) -> None:
        """Test initialization with caching disabled."""
        use_case = EnrichDatasetWithLMv3(
            lmv3_engine=mock_lmv3_engine,
            image_storage=mock_image_storage,
            checkpoint="test-checkpoint",
            enable_cache=False,
        )

        # When cache is disabled, the engine should be used directly
        assert use_case.lmv3_engine is mock_lmv3_engine
        assert use_case.image_storage is mock_image_storage

    @patch(
        "notarius.application.use_cases.inference.add_lmv3_preds_to_dataset.create_lmv3_cache_backend"
    )
    def test_init_with_cache_enabled(
        self,
        mock_create_cache: MagicMock,
        mock_lmv3_engine: MagicMock,
        mock_image_storage: MagicMock,
    ) -> None:
        """Test initialization with caching enabled."""
        mock_backend = MagicMock()
        mock_keygen = MagicMock()
        mock_create_cache.return_value = (mock_backend, mock_keygen)

        use_case = EnrichDatasetWithLMv3(
            lmv3_engine=mock_lmv3_engine,
            image_storage=mock_image_storage,
            checkpoint="test-checkpoint",
            enable_cache=True,
        )

        # When cache is enabled, the engine should be wrapped
        assert isinstance(use_case.lmv3_engine, CachedEngine)
        mock_create_cache.assert_called_once_with("test-checkpoint")

    async def test_execute_processes_all_items(
        self,
        mock_lmv3_engine: MagicMock,
        mock_image_storage: MagicMock,
        sample_dataset: BaseDataset[BaseDataItem],
    ) -> None:
        """Test that execute processes all dataset items."""
        use_case = EnrichDatasetWithLMv3(
            lmv3_engine=mock_lmv3_engine,
            image_storage=mock_image_storage,
            checkpoint="test-checkpoint",
            enable_cache=False,
        )

        request = EnrichWithLMv3Request(dataset=sample_dataset)
        response = await use_case.execute(request)

        assert len(response.dataset.items) == 2
        assert mock_lmv3_engine.process.call_count == 2
        assert mock_image_storage.load_image.call_count == 2

    async def test_execute_creates_prediction_data_items(
        self,
        mock_lmv3_engine: MagicMock,
        mock_image_storage: MagicMock,
        mock_schematism_page: SchematismPage,
        sample_dataset: BaseDataset[BaseDataItem],
    ) -> None:
        """Test that execute creates PredictionDataItem instances."""
        use_case = EnrichDatasetWithLMv3(
            lmv3_engine=mock_lmv3_engine,
            image_storage=mock_image_storage,
            checkpoint="test-checkpoint",
            enable_cache=False,
        )

        request = EnrichWithLMv3Request(dataset=sample_dataset)
        response = await use_case.execute(request)

        # All items should be PredictionDataItem with predictions
        for item in response.dataset.items:
            assert isinstance(item, PredictionDataItem)
            assert item.predictions == mock_schematism_page

    async def test_execute_preserves_original_data(
        self,
        mock_lmv3_engine: MagicMock,
        mock_image_storage: MagicMock,
        sample_dataset: BaseDataset[BaseDataItem],
    ) -> None:
        """Test that execute preserves original item data."""
        use_case = EnrichDatasetWithLMv3(
            lmv3_engine=mock_lmv3_engine,
            image_storage=mock_image_storage,
            checkpoint="test-checkpoint",
            enable_cache=False,
        )

        request = EnrichWithLMv3Request(dataset=sample_dataset)
        response = await use_case.execute(request)

        # Original data should be preserved
        assert response.dataset.items[0].image_path == "/path/to/image1.jpg"
        assert response.dataset.items[0].text == "Sample OCR text 1"
        assert response.dataset.items[0].metadata.sample_id == 1

        assert response.dataset.items[1].image_path == "/path/to/image2.jpg"
        assert response.dataset.items[1].text == "Sample OCR text 2"
        assert response.dataset.items[1].metadata.sample_id == 2

    async def test_execute_skips_items_without_image_path(
        self,
        mock_lmv3_engine: MagicMock,
        mock_image_storage: MagicMock,
        dataset_with_missing_paths: BaseDataset[BaseDataItem],
    ) -> None:
        """Test that items without image paths are skipped."""
        use_case = EnrichDatasetWithLMv3(
            lmv3_engine=mock_lmv3_engine,
            image_storage=mock_image_storage,
            checkpoint="test-checkpoint",
            enable_cache=False,
        )

        request = EnrichWithLMv3Request(dataset=dataset_with_missing_paths)
        response = await use_case.execute(request)

        # Only one item has an image path
        assert len(response.dataset.items) == 1
        assert mock_lmv3_engine.process.call_count == 1

    async def test_execute_with_empty_dataset(
        self,
        mock_lmv3_engine: MagicMock,
        mock_image_storage: MagicMock,
        empty_dataset: BaseDataset[BaseDataItem],
    ) -> None:
        """Test execution with empty dataset."""
        use_case = EnrichDatasetWithLMv3(
            lmv3_engine=mock_lmv3_engine,
            image_storage=mock_image_storage,
            checkpoint="test-checkpoint",
            enable_cache=False,
        )

        request = EnrichWithLMv3Request(dataset=empty_dataset)
        response = await use_case.execute(request)

        assert len(response.dataset.items) == 0
        assert response.lmv3_executions == 0
        assert response.cache_hits == 0
        mock_lmv3_engine.process.assert_not_called()

    async def test_execute_returns_correct_statistics_without_cache(
        self,
        mock_lmv3_engine: MagicMock,
        mock_image_storage: MagicMock,
        sample_dataset: BaseDataset[BaseDataItem],
    ) -> None:
        """Test that statistics are correct when cache is disabled."""
        use_case = EnrichDatasetWithLMv3(
            lmv3_engine=mock_lmv3_engine,
            image_storage=mock_image_storage,
            checkpoint="test-checkpoint",
            enable_cache=False,
        )

        request = EnrichWithLMv3Request(dataset=sample_dataset)
        response = await use_case.execute(request)

        # Without cache, all executions should be counted as LMv3 executions
        assert response.lmv3_executions == 2
        assert response.cache_hits == 0

    async def test_execute_converts_image_to_rgb(
        self,
        mock_lmv3_engine: MagicMock,
        mock_image_storage: MagicMock,
        sample_dataset: BaseDataset[BaseDataItem],
    ) -> None:
        """Test that images are converted to RGB mode."""
        # Create a grayscale image that will be converted to RGB
        grayscale_image = Image.new("L", (100, 100), color=128)
        rgb_image = Image.new("RGB", (100, 100), color="white")
        grayscale_image.convert = MagicMock(return_value=rgb_image)
        mock_image_storage.load_image.return_value = grayscale_image

        use_case = EnrichDatasetWithLMv3(
            lmv3_engine=mock_lmv3_engine,
            image_storage=mock_image_storage,
            checkpoint="test-checkpoint",
            enable_cache=False,
        )

        request = EnrichWithLMv3Request(dataset=sample_dataset)
        await use_case.execute(request)

        # Verify convert("RGB") was called
        grayscale_image.convert.assert_called_with("RGB")

    @patch(
        "notarius.application.use_cases.inference.add_lmv3_preds_to_dataset.create_lmv3_cache_backend"
    )
    async def test_execute_returns_correct_statistics_with_cache(
        self,
        mock_create_cache: MagicMock,
        mock_lmv3_engine: MagicMock,
        mock_image_storage: MagicMock,
        sample_dataset: BaseDataset[BaseDataItem],
        mock_schematism_page: SchematismPage,
    ) -> None:
        """Test that statistics are correct when cache is enabled."""
        mock_backend = MagicMock()
        mock_keygen = MagicMock()
        mock_create_cache.return_value = (mock_backend, mock_keygen)

        # Mock cache to return None (cache miss) for all requests
        mock_backend.get.return_value = None
        mock_backend.set.return_value = True
        mock_keygen.generate_key.return_value = "test_key"

        use_case = EnrichDatasetWithLMv3(
            lmv3_engine=mock_lmv3_engine,
            image_storage=mock_image_storage,
            checkpoint="test-checkpoint",
            enable_cache=True,
        )

        request = EnrichWithLMv3Request(dataset=sample_dataset)
        response = await use_case.execute(request)

        # Stats should come from CachedEngine - all cache misses
        assert response.lmv3_executions == 2
        assert response.cache_hits == 0


class TestEnrichDatasetWithLMv3Integration:
    """Integration-style tests for EnrichDatasetWithLMv3."""

    async def test_full_workflow_produces_valid_predictions(
        self,
        mock_image_storage: MagicMock,
    ) -> None:
        """Test complete workflow produces valid PredictionDataItems."""
        # Create mock engine with different predictions for each item
        mock_engine = MagicMock(spec=LMv3Engine)

        page1 = SchematismPage(
            entries=[SchematismEntry(deanery="Deanery 1", parish="Parish 1")]
        )
        page2 = SchematismPage(
            entries=[SchematismEntry(deanery="Deanery 2", parish="Parish 2")]
        )

        mock_engine.process.side_effect = [
            LMv3Response(output=page1),
            LMv3Response(output=page2),
        ]

        use_case = EnrichDatasetWithLMv3(
            lmv3_engine=mock_engine,
            image_storage=mock_image_storage,
            checkpoint="test-checkpoint",
            enable_cache=False,
        )

        dataset = BaseDataset[BaseDataItem](
            items=[
                BaseDataItem(
                    image_path="/path/page1.jpg",
                    text="OCR text 1",
                    metadata=BaseMetaData(
                        sample_id=1, schematism_name="test", filename="p1.jpg"
                    ),
                ),
                BaseDataItem(
                    image_path="/path/page2.jpg",
                    text="OCR text 2",
                    metadata=BaseMetaData(
                        sample_id=2, schematism_name="test", filename="p2.jpg"
                    ),
                ),
            ]
        )

        request = EnrichWithLMv3Request(dataset=dataset)
        response = await use_case.execute(request)

        assert len(response.dataset.items) == 2

        # Verify predictions are correctly assigned
        assert response.dataset.items[0].predictions.entries[0].deanery == "Deanery 1"
        assert response.dataset.items[1].predictions.entries[0].deanery == "Deanery 2"

        # Verify original data is preserved
        assert response.dataset.items[0].text == "OCR text 1"
        assert response.dataset.items[1].text == "OCR text 2"

        assert response.lmv3_executions == 2
        assert response.cache_hits == 0
