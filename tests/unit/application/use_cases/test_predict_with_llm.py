"""Tests for PredictDatasetWithLLM use case."""

import pytest
from dataclasses import dataclass
from unittest.mock import MagicMock, patch
from PIL import Image

from notarius.application.use_cases.inference.add_llm_preds_to_dataset import (
    PredictDatasetWithLLM,
    PredictWithLLMRequest,
    PredictWithLLMResponse,
)
from notarius.application.ports.outbound.cached_engine import CachedEngine
from notarius.infrastructure.llm.engine_adapter import (
    LLMEngine,
    CompletionResult,
)
from notarius.infrastructure.llm.conversation import Conversation
from notarius.domain.entities.completions import BaseProviderResponse
from notarius.domain.entities.schematism import (
    SchematismPage,
    SchematismEntry,
    PageContext,
)
from notarius.domain.entities.messages import ChatMessage, TextContent
from notarius.schemas.data.pipeline import (
    BaseDataset,
    BaseDataItem,
    PredictionDataItem,
    BaseMetaData,
)


# Mock provider response for testing
@dataclass(frozen=True)
class MockLLMResponse(BaseProviderResponse[SchematismPage]):
    """Mock LLM provider response."""

    def to_string(self) -> str:
        return (
            self.structured_response.model_dump_json()
            if self.structured_response
            else ""
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
        page_number="1",
        entries=[
            SchematismEntry(
                deanery="Test Deanery",
                parish="Test Parish",
            )
        ],
    )


@pytest.fixture
def mock_schematism_page_with_context() -> SchematismPage:
    """Create a mock SchematismPage with context for multi-page tests."""
    return SchematismPage(
        page_number="1",
        entries=[
            SchematismEntry(deanery="Test Deanery", parish="Test Parish"),
        ],
        context=PageContext(
            active_deanery="Test Deanery",
            last_page_number="1",
        ),
    )


@pytest.fixture
def mock_llm_engine(mock_schematism_page: SchematismPage) -> MagicMock:
    """Create a mock LLM engine."""
    engine = MagicMock(spec=LLMEngine)

    mock_response = MockLLMResponse(
        structured_response=mock_schematism_page,
        text_response=None,
    )
    mock_conversation = Conversation.from_messages(
        [ChatMessage(role="assistant", content=[TextContent(text="response")])]
    )

    engine.process.return_value = CompletionResult(
        output=mock_response,
        conversation=mock_conversation,
    )
    # Mock stats property for non-cached engine
    engine.stats = {"calls": 0, "errors": 0}
    return engine


@pytest.fixture
def mock_image_storage(mock_image: Image.Image) -> MagicMock:
    """Create a mock image storage resource."""
    storage = MagicMock()
    storage.load_image.return_value = mock_image
    return storage


@pytest.fixture
def mock_prompt_renderer() -> MagicMock:
    """Create a mock Jinja2 prompt renderer."""
    renderer = MagicMock()
    renderer.render_prompt.return_value = "Rendered prompt text"
    return renderer


@pytest.fixture
def sample_metadata() -> BaseMetaData:
    """Create sample metadata."""
    return BaseMetaData(
        sample_id=1,
        schematism_name="test_schematism",
        filename="test_file.jpg",
    )


@pytest.fixture
def sample_lmv3_dataset(
    sample_metadata: BaseMetaData, mock_schematism_page: SchematismPage
) -> BaseDataset[PredictionDataItem]:
    """Create a sample LMv3 dataset with predictions."""
    items = [
        PredictionDataItem(
            image_path="/path/to/image1.jpg",
            text="OCR text 1",
            predictions=mock_schematism_page,
            metadata=sample_metadata,
        ),
        PredictionDataItem(
            image_path="/path/to/image2.jpg",
            text="OCR text 2",
            predictions=mock_schematism_page,
            metadata=BaseMetaData(
                sample_id=2,
                schematism_name="test_schematism",
                filename="test_file2.jpg",
            ),
        ),
    ]
    return BaseDataset[PredictionDataItem](items=items)


@pytest.fixture
def sample_ocr_dataset(sample_metadata: BaseMetaData) -> BaseDataset[BaseDataItem]:
    """Create a sample OCR dataset."""
    items = [
        BaseDataItem(
            image_path="/path/to/image1.jpg",
            text="OCR text from page 1",
            metadata=sample_metadata,
        ),
        BaseDataItem(
            image_path="/path/to/image2.jpg",
            text="OCR text from page 2",
            metadata=BaseMetaData(
                sample_id=2,
                schematism_name="test_schematism",
                filename="test_file2.jpg",
            ),
        ),
    ]
    return BaseDataset[BaseDataItem](items=items)


@pytest.fixture
def empty_lmv3_dataset() -> BaseDataset[PredictionDataItem]:
    """Create an empty LMv3 dataset."""
    return BaseDataset[PredictionDataItem](items=[])


@pytest.fixture
def empty_ocr_dataset() -> BaseDataset[BaseDataItem]:
    """Create an empty OCR dataset."""
    return BaseDataset[BaseDataItem](items=[])


class TestPredictWithLLMRequest:
    """Test suite for PredictWithLLMRequest dataclass."""

    def test_request_with_defaults(
        self,
        sample_lmv3_dataset: BaseDataset[PredictionDataItem],
        sample_ocr_dataset: BaseDataset[BaseDataItem],
    ) -> None:
        """Test request creation with default values."""
        request = PredictWithLLMRequest(
            lmv3_dataset=sample_lmv3_dataset,
            ocr_dataset=sample_ocr_dataset,
        )

        assert request.lmv3_dataset is sample_lmv3_dataset
        assert request.ocr_dataset is sample_ocr_dataset
        assert request.system_prompt == "system.j2"
        assert request.user_prompt == "user.j2"
        assert request.use_lmv3_hints is True
        assert request.accumulate_context is False

    def test_request_with_custom_prompts(
        self,
        sample_lmv3_dataset: BaseDataset[PredictionDataItem],
        sample_ocr_dataset: BaseDataset[BaseDataItem],
    ) -> None:
        """Test request creation with custom prompt templates."""
        request = PredictWithLLMRequest(
            lmv3_dataset=sample_lmv3_dataset,
            ocr_dataset=sample_ocr_dataset,
            system_prompt="custom_system.j2",
            user_prompt="custom_user.j2",
        )

        assert request.system_prompt == "custom_system.j2"
        assert request.user_prompt == "custom_user.j2"

    def test_request_with_context_accumulation(
        self,
        sample_lmv3_dataset: BaseDataset[PredictionDataItem],
        sample_ocr_dataset: BaseDataset[BaseDataItem],
    ) -> None:
        """Test request creation with context accumulation enabled."""
        request = PredictWithLLMRequest(
            lmv3_dataset=sample_lmv3_dataset,
            ocr_dataset=sample_ocr_dataset,
            accumulate_context=True,
        )

        assert request.accumulate_context is True


class TestPredictWithLLMResponse:
    """Test suite for PredictWithLLMResponse dataclass."""

    def test_response_creation(
        self,
        sample_lmv3_dataset: BaseDataset[PredictionDataItem],
    ) -> None:
        """Test response creation with all fields."""
        response = PredictWithLLMResponse(
            dataset=sample_lmv3_dataset,
            llm_executions=5,
            cache_hits=3,
            success_rate=0.8,
        )

        assert response.dataset is sample_lmv3_dataset
        assert response.llm_executions == 5
        assert response.cache_hits == 3
        assert response.success_rate == 0.8


class TestPredictDatasetWithLLM:
    """Test suite for PredictDatasetWithLLM use case."""

    def test_init_with_cache_disabled(
        self,
        mock_llm_engine: MagicMock,
        mock_image_storage: MagicMock,
    ) -> None:
        """Test initialization with caching disabled."""
        use_case = PredictDatasetWithLLM(
            llm_engine=mock_llm_engine,
            image_storage=mock_image_storage,
            model_name="test-model",
            enable_cache=False,
        )

        assert use_case.llm_engine is mock_llm_engine
        assert use_case.image_storage is mock_image_storage

    @patch(
        "notarius.application.use_cases.inference.add_llm_preds_to_dataset.create_llm_cache_backend"
    )
    def test_init_with_cache_enabled(
        self,
        mock_create_cache: MagicMock,
        mock_llm_engine: MagicMock,
        mock_image_storage: MagicMock,
    ) -> None:
        """Test initialization with caching enabled."""
        mock_backend = MagicMock()
        mock_keygen = MagicMock()
        mock_create_cache.return_value = (mock_backend, mock_keygen)

        use_case = PredictDatasetWithLLM(
            llm_engine=mock_llm_engine,
            image_storage=mock_image_storage,
            model_name="test-model",
            enable_cache=True,
        )

        assert isinstance(use_case.llm_engine, CachedEngine)
        mock_create_cache.assert_called_once_with("test-model")

    def test_init_with_custom_prompt_renderer(
        self,
        mock_llm_engine: MagicMock,
        mock_image_storage: MagicMock,
        mock_prompt_renderer: MagicMock,
    ) -> None:
        """Test initialization with custom prompt renderer."""
        use_case = PredictDatasetWithLLM(
            llm_engine=mock_llm_engine,
            image_storage=mock_image_storage,
            model_name="test-model",
            prompt_renderer=mock_prompt_renderer,
            enable_cache=False,
        )

        assert use_case.prompt_renderer is mock_prompt_renderer

    async def test_execute_processes_all_items(
        self,
        mock_llm_engine: MagicMock,
        mock_image_storage: MagicMock,
        mock_prompt_renderer: MagicMock,
        sample_lmv3_dataset: BaseDataset[PredictionDataItem],
        sample_ocr_dataset: BaseDataset[BaseDataItem],
    ) -> None:
        """Test that execute processes all dataset items."""
        use_case = PredictDatasetWithLLM(
            llm_engine=mock_llm_engine,
            image_storage=mock_image_storage,
            model_name="test-model",
            prompt_renderer=mock_prompt_renderer,
            enable_cache=False,
        )

        request = PredictWithLLMRequest(
            lmv3_dataset=sample_lmv3_dataset,
            ocr_dataset=sample_ocr_dataset,
        )
        response = await use_case.execute(request)

        assert len(response.dataset.items) == 2
        assert mock_llm_engine.process.call_count == 2
        assert mock_image_storage.load_image.call_count == 2

    async def test_execute_creates_prediction_data_items(
        self,
        mock_llm_engine: MagicMock,
        mock_image_storage: MagicMock,
        mock_prompt_renderer: MagicMock,
        sample_lmv3_dataset: BaseDataset[PredictionDataItem],
        sample_ocr_dataset: BaseDataset[BaseDataItem],
    ) -> None:
        """Test that execute creates PredictionDataItem instances."""
        use_case = PredictDatasetWithLLM(
            llm_engine=mock_llm_engine,
            image_storage=mock_image_storage,
            model_name="test-model",
            prompt_renderer=mock_prompt_renderer,
            enable_cache=False,
        )

        request = PredictWithLLMRequest(
            lmv3_dataset=sample_lmv3_dataset,
            ocr_dataset=sample_ocr_dataset,
        )
        response = await use_case.execute(request)

        for item in response.dataset.items:
            assert isinstance(item, PredictionDataItem)
            assert item.predictions is not None

    async def test_execute_uses_ocr_text(
        self,
        mock_llm_engine: MagicMock,
        mock_image_storage: MagicMock,
        mock_prompt_renderer: MagicMock,
        sample_lmv3_dataset: BaseDataset[PredictionDataItem],
        sample_ocr_dataset: BaseDataset[BaseDataItem],
    ) -> None:
        """Test that execute uses OCR text in the prompt context."""
        use_case = PredictDatasetWithLLM(
            llm_engine=mock_llm_engine,
            image_storage=mock_image_storage,
            model_name="test-model",
            prompt_renderer=mock_prompt_renderer,
            enable_cache=False,
        )

        request = PredictWithLLMRequest(
            lmv3_dataset=sample_lmv3_dataset,
            ocr_dataset=sample_ocr_dataset,
        )
        await use_case.execute(request)

        # Verify prompt renderer was called with OCR text in context
        calls = mock_prompt_renderer.render_prompt.call_args_list
        # Filter user prompt calls (not system prompt)
        user_prompt_calls = [
            c for c in calls if c[1].get("context", {}).get("ocr_text")
        ]
        assert len(user_prompt_calls) == 2
        assert user_prompt_calls[0][1]["context"]["ocr_text"] == "OCR text from page 1"

    async def test_execute_with_lmv3_hints(
        self,
        mock_llm_engine: MagicMock,
        mock_image_storage: MagicMock,
        mock_prompt_renderer: MagicMock,
        sample_lmv3_dataset: BaseDataset[PredictionDataItem],
        sample_ocr_dataset: BaseDataset[BaseDataItem],
    ) -> None:
        """Test that LMv3 hints are passed when enabled."""
        use_case = PredictDatasetWithLLM(
            llm_engine=mock_llm_engine,
            image_storage=mock_image_storage,
            model_name="test-model",
            prompt_renderer=mock_prompt_renderer,
            enable_cache=False,
        )

        request = PredictWithLLMRequest(
            lmv3_dataset=sample_lmv3_dataset,
            ocr_dataset=sample_ocr_dataset,
            use_lmv3_hints=True,
        )
        await use_case.execute(request)

        # Verify hints are passed in context
        calls = mock_prompt_renderer.render_prompt.call_args_list
        user_prompt_calls = [c for c in calls if c[1].get("context", {}).get("hints")]
        assert len(user_prompt_calls) == 2

    async def test_execute_without_lmv3_hints(
        self,
        mock_llm_engine: MagicMock,
        mock_image_storage: MagicMock,
        mock_prompt_renderer: MagicMock,
        sample_lmv3_dataset: BaseDataset[PredictionDataItem],
        sample_ocr_dataset: BaseDataset[BaseDataItem],
    ) -> None:
        """Test that LMv3 hints are not passed when disabled."""
        use_case = PredictDatasetWithLLM(
            llm_engine=mock_llm_engine,
            image_storage=mock_image_storage,
            model_name="test-model",
            prompt_renderer=mock_prompt_renderer,
            enable_cache=False,
        )

        request = PredictWithLLMRequest(
            lmv3_dataset=sample_lmv3_dataset,
            ocr_dataset=sample_ocr_dataset,
            use_lmv3_hints=False,
        )
        await use_case.execute(request)

        # Verify hints are NOT passed in context
        calls = mock_prompt_renderer.render_prompt.call_args_list
        user_prompt_calls = [c for c in calls if "hints" in c[1].get("context", {})]
        assert len(user_prompt_calls) == 0

    async def test_execute_with_empty_datasets(
        self,
        mock_llm_engine: MagicMock,
        mock_image_storage: MagicMock,
        mock_prompt_renderer: MagicMock,
        empty_lmv3_dataset: BaseDataset[PredictionDataItem],
        empty_ocr_dataset: BaseDataset[BaseDataItem],
    ) -> None:
        """Test execution with empty datasets."""
        use_case = PredictDatasetWithLLM(
            llm_engine=mock_llm_engine,
            image_storage=mock_image_storage,
            model_name="test-model",
            prompt_renderer=mock_prompt_renderer,
            enable_cache=False,
        )

        request = PredictWithLLMRequest(
            lmv3_dataset=empty_lmv3_dataset,
            ocr_dataset=empty_ocr_dataset,
        )
        response = await use_case.execute(request)

        assert len(response.dataset.items) == 0
        assert response.llm_executions == 0
        assert response.cache_hits == 0
        assert response.success_rate == 0.0
        mock_llm_engine.process.assert_not_called()

    async def test_execute_returns_correct_statistics_without_cache(
        self,
        mock_llm_engine: MagicMock,
        mock_image_storage: MagicMock,
        mock_prompt_renderer: MagicMock,
        sample_lmv3_dataset: BaseDataset[PredictionDataItem],
        sample_ocr_dataset: BaseDataset[BaseDataItem],
    ) -> None:
        """Test that statistics are correct when cache is disabled."""
        # Update mock stats to expected value for this test (2 items processed)
        mock_llm_engine.stats = {"calls": 2, "errors": 0}

        use_case = PredictDatasetWithLLM(
            llm_engine=mock_llm_engine,
            image_storage=mock_image_storage,
            model_name="test-model",
            prompt_renderer=mock_prompt_renderer,
            enable_cache=False,
        )

        request = PredictWithLLMRequest(
            lmv3_dataset=sample_lmv3_dataset,
            ocr_dataset=sample_ocr_dataset,
        )
        response = await use_case.execute(request)

        assert response.llm_executions == 2
        assert response.cache_hits == 0
        assert response.success_rate == 1.0

    async def test_execute_uses_structured_output(
        self,
        mock_llm_engine: MagicMock,
        mock_image_storage: MagicMock,
        mock_prompt_renderer: MagicMock,
        sample_lmv3_dataset: BaseDataset[PredictionDataItem],
        sample_ocr_dataset: BaseDataset[BaseDataItem],
    ) -> None:
        """Test that execute requests structured output (SchematismPage)."""
        use_case = PredictDatasetWithLLM(
            llm_engine=mock_llm_engine,
            image_storage=mock_image_storage,
            model_name="test-model",
            prompt_renderer=mock_prompt_renderer,
            enable_cache=False,
        )

        request = PredictWithLLMRequest(
            lmv3_dataset=sample_lmv3_dataset,
            ocr_dataset=sample_ocr_dataset,
        )
        await use_case.execute(request)

        # Verify CompletionRequest was created with structured_output=SchematismPage
        call_args = mock_llm_engine.process.call_args_list
        for call in call_args:
            completion_request = call[0][0]
            assert completion_request.structured_output == SchematismPage


class TestPredictDatasetWithLLMContextAccumulation:
    """Test suite for context accumulation functionality."""

    async def test_execute_with_context_accumulation(
        self,
        mock_image_storage: MagicMock,
        mock_prompt_renderer: MagicMock,
        sample_lmv3_dataset: BaseDataset[PredictionDataItem],
        sample_ocr_dataset: BaseDataset[BaseDataItem],
        mock_schematism_page_with_context: SchematismPage,
    ) -> None:
        """Test that context is accumulated across pages."""
        mock_engine = MagicMock(spec=LLMEngine)

        # Create responses with context
        mock_response = MockLLMResponse(
            structured_response=mock_schematism_page_with_context,
            text_response=None,
        )
        mock_conversation = Conversation.from_messages(
            [ChatMessage(role="assistant", content=[TextContent(text="response")])]
        )

        mock_engine.process.return_value = CompletionResult(
            output=mock_response,
            conversation=mock_conversation,
        )
        # Mock stats property for non-cached engine
        mock_engine.stats = {"calls": 2, "errors": 0}

        use_case = PredictDatasetWithLLM(
            llm_engine=mock_engine,
            image_storage=mock_image_storage,
            model_name="test-model",
            prompt_renderer=mock_prompt_renderer,
            enable_cache=False,
        )

        request = PredictWithLLMRequest(
            lmv3_dataset=sample_lmv3_dataset,
            ocr_dataset=sample_ocr_dataset,
            accumulate_context=True,
        )
        response = await use_case.execute(request)

        # Verify both items were processed
        assert len(response.dataset.items) == 2

        # Verify previous_context was passed after first item
        calls = mock_prompt_renderer.render_prompt.call_args_list
        user_prompt_calls = [c for c in calls if c[1].get("template_name") == "user.j2"]

        # Second user prompt should have previous_context
        if len(user_prompt_calls) >= 2:
            second_call_context = user_prompt_calls[1][1]["context"]
            assert "previous_context" in second_call_context


class TestPredictDatasetWithLLMIntegration:
    """Integration-style tests for PredictDatasetWithLLM."""

    async def test_full_workflow_produces_valid_predictions(
        self,
        mock_image_storage: MagicMock,
    ) -> None:
        """Test complete workflow produces valid PredictionDataItems."""
        mock_engine = MagicMock(spec=LLMEngine)
        mock_renderer = MagicMock()
        mock_renderer.render_prompt.return_value = "Rendered prompt"

        page1 = SchematismPage(
            page_number="1",
            entries=[SchematismEntry(deanery="Deanery A", parish="Parish 1")],
        )
        page2 = SchematismPage(
            page_number="2",
            entries=[SchematismEntry(deanery="Deanery A", parish="Parish 2")],
        )

        mock_response1 = MockLLMResponse(structured_response=page1, text_response=None)
        mock_response2 = MockLLMResponse(structured_response=page2, text_response=None)

        mock_conversation = Conversation.from_messages(
            [ChatMessage(role="assistant", content=[TextContent(text="resp")])]
        )

        mock_engine.process.side_effect = [
            CompletionResult(output=mock_response1, conversation=mock_conversation),
            CompletionResult(output=mock_response2, conversation=mock_conversation),
        ]
        # Mock stats property for non-cached engine
        mock_engine.stats = {"calls": 2, "errors": 0}

        use_case = PredictDatasetWithLLM(
            llm_engine=mock_engine,
            image_storage=mock_image_storage,
            model_name="test-model",
            prompt_renderer=mock_renderer,
            enable_cache=False,
        )

        lmv3_dataset = BaseDataset[PredictionDataItem](
            items=[
                PredictionDataItem(
                    image_path="/path/page1.jpg",
                    text="text1",
                    predictions=page1,
                    metadata=BaseMetaData(
                        sample_id=1, schematism_name="test", filename="p1.jpg"
                    ),
                ),
                PredictionDataItem(
                    image_path="/path/page2.jpg",
                    text="text2",
                    predictions=page2,
                    metadata=BaseMetaData(
                        sample_id=2, schematism_name="test", filename="p2.jpg"
                    ),
                ),
            ]
        )

        ocr_dataset = BaseDataset[BaseDataItem](
            items=[
                BaseDataItem(
                    image_path="/path/page1.jpg",
                    text="OCR for page 1",
                    metadata=BaseMetaData(
                        sample_id=1, schematism_name="test", filename="p1.jpg"
                    ),
                ),
                BaseDataItem(
                    image_path="/path/page2.jpg",
                    text="OCR for page 2",
                    metadata=BaseMetaData(
                        sample_id=2, schematism_name="test", filename="p2.jpg"
                    ),
                ),
            ]
        )

        request = PredictWithLLMRequest(
            lmv3_dataset=lmv3_dataset,
            ocr_dataset=ocr_dataset,
        )
        response = await use_case.execute(request)

        assert len(response.dataset.items) == 2
        assert response.dataset.items[0].predictions.entries[0].parish == "Parish 1"
        assert response.dataset.items[1].predictions.entries[0].parish == "Parish 2"
        assert response.llm_executions == 2
        assert response.success_rate == 1.0
