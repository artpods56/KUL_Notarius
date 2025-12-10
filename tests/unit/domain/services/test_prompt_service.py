"""Tests for PromptConstructionService."""

from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TypedDict
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image
from PIL.Image import Image as PILImage

from notarius.domain.services.prompt_service import PromptConstructionService
from notarius.infrastructure.llm.prompt_manager import Jinja2PromptRenderer


PromptWithTemplate = TypedDict(
    "PromptWithTemplate",
    {
        "template": str,
        "text": str,
        "resolved_prompt": str,
    },
)


@pytest.fixture
def prompt_text() -> str:
    return "Lorem ipsum"


@pytest.fixture
def system_prompt_template() -> PromptWithTemplate:
    return {
        "template": "system.j2",
        "text": "System text: {{ instruction }}",
        "resolved_prompt": "System text: ",
    }


@pytest.fixture
def user_prompt_template() -> PromptWithTemplate:
    return {
        "template": "user.j2",
        "text": "User text: {{ query }}",
        "resolved_prompt": "User text: ",
    }


class TestPromptConstructionService:
    """Test suite for PromptConstructionService class."""

    @pytest.fixture
    def tmp_template_dir(
        self,
        user_prompt_template: PromptWithTemplate,
        system_prompt_template: PromptWithTemplate,
    ):
        """Create a temporary directory with test templates."""
        with TemporaryDirectory() as tmpdir:
            template_dir = Path(tmpdir)

            # Create test templates
            system_template = template_dir / system_prompt_template["template"]
            _ = system_template.write_text(system_prompt_template["text"])

            user_template = template_dir / user_prompt_template["template"]
            _ = user_template.write_text(user_prompt_template["text"])

            yield template_dir

    @pytest.fixture
    def prompt_renderer(self, tmp_template_dir: Path) -> Jinja2PromptRenderer:
        """Create a text renderer for testing."""
        return Jinja2PromptRenderer(template_dir=tmp_template_dir)

    @pytest.fixture
    def service(
        self, prompt_renderer: Jinja2PromptRenderer
    ) -> PromptConstructionService:
        """Create a service instance for testing."""
        return PromptConstructionService(
            prompt_renderer=prompt_renderer,
        )

    @pytest.fixture
    def sample_image(self) -> PILImage:
        """Create a sample PIL image for testing."""
        return Image.new("RGB", (100, 100), color="green")

    def test_init(self, prompt_renderer: Jinja2PromptRenderer) -> None:
        """Test service initialization."""
        service = PromptConstructionService(
            prompt_renderer=prompt_renderer,
        )
        assert service.prompt_renderer is prompt_renderer
        assert isinstance(service.prompt_renderer, Jinja2PromptRenderer)

    def test_build_messages_text_only(
        self,
        service: PromptConstructionService,
        system_prompt_template: PromptWithTemplate,
        user_prompt_template: PromptWithTemplate,
        prompt_text: str,
    ) -> None:
        """Test building messages with text only."""
        context = {"instruction": "Analyze text", "query": "What is this?"}

        messages = service.build_messages(
            text=prompt_text,
            system_template=system_prompt_template["template"],
            user_template=user_prompt_template["template"],
            context=context,
        )

        assert len(messages) == 2

        system_message = messages[0]
        assert system_message.get("role") == "system"
        assert isinstance(system_message.get("text"), str)
        assert (
            system_message.get("text")
            == system_prompt_template["resolved_prompt"] + context["instruction"]
        )

        user_message = messages[1]
        assert user_message.get("role") == "user"
        assert isinstance(user_message.get("text"), str)
        assert (
            system_message.get("text")
            == system_prompt_template["resolved_prompt"] + context["instruction"]
        )

    def test_build_messages_image_only(
        self,
        service: PromptConstructionService,
        sample_image: PILImage,
        system_prompt_template: PromptWithTemplate,
        user_prompt_template: PromptWithTemplate,
    ) -> None:
        """Test building messages with image only."""
        context = {"instruction": "Analyze image", "query": "Describe this"}

        messages = service.build_messages(
            image=sample_image,
            system_template=system_prompt_template["template"],
            user_template=user_prompt_template["template"],
            context=context,
        )
        assert len(messages) == 2

        system_message = messages[0]
        assert system_message.get("role") == "system"
        assert (
            system_message.get("text")
            == system_prompt_template["resolved_prompt"] + context["instruction"]
        )

        user_message = messages[1]
        assert user_message.get("role") == "user"
        assert isinstance(user_message.get("text"), list)

        user_message_content = user_message.get("text")
        if isinstance(user_message_content, list):
            text_part, image_part = user_message_content
            assert (
                text_part.get("text")
                == user_prompt_template["resolved_prompt"] + context["query"]
            )
            assert text_part.get("type") == "text"

            assert image_part.get("type") == "image_url"

            image_url = image_part.get("image_url")
            assert isinstance(image_url, dict)
            assert "url" in image_url
            assert image_url["url"] is not None

    def test_build_messages_multimodal(
        self,
        service: PromptConstructionService,
        sample_image: PILImage,
        system_prompt_template: PromptWithTemplate,
        user_prompt_template: PromptWithTemplate,
    ) -> None:
        """Test building messages with both image and text."""
        context = {"instruction": "Analyze", "query": "What do you see?"}

        messages = service.build_messages(
            image=sample_image,
            text="additional context",
            system_template=system_prompt_template["template"],
            user_template=user_prompt_template["template"],
            context=context,
        )

        assert len(messages) == 2
        # When image is provided, it takes precedence
        assert messages[1]["role"] == "user"
        assert isinstance(messages[1]["text"], list)

    def test_build_messages_without_context(
        self,
        service: PromptConstructionService,
        system_prompt_template: PromptWithTemplate,
        user_prompt_template: PromptWithTemplate,
        prompt_text: str,
    ) -> None:
        """Test building messages without context (uses empty dict)."""
        messages = service.build_messages(
            text=prompt_text,
            system_template=system_prompt_template["template"],
            user_template=user_prompt_template["template"],
        )
        assert len(messages) == 2

        system_message = messages[0]
        assert system_message.get("role") == "system"
        # Without context, variables are rendered as empty strings
        assert system_message.get("text") == system_prompt_template["resolved_prompt"]

        user_message = messages[1]
        assert user_message.get("role") == "user"
        assert isinstance(user_message.get("text"), str)
        assert user_message.get("text") == user_prompt_template["resolved_prompt"]

    def test_build_messages_missing_image_and_text_raises_error(
        self,
        service: PromptConstructionService,
        system_prompt_template: PromptWithTemplate,
        user_prompt_template: PromptWithTemplate,
    ) -> None:
        """Test that missing both image and text raises ValueError."""
        context = {"instruction": "Test", "query": "Test"}

        with pytest.raises(
            ValueError, match="At least one of 'image' or 'text' must be provided"
        ):
            _ = service.build_messages(
                system_template=system_prompt_template["template"],
                user_template=user_prompt_template["template"],
                context=context,
            )

    def test_build_messages_calls_renderer(
        self,
        service: PromptConstructionService,
        sample_image: PILImage,
        system_prompt_template: PromptWithTemplate,
        user_prompt_template: PromptWithTemplate,
    ) -> None:
        """Test that service calls text renderer correctly."""
        with patch.object(service.prompt_renderer, "render_prompt") as mock_render:
            context = {"key": "value"}
            _ = service.build_messages(
                image=sample_image,
                system_template=system_prompt_template["template"],
                user_template=user_prompt_template["template"],
                context=context,
            )

            # Verify renderer was called twice
            assert mock_render.call_count == 2
            mock_render.assert_any_call(system_prompt_template["template"], context)
            mock_render.assert_any_call(user_prompt_template["template"], context)

    @patch("notarius.domain.services.prompt_service.construct_system_message")
    @patch("notarius.domain.services.prompt_service.construct_user_image_message")
    @patch("notarius.domain.services.prompt_service.construct_user_text_message")
    def test_build_messages_calls_provider_methods(
        self,
        mock_text_msg: MagicMock,
        mock_image_msg: MagicMock,
        mock_system_msg: MagicMock,
        service: PromptConstructionService,
        sample_image: PILImage,
        system_prompt_template: PromptWithTemplate,
        user_prompt_template: PromptWithTemplate,
    ) -> None:
        """Test that service calls provider message construction methods."""
        # Set up return values
        mock_system_msg.return_value = {"role": "system", "text": "system"}
        mock_image_msg.return_value = {"role": "user", "text": []}

        context = {"instruction": "Test", "query": "Query"}

        _ = service.build_messages(
            image=sample_image,
            system_template=system_prompt_template["template"],
            user_template=user_prompt_template["template"],
            context=context,
        )

        # Verify provider methods were called
        expected_system = (
            system_prompt_template["resolved_prompt"] + context["instruction"]
        )
        expected_user = user_prompt_template["resolved_prompt"] + context["query"]
        mock_system_msg.assert_called_once_with(expected_system)
        mock_image_msg.assert_called_once_with(sample_image, expected_user)
        mock_text_msg.assert_not_called()

    @patch("notarius.domain.services.prompt_service.construct_system_message")
    @patch("notarius.domain.services.prompt_service.construct_user_image_message")
    @patch("notarius.domain.services.prompt_service.construct_user_text_message")
    def test_build_messages_text_calls_text_method(
        self,
        mock_text_msg: MagicMock,
        mock_image_msg: MagicMock,
        mock_system_msg: MagicMock,
        service: PromptConstructionService,
        system_prompt_template: PromptWithTemplate,
        user_prompt_template: PromptWithTemplate,
    ) -> None:
        """Test that text-only input calls construct_user_text_message."""
        # Set up return values
        mock_system_msg.return_value = {"role": "system", "text": "system"}
        mock_text_msg.return_value = {"role": "user", "text": "text"}

        context = {"instruction": "Test", "query": "Query"}

        _ = service.build_messages(
            text="test",
            system_template=system_prompt_template["template"],
            user_template=user_prompt_template["template"],
            context=context,
        )

        expected_system = (
            system_prompt_template["resolved_prompt"] + context["instruction"]
        )
        expected_user = user_prompt_template["resolved_prompt"] + context["query"]
        mock_system_msg.assert_called_once_with(expected_system)
        mock_text_msg.assert_called_once_with(expected_user)
        mock_image_msg.assert_not_called()

    @patch("notarius.domain.services.prompt_service.construct_system_message")
    @patch("notarius.domain.services.prompt_service.construct_user_image_message")
    @patch("notarius.domain.services.prompt_service.construct_user_text_message")
    def test_build_messages_image_calls_image_method(
        self,
        mock_text_msg: MagicMock,
        mock_image_msg: MagicMock,
        mock_system_msg: MagicMock,
        service: PromptConstructionService,
        sample_image: PILImage,
        system_prompt_template: PromptWithTemplate,
        user_prompt_template: PromptWithTemplate,
    ) -> None:
        """Test that image input calls construct_user_image_message."""
        # Set up return values
        mock_system_msg.return_value = {"role": "system", "text": "system"}
        mock_image_msg.return_value = {"role": "user", "text": []}

        context = {"instruction": "Test", "query": "Query"}

        _ = service.build_messages(
            image=sample_image,
            system_template=system_prompt_template["template"],
            user_template=user_prompt_template["template"],
            context=context,
        )

        expected_system = (
            system_prompt_template["resolved_prompt"] + context["instruction"]
        )
        expected_user = user_prompt_template["resolved_prompt"] + context["query"]
        mock_system_msg.assert_called_once_with(expected_system)
        mock_image_msg.assert_called_once_with(sample_image, expected_user)
        mock_text_msg.assert_not_called()

    def test_build_messages_with_custom_templates(
        self, tmp_template_dir: Path, service: PromptConstructionService
    ) -> None:
        """Test building messages with custom template names."""
        # Create custom templates
        _ = (tmp_template_dir / "custom_system.j2").write_text("Custom: {{ var }}")
        _ = (tmp_template_dir / "custom_user.j2").write_text("User: {{ var }}")

        context = {"var": "value"}

        messages = service.build_messages(
            text="test",
            system_template="custom_system.j2",
            user_template="custom_user.j2",
            context=context,
        )

        system_message, user_message = messages

        assert system_message.get("text") == "top_level"

        assert user_message.get("text") == "nested_value"

    def test_build_messages_with_complex_context(self) -> None:
        """Test building messages with complex nested context."""
        # Create template that uses nested context
        with TemporaryDirectory() as tmpdir:
            template_dir = Path(tmpdir)
            _ = (template_dir / "nested_system.j2").write_text("{{ data.key }}")
            _ = (template_dir / "nested_user.j2").write_text("{{ data.nested.value }}")

            renderer = Jinja2PromptRenderer(template_dir=template_dir)
            service_nested = PromptConstructionService(
                prompt_renderer=renderer,
            )

            complex_context = {
                "data": {"key": "top_level", "nested": {"value": "nested_value"}}
            }

            messages = service_nested.build_messages(
                text="test",
                system_template="nested_system.j2",
                user_template="nested_user.j2",
                context=complex_context,
            )

            system_message, user_message = messages

            assert system_message.get("text") == "top_level"

            assert user_message.get("text") == "nested_value"

    def test_build_messages_returns_correct_type(
        self,
        service: PromptConstructionService,
        system_prompt_template: PromptWithTemplate,
        user_prompt_template: PromptWithTemplate,
        prompt_text: str,
    ) -> None:
        """Test that build_messages returns list of ChatCompletionMessageParam."""
        messages = service.build_messages(
            text=prompt_text,
            system_template=system_prompt_template["template"],
            user_template=user_prompt_template["template"],
            context={"instruction": "Test", "query": "Query"},
        )

        assert isinstance(messages, list)
        assert len(messages) > 0
        for message in messages:
            assert isinstance(message, dict)
            assert "role" in message
            assert "text" in message
