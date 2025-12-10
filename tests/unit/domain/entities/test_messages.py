"""Tests for domain message entities."""

import pytest

from notarius.domain.entities.messages import (
    TextContent,
    ImageContent,
    ChatMessage,
    MessageContent,
    ChatMessageList,
)


class TestTextContent:
    """Test suite for TextContent entity."""

    def test_create_text_content_with_defaults(self):
        """Test creating TextContent with default type."""
        content = TextContent(text="Hello, world!")

        assert content.text == "Hello, world!"
        assert content.type == "input_text"

    def test_create_text_content_explicit_type(self):
        """Test creating TextContent with explicit type."""
        content = TextContent(text="Test message", type="input_text")

        assert content.text == "Test message"
        assert content.type == "input_text"

    def test_text_content_is_frozen(self):
        """Test that TextContent is immutable."""
        content = TextContent(text="Immutable")

        with pytest.raises(Exception):  # FrozenInstanceError or AttributeError
            content.text = "Modified"

    def test_text_content_with_empty_string(self):
        """Test TextContent with empty string."""
        content = TextContent(text="")

        assert content.text == ""
        assert content.type == "input_text"

    def test_text_content_with_multiline_text(self):
        """Test TextContent with multiline text."""
        multiline = "Line 1\nLine 2\nLine 3"
        content = TextContent(text=multiline)

        assert content.text == multiline
        assert "\n" in content.text

    def test_text_content_with_unicode(self):
        """Test TextContent with unicode characters."""
        content = TextContent(text="Hello 世界 🌍")

        assert content.text == "Hello 世界 🌍"


class TestImageContent:
    """Test suite for ImageContent entity."""

    def test_create_image_content_with_defaults(self):
        """Test creating ImageContent with default values."""
        content = ImageContent(image_url="https://example.com/image.jpg")

        assert content.image_url == "https://example.com/image.jpg"
        assert content.detail == "auto"
        assert content.type == "input_image"

    def test_create_image_content_with_custom_detail(self):
        """Test creating ImageContent with custom detail level."""
        content = ImageContent(
            image_url="data:image/jpeg;base64,/9j/4AAQ...",
            detail="high",
        )

        assert content.detail == "high"

    @pytest.mark.parametrize("detail_level", ["auto", "low", "high"])
    def test_image_content_all_detail_levels(self, detail_level):
        """Test all valid detail levels."""
        content = ImageContent(
            image_url="https://example.com/image.jpg",
            detail=detail_level,
        )

        assert content.detail == detail_level

    def test_image_content_is_frozen(self):
        """Test that ImageContent is immutable."""
        content = ImageContent(image_url="https://example.com/image.jpg")

        with pytest.raises(Exception):  # FrozenInstanceError or AttributeError
            content.image_url = "https://example.com/modified.jpg"

    def test_image_content_with_data_url(self):
        """Test ImageContent with base64 data URL."""
        data_url = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAUA"
        content = ImageContent(image_url=data_url)

        assert content.image_url == data_url
        assert content.image_url.startswith("data:image/")

    def test_image_content_with_http_url(self):
        """Test ImageContent with HTTP URL."""
        url = "http://example.com/image.png"
        content = ImageContent(image_url=url)

        assert content.image_url == url


class TestChatMessage:
    """Test suite for ChatMessage entity."""

    @pytest.fixture
    def sample_text_content(self) -> MessageContent:
        """Fixture for simple text content."""
        return [TextContent(text="Hello!")]

    @pytest.fixture
    def sample_multimodal_content(self) -> MessageContent:
        """Fixture for multimodal content."""
        return [
            TextContent(text="What's in this image?"),
            ImageContent(image_url="https://example.com/image.jpg"),
        ]

    def test_create_simple_text_message(self, sample_text_content):
        """Test creating a simple text message."""
        message = ChatMessage(
            role="user",
            content=sample_text_content,
        )

        assert message.role == "user"
        assert len(message.content) == 1
        assert isinstance(message.content[0], TextContent)
        assert message.content[0].text == "Hello!"

    def test_create_multimodal_message(self, sample_multimodal_content):
        """Test creating a multimodal message with text and image."""
        message = ChatMessage(
            role="user",
            content=sample_multimodal_content,
        )

        assert message.role == "user"
        assert len(message.content) == 2
        assert isinstance(message.content[0], TextContent)
        assert isinstance(message.content[1], ImageContent)

    @pytest.mark.parametrize("role", ["user", "system", "developer"])
    def test_create_message_with_all_roles(self, role, sample_text_content):
        """Test creating messages with all valid roles."""
        message = ChatMessage(
            role=role,
            content=sample_text_content,
        )

        assert message.role == role

    def test_message_is_frozen(self, sample_text_content):
        """Test that ChatMessage is immutable."""
        message = ChatMessage(
            role="user",
            content=sample_text_content,
        )

        with pytest.raises(Exception):  # FrozenInstanceError or AttributeError
            message.role = "system"

    def test_message_with_empty_content_list(self):
        """Test creating a message with empty content list."""
        message = ChatMessage(role="user", content=[])

        assert message.content == []
        assert len(message.content) == 0

    def test_message_with_multiple_text_parts(self):
        """Test message with multiple text content parts."""
        content = [
            TextContent(text="First part"),
            TextContent(text="Second part"),
            TextContent(text="Third part"),
        ]
        message = ChatMessage(role="user", content=content)

        assert len(message.content) == 3
        assert all(isinstance(part, TextContent) for part in message.content)

    def test_message_with_multiple_images(self):
        """Test message with multiple image content parts."""
        content = [
            ImageContent(image_url="https://example.com/image1.jpg"),
            ImageContent(image_url="https://example.com/image2.jpg", detail="high"),
        ]
        message = ChatMessage(role="user", content=content)

        assert len(message.content) == 2
        assert all(isinstance(part, ImageContent) for part in message.content)
        assert message.content[1].detail == "high"

    def test_system_message_creation(self):
        """Test creating a system message."""
        content = [TextContent(text="You are a helpful assistant.")]
        message = ChatMessage(role="system", content=content)

        assert message.role == "system"
        assert message.content[0].text == "You are a helpful assistant."

    def test_developer_message_creation(self):
        """Test creating a developer message."""
        content = [TextContent(text="Use structured output format.")]
        message = ChatMessage(role="developer", content=content)

        assert message.role == "developer"


class TestChatMessageList:
    """Test suite for ChatMessageList type alias."""

    def test_create_conversation(self):
        """Test creating a full input."""
        conversation: ChatMessageList = [
            ChatMessage(
                role="system",
                content=[TextContent(text="You are helpful.")],
            ),
            ChatMessage(
                role="user",
                content=[TextContent(text="Hello!")],
            ),
            ChatMessage(
                role="user",
                content=[
                    TextContent(text="What's this?"),
                    ImageContent(image_url="https://example.com/img.jpg"),
                ],
            ),
        ]

        assert len(conversation) == 3
        assert conversation[0].role == "system"
        assert conversation[1].role == "user"
        assert len(conversation[2].content) == 2

    def test_empty_conversation(self):
        """Test creating an empty input."""
        conversation: ChatMessageList = []

        assert len(conversation) == 0
        assert conversation == []


class TestMessageContentFactories:
    """Test suite for common message content patterns using fixtures."""

    @pytest.fixture
    def text_only_content(self) -> MessageContent:
        """Factory for text-only content."""
        return [TextContent(text="Simple text message")]

    @pytest.fixture
    def image_only_content(self) -> MessageContent:
        """Factory for image-only content."""
        return [ImageContent(image_url="https://example.com/image.jpg")]

    @pytest.fixture
    def text_and_image_content(self) -> MessageContent:
        """Factory for text + image content."""
        return [
            TextContent(text="Describe this image:"),
            ImageContent(
                image_url="data:image/jpeg;base64,/9j/4AAQ...",
                detail="high",
            ),
        ]

    def test_text_only_pattern(self, text_only_content):
        """Test text-only message pattern."""
        message = ChatMessage(role="user", content=text_only_content)

        assert len(message.content) == 1
        assert isinstance(message.content[0], TextContent)

    def test_image_only_pattern(self, image_only_content):
        """Test image-only message pattern."""
        message = ChatMessage(role="user", content=image_only_content)

        assert len(message.content) == 1
        assert isinstance(message.content[0], ImageContent)

    def test_text_and_image_pattern(self, text_and_image_content):
        """Test text + image message pattern."""
        message = ChatMessage(role="user", content=text_and_image_content)

        assert len(message.content) == 2
        assert isinstance(message.content[0], TextContent)
        assert isinstance(message.content[1], ImageContent)
        assert message.content[1].detail == "high"
