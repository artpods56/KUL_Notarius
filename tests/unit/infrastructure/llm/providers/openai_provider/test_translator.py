"""Tests for OpenAI message translator.

This module tests the translation of domain message entities to OpenAI's
native format using singledispatch pattern.
"""

import pytest
from openai.types.responses import (
    ResponseInputTextParam,
    ResponseInputImageParam,
)
from openai.types.responses.response_input_param import Message

from notarius.domain.entities.messages import (
    TextContent,
    ImageContent,
    ChatMessage,
    MessageContent,
    ChatMessageList,
)
from notarius.infrastructure.llm.providers.openai_provider.translator import (
    translate,
    content_to_openai,
    messages_to_openai,
)


class TestTranslateSingledispatch:
    """Test suite for the singledispatch translate function."""

    def test_translate_text_content(self):
        """Test translating TextContent to OpenAI format."""
        text_part = TextContent(text="Hello, world!")

        result = translate(text_part)

        # TypedDict doesn't support isinstance, check structure instead
        assert isinstance(result, dict)
        assert result["text"] == "Hello, world!"
        assert result["type"] == "input_text"

    def test_translate_text_content_with_special_characters(self):
        """Test translating TextContent with special characters."""
        text_part = TextContent(text="Special chars: \n\t\"'\\")

        result = translate(text_part)

        assert result["text"] == "Special chars: \n\t\"'\\"
        assert result["type"] == "input_text"

    def test_translate_text_content_with_unicode(self):
        """Test translating TextContent with unicode."""
        text_part = TextContent(text="Unicode: 世界 🌍 café")

        result = translate(text_part)

        assert result["text"] == "Unicode: 世界 🌍 café"

    def test_translate_image_content_with_url(self):
        """Test translating ImageContent with HTTP URL."""
        image_part = ImageContent(
            image_url="https://example.com/image.jpg",
            detail="high",
        )

        result = translate(image_part)

        # TypedDict doesn't support isinstance, check structure instead
        assert isinstance(result, dict)
        assert result["image_url"] == "https://example.com/image.jpg"
        assert result["detail"] == "high"
        assert result["type"] == "input_image"

    def test_translate_image_content_with_data_url(self):
        """Test translating ImageContent with base64 data URL."""
        data_url = "data:image/jpeg;base64,/9j/4AAQSkZJRg=="
        image_part = ImageContent(image_url=data_url)

        result = translate(image_part)

        assert result["image_url"] == data_url
        assert result["detail"] == "auto"  # default value

    @pytest.mark.parametrize("detail_level", ["auto", "low", "high"])
    def test_translate_image_content_all_detail_levels(self, detail_level):
        """Test translating ImageContent with all detail levels."""
        image_part = ImageContent(
            image_url="https://example.com/image.jpg",
            detail=detail_level,
        )

        result = translate(image_part)

        assert result["detail"] == detail_level

    def test_translate_unsupported_type_raises_error(self):
        """Test that translating unsupported type raises ValueError."""

        class UnsupportedContent:
            pass

        unsupported = UnsupportedContent()

        with pytest.raises(ValueError, match="No translator for"):
            translate(unsupported)


class TestContentToOpenAI:
    """Test suite for content_to_openai function."""

    @pytest.fixture
    def text_only_content(self) -> MessageContent:
        """Fixture for text-only content."""
        return [TextContent(text="Simple text message")]

    @pytest.fixture
    def image_only_content(self) -> MessageContent:
        """Fixture for image-only content."""
        return [ImageContent(image_url="https://example.com/image.jpg")]

    @pytest.fixture
    def multimodal_content(self) -> MessageContent:
        """Fixture for multimodal content."""
        return [
            TextContent(text="What's in this image?"),
            ImageContent(
                image_url="https://example.com/photo.jpg",
                detail="high",
            ),
        ]

    @pytest.fixture
    def complex_multimodal_content(self) -> MessageContent:
        """Fixture for complex multimodal content with multiple parts."""
        return [
            TextContent(text="First instruction"),
            ImageContent(image_url="https://example.com/img1.jpg"),
            TextContent(text="Second instruction"),
            ImageContent(
                image_url="https://example.com/img2.jpg",
                detail="low",
            ),
            TextContent(text="Final instruction"),
        ]

    def test_content_to_openai_text_only(self, text_only_content):
        """Test converting text-only content."""
        result = content_to_openai(text_only_content)

        assert len(result) == 1
        assert result[0]["type"] == "input_text"
        assert result[0]["text"] == "Simple text message"

    def test_content_to_openai_image_only(self, image_only_content):
        """Test converting image-only content."""
        result = content_to_openai(image_only_content)

        assert len(result) == 1
        assert result[0]["type"] == "input_image"
        assert result[0]["image_url"] == "https://example.com/image.jpg"

    def test_content_to_openai_multimodal(self, multimodal_content):
        """Test converting multimodal content (text + image)."""
        result = content_to_openai(multimodal_content)

        assert len(result) == 2
        assert result[0]["type"] == "input_text"
        assert result[0]["text"] == "What's in this image?"
        assert result[1]["type"] == "input_image"
        assert result[1]["detail"] == "high"

    def test_content_to_openai_complex_multimodal(self, complex_multimodal_content):
        """Test converting complex multimodal content."""
        result = content_to_openai(complex_multimodal_content)

        assert len(result) == 5
        # Verify alternating pattern
        assert result[0]["type"] == "input_text"
        assert result[1]["type"] == "input_image"
        assert result[2]["type"] == "input_text"
        assert result[3]["type"] == "input_image"
        assert result[4]["type"] == "input_text"
        # Verify detail levels
        assert result[3]["detail"] == "low"

    def test_content_to_openai_empty_list(self):
        """Test converting empty content list."""
        result = content_to_openai([])

        assert result == []
        assert len(result) == 0

    def test_content_to_openai_preserves_order(self):
        """Test that content order is preserved during conversion."""
        content = [
            TextContent(text="First"),
            TextContent(text="Second"),
            TextContent(text="Third"),
        ]

        result = content_to_openai(content)

        assert result[0]["text"] == "First"
        assert result[1]["text"] == "Second"
        assert result[2]["text"] == "Third"


class TestMessagesToOpenAI:
    """Test suite for messages_to_openai function."""

    @pytest.fixture
    def simple_conversation(self) -> ChatMessageList:
        """Fixture for simple text-only input."""
        return [
            ChatMessage(
                role="system",
                content=[TextContent(text="You are a helpful assistant.")],
            ),
            ChatMessage(
                role="user",
                content=[TextContent(text="Hello!")],
            ),
        ]

    @pytest.fixture
    def multimodal_conversation(self) -> ChatMessageList:
        """Fixture for multimodal input."""
        return [
            ChatMessage(
                role="system",
                content=[TextContent(text="You analyze images.")],
            ),
            ChatMessage(
                role="user",
                content=[
                    TextContent(text="What's in this image?"),
                    ImageContent(
                        image_url="https://example.com/photo.jpg",
                        detail="high",
                    ),
                ],
            ),
        ]

    @pytest.fixture
    def developer_role_conversation(self) -> ChatMessageList:
        """Fixture for input with developer role."""
        return [
            ChatMessage(
                role="developer",
                content=[TextContent(text="Use JSON format.")],
            ),
            ChatMessage(
                role="user",
                content=[TextContent(text="Extract data from image.")],
            ),
        ]

    def test_messages_to_openai_simple_conversation(self, simple_conversation):
        """Test converting simple text-only input."""
        result = messages_to_openai(simple_conversation)

        assert len(result) == 2
        # TypedDict doesn't support isinstance, check structure instead
        assert isinstance(result[0], dict)
        assert result[0]["role"] == "system"
        assert len(result[0]["content"]) == 1
        assert result[0]["content"][0]["text"] == "You are a helpful assistant."
        assert result[1]["role"] == "user"
        assert result[1]["content"][0]["text"] == "Hello!"

    def test_messages_to_openai_multimodal_conversation(self, multimodal_conversation):
        """Test converting multimodal input."""
        result = messages_to_openai(multimodal_conversation)

        assert len(result) == 2
        # System message
        assert result[0]["role"] == "system"
        assert len(result[0]["content"]) == 1
        # User message with text and image
        assert result[1]["role"] == "user"
        assert len(result[1]["content"]) == 2
        assert result[1]["content"][0]["type"] == "input_text"
        assert result[1]["content"][1]["type"] == "input_image"
        assert result[1]["content"][1]["detail"] == "high"

    def test_messages_to_openai_developer_role(self, developer_role_conversation):
        """Test converting input with developer role."""
        result = messages_to_openai(developer_role_conversation)

        assert len(result) == 2
        assert result[0]["role"] == "developer"
        assert result[1]["role"] == "user"

    def test_messages_to_openai_empty_list(self):
        """Test converting empty message list."""
        result = messages_to_openai([])

        assert result == []

    def test_messages_to_openai_preserves_message_order(self):
        """Test that message order is preserved."""
        conversation = [
            ChatMessage(role="system", content=[TextContent(text="First")]),
            ChatMessage(role="user", content=[TextContent(text="Second")]),
            ChatMessage(role="user", content=[TextContent(text="Third")]),
        ]

        result = messages_to_openai(conversation)

        assert len(result) == 3
        assert result[0]["content"][0]["text"] == "First"
        assert result[1]["content"][0]["text"] == "Second"
        assert result[2]["content"][0]["text"] == "Third"

    def test_messages_to_openai_with_multiple_images(self):
        """Test converting message with multiple images."""
        conversation = [
            ChatMessage(
                role="user",
                content=[
                    TextContent(text="Compare these images:"),
                    ImageContent(image_url="https://example.com/img1.jpg"),
                    ImageContent(
                        image_url="https://example.com/img2.jpg",
                        detail="low",
                    ),
                ],
            ),
        ]

        result = messages_to_openai(conversation)

        assert len(result) == 1
        assert len(result[0]["content"]) == 3
        assert result[0]["content"][0]["type"] == "input_text"
        assert result[0]["content"][1]["type"] == "input_image"
        assert result[0]["content"][2]["type"] == "input_image"
        assert result[0]["content"][2]["detail"] == "low"


class TestIntegrationScenarios:
    """Integration tests for common usage scenarios."""

    @pytest.fixture
    def full_annotation_workflow(self) -> ChatMessageList:
        """Fixture simulating a full document annotation workflow."""
        return [
            ChatMessage(
                role="system",
                content=[
                    TextContent(
                        text="You are an expert at extracting structured data from historical documents."
                    )
                ],
            ),
            ChatMessage(
                role="developer",
                content=[
                    TextContent(
                        text="Output must be valid JSON with fields: parish, deanery, dedication."
                    )
                ],
            ),
            ChatMessage(
                role="user",
                content=[
                    TextContent(text="Extract information from this schematism page:"),
                    ImageContent(
                        image_url="data:image/jpeg;base64,/9j/4AAQSkZJRg==",
                        detail="high",
                    ),
                    TextContent(text="OCR text: St. Mary Parish, Warsaw Deanery"),
                ],
            ),
        ]

    def test_full_annotation_workflow(self, full_annotation_workflow):
        """Test complete annotation workflow translation."""
        result = messages_to_openai(full_annotation_workflow)

        assert len(result) == 3
        # System message
        assert result[0]["role"] == "system"
        assert "historical documents" in result[0]["content"][0]["text"]
        # Developer message
        assert result[1]["role"] == "developer"
        assert "JSON" in result[1]["content"][0]["text"]
        # User message with multimodal content
        assert result[2]["role"] == "user"
        assert len(result[2]["content"]) == 3
        assert result[2]["content"][1]["type"] == "input_image"
        assert result[2]["content"][1]["detail"] == "high"
        assert "OCR text" in result[2]["content"][2]["text"]

    def test_data_url_vs_http_url(self):
        """Test handling both data URLs and HTTP URLs."""
        conversation = [
            ChatMessage(
                role="user",
                content=[
                    ImageContent(
                        image_url="https://example.com/remote.jpg",
                        detail="auto",
                    ),
                    ImageContent(
                        image_url="data:image/png;base64,iVBORw0KGgo=",
                        detail="high",
                    ),
                ],
            ),
        ]

        result = messages_to_openai(conversation)

        assert len(result[0]["content"]) == 2
        # HTTP URL
        assert result[0]["content"][0]["image_url"].startswith("https://")
        assert result[0]["content"][0]["detail"] == "auto"
        # Data URL
        assert result[0]["content"][1]["image_url"].startswith("data:image/")
        assert result[0]["content"][1]["detail"] == "high"


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_translate_with_empty_text(self):
        """Test translating TextContent with empty string."""
        text_part = TextContent(text="")

        result = translate(text_part)

        assert result["text"] == ""
        assert result["type"] == "input_text"

    def test_translate_with_very_long_text(self):
        """Test translating very long text content."""
        long_text = "A" * 10000
        text_part = TextContent(text=long_text)

        result = translate(text_part)

        assert len(result["text"]) == 10000
        assert result["text"] == long_text

    def test_content_to_openai_with_single_item(self):
        """Test content conversion with single item."""
        content = [TextContent(text="Single item")]

        result = content_to_openai(content)

        assert len(result) == 1

    def test_messages_to_openai_single_message(self):
        """Test conversion with single message."""
        messages = [ChatMessage(role="user", content=[TextContent(text="Hello")])]

        result = messages_to_openai(messages)

        assert len(result) == 1
        assert result[0]["role"] == "user"

    def test_all_roles_in_conversation(self):
        """Test input using all available roles."""
        conversation = [
            ChatMessage(role="system", content=[TextContent(text="System")]),
            ChatMessage(role="developer", content=[TextContent(text="Developer")]),
            ChatMessage(role="user", content=[TextContent(text="User")]),
        ]

        result = messages_to_openai(conversation)

        assert len(result) == 3
        assert result[0]["role"] == "system"
        assert result[1]["role"] == "developer"
        assert result[2]["role"] == "user"
