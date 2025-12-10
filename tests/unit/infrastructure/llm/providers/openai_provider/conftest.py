"""Shared fixtures for OpenAI provider tests."""

import pytest

from notarius.domain.entities.messages import (
    TextContent,
    ImageContent,
    ChatMessage,
    MessageContent,
    ChatMessageList,
)


# ============================================================================
# Content Part Fixtures
# ============================================================================


@pytest.fixture
def text_content_simple() -> TextContent:
    """Simple text content fixture."""
    return TextContent(text="Hello, world!")


@pytest.fixture
def text_content_multiline() -> TextContent:
    """Multiline text content fixture."""
    return TextContent(text="Line 1\nLine 2\nLine 3")


@pytest.fixture
def text_content_unicode() -> TextContent:
    """Unicode text content fixture."""
    return TextContent(text="Hello 世界 🌍 café")


@pytest.fixture
def image_content_http() -> ImageContent:
    """Image content with HTTP URL fixture."""
    return ImageContent(
        image_url="https://example.com/image.jpg",
        detail="auto",
    )


@pytest.fixture
def image_content_data_url() -> ImageContent:
    """Image content with base64 data URL fixture."""
    return ImageContent(
        image_url="data:image/jpeg;base64,/9j/4AAQSkZJRg==",
        detail="high",
    )


@pytest.fixture
def image_content_high_detail() -> ImageContent:
    """Image content with high detail setting."""
    return ImageContent(
        image_url="https://example.com/photo.jpg",
        detail="high",
    )


# ============================================================================
# MessageContent Fixtures (Lists of ContentParts)
# ============================================================================


@pytest.fixture
def message_content_text_only() -> MessageContent:
    """Text-only message content."""
    return [TextContent(text="Simple text message")]


@pytest.fixture
def message_content_image_only() -> MessageContent:
    """Image-only message content."""
    return [ImageContent(image_url="https://example.com/image.jpg")]


@pytest.fixture
def message_content_multimodal() -> MessageContent:
    """Multimodal message content (text + image)."""
    return [
        TextContent(text="What's in this image?"),
        ImageContent(
            image_url="https://example.com/photo.jpg",
            detail="high",
        ),
    ]


@pytest.fixture
def message_content_complex() -> MessageContent:
    """Complex multimodal content with multiple parts."""
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


@pytest.fixture
def message_content_multiple_images() -> MessageContent:
    """Message content with multiple images."""
    return [
        TextContent(text="Compare these images:"),
        ImageContent(image_url="https://example.com/img1.jpg"),
        ImageContent(
            image_url="https://example.com/img2.jpg",
            detail="low",
        ),
    ]


# ============================================================================
# ChatMessage Fixtures
# ============================================================================


@pytest.fixture
def chat_message_user() -> ChatMessage:
    """Simple user message."""
    return ChatMessage(
        role="user",
        content=[TextContent(text="Hello!")],
    )


@pytest.fixture
def chat_message_system() -> ChatMessage:
    """System message."""
    return ChatMessage(
        role="system",
        content=[TextContent(text="You are a helpful assistant.")],
    )


@pytest.fixture
def chat_message_developer() -> ChatMessage:
    """Developer message."""
    return ChatMessage(
        role="developer",
        content=[TextContent(text="Use JSON format.")],
    )


@pytest.fixture
def chat_message_multimodal() -> ChatMessage:
    """Multimodal user message."""
    return ChatMessage(
        role="user",
        content=[
            TextContent(text="What's in this image?"),
            ImageContent(
                image_url="https://example.com/photo.jpg",
                detail="high",
            ),
        ],
    )


# ============================================================================
# ChatMessageList Fixtures (Conversations)
# ============================================================================


@pytest.fixture
def conversation_simple() -> ChatMessageList:
    """Simple text-only input."""
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
def conversation_multimodal() -> ChatMessageList:
    """Multimodal input with images."""
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
def conversation_developer_role() -> ChatMessageList:
    """Conversation with developer role."""
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


@pytest.fixture
def conversation_annotation_workflow() -> ChatMessageList:
    """Full annotation workflow input."""
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
