"""Domain entities for LLM messages.

These domain types are provider-agnostic and represent the core message
structure used throughout the application. Provider-specific adapters
translate between these domain types and their native formats.

Supports both simple text messages and multimodal messages (text + images).
"""

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class TextContent:
    """Text content part of a multimodal message."""

    text: str
    type: Literal["input_text"] = "input_text"


@dataclass(frozen=True)
class ImageContent:
    """Image content part of a multimodal message."""

    image_url: str
    detail: Literal["auto", "low", "high"] = "auto"
    type: Literal["input_image"] = "input_image"


# Union of all content types
ContentPart = TextContent | ImageContent

MessageContent = list[ContentPart]


@dataclass(frozen=True)
class ChatMessage:
    """A message in a input.

    Supports both simple text messages and multimodal messages with images.

    Attributes:
        role: The role of the message sender
        content: Either a simple text string or a list of content parts (text, images)

    Examples:
        # Simple text message
        ChatMessage(role="user", content="Hello!")

        # Multimodal message with text and image
        ChatMessage(
            role="user",
            content=[
                TextContent(text="What's in this image?"),
                ImageContent(image_url="data:image/jpeg;base64,...")
            ]
        )
    """

    role: Literal["user", "system", "developer", "assistant"]
    content: MessageContent


ChatMessageList = list[ChatMessage] | tuple[ChatMessage, ...]


