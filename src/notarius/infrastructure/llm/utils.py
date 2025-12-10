import base64
from io import BytesIO
from pathlib import Path
from typing import Any, Literal
from collections.abc import Sequence
import uuid
from PIL import Image
from openai.types.chat import (
    ChatCompletionMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
    ChatCompletionContentPartTextParam,
    ChatCompletionContentPartImageParam,
)
from openai.types.responses import (
    EasyInputMessageParam,
    ResponseInputImage,
    ResponseInputMessageItem,
    ResponseInputParam,
    ResponseInputText,
    ResponseInputMessageContentList,
    ResponseInputItemParam,
    ResponseInputImageParam,
    ResponseInputTextParam,
)

from notarius.domain.entities.messages import ChatMessage, TextContent, ImageContent
from notarius.infrastructure.llm.prompt_manager import Jinja2PromptRenderer


# ... (all your other imports)


def make_all_properties_required(schema: dict) -> dict:
    """
    Recursively modifies a JSON schema to make all properties required,
    to comply with strict API validation rules (like some Azure OpenAI endpoints).
    """
    if "properties" in schema and isinstance(schema["properties"], dict):
        # Make all properties in the current level required
        schema["required"] = list(schema["properties"].keys())

        # Recurse into nested properties (for nested objects)
        for prop_name, prop_schema in schema["properties"].items():
            if isinstance(prop_schema, dict):
                make_all_properties_required(prop_schema)

    # Recurse into array items (for lists of objects)
    if "items" in schema and isinstance(schema["items"], dict):
        make_all_properties_required(schema["items"])

    # Recurse into $defs (for referenced schemas)
    if "$defs" in schema and isinstance(schema["$defs"], dict):
        for def_name, def_schema in schema["$defs"].items():
            if isinstance(def_schema, dict):
                make_all_properties_required(def_schema)

    return schema


def parse_model_name(model_name: str) -> str:
    """Parse model name to ensure it is in a valid file format.

    Examples:
        /ml_models/gemma-3-27b-it-Q4_K_M.gguf -> gemma-3-27b-it-Q4_K_M
        gpt-4/turbo -> gpt-4_turbo
    """
    if Path(model_name).is_absolute():
        model_name = model_name.split("/")[-1].split(".")[0]

    return model_name.replace("/", "_")


def encode_image_to_base64(pil_image) -> str:
    """Convert PIL image to base64 string."""
    buffer = BytesIO()
    pil_image.convert("RGB").save(buffer, format="JPEG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def generate_id() -> str:
    return str(uuid.uuid4())


def construct_text_message(
    text: str, role: Literal["user", "system", "developer", "assistant"]
) -> ChatMessage:
    """Construct text-only message from template using domain types."""

    return ChatMessage(
        role=role,
        content=[TextContent(text=text)],
    )


def construct_image_message(
    pil_image: Image.Image,
    text: str,
    role: Literal["user", "system", "developer", "assistant"],
    detail: Literal["auto", "low", "high"] = "auto",
) -> ChatMessage:
    """Construct multimodal message with image and text using domain types.

    Args:
        pil_image: PIL Image object
        text: Text prompt to accompany the image
        role: ChatMessage role
        detail: Image detail level for processing

    Returns:
        Domain ChatMessage with text and image content parts
    """

    # Ensure image is in RGB format
    if pil_image.mode != "RGB":
        pil_image = pil_image.convert("RGB")

    # Convert to base64 data URL
    base64_image = encode_image_to_base64(pil_image)
    image_url = f"data:image/jpeg;base64,{base64_image}"

    return ChatMessage(
        role=role,
        content=[
            TextContent(text=text),
            ImageContent(image_url=image_url, detail=detail),
        ],
    )
