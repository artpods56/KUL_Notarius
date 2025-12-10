"""Service for constructing LLM prompts and messages."""

from typing import Any, final

from PIL.Image import Image as PILImage
from openai.types.chat import ChatCompletionMessageParam
from structlog import get_logger

from notarius.infrastructure.llm.prompt_manager import Jinja2PromptRenderer
from notarius.infrastructure.llm.utils import (
    construct_system_message,
    construct_user_text_message,
    construct_image_message,
)

from notarius.shared.logger import Logger

logger: Logger = get_logger(__name__)


@final
class PromptConstructionService:
    """Service for rendering prompts and constructing chat messages.

    Encapsulates the logic for:
    - Rendering Jinja2 templates with context
    - Building system and user messages
    - Handling multimodal (image + text) inputs
    """

    def __init__(
        self,
        prompt_renderer: Jinja2PromptRenderer,
    ):
        """Initialize the service.

        Args:
            prompt_renderer: Renderer for Jinja2 templates
        """
        self.prompt_renderer = prompt_renderer

    def build_messages(
        self,
        image: PILImage | None = None,
        text: str | None = None,
        system_template: str = "system.j2",
        user_template: str = "user.j2",
        context: dict[str, Any] | None = None,
    ) -> list[ChatCompletionMessageParam]:
        """Build chat completion messages from inputs.

        Args:
            image: Optional PIL Image for vision processing
            text: Optional text text (currently unused in message construction)
            system_template: Name of system text template
            user_template: Name of user text template
            context: Context dictionary for template rendering

        Returns:
            List of chat completion messages

        Raises:
            ValueError: If neither image nor text is provided
        """
        if image is None and text is None:
            raise ValueError("At least one of 'image' or 'text' must be provided")

        if context is None:
            context = {}

        # Render prompts from templates
        system_prompt = self.prompt_renderer.render_prompt(system_template, context)
        user_prompt = self.prompt_renderer.render_prompt(user_template, context)

        messages: list[ChatCompletionMessageParam] = [
            construct_system_message(system_prompt)
        ]

        if image is not None:
            messages.append(construct_image_message(image, user_prompt))
        else:
            messages.append(construct_user_text_message(user_prompt))

        return messages
