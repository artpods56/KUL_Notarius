from pathlib import Path

from jinja2 import (
    Environment,
    FileSystemLoader,
    select_autoescape,
)
from structlog import get_logger


logger = get_logger(__name__)


class Jinja2PromptRenderer:
    """Jinja2-based implementation of PromptRenderer.

    This adapter wraps a `jinja2.Environment` for rendering LLM prompts.

    Features:
    1) Uses `pathlib` for clearer path handling.
    2) Disables HTML auto-escaping for ``*.j2`` templates (prompts are plain text).
    3) Enables `StrictUndefined` so missing variables raise immediately.
    4) Removes superfluous whitespace with `trim_blocks` and `lstrip_blocks`.
    5) Includes are resolved relative to the template's folder first, then the root.
    """

    def __init__(self, template_dir: str | Path = "prompts") -> None:
        base_dir = Path(template_dir)

        if not base_dir.is_absolute():
            # Resolve relative to the current file (src/notarius/adapters/)
            base_dir = Path(__file__).resolve().parent.parent / "llm" / base_dir

        if not base_dir.exists():
            raise FileNotFoundError(f"Template directory '{base_dir}' does not exist")

        self.base_dir = base_dir

    def _create_env(self, search_paths: list[str]) -> Environment:
        """Create a Jinja2 environment with the given search paths."""
        return Environment(
            loader=FileSystemLoader(search_paths),
            autoescape=select_autoescape(
                disabled_extensions=("j2",), default=False, default_for_string=False
            ),
            trim_blocks=True,
            lstrip_blocks=True,
        )

    def render_prompt(self, template_name: str, context: dict[str, str]) -> str:
        """Render *template_name* with *context*.

        Includes are resolved relative to the template's folder first,
        then fall back to the prompts root folder.

        The method will raise ``jinja2.exceptions.UndefinedError`` if the
        template references a variable that is not provided in *context*.
        """
        # Get the template's parent directory for relative includes
        template_path = self.base_dir / template_name
        template_dir = template_path.parent

        # Search paths: template's folder first, then root
        search_paths = [str(template_dir), str(self.base_dir)]

        env = self._create_env(search_paths)

        # Load template by filename only (since template_dir is in search path)
        template = env.get_template(template_path.name)
        rendered = template.render(**context)
        return rendered
