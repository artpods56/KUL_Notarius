"""Dagster IO Manager using dill for serialization.

Dill extends pickle to serialize a wider range of Python objects including:
- Lambda functions
- Nested functions
- Class instances with complex state
- Interactive shell code

This is particularly useful for ML/AI pipelines where objects may contain
complex state, custom classes, or functions that pickle cannot handle.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import dill
from dagster import ConfigurableIOManager, InputContext, OutputContext
from dagster._core.storage.upath_io_manager import is_dict_type
from notarius.shared.constants import TMP_DIR


class DillIOManager(ConfigurableIOManager):
    """IO Manager that uses dill for serialization instead of pickle.

    Dill provides more robust serialization than pickle, handling:
    - Lambda functions and closures
    - Nested functions and classes
    - Interactive interpreter code
    - More complex object graphs

    Args:
        base_dir: Base directory for storing serialized objects.
                  Defaults to '/tmp/dagster_io' if not specified.
    """

    base_dir: str = str(TMP_DIR / "dagster_io")

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Ensure base directory exists
        Path(self.base_dir).mkdir(parents=True, exist_ok=True)

    def _get_path(self, context: OutputContext | InputContext) -> Path:
        """Generate file path for storing the asset.

        Args:
            context: Dagster context containing asset key information

        Returns:
            Path where the asset should be stored
        """
        # Get asset key parts (e.g., ["namespace", "asset_name"])
        parts = context.asset_key.path if context.asset_key else ["output"]

        # Create nested directory structure based on asset key
        path_parts = [self.base_dir, *parts[:-1]]
        directory = Path(*path_parts) if len(path_parts) > 1 else Path(self.base_dir)
        directory.mkdir(parents=True, exist_ok=True)

        # Use .dill extension to indicate dill serialization
        filename = f"{parts[-1]}.dill"
        return directory / filename

    def handle_output(self, context: OutputContext, obj: Any) -> None:
        """Store the output object using dill serialization.

        Args:
            context: Dagster output context
            obj: Object to serialize and store
        """
        if obj is None:
            return

        filepath = self._get_path(context)

        context.log.info(f"Storing asset to {filepath}")

        # Serialize with dill
        try:
            with open(filepath, "wb") as f:
                dill.dump(obj, f)

            # Log file size for debugging
            file_size = os.path.getsize(filepath)
            context.log.debug(f"Serialized object size: {file_size:,} bytes")

        except Exception as e:
            context.log.error(f"Failed to serialize object: {str(e)}")
            raise

    def load_input(self, context: InputContext) -> Any:
        """Load the input object using dill deserialization.

        Args:
            context: Dagster input context

        Returns:
            Deserialized object

        Raises:
            FileNotFoundError: If the asset file doesn't exist
        """
        filepath = self._get_path(context)

        if not filepath.exists():
            raise FileNotFoundError(
                f"Asset file not found: {filepath}. " f"Asset key: {context.asset_key}"
            )

        context.log.info(f"Loading asset from {filepath}")

        try:
            with open(filepath, "rb") as f:
                obj = dill.load(f)

            # Log loaded object type for debugging
            context.log.debug(f"Loaded object type: {type(obj).__name__}")
            return obj

        except Exception as e:
            context.log.error(f"Failed to deserialize object: {str(e)}")
            raise


class DillPickleIOManager(DillIOManager):
    """Alias for DillIOManager with a more descriptive name.

    Use this when you want to emphasize that you're using dill
    as a pickle replacement.
    """

    pass


def dill_io_manager(base_dir: str = "/tmp/dagster_io") -> DillIOManager:
    """Factory function to create a DillIOManager.

    Args:
        base_dir: Base directory for storing serialized objects

    Returns:
        Configured DillIOManager instance

    Example:
        ```python
        from dagster import Definitions
        from notarius.orchestration.dill_io_manager import dill_io_manager

        defs = Definitions(
            assets=[...],
            resources={
                "io_manager": dill_io_manager(base_dir="/path/to/storage"),
            },
        )
        ```
    """
    return DillIOManager(base_dir=base_dir)
