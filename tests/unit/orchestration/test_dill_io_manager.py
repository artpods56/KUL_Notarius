"""Tests for DillIOManager."""

import tempfile
from pathlib import Path
from typing import Callable

import pytest
from dagster import AssetKey, DagsterInstance, InputContext, OutputContext, build_input_context, build_output_context

from notarius.orchestration.dill_io_manager import DillIOManager, dill_io_manager


class ComplexObject:
    """Test object with complex state that might be hard to pickle."""

    def __init__(self, name: str, func: Callable[[int], int]):
        self.name = name
        self.func = func  # Functions can be tricky for pickle
        self.nested_data = {
            "lambda": lambda x: x * 2,  # Lambda functions
            "list_comp": [i**2 for i in range(10)],
        }


@pytest.fixture
def temp_storage_dir():
    """Create a temporary directory for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def io_manager(temp_storage_dir):
    """Create a DillIOManager instance for testing."""
    return DillIOManager(base_dir=temp_storage_dir)


@pytest.fixture
def output_context():
    """Create a mock output context."""
    return build_output_context(
        name="test_asset",
        asset_key=AssetKey(["test", "asset"]),
    )


@pytest.fixture
def input_context():
    """Create a mock input context."""
    return build_input_context(
        name="test_asset",
        asset_key=AssetKey(["test", "asset"]),
    )


def test_dill_io_manager_creation(temp_storage_dir):
    """Test that DillIOManager can be created."""
    manager = DillIOManager(base_dir=temp_storage_dir)
    assert manager.base_dir == temp_storage_dir
    assert Path(temp_storage_dir).exists()


def test_dill_io_manager_factory(temp_storage_dir):
    """Test the factory function."""
    manager = dill_io_manager(base_dir=temp_storage_dir)
    assert isinstance(manager, DillIOManager)
    assert manager.base_dir == temp_storage_dir


def test_handle_simple_output(io_manager, output_context, temp_storage_dir):
    """Test storing a simple object."""
    test_data = {"key": "value", "number": 42}

    io_manager.handle_output(output_context, test_data)

    # Check that file was created
    expected_path = Path(temp_storage_dir) / "test" / "asset.dill"
    assert expected_path.exists()


def test_load_simple_input(io_manager, output_context, input_context, temp_storage_dir):
    """Test loading a simple object."""
    test_data = {"key": "value", "number": 42}

    # Store the object
    io_manager.handle_output(output_context, test_data)

    # Load it back
    loaded_data = io_manager.load_input(input_context)

    assert loaded_data == test_data


def test_handle_complex_object(io_manager, output_context, input_context):
    """Test storing and loading a complex object with functions."""
    # Create a complex object with a lambda
    complex_obj = ComplexObject(name="test", func=lambda x: x**2)

    # Store the object
    io_manager.handle_output(output_context, complex_obj)

    # Load it back
    loaded_obj = io_manager.load_input(input_context)

    # Verify the object was preserved
    assert loaded_obj.name == "test"
    assert loaded_obj.func(3) == 9  # Lambda should work
    assert loaded_obj.nested_data["lambda"](5) == 10  # Nested lambda should work


def test_handle_lambda_function(io_manager, output_context, input_context):
    """Test that dill can handle lambda functions (pickle cannot)."""
    # This is where dill shines - it can serialize lambdas
    test_lambda = lambda x: x * 2 + 1

    # Store the lambda
    io_manager.handle_output(output_context, test_lambda)

    # Load it back
    loaded_lambda = io_manager.load_input(input_context)

    # Verify it works
    assert loaded_lambda(5) == 11
    assert loaded_lambda(10) == 21


def test_handle_nested_functions(io_manager, output_context, input_context):
    """Test that dill can handle nested functions."""

    def outer_func(x):
        def inner_func(y):
            return x + y

        return inner_func

    # Create a closure
    closure = outer_func(10)

    # Store the closure
    io_manager.handle_output(output_context, closure)

    # Load it back
    loaded_closure = io_manager.load_input(input_context)

    # Verify it works
    assert loaded_closure(5) == 15


def test_handle_none_output(io_manager, output_context):
    """Test that None values are handled gracefully."""
    # Should not raise an error
    io_manager.handle_output(output_context, None)


def test_load_nonexistent_file(io_manager, input_context):
    """Test that loading a non-existent file raises appropriate error."""
    with pytest.raises(FileNotFoundError, match="Asset file not found"):
        io_manager.load_input(input_context)


def test_path_generation_with_nested_keys(temp_storage_dir):
    """Test that nested asset keys create proper directory structure."""
    manager = DillIOManager(base_dir=temp_storage_dir)

    context = build_output_context(
        name="nested_asset",
        asset_key=AssetKey(["level1", "level2", "level3", "asset"]),
    )

    test_data = {"nested": "data"}
    manager.handle_output(context, test_data)

    # Check directory structure
    expected_path = (
        Path(temp_storage_dir) / "level1" / "level2" / "level3" / "asset.dill"
    )
    assert expected_path.exists()


def test_roundtrip_with_list_of_objects(io_manager, output_context, input_context):
    """Test storing and loading a list of complex objects."""
    test_objects = [
        ComplexObject(name=f"obj_{i}", func=lambda x, i=i: x * i) for i in range(5)
    ]

    # Store the list
    io_manager.handle_output(output_context, test_objects)

    # Load it back
    loaded_objects = io_manager.load_input(input_context)

    # Verify all objects were preserved
    assert len(loaded_objects) == 5
    for i, obj in enumerate(loaded_objects):
        assert obj.name == f"obj_{i}"
        assert obj.func(2) == 2 * i


def test_integration_with_dagster_types(temp_storage_dir):
    """Test that the IO manager integrates properly with Dagster."""
    from dagster import asset, materialize

    manager = dill_io_manager(base_dir=temp_storage_dir)

    @asset
    def test_asset() -> dict:
        """Test asset that returns a dict."""
        return {"result": "success", "lambda": lambda x: x * 2}

    # Materialize the asset with our IO manager
    result = materialize(
        [test_asset],
        resources={"io_manager": manager},
    )

    assert result.success
