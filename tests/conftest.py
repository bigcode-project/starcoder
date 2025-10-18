"""Shared pytest fixtures for StarCoder tests.

This module provides common fixtures used across unit and integration tests.
"""

import os
import shutil
import tempfile
from pathlib import Path
from typing import Generator, Dict, Any
from unittest.mock import MagicMock

import pytest


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for test files.

    Yields:
        Path: Path to the temporary directory.

    Example:
        def test_file_creation(temp_dir):
            test_file = temp_dir / "test.txt"
            test_file.write_text("test content")
            assert test_file.exists()
    """
    tmp_dir = tempfile.mkdtemp()
    try:
        yield Path(tmp_dir)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


@pytest.fixture
def temp_file(temp_dir: Path) -> Generator[Path, None, None]:
    """Create a temporary file for testing.

    Args:
        temp_dir: Temporary directory fixture.

    Yields:
        Path: Path to the temporary file.

    Example:
        def test_file_reading(temp_file):
            temp_file.write_text("test content")
            assert temp_file.read_text() == "test content"
    """
    tmp_file = temp_dir / "test_file.txt"
    tmp_file.touch()
    yield tmp_file


@pytest.fixture
def mock_config() -> Dict[str, Any]:
    """Provide a mock configuration for testing.

    Returns:
        Dict: Mock configuration dictionary.

    Example:
        def test_config_usage(mock_config):
            assert mock_config["model_name"] == "test-model"
    """
    return {
        "model_name": "test-model",
        "max_length": 100,
        "batch_size": 4,
        "learning_rate": 1e-4,
        "num_epochs": 3,
        "device": "cpu",
    }


@pytest.fixture
def mock_tokenizer() -> MagicMock:
    """Create a mock tokenizer for testing.

    Returns:
        MagicMock: Mock tokenizer object.

    Example:
        def test_tokenization(mock_tokenizer):
            result = mock_tokenizer.encode("test text")
            assert result == [1, 2, 3, 4]
    """
    tokenizer = MagicMock()
    tokenizer.encode.return_value = [1, 2, 3, 4]
    tokenizer.decode.return_value = "decoded text"
    tokenizer.vocab_size = 50000
    tokenizer.pad_token_id = 0
    tokenizer.eos_token_id = 2
    return tokenizer


@pytest.fixture
def mock_model() -> MagicMock:
    """Create a mock model for testing.

    Returns:
        MagicMock: Mock model object.

    Example:
        def test_model_inference(mock_model):
            result = mock_model.generate([1, 2, 3])
            assert result is not None
    """
    model = MagicMock()
    model.generate.return_value = [[1, 2, 3, 4, 5]]
    model.config.return_value = {"hidden_size": 768}
    return model


@pytest.fixture
def sample_text() -> str:
    """Provide sample text for testing.

    Returns:
        str: Sample text string.

    Example:
        def test_text_processing(sample_text):
            assert len(sample_text) > 0
    """
    return "def hello_world():\n    print('Hello, World!')"


@pytest.fixture
def sample_code_snippet() -> str:
    """Provide a sample code snippet for testing.

    Returns:
        str: Sample Python code snippet.

    Example:
        def test_code_parsing(sample_code_snippet):
            assert "def" in sample_code_snippet
    """
    return """
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

def main():
    result = fibonacci(10)
    print(f"Fibonacci(10) = {result}")

if __name__ == "__main__":
    main()
"""


@pytest.fixture
def mock_dataset() -> MagicMock:
    """Create a mock dataset for testing.

    Returns:
        MagicMock: Mock dataset object.

    Example:
        def test_dataset_loading(mock_dataset):
            assert len(mock_dataset) > 0
    """
    dataset = MagicMock()
    dataset.__len__.return_value = 100
    dataset.__getitem__.return_value = {
        "text": "sample text",
        "labels": [1, 2, 3],
    }
    return dataset


@pytest.fixture
def mock_training_args() -> Dict[str, Any]:
    """Provide mock training arguments for testing.

    Returns:
        Dict: Mock training arguments.

    Example:
        def test_training_setup(mock_training_args):
            assert mock_training_args["learning_rate"] > 0
    """
    return {
        "output_dir": "./test_output",
        "num_train_epochs": 3,
        "per_device_train_batch_size": 4,
        "per_device_eval_batch_size": 4,
        "learning_rate": 5e-5,
        "warmup_steps": 100,
        "logging_steps": 10,
        "save_steps": 100,
        "evaluation_strategy": "steps",
        "eval_steps": 50,
    }


@pytest.fixture(autouse=True)
def reset_environment() -> Generator[None, None, None]:
    """Reset environment variables after each test.

    Yields:
        None

    Note:
        This fixture runs automatically for all tests.
    """
    original_env = os.environ.copy()
    yield
    os.environ.clear()
    os.environ.update(original_env)


@pytest.fixture
def mock_huggingface_hub() -> MagicMock:
    """Create a mock HuggingFace Hub client.

    Returns:
        MagicMock: Mock HuggingFace Hub client.

    Example:
        def test_model_upload(mock_huggingface_hub):
            mock_huggingface_hub.push_to_hub("model")
            mock_huggingface_hub.push_to_hub.assert_called_once()
    """
    hub = MagicMock()
    hub.list_models.return_value = ["model1", "model2"]
    hub.model_info.return_value = {"id": "test-model", "downloads": 1000}
    return hub
