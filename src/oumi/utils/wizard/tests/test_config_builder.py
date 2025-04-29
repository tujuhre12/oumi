# Copyright 2025 - Oumi
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for the ConfigBuilder class."""

import os
import tempfile
from pathlib import Path
import pytest
import yaml
from unittest.mock import patch

from oumi.utils.wizard.config_builder import (
    ConfigBuilder,
    ConfigType,
    TrainingMethodType,
)
from oumi.utils.wizard import (
    create_train_config,
    create_eval_config,
    create_infer_config,
)


def test_config_builder_initialization():
    """Test that the ConfigBuilder initializes correctly."""
    # Test train config
    builder = ConfigBuilder(config_type=ConfigType.TRAIN)
    assert builder.config_type == ConfigType.TRAIN

    # Test eval config
    builder = ConfigBuilder(config_type=ConfigType.EVAL)
    assert builder.config_type == ConfigType.EVAL

    # Test infer config
    builder = ConfigBuilder(config_type=ConfigType.INFER)
    assert builder.config_type == ConfigType.INFER


def test_set_model():
    """Test setting the model."""
    builder = ConfigBuilder(config_type=ConfigType.TRAIN)
    builder.set_model("meta-llama/Llama-3.1-8B-Instruct")

    assert builder.model_name == "meta-llama/Llama-3.1-8B-Instruct"
    assert builder.config.model.model_name == "meta-llama/Llama-3.1-8B-Instruct"


def test_set_training_type():
    """Test setting the training type."""
    builder = ConfigBuilder(config_type=ConfigType.TRAIN)
    builder.set_training_type("lora")

    assert builder.training_type == TrainingMethodType.LORA
    assert builder.config.peft is not None
    assert builder.config.peft.lora_r == 16  # Check default LoRA rank


def test_set_dataset():
    """Test setting the dataset."""
    builder = ConfigBuilder(config_type=ConfigType.TRAIN)
    builder.set_dataset("yahma/alpaca-cleaned")

    assert builder.dataset_name == "yahma/alpaca-cleaned"
    assert builder.config.data.train.datasets[0].dataset_name == "yahma/alpaca-cleaned"


def test_build_yaml():
    """Test building the YAML configuration."""
    builder = ConfigBuilder(config_type=ConfigType.TRAIN)
    builder.set_model("meta-llama/Llama-3.1-8B-Instruct")
    builder.set_training_type("qlora")
    builder.set_dataset("yahma/alpaca-cleaned")
    builder.set_description("Test configuration")

    yaml_str = builder.build_yaml()

    # Check that the YAML is valid
    config_dict = yaml.safe_load(yaml_str)

    # Check that the description is included as a comment
    assert "# Test configuration" in yaml_str

    # Check that the model is set correctly
    assert config_dict["model"]["model_name"] == "meta-llama/Llama-3.1-8B-Instruct"

    # Check that the dataset is set correctly
    assert (
        config_dict["data"]["train"]["datasets"][0]["dataset_name"]
        == "yahma/alpaca-cleaned"
    )


def test_model_size_estimation():
    """Test model size estimation from model names."""
    builder = ConfigBuilder()

    # Test exact matches
    assert builder._estimate_model_size("meta-llama/Llama-3.1-8B-Instruct") == 8.0
    assert builder._estimate_model_size("meta-llama/Llama-3-70B") == 70.0
    assert builder._estimate_model_size("model-7b") == 7.0

    # Test known model families
    assert builder._estimate_model_size("gpt2") == 0.125
    assert builder._estimate_model_size("phi-3-mini") == 3.8

    # Test unknown model
    assert builder._estimate_model_size("unknown-model") is None


@patch("torch.cuda.is_available")
@patch("torch.cuda.device_count")
@patch("torch.cuda.get_device_properties")
def test_get_gpu_resources(mock_props, mock_count, mock_available):
    """Test GPU resource detection."""
    # Mock GPU unavailable
    mock_available.return_value = False

    builder = ConfigBuilder()
    gpu_info = builder._get_gpu_resources()

    assert gpu_info["count"] == 0
    assert gpu_info["total_memory_gb"] == 0

    # Mock GPU available with 2 devices
    mock_available.return_value = True
    mock_count.return_value = 2

    # Create mock properties for GPUs
    class MockProps:
        def __init__(self, name, total_memory):
            self.name = name
            self.total_memory = total_memory

    mock_props.side_effect = [
        MockProps("NVIDIA A100", 40 * 1024**3),  # 40 GB
        MockProps("NVIDIA A100", 40 * 1024**3),  # 40 GB
    ]

    # Re-test with mocked GPUs
    gpu_info = builder._get_gpu_resources()

    assert gpu_info["count"] == 2
    assert len(gpu_info["devices"]) == 2
    assert gpu_info["total_memory_gb"] == 80.0


def test_save_config():
    """Test saving the configuration to a file."""
    builder = ConfigBuilder(config_type=ConfigType.TRAIN)
    builder.set_model("meta-llama/Llama-3.1-8B-Instruct")
    builder.set_training_type("qlora")
    builder.set_dataset("yahma/alpaca-cleaned")

    # Create a temporary file
    with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as tmp:
        try:
            # Save the config to the temporary file
            builder.save(tmp.name)

            # Check that the file exists
            assert os.path.exists(tmp.name)

            # Load the file and check its contents
            with open(tmp.name, "r") as f:
                content = f.read()
                config_dict = yaml.safe_load(content)

                # Verify the config matches our expectations
                assert (
                    config_dict["model"]["model_name"]
                    == "meta-llama/Llama-3.1-8B-Instruct"
                )
                assert (
                    config_dict["data"]["train"]["datasets"][0]["dataset_name"]
                    == "yahma/alpaca-cleaned"
                )
        finally:
            # Clean up the temporary file
            os.unlink(tmp.name)


def test_programmatic_api():
    """Test the programmatic API functions."""
    # Test creating a training config
    train_config = create_train_config(
        model="meta-llama/Llama-3.1-8B-Instruct",
        training_type="lora",
        dataset="yahma/alpaca-cleaned",
        description="Test training config",
    )

    # Verify it's a ConfigBuilder
    assert isinstance(train_config, ConfigBuilder)
    assert train_config.config_type == ConfigType.TRAIN
    assert train_config.model_name == "meta-llama/Llama-3.1-8B-Instruct"
    assert train_config.training_type == TrainingMethodType.LORA
    assert train_config.dataset_name == "yahma/alpaca-cleaned"
    assert train_config.description == "Test training config"

    # Test creating an evaluation config
    eval_config = create_eval_config(
        model="meta-llama/Llama-3.1-8B-Instruct", description="Test eval config"
    )

    # Verify it's a ConfigBuilder
    assert isinstance(eval_config, ConfigBuilder)
    assert eval_config.config_type == ConfigType.EVAL
    assert eval_config.model_name == "meta-llama/Llama-3.1-8B-Instruct"
    assert eval_config.description == "Test eval config"

    # Test creating an inference config
    infer_config = create_infer_config(model="meta-llama/Llama-3.1-8B-Instruct")

    # Verify it's a ConfigBuilder
    assert isinstance(infer_config, ConfigBuilder)
    assert infer_config.config_type == ConfigType.INFER
    assert infer_config.model_name == "meta-llama/Llama-3.1-8B-Instruct"


def test_memory_requirements_estimation():
    """Test memory requirements estimation."""
    builder = ConfigBuilder()

    # Test with a specific model size
    memory_req = builder._estimate_memory_requirements(model_size_billions=7.0)

    # Check that we have estimates for all training types
    assert "full" in memory_req
    assert "lora" in memory_req
    assert "qlora" in memory_req

    # Check that the estimates are reasonable
    # Full training should need more memory than LoRA, which should need more than QLoRA
    assert (
        memory_req["full"]["per_gpu_estimate_gb"]
        > memory_req["lora"]["per_gpu_estimate_gb"]
    )
    assert (
        memory_req["lora"]["per_gpu_estimate_gb"]
        > memory_req["qlora"]["per_gpu_estimate_gb"]
    )


def test_lint():
    """Test the lint method for error detection."""
    # Create a builder with missing model (should trigger errors)
    builder = ConfigBuilder(config_type=ConfigType.TRAIN)
    issues = builder.lint()

    # Should have error about missing model
    assert len(issues["errors"]) > 0
    assert any("Missing model name" in error for error in issues["errors"])

    # Create a valid builder
    builder = ConfigBuilder(config_type=ConfigType.TRAIN)
    builder.set_model("meta-llama/Llama-3.1-8B-Instruct")
    builder.set_training_type("full")

    # Configure invalid settings (batch size too small)
    builder.config.training.per_device_train_batch_size = 0

    # Check for validation errors
    issues = builder.lint()
    assert len(issues["errors"]) > 0
    assert any("Batch size must be at least 1" in error for error in issues["errors"])


def test_analyze():
    """Test the analyze method for performance analysis."""
    builder = ConfigBuilder(config_type=ConfigType.TRAIN)
    builder.set_model("meta-llama/Llama-3.1-8B-Instruct")
    builder.set_training_type("qlora")
    builder.set_dataset("yahma/alpaca-cleaned")

    # Run the analysis
    analysis = builder.analyze()

    # Check that we have all the expected sections
    assert "overview" in analysis
    assert "performance" in analysis
    assert "resources" in analysis
    assert "recommendations" in analysis

    # Check that basic info is captured
    assert analysis["overview"]["model_name"] == "meta-llama/Llama-3.1-8B-Instruct"
    assert analysis["overview"]["training_type"] == "qlora"

    # Performance metrics should be populated
    if "performance" in analysis:
        assert "per_device_batch_size" in analysis["performance"]
        assert "learning_rate" in analysis["performance"]


def test_validate():
    """Test the validate method."""
    # Create an invalid builder
    builder = ConfigBuilder(config_type=ConfigType.TRAIN)
    assert not builder.validate()  # Should fail validation

    # Create a valid builder
    builder = ConfigBuilder(config_type=ConfigType.TRAIN)
    builder.set_model("meta-llama/Llama-3.1-8B-Instruct")
    builder.set_training_type("qlora")

    # Should pass validation
    assert builder.validate()
