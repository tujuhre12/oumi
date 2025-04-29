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

"""Tests for the config wizard CLI."""

import os
import tempfile
from pathlib import Path

import pytest
from typer.testing import CliRunner

from oumi.cli.config import ConfigCreateType
from oumi.cli.main import get_app
from oumi.utils.wizard.config_builder import (
    ConfigBuilder,
    ConfigType,
    TrainingMethodType,
)


@pytest.fixture
def runner():
    """Create a CLI runner for tests."""
    return CliRunner()


def test_config_builder_init():
    """Test ConfigBuilder initialization."""
    builder = ConfigBuilder(ConfigType.TRAIN)
    assert builder.config_type == ConfigType.TRAIN

    builder = ConfigBuilder(ConfigType.EVAL)
    assert builder.config_type == ConfigType.EVAL

    builder = ConfigBuilder(ConfigType.INFER)
    assert builder.config_type == ConfigType.INFER


def test_config_builder_set_model():
    """Test ConfigBuilder.set_model."""
    builder = ConfigBuilder(ConfigType.TRAIN)
    builder.set_model("llama/test-model")

    assert builder.model_name == "llama/test-model"
    assert builder.config.model.model_name == "llama/test-model"


def test_config_builder_set_training_type():
    """Test ConfigBuilder.set_training_type."""
    builder = ConfigBuilder(ConfigType.TRAIN)
    builder.set_model("llama/test-model")

    # Test setting to LoRA
    builder.set_training_type("lora")
    assert builder.training_type == TrainingMethodType.LORA
    assert hasattr(builder.config.peft, "lora_r")
    assert builder.config.peft.lora_r == 16
    assert builder.config.fsdp.enable_fsdp is False

    # Test setting to full fine-tuning
    builder = ConfigBuilder(ConfigType.TRAIN)
    builder.set_model("llama/test-model")
    builder.set_training_type("full")
    assert builder.training_type == TrainingMethodType.FULL
    assert not hasattr(builder.config.peft, "q_lora") or not builder.config.peft.q_lora
    assert builder.config.fsdp.enable_fsdp is True


def test_config_builder_set_dataset():
    """Test ConfigBuilder.set_dataset."""
    builder = ConfigBuilder(ConfigType.TRAIN)
    builder.set_model("llama/test-model")
    builder.set_dataset("test-dataset")

    assert builder.dataset_name == "test-dataset"
    assert len(builder.config.data.train.datasets) == 1
    assert builder.config.data.train.datasets[0].dataset_name == "test-dataset"


def test_config_builder_build():
    """Test ConfigBuilder.build."""
    builder = ConfigBuilder(ConfigType.TRAIN)
    builder.set_model("llama/test-model")
    builder.set_training_type("lora")
    builder.set_dataset("test-dataset")

    config = builder.build()
    assert config.model.model_name == "llama/test-model"
    assert config.data.train.datasets[0].dataset_name == "test-dataset"
    assert config.peft.lora_r == 16
    assert config.training.trainer_type.value == "trl_sft"


def test_config_builder_build_yaml():
    """Test ConfigBuilder.build_yaml."""
    builder = ConfigBuilder(ConfigType.TRAIN)
    builder.set_model("llama/test-model")
    builder.set_training_type("lora")
    builder.set_dataset("test-dataset")

    yaml_str = builder.build_yaml()
    assert isinstance(yaml_str, str)
    assert "llama/test-model" in yaml_str
    assert "test-dataset" in yaml_str
    assert "r: 16" in yaml_str or "r: 16" in yaml_str.lower()
