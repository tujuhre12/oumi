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

"""Configuration wizard utilities.

This module provides the Oumi Configuration Wizard, a tool for creating valid configuration
files for training, evaluation, and inference through both a CLI interface and a programmatic API.

Example usage:

    # Programmatic API
    from oumi.utils.wizard import create_train_config, create_eval_config, create_infer_config

    # Create a training configuration with LoRA
    config = create_train_config(
        model="meta-llama/Llama-3.1-8B-Instruct",
        training_type="lora",
        dataset="yahma/alpaca-cleaned",
    )

    # Save to YAML file
    config.save("my_config.yaml")

    # Create an evaluation configuration
    eval_config = create_eval_config(
        model="meta-llama/Llama-3.1-8B-Instruct",
    )

    # Create an inference configuration
    infer_config = create_infer_config(
        model="meta-llama/Llama-3.1-8B-Instruct",
    )
"""

from oumi.utils.wizard.config_builder import (
    ConfigBuilder,
    ConfigType,
    TrainingMethodType,
)


def create_train_config(
    model: str,
    training_type: str = "auto",
    dataset: str = None,
    description: str = None,
) -> ConfigBuilder:
    """Create a training configuration.

    Args:
        model: Model name or HF identifier
        training_type: Training type (full, lora, qlora, auto)
        dataset: Optional dataset name or HF identifier
        description: Optional description of the configuration

    Returns:
        ConfigBuilder object with the configuration
    """
    builder = ConfigBuilder(ConfigType.TRAIN)
    builder.set_model(model)
    builder.set_training_type(training_type)

    if dataset:
        builder.set_dataset(dataset)

    if description:
        builder.set_description(description)

    return builder


def create_eval_config(
    model: str,
    description: str = None,
) -> ConfigBuilder:
    """Create an evaluation configuration.

    Args:
        model: Model name or HF identifier
        description: Optional description of the configuration

    Returns:
        ConfigBuilder object with the configuration
    """
    builder = ConfigBuilder(ConfigType.EVAL)
    builder.set_model(model)

    if description:
        builder.set_description(description)

    return builder


def create_infer_config(
    model: str,
    description: str = None,
) -> ConfigBuilder:
    """Create an inference configuration.

    Args:
        model: Model name or HF identifier
        description: Optional description of the configuration

    Returns:
        ConfigBuilder object with the configuration
    """
    builder = ConfigBuilder(ConfigType.INFER)
    builder.set_model(model)

    if description:
        builder.set_description(description)

    return builder
