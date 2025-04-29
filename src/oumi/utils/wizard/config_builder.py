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

"""Configuration builder for the Oumi config wizard."""

import io
import os
from enum import Enum, auto
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import jinja2
import torch
import yaml
from omegaconf import OmegaConf

from oumi.core.configs import (
    DataParams,
    DatasetParams,
    DatasetSplitParams,
    EvaluationConfig,
    FSDPParams,
    InferenceConfig,
    ModelParams,
    PeftParams,
    TrainingConfig,
    TrainingParams,
)
from oumi.core.configs.params.training_params import TrainerType
from oumi.utils.wizard import templates


class ConfigType(Enum):
    """Configuration types."""

    TRAIN = auto()
    EVAL = auto()
    INFER = auto()


class TrainingMethodType(Enum):
    """Training method types."""

    FULL = "full"  # Full model fine-tuning
    LORA = "lora"  # LoRA adapter fine-tuning
    QLORA = "qlora"  # Quantized LoRA fine-tuning
    AUTO = "auto"  # Automatically determine based on model size and available resources


class ConfigBuilder:
    """Builder for Oumi configurations."""

    def __init__(self, config_type: ConfigType = ConfigType.TRAIN):
        """Initialize the config builder.

        Args:
            config_type: Type of configuration to build
        """
        self.config_type = config_type
        self.model_name: Optional[str] = None
        self.training_type: Optional[TrainingMethodType] = None
        self.dataset_name: Optional[str] = None

        # Initialize config objects based on type
        if config_type == ConfigType.TRAIN:
            self.config = TrainingConfig()
        elif config_type == ConfigType.EVAL:
            self.config = EvaluationConfig()
        elif config_type == ConfigType.INFER:
            self.config = InferenceConfig()
        else:
            raise ValueError(f"Unknown config type: {config_type}")

    def set_model(self, model_name: str) -> "ConfigBuilder":
        """Set the model name.

        Args:
            model_name: Name or HF identifier of the model

        Returns:
            self for method chaining
        """
        self.model_name = model_name

        model_params = ModelParams(
            model_name=model_name,
            model_max_length=8192,  # Default reasonable value
            torch_dtype_str="bfloat16" if torch.cuda.is_available() else "float32",
            attn_implementation="sdpa",
        )

        if self.config_type == ConfigType.TRAIN:
            self.config.model = model_params
        elif self.config_type == ConfigType.EVAL:
            self.config.model = model_params
        elif self.config_type == ConfigType.INFER:
            self.config.model = model_params

        return self

    def set_training_type(self, training_type: str) -> "ConfigBuilder":
        """Set the training type.

        Args:
            training_type: Type of training (full, lora, qlora, auto)

        Returns:
            self for method chaining
        """
        if self.config_type != ConfigType.TRAIN:
            raise ValueError("Training type can only be set for training configs")

        try:
            self.training_type = TrainingMethodType(training_type.lower())
        except ValueError:
            raise ValueError(f"Invalid training type: {training_type}")

        # Configure training params based on type
        if self.training_type == TrainingMethodType.FULL:
            # Full model fine-tuning
            self.config.training = TrainingParams(
                trainer_type=TrainerType.TRL_SFT,
                per_device_train_batch_size=1,
                gradient_accumulation_steps=16,
                learning_rate=2.0e-5,
                num_train_epochs=3,
                max_steps=-1,
                enable_gradient_checkpointing=True,
            )
            # FSDP for full fine-tuning is recommended
            self.config.fsdp = FSDPParams(
                enable_fsdp=True,
                sharding_strategy="HYBRID_SHARD",
            )
            # No PEFT for full fine-tuning
            self.config.peft = PeftParams()

        elif self.training_type == TrainingMethodType.LORA:
            # LoRA adapter fine-tuning
            self.config.training = TrainingParams(
                trainer_type=TrainerType.TRL_SFT,
                per_device_train_batch_size=4,
                gradient_accumulation_steps=8,
                learning_rate=1.0e-4,
                num_train_epochs=3,
                max_steps=-1,
            )
            # Configure PEFT params for LoRA
            self.config.peft = PeftParams(
                lora_r=16,
                lora_alpha=32,
                lora_dropout=0.05,
                lora_target_modules=[
                    "q_proj",
                    "k_proj",
                    "v_proj",
                    "o_proj",
                    "gate_proj",
                    "up_proj",
                    "down_proj",
                ],
            )
            # Disable FSDP for PEFT
            self.config.fsdp = FSDPParams(enable_fsdp=False)

        elif self.training_type == TrainingMethodType.QLORA:
            # QLoRA adapter fine-tuning
            self.config.training = TrainingParams(
                trainer_type=TrainerType.TRL_SFT,
                per_device_train_batch_size=4,
                gradient_accumulation_steps=8,
                learning_rate=1.0e-4,
                num_train_epochs=3,
                max_steps=-1,
                bf16=False,  # 4-bit quantization requires fp16
                fp16=True,
            )
            # Configure PEFT params for QLoRA
            self.config.peft = PeftParams(
                lora_r=16,
                lora_alpha=32,
                lora_dropout=0.05,
                lora_target_modules=[
                    "q_proj",
                    "k_proj",
                    "v_proj",
                    "o_proj",
                    "gate_proj",
                    "up_proj",
                    "down_proj",
                ],
                # QLoRA specific params
                q_lora=True,
                q_lora_bits=4,
                bnb_4bit_compute_dtype="float16",
                bnb_4bit_quant_type="nf4",
                use_bnb_nested_quant=True,
            )
            # Disable FSDP for PEFT
            self.config.fsdp = FSDPParams(enable_fsdp=False)

        elif self.training_type == TrainingMethodType.AUTO:
            # Will determine based on model size and resources
            # For now, default to QLoRA as it's most memory efficient
            self.set_training_type("qlora")  # Recursive call with specific type

        return self

    def set_dataset(self, dataset_name: str) -> "ConfigBuilder":
        """Set the dataset.

        Args:
            dataset_name: Name or HF identifier of the dataset

        Returns:
            self for method chaining
        """
        if self.config_type != ConfigType.TRAIN:
            raise ValueError("Dataset can only be set for training configs")

        self.dataset_name = dataset_name

        # Create dataset params
        dataset_params = DatasetParams(
            dataset_name=dataset_name,
        )

        # Add to data params
        self.config.data = DataParams(
            train=DatasetSplitParams(
                datasets=[dataset_params],
            )
        )

        return self

    def build(self) -> Any:
        """Build the configuration object.

        Returns:
            Configuration object based on the config type
        """
        # Apply any final validations or defaults
        if self.config_type == ConfigType.TRAIN:
            # Ensure we have the minimum required fields
            if not self.model_name:
                raise ValueError("Model name is required for training configs")

            # If training type wasn't explicitly set, use auto
            if not self.training_type:
                self.set_training_type("auto")

        # Return the config object
        return self.config

    def build_yaml(self) -> str:
        """Build the configuration as YAML.

        Returns:
            YAML string representation of the configuration
        """
        # Instead of using the object serialization, we'll use templates
        # This gives us more control over the output format
        if self.config_type == ConfigType.TRAIN:
            if not self.training_type:
                self.set_training_type("auto")
            return templates.get_training_template(
                self.model_name, self.training_type.value, self.dataset_name
            )
        elif self.config_type == ConfigType.EVAL:
            return templates.get_evaluation_template(self.model_name)
        elif self.config_type == ConfigType.INFER:
            return templates.get_inference_template(self.model_name)
        else:
            # Fallback to using the object serialization
            config = self.build()
            string_io = io.StringIO()
            config.to_yaml(string_io)
            string_io.seek(0)
            return string_io.read()
