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
import re
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
        self.description: Optional[str] = None

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

        # Get GPU resources and memory information
        gpu_info = self._get_gpu_resources()
        estimated_batch_sizes = gpu_info["estimated_batch_size"]
        model_size = (
            self._estimate_model_size(self.model_name) if self.model_name else None
        )

        # Configure training params based on type
        if self.training_type == TrainingMethodType.FULL:
            # Adaptive batch size based on available memory
            batch_size = 1  # Default conservative value
            grad_accum = 16  # Default conservative value

            # If we have memory estimates, use them
            if "full" in estimated_batch_sizes:
                batch_size = estimated_batch_sizes["full"]
                # For small batch sizes, increase gradient accumulation
                if batch_size <= 2:
                    grad_accum = 32 if batch_size == 1 else 16
                else:
                    grad_accum = max(
                        1, 32 // batch_size
                    )  # Target effective batch size of ~32

            # Full model fine-tuning
            self.config.training = TrainingParams(
                trainer_type=TrainerType.TRL_SFT,
                per_device_train_batch_size=batch_size,
                gradient_accumulation_steps=grad_accum,
                learning_rate=2.0e-5,
                num_train_epochs=3,
                max_steps=-1,
                enable_gradient_checkpointing=True,
            )

            # FSDP settings based on model size and GPU count
            enable_fsdp = gpu_info["count"] > 1 or (model_size and model_size > 7)

            # FSDP for full fine-tuning is recommended for multi-GPU or large models
            self.config.fsdp = FSDPParams(
                enable_fsdp=enable_fsdp,
                sharding_strategy="HYBRID_SHARD" if enable_fsdp else "FULL_SHARD",
            )

            # No PEFT for full fine-tuning
            self.config.peft = PeftParams()

        elif self.training_type == TrainingMethodType.LORA:
            # Adaptive batch size based on available memory
            batch_size = 4  # Default reasonable value
            grad_accum = 8  # Default reasonable value

            # If we have memory estimates, use them
            if "lora" in estimated_batch_sizes:
                batch_size = estimated_batch_sizes["lora"]
                # For small batch sizes, increase gradient accumulation
                if batch_size <= 2:
                    grad_accum = 16 if batch_size == 1 else 8
                else:
                    grad_accum = max(
                        1, 32 // batch_size
                    )  # Target effective batch size of ~32

            # LoRA adapter fine-tuning
            self.config.training = TrainingParams(
                trainer_type=TrainerType.TRL_SFT,
                per_device_train_batch_size=batch_size,
                gradient_accumulation_steps=grad_accum,
                learning_rate=1.0e-4,
                num_train_epochs=3,
                max_steps=-1,
            )

            # Configure PEFT params for LoRA
            # For larger models, adjust LoRA rank
            lora_r = 16
            if model_size and model_size > 30:
                lora_r = 8  # Smaller rank for very large models to save memory

            self.config.peft = PeftParams(
                lora_r=lora_r,
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

            # Determine if we should use FSDP with LoRA
            use_fsdp_with_lora = (
                gpu_info["count"] >= 4 and model_size and model_size > 30
            )

            # Disable FSDP for PEFT by default (except for very large models with many GPUs)
            self.config.fsdp = FSDPParams(
                enable_fsdp=use_fsdp_with_lora,
                sharding_strategy="FULL_SHARD" if use_fsdp_with_lora else "",
            )

        elif self.training_type == TrainingMethodType.QLORA:
            # Adaptive batch size based on available memory
            batch_size = 4  # Default reasonable value
            grad_accum = 8  # Default reasonable value

            # If we have memory estimates, use them
            if "qlora" in estimated_batch_sizes:
                batch_size = estimated_batch_sizes["qlora"]
                # For small batch sizes, increase gradient accumulation
                if batch_size <= 2:
                    grad_accum = 16 if batch_size == 1 else 8
                else:
                    grad_accum = max(
                        1, 32 // batch_size
                    )  # Target effective batch size of ~32

            # QLoRA adapter fine-tuning
            self.config.training = TrainingParams(
                trainer_type=TrainerType.TRL_SFT,
                per_device_train_batch_size=batch_size,
                gradient_accumulation_steps=grad_accum,
                learning_rate=1.0e-4,
                num_train_epochs=3,
                max_steps=-1,
                bf16=False,  # 4-bit quantization requires fp16
                fp16=True,
            )

            # Configure PEFT params for QLoRA
            # For larger models, adjust LoRA rank
            lora_r = 16
            if model_size and model_size > 30:
                lora_r = 8  # Smaller rank for very large models to save memory

            self.config.peft = PeftParams(
                lora_r=lora_r,
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

            # Disable FSDP for QLoRA (not compatible)
            self.config.fsdp = FSDPParams(enable_fsdp=False)

        elif self.training_type == TrainingMethodType.AUTO:
            # Determine the best training method based on model size and available resources
            model_size = self._estimate_model_size(self.model_name)
            gpu_info = self._get_gpu_resources()

            # Choose training method based on model size and available GPU memory
            if model_size is None or gpu_info["total_memory_gb"] < 8:
                # Default to QLoRA for unknown models or with limited GPU memory
                self.set_training_type("qlora")
            elif model_size < 3 and gpu_info["total_memory_gb"] >= 24:
                # Small models with sufficient memory can use full fine-tuning
                self.set_training_type("full")
            elif model_size < 13 and gpu_info["total_memory_gb"] >= 40:
                # Medium models with sufficient memory can use full fine-tuning
                self.set_training_type("full")
            elif (
                model_size < 40
                and gpu_info["count"] >= 2
                and gpu_info["total_memory_gb"] >= 80
            ):
                # Large models with multiple GPUs can use LoRA
                self.set_training_type("lora")
            else:
                # Default to QLoRA for anything else
                self.set_training_type("qlora")

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

    def _estimate_model_size(self, model_name: str) -> Optional[float]:
        """Estimate the model size in billions of parameters based on the model name.

        Args:
            model_name: The name of the model

        Returns:
            Estimated size in billions of parameters, or None if unknown
        """
        # Look for common size indicators in model name (e.g., "7b", "13B", "70B", etc.)
        size_pattern = r"[-_/\s](\d+)\.?(\d)?[bB]"
        match = re.search(size_pattern, model_name)

        if match:
            # Extract the size from the match
            if match.group(2):  # Has decimal point
                size = float(f"{match.group(1)}.{match.group(2)}")
            else:
                size = float(match.group(1))
            return size

        # Check for known model families with specific sizes
        known_models = {
            # Small models
            "gpt2": 0.125,
            "gpt2-medium": 0.35,
            "gpt2-large": 0.774,
            "gpt2-xl": 1.5,
            "tiny": 0.1,  # Generic tiny models
            "small": 0.3,  # Generic small models
            # Medium models
            "phi-1": 1.3,
            "phi-1.5": 1.3,
            "phi-2": 2.7,
            "phi-3-mini": 3.8,
            "gemma-2-2b": 2.0,
            "gemma-2b": 2.0,
            "gemma-2-9b": 9.0,
            "gemma-9b": 9.0,
            "mistral-7b": 7.0,
            "mistral-v0.1": 7.0,
            "mistral-v0.2": 7.0,
            # Large models
            "phi-3": 14.0,
            "mixtral-8x7b": 56.0,  # 8 experts of 7B parameters
            "mixtral": 46.7,  # Generically assuming Mixtral
            "gemma-2-27b": 27.0,
            "gemma-27b": 27.0,
            "qwen1.5-7b": 7.0,
            "qwen1.5-14b": 14.0,
            "qwen1.5-72b": 72.0,
            "qwen2-7b": 7.0,
            "qwen2-57b": 57.0,
            "qwen3-32b": 32.0,
            # Very large models
            "gpt3": 175.0,
            "gpt-3.5": 175.0,
            "gpt-4": 1700.0,  # Estimated
            "llama-3-70b": 70.0,
            "llama-3.1-70b": 70.0,
            "llama-3.1-405b": 405.0,
        }

        # Try to match with known models
        for key, size in known_models.items():
            if key in model_name.lower():
                return size

        # Unknown model size
        return None

    def _estimate_memory_requirements(
        self, model_size_billions: Optional[float] = None
    ) -> Dict[str, float]:
        """Estimate memory requirements for different training methods.

        Args:
            model_size_billions: Model size in billions of parameters (if None, uses self.model_name)

        Returns:
            Dictionary with memory estimates in GB for different training methods
        """
        if model_size_billions is None and self.model_name:
            model_size_billions = self._estimate_model_size(self.model_name)

        if model_size_billions is None:
            # Default to a medium-sized model if we can't estimate
            model_size_billions = 7.0

        # Memory estimates are approximate
        # Full training typically needs ~4x model size for optimizer states, gradients, etc.
        # LoRA reduces memory by ~3x compared to full fine-tuning
        # QLoRA reduces memory by ~10x compared to full fine-tuning

        # Size of parameters in FP16 (2 bytes per parameter)
        model_size_gb = (
            model_size_billions * 2 / 1024
        )  # Convert billions of parameters to GB

        # Estimate for different training methods
        return {
            "full": {
                # Model weights + optimizer states + gradients + activations
                "per_gpu_estimate_gb": model_size_gb * 4,
                "minimum_total_gb": model_size_gb * 4,
            },
            "lora": {
                # Base model (frozen, so no optimizer states for most params) + LoRA params + optimizer states for LoRA
                "per_gpu_estimate_gb": model_size_gb * 1.5,
                "minimum_total_gb": model_size_gb * 1.5,
            },
            "qlora": {
                # Quantized model (4-bit) + LoRA params + optimizer states for LoRA
                "per_gpu_estimate_gb": model_size_gb * 0.5,
                "minimum_total_gb": model_size_gb * 0.5,
            },
        }

    def _get_gpu_resources(self) -> Dict[str, Any]:
        """Get information about available GPU resources.

        Returns:
            Dictionary with GPU count, total memory, and free memory
        """
        result = {
            "count": 0,
            "total_memory_gb": 0,
            "free_memory_gb": 0,
            "devices": [],
            "estimated_max_model_size_full": 0,  # Maximum model size (in B params) for full fine-tuning
            "estimated_max_model_size_lora": 0,  # Maximum model size (in B params) for LoRA
            "estimated_max_model_size_qlora": 0,  # Maximum model size (in B params) for QLoRA
            "estimated_batch_size": {},  # Estimated batch size for each training method
        }

        if not torch.cuda.is_available():
            return result

        # Get GPU count
        gpu_count = torch.cuda.device_count()
        result["count"] = gpu_count

        # Get memory information for each device
        for i in range(gpu_count):
            try:
                # Get device properties
                props = torch.cuda.get_device_properties(i)
                dev_info = {
                    "name": props.name,
                    "total_memory_gb": props.total_memory / (1024**3),  # Convert to GB
                }

                # Attempt to get free memory - might not be available on all platforms
                try:
                    torch.cuda.set_device(i)
                    torch.cuda.empty_cache()
                    free_mem = torch.cuda.memory_reserved(
                        i
                    ) - torch.cuda.memory_allocated(i)
                    dev_info["free_memory_gb"] = free_mem / (1024**3)  # Convert to GB
                except:
                    dev_info["free_memory_gb"] = 0

                result["devices"].append(dev_info)
                result["total_memory_gb"] += dev_info["total_memory_gb"]
                result["free_memory_gb"] += dev_info.get("free_memory_gb", 0)
            except Exception:
                # Skip this device if we can't get its properties
                continue

        # Estimate maximum model size that can be trained with each method
        # Assume we need 20% of GPU memory for other things (activations, etc.)
        usable_memory_gb = result["total_memory_gb"] * 0.8

        # Full fine-tuning: model weights + optimizer states + gradients (approximately 4x model size)
        result["estimated_max_model_size_full"] = (
            (usable_memory_gb / 4.0) * 1024 / 2
        )  # Convert GB to billions of parameters

        # LoRA: model weights + small optimizer states (approximately 1.5x model size)
        result["estimated_max_model_size_lora"] = (usable_memory_gb / 1.5) * 1024 / 2

        # QLoRA: 4-bit quantized model + small optimizer states (approximately 0.5x model size)
        result["estimated_max_model_size_qlora"] = (usable_memory_gb / 0.5) * 1024 / 2

        # Estimate batch sizes for a given model (if model_name is set)
        if self.model_name:
            model_size = self._estimate_model_size(self.model_name)
            if model_size:
                memory_requirements = self._estimate_memory_requirements(model_size)

                # Calculate batch size estimates for each training type
                for method, mem_estimate in memory_requirements.items():
                    # Simple heuristic: How many times can the required memory fit in the usable memory
                    # Subtract 20% for safety margin and reserve at least 2GB
                    safety_margin = max(usable_memory_gb * 0.2, 2.0)
                    available_mem = usable_memory_gb - safety_margin

                    # Base batch size on how many models can fit (minimum 1)
                    estimated_batch_size = max(
                        1, int(available_mem / mem_estimate["per_gpu_estimate_gb"])
                    )

                    # For very large models, we might need gradient accumulation instead
                    if estimated_batch_size < 1:
                        estimated_batch_size = 1

                    result["estimated_batch_size"][method] = estimated_batch_size

        return result

    def set_description(self, description: str) -> "ConfigBuilder":
        """Set a description for the configuration file.

        Args:
            description: Description of the configuration

        Returns:
            self for method chaining
        """
        self.description = description
        return self

    def build_yaml(self) -> str:
        """Build the configuration as YAML.

        Returns:
            YAML string representation of the configuration
        """
        # Instead of using the object serialization, we'll use templates
        # This gives us more control over the output format

        # Add description as YAML comments if provided
        description_yaml = ""
        if self.description:
            # Split description into lines and prefix each with #
            description_lines = self.description.strip().split("\n")
            description_yaml = "\n".join([f"# {line}" for line in description_lines])
            description_yaml += "\n\n"

        # Generate the base YAML based on config type
        if self.config_type == ConfigType.TRAIN:
            if not self.training_type:
                self.set_training_type("auto")
            base_yaml = templates.get_training_template(
                self.model_name, self.training_type.value, self.dataset_name
            )
        elif self.config_type == ConfigType.EVAL:
            base_yaml = templates.get_evaluation_template(self.model_name)
        elif self.config_type == ConfigType.INFER:
            base_yaml = templates.get_inference_template(self.model_name)
        else:
            # Fallback to using the object serialization
            config = self.build()
            string_io = io.StringIO()
            config.to_yaml(string_io)
            string_io.seek(0)
            base_yaml = string_io.read()

        # Combine description and base YAML
        return description_yaml + base_yaml

    def save(self, file_path: Union[str, Path]) -> None:
        """Save the configuration to a YAML file.

        Args:
            file_path: Path to the output file
        """
        file_path = Path(file_path)

        # Create parent directories if they don't exist
        file_path.parent.mkdir(parents=True, exist_ok=True)

        # Generate YAML and write to file
        yaml_content = self.build_yaml()
        with open(file_path, "w") as f:
            f.write(yaml_content)

        return file_path
