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

"""CLI commands for the Config Wizard.

The Config Wizard provides an interactive way to create Oumi configuration files
for training, evaluation, and inference. It guides users through the configuration
process, offering sensible defaults and validating inputs.

Example usage:
    # Create a training configuration for a model
    oumi config create train --model meta-llama/Llama-3.1-8B-Instruct

    # Create a specific training configuration with LoRA
    oumi config create train --model meta-llama/Llama-3.1-8B-Instruct --training-type lora

    # Create an evaluation configuration
    oumi config create eval --model meta-llama/Llama-3.1-8B-Instruct

    # Create an inference configuration and save to file
    oumi config create infer --model meta-llama/Llama-3.1-8B-Instruct --output my_config.yaml
"""

import os
import sys
from enum import Enum
from pathlib import Path
from typing import Optional

import typer
from prompt_toolkit import PromptSession
from prompt_toolkit.completion import PathCompleter, WordCompleter
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.styles import Style
from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table

from oumi.cli.cli_utils import CONSOLE
from oumi.utils.wizard import config_builder, templates
from oumi.utils.wizard.config_builder import ConfigType, TrainingMethodType

# Style for prompt toolkit
prompt_style = Style.from_dict(
    {
        "prompt": "ansicyan bold",
        "input": "ansiwhite",
    }
)

# Create a session for prompt toolkit
session = PromptSession()


class ConfigCreateType(str, Enum):
    """Enum for config wizard types."""

    TRAIN = "train"
    EVAL = "eval"
    INFER = "infer"


def _get_model_list():
    """Get a list of supported models for autocompletion."""
    # This will be expanded in Milestone 2 to dynamically fetch from HF
    return [
        "meta-llama/Llama-3.1-8B-Instruct",
        "meta-llama/Llama-3.1-70B-Instruct",
        "meta-llama/Llama-3.1-405B-Instruct",
        "meta-llama/Llama-3.2-1B-Instruct",
        "meta-llama/Llama-3.2-3B-Instruct",
        "microsoft/phi-3-mini-4k-instruct",
        "google/gemma-2-2b-it",
        "google/gemma-2-9b-it",
        "google/gemma-2-27b-it",
        "mistralai/Mistral-7B-v0.3",
        "Qwen/Qwen1.5-7B-Chat",
        "Qwen/Qwen1.5-14B-Chat",
        "Qwen/Qwen1.5-72B-Chat",
        "Qwen/Qwen2-7B-Instruct",
        "Qwen/Qwen3-32B-Instruct",
    ]


def _get_training_type_list():
    """Get a list of training types for autocompletion."""
    return ["full", "lora", "qlora", "auto"]


def _get_dataset_list():
    """Get a list of common datasets for autocompletion."""
    return [
        "yahma/alpaca-cleaned",
        "databricks/databricks-dolly-15k",
        "tatsu-lab/alpaca",
        "HuggingFaceH4/ultrachat_200k",
        "epfLLM/tulu-v2-sft-mixture",
        "lmsys/chatbot_arena_conversations",
        "openaccess-ai-collective/aya-collection-all",
        "mlabonne/guanaco-llama2-1k",
    ]


def create(
    config_type: ConfigCreateType = typer.Argument(
        ..., help="Type of config to create"
    ),
    model: Optional[str] = typer.Option(
        None,
        "--model",
        "-m",
        help="Model identifier (e.g., meta-llama/Llama-3.1-8B-Instruct)",
    ),
    training_type: Optional[str] = typer.Option(
        None, "--training-type", "-t", help="Training type (full, lora, qlora, auto)"
    ),
    output_file: Optional[Path] = typer.Option(
        None, "--output", "-o", help="Output file path"
    ),
):
    """Create a configuration file interactively."""
    CONSOLE.print(
        Panel.fit("Oumi Config Wizard", title="Welcome", border_style="green")
    )
    CONSOLE.print("Let's create a new configuration file interactively.\n")

    # Get config_type
    wizard_config_type = ConfigType.TRAIN
    if config_type == ConfigCreateType.EVAL:
        wizard_config_type = ConfigType.EVAL
    elif config_type == ConfigCreateType.INFER:
        wizard_config_type = ConfigType.INFER

    # Get model if not provided
    if not model:
        model_completer = WordCompleter(_get_model_list(), ignore_case=True)
        model = session.prompt(
            HTML("<prompt>Model name or HF identifier: </prompt>"),
            completer=model_completer,
            style=prompt_style,
        ).strip()

    # Get training type if needed and not provided
    if wizard_config_type == ConfigType.TRAIN and not training_type:
        training_type_completer = WordCompleter(
            _get_training_type_list(), ignore_case=True
        )
        training_type = session.prompt(
            HTML("<prompt>Training type (full, lora, qlora, auto): </prompt>"),
            completer=training_type_completer,
            default="auto",
            style=prompt_style,
        ).strip()

    # Create the config builder
    builder = config_builder.ConfigBuilder(config_type=wizard_config_type)
    builder.set_model(model)

    # Display hardware information for training configs
    if wizard_config_type == ConfigType.TRAIN:
        gpu_info = builder._get_gpu_resources()

        # Show hardware detection results
        hardware_table = Table(title="Detected Hardware")
        hardware_table.add_column("Resource", style="cyan")
        hardware_table.add_column("Value", style="green")

        hardware_table.add_row("GPU Count", str(gpu_info["count"]))
        hardware_table.add_row(
            "Total GPU Memory", f"{gpu_info['total_memory_gb']:.2f} GB"
        )
        hardware_table.add_row(
            "Available Memory", f"{gpu_info['free_memory_gb']:.2f} GB"
        )

        # Add individual GPUs if available
        if gpu_info["devices"]:
            for i, device in enumerate(gpu_info["devices"]):
                hardware_table.add_row(
                    f"GPU {i}", f"{device['name']} ({device['total_memory_gb']:.2f} GB)"
                )

        CONSOLE.print(hardware_table)

        # Set training type
        if training_type:
            builder.set_training_type(training_type)

            # If auto was selected, explain the choice
            if training_type.lower() == "auto":
                model_size = builder._estimate_model_size(model)
                selected_type = builder.training_type
                gpu_info = builder._get_gpu_resources()
                memory_requirements = builder._estimate_memory_requirements(model_size)

                CONSOLE.print("\n[bold green]Auto Training Type Selection[/bold green]")
                CONSOLE.print(
                    f"• Estimated model size: [cyan]{model_size if model_size else 'Unknown'} billion parameters[/cyan]"
                )
                CONSOLE.print(
                    f"• Selected training type: [cyan]{selected_type.value}[/cyan]"
                )

                # Show memory requirements for each training type
                CONSOLE.print("\n[bold]Estimated Memory Requirements:[/bold]")
                for method, req in memory_requirements.items():
                    per_gpu = req["per_gpu_estimate_gb"]
                    total = req["minimum_total_gb"]
                    estimated_batch_size = gpu_info["estimated_batch_size"].get(
                        method, 1
                    )

                    if method == selected_type.value:
                        style = "bold green"
                    else:
                        style = ""

                    CONSOLE.print(
                        f"• [white]{method.upper()}[/white]: [yellow]{per_gpu:.1f} GB[/yellow] per GPU ([cyan]{total:.1f} GB[/cyan] total)"
                        + f", [magenta]batch size ≈ {estimated_batch_size}[/magenta]",
                        style=style,
                    )

                # Show specific reason for selection
                CONSOLE.print("\n[bold]Recommendation Reasoning:[/bold]")
                if selected_type == TrainingMethodType.FULL:
                    CONSOLE.print(
                        "• [green]Full fine-tuning[/green] selected because your hardware can handle the full model"
                    )
                    CONSOLE.print(
                        f"  - The model requires approximately [yellow]{memory_requirements['full']['per_gpu_estimate_gb']:.1f} GB[/yellow] per GPU"
                    )
                    CONSOLE.print(
                        f"  - You have [cyan]{gpu_info['total_memory_gb']:.1f} GB[/cyan] total GPU memory available"
                    )

                    if gpu_info["count"] > 1:
                        CONSOLE.print(
                            f"  - Multi-GPU training with [cyan]{gpu_info['count']}[/cyan] GPUs enables efficient parallelization"
                        )

                    # Show batch size and gradient accumulation from the config
                    train_config = builder.config.training
                    CONSOLE.print(
                        f"  - Batch size: [magenta]{train_config.per_device_train_batch_size}[/magenta], "
                        + f"Gradient accumulation: [magenta]{train_config.gradient_accumulation_steps}[/magenta]"
                    )

                elif selected_type == TrainingMethodType.LORA:
                    CONSOLE.print(
                        "• [yellow]LoRA fine-tuning[/yellow] selected as a balance between efficiency and performance"
                    )

                    if model_size and model_size > 13:
                        CONSOLE.print(
                            f"  - Model size ([cyan]{model_size}B[/cyan]) is too large for full fine-tuning on your hardware"
                        )

                    CONSOLE.print(
                        f"  - LoRA reduces memory usage by training only a small number of adapter parameters"
                    )
                    CONSOLE.print(
                        f"  - Memory required: [yellow]{memory_requirements['lora']['per_gpu_estimate_gb']:.1f} GB[/yellow] per GPU"
                    )

                    # Show batch size and gradient accumulation from the config
                    train_config = builder.config.training
                    CONSOLE.print(
                        f"  - Batch size: [magenta]{train_config.per_device_train_batch_size}[/magenta], "
                        + f"Gradient accumulation: [magenta]{train_config.gradient_accumulation_steps}[/magenta]"
                    )

                elif selected_type == TrainingMethodType.QLORA:
                    CONSOLE.print(
                        "• [blue]QLoRA fine-tuning[/blue] selected for memory efficiency"
                    )

                    if gpu_info["total_memory_gb"] < 24:
                        CONSOLE.print(
                            f"  - Limited GPU memory ([cyan]{gpu_info['total_memory_gb']:.1f} GB[/cyan]) requires memory-efficient training"
                        )
                    elif model_size and model_size > 30:
                        CONSOLE.print(
                            f"  - Very large model ([cyan]{model_size}B[/cyan]) requires quantization for efficient training"
                        )

                    CONSOLE.print(
                        f"  - QLoRA quantizes the base model to 4-bit and uses parameter-efficient LoRA adapters"
                    )
                    CONSOLE.print(
                        f"  - Memory required: [yellow]{memory_requirements['qlora']['per_gpu_estimate_gb']:.1f} GB[/yellow] per GPU"
                    )

                    # Show batch size and gradient accumulation from the config
                    train_config = builder.config.training
                    CONSOLE.print(
                        f"  - Batch size: [magenta]{train_config.per_device_train_batch_size}[/magenta], "
                        + f"Gradient accumulation: [magenta]{train_config.gradient_accumulation_steps}[/magenta]"
                    )

                # Hardware-specific recommendations
                CONSOLE.print("\n[bold]Hardware Utilization:[/bold]")
                if gpu_info["count"] > 1:
                    CONSOLE.print(
                        f"• Using [cyan]{gpu_info['count']} GPUs[/cyan] for distributed training"
                    )

                    if builder.config.fsdp.enable_fsdp:
                        CONSOLE.print(
                            f"• FSDP enabled with [cyan]{builder.config.fsdp.sharding_strategy}[/cyan] sharding strategy"
                        )
                    elif selected_type != TrainingMethodType.FULL:
                        CONSOLE.print(
                            f"• FSDP is disabled for {selected_type.value.upper()} fine-tuning"
                        )
                else:
                    CONSOLE.print(f"• Using a single GPU for training")

                CONSOLE.print("")

    # Get additional parameters interactively
    if wizard_config_type == ConfigType.TRAIN:
        dataset_completer = WordCompleter(_get_dataset_list(), ignore_case=True)
        dataset = session.prompt(
            HTML(
                "<prompt>Dataset name or HF identifier (press Enter to skip): </prompt>"
            ),
            completer=dataset_completer,
            style=prompt_style,
        ).strip()
        if dataset:
            builder.set_dataset(dataset)

    # Add a description (optional for all config types)
    description = session.prompt(
        HTML("<prompt>Config description (optional, press Enter to skip): </prompt>"),
        style=prompt_style,
    ).strip()
    if description:
        builder.set_description(description)

    # Preview the configuration
    config_yaml = builder.build_yaml()
    CONSOLE.print("\nGenerated Configuration:", style="bold green")
    CONSOLE.print(Syntax(config_yaml, "yaml", theme="monokai", line_numbers=True))

    # Save to file if requested
    if not output_file:
        output_path_completer = PathCompleter(
            only_directories=False,
            expanduser=True,
            file_filter=lambda x: x.endswith(".yaml"),
        )
        output_file_str = session.prompt(
            HTML("<prompt>Save to file (or press Enter to print only): </prompt>"),
            completer=output_path_completer,
            style=prompt_style,
        ).strip()
        if output_file_str:
            output_file = Path(output_file_str)

    if output_file:
        # Create parent directories if they don't exist
        output_file.parent.mkdir(parents=True, exist_ok=True)

        # Write the config to the file
        with open(output_file, "w") as f:
            f.write(config_yaml)

        CONSOLE.print(f"\nConfiguration saved to: {output_file}", style="bold green")

    return 0
