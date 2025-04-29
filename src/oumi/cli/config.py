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

"""CLI commands for the config wizard."""

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
from rich.panel import Panel
from rich.syntax import Syntax

from oumi.cli.cli_utils import CONSOLE
from oumi.utils.wizard import config_builder, templates
from oumi.utils.wizard.config_builder import ConfigType

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

    if wizard_config_type == ConfigType.TRAIN and training_type:
        builder.set_training_type(training_type)

    # Get additional parameters interactively
    if wizard_config_type == ConfigType.TRAIN:
        dataset = session.prompt(
            HTML(
                "<prompt>Dataset name or HF identifier (press Enter to skip): </prompt>"
            ),
            style=prompt_style,
        ).strip()
        if dataset:
            builder.set_dataset(dataset)

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
