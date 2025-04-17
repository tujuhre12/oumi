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

import os
from typing import Annotated, Final, List, Optional

import typer
from rich.table import Table

import oumi.cli.cli_utils as cli_utils
from oumi.cli.alias import AliasType, try_get_config_name_for_alias
from oumi.core.configs.inference_engine_type import InferenceEngineType
from oumi.utils.logging import logger

_DEFAULT_CLI_PDF_DPI: Final[int] = 200


# Get list of common inference configurations for autocompletion
def get_config_examples() -> List[str]:
    """Get a list of example config paths for autocompletion."""
    return [
        "configs/recipes/llama3_1/inference/8b_infer.yaml",
        "configs/recipes/llama3_1/inference/8b_vllm_infer.yaml",
        "configs/apis/anthropic/infer_claude3_7.yaml",
    ]


# Get list of inference engines for autocompletion
def get_inference_engines() -> List[str]:
    """Get a list of available inference engines for autocompletion."""
    return [engine.name for engine in InferenceEngineType]


def infer(
    ctx: typer.Context,
    config: Annotated[
        str,
        typer.Option(
            *cli_utils.CONFIG_FLAGS,
            help="Path to the configuration file for inference.",
            autocompletion=get_config_examples,
        ),
    ],
    engine: Annotated[
        Optional[str],
        typer.Option(
            "--engine", "-e", 
            help="Inference engine to use", 
            autocompletion=get_inference_engines
        ),
    ] = None,
    model: Annotated[
        Optional[str],
        typer.Option(
            "--model", "-m",
            help="Model to use for inference"
        ),
    ] = None,
    interactive: Annotated[
        bool,
        typer.Option("-i", "--interactive", help="Run in an interactive session."),
    ] = False,
    image: Annotated[
        Optional[str],
        typer.Option(
            "--image",
            help=(
                "File path or URL of an input image to be used with image+text VLLMs. "
                "Only used in interactive mode."
            ),
        ),
    ] = None,
    system_prompt: Annotated[
        Optional[str],
        typer.Option(
            "--system-prompt",
            help=(
                "System prompt for task-specific instructions. "
                "Only used in interactive mode."
            ),
        ),
    ] = None,
    param: Annotated[
        List[str],
        typer.Option(
            "--param", "-p", 
            help="Override config parameters (e.g. generation.temperature=0.7)",
            autocompletion=lambda: [
                "model.model_name", 
                "inference_engine.engine_type",
                "generation.temperature",
                "generation.top_p",
                "generation.max_tokens",
                "input_path",
                "output_path"
            ]
        ),
    ] = None,
    level: cli_utils.LOG_LEVEL_TYPE = None,
):
    """Run inference on a model.

    If `input_filepath` is provided in the configuration file, inference will run on
    those input examples. Otherwise, inference will run interactively with user-provided
    inputs.

    Args:
        ctx: The Typer context object.
        config: Path to the configuration file for inference.
        engine: Inference engine to use (e.g. VLLM, NATIVE, ANTHROPIC).
        model: Model to use for inference.
        interactive: Whether to run in an interactive session.
        image: Path to the input image for `image+text` VLLMs.
        system_prompt: System prompt for task-specific instructions.
        param: Override config parameters (e.g. generation.temperature=0.7).
        level: The logging level for the specified command.
    """
    extra_args = cli_utils.parse_extra_cli_args(ctx)

    # Add engine and model to extra_args if provided
    if engine:
        extra_args.append(f"inference_engine.engine_type={engine}")
    
    if model:
        extra_args.append(f"model.model_name={model}")
    
    # Add any additional parameters passed via --param
    if param:
        extra_args.extend(param)

    config = str(
        cli_utils.resolve_and_fetch_config(
            try_get_config_name_for_alias(config, AliasType.INFER),
        )
    )

    # Delayed imports
    from oumi import infer as oumi_infer
    from oumi import infer_interactive as oumi_infer_interactive
    from oumi.core.configs import InferenceConfig
    from oumi.utils.image_utils import (
        create_png_bytes_from_image_list,
        load_image_png_bytes_from_path,
        load_image_png_bytes_from_url,
        load_pdf_pages_from_path,
        load_pdf_pages_from_url,
    )
    # End imports

    parsed_config: InferenceConfig = InferenceConfig.from_yaml_and_arg_list(
        config, extra_args, logger=logger
    )
    parsed_config.finalize_and_validate()
    # https://stackoverflow.com/questions/62691279/how-to-disable-tokenizers-parallelism-true-false-warning
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    input_image_png_bytes: Optional[list[bytes]] = None
    if image:
        image_lower = image.lower()
        if image_lower.startswith("http://") or image_lower.startswith("https://"):
            if image_lower.endswith(".pdf"):
                input_image_png_bytes = create_png_bytes_from_image_list(
                    load_pdf_pages_from_url(image, dpi=_DEFAULT_CLI_PDF_DPI)
                )
            else:
                input_image_png_bytes = [load_image_png_bytes_from_url(image)]
        else:
            if image_lower.endswith(".pdf"):
                input_image_png_bytes = create_png_bytes_from_image_list(
                    load_pdf_pages_from_path(image, dpi=_DEFAULT_CLI_PDF_DPI)
                )
            else:
                input_image_png_bytes = [load_image_png_bytes_from_path(image)]
    if parsed_config.input_path:
        if interactive:
            logger.warning(
                "Input path provided, skipping interactive inference. "
                "To run in interactive mode, do not provide an input path."
            )
        generations = oumi_infer(parsed_config)
        # Don't print results if output_filepath is provided.
        if parsed_config.output_path:
            return
        table = Table(
            title="Inference Results",
            title_style="bold magenta",
            show_edge=False,
            show_lines=True,
        )
        table.add_column("Conversation", style="green")
        for generation in generations:
            table.add_row(repr(generation))
        cli_utils.CONSOLE.print(table)
        return
    if not interactive:
        logger.warning(
            "No input path provided, running in interactive mode. "
            "To run with an input path, provide one in the configuration file."
        )
    return oumi_infer_interactive(
        parsed_config,
        input_image_bytes=input_image_png_bytes,
        system_prompt=system_prompt,
    )
