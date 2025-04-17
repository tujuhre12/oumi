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
from typing import Annotated, Final, Optional

import typer
from rich.table import Table

import oumi.cli.cli_utils as cli_utils
from oumi.cli.alias import AliasType, try_get_config_name_for_alias
from oumi.cli.cli_utils import SHORTHAND_MAPPINGS
from oumi.utils.logging import logger

_DEFAULT_CLI_PDF_DPI: Final[int] = 200


def infer(
    ctx: typer.Context,
    config: Annotated[
        str,
        typer.Option(
            *cli_utils.CONFIG_FLAGS,
            help="Path to the configuration file for inference.",
        ),
    ],
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
    # Add explicit shorthand options for common parameters
    model: Annotated[
        Optional[str],
        typer.Option(
            "--model", help=SHORTHAND_MAPPINGS["model"]["help"]
        ),
    ] = None,
    temperature: Annotated[
        Optional[str],
        typer.Option(
            "--temperature", help=SHORTHAND_MAPPINGS["temperature"]["help"]
        ),
    ] = None,
    top_p: Annotated[
        Optional[str],
        typer.Option(
            "--top_p", help=SHORTHAND_MAPPINGS["top_p"]["help"]
        ),
    ] = None,
    top_k: Annotated[
        Optional[str],
        typer.Option(
            "--top_k", help=SHORTHAND_MAPPINGS["top_k"]["help"]
        ),
    ] = None,
    max_tokens: Annotated[
        Optional[str],
        typer.Option(
            "--max_tokens", help=SHORTHAND_MAPPINGS["max_tokens"]["help"]
        ),
    ] = None,
    chat_template: Annotated[
        Optional[str],
        typer.Option(
            "--chat_template", help=SHORTHAND_MAPPINGS["chat_template"]["help"]
        ),
    ] = None,
    engine: Annotated[
        Optional[str],
        typer.Option(
            "--engine", help=SHORTHAND_MAPPINGS["engine"]["help"]
        ),
    ] = None,
    input_file: Annotated[
        Optional[str],
        typer.Option(
            "--input", help=SHORTHAND_MAPPINGS["input"]["help"]
        ),
    ] = None,
    output_file: Annotated[
        Optional[str],
        typer.Option(
            "--output", help=SHORTHAND_MAPPINGS["output"]["help"]
        ),
    ] = None,
    level: cli_utils.LOG_LEVEL_TYPE = None,
):
    """Run inference on a model.

    If `input_filepath` is provided in the configuration file, inference will run on
    those input examples. Otherwise, inference will run interactively with user-provided
    inputs.

    Shorthand Arguments:
        --model VALUE: Shorthand for --model.model_name VALUE
        --tokenizer VALUE: Shorthand for --model.tokenizer_name VALUE
        --max_length VALUE: Shorthand for --model.model_max_length VALUE
        --chat_template VALUE: Shorthand for --model.chat_template VALUE
        --temperature VALUE: Shorthand for --generation.temperature VALUE
        --top_p VALUE: Shorthand for --generation.top_p VALUE
        --top_k VALUE: Shorthand for --generation.top_k VALUE
        --max_tokens VALUE: Shorthand for --generation.max_new_tokens VALUE
        --engine VALUE: Shorthand for --engine VALUE
        --input VALUE: Shorthand for --input_path VALUE
        --output VALUE: Shorthand for --output_path VALUE

    Examples:
        # Using shorthand arguments
        oumi infer --config config.yaml --model llama3-70b-instruct --temperature 0.7

        # Using full arguments (still supported)
        oumi infer --config config.yaml --model.model_name llama3-70b-instruct --generation.temperature 0.7

    Args:
        ctx: The Typer context object.
        config: Path to the configuration file for inference.
        output_dir: Directory to save configs
        (defaults to OUMI_DIR env var or ~/.oumi/fetch).
        interactive: Whether to run in an interactive session.
        image: Path to the input image for `image+text` VLLMs.
        system_prompt: System prompt for task-specific instructions.
        level: The logging level for the specified command.
    """
    # Parse extra CLI args
    extra_args = cli_utils.parse_extra_cli_args(ctx)

    # Add shorthand arguments to extra_args if provided
    shorthand_args = {}
    if model is not None:
        shorthand_args["model.model_name"] = model
    if temperature is not None:
        shorthand_args["generation.temperature"] = temperature
    if top_p is not None:
        shorthand_args["generation.top_p"] = top_p
    if top_k is not None:
        shorthand_args["generation.top_k"] = top_k
    if max_tokens is not None:
        shorthand_args["generation.max_new_tokens"] = max_tokens
    if chat_template is not None:
        shorthand_args["model.chat_template"] = chat_template
    if engine is not None:
        shorthand_args["engine"] = engine
    if input_file is not None:
        shorthand_args["input_path"] = input_file
    if output_file is not None:
        shorthand_args["output_path"] = output_file
    
    # Convert shorthand args to CLI format
    for key, value in shorthand_args.items():
        extra_args.append(f"{key}={value}")

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
