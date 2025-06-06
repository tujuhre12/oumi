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

from typing import Annotated, Optional

import typer

import oumi.cli.cli_utils as cli_utils
from oumi.cli.alias import AliasType, try_get_config_name_for_alias
from oumi.cli.cli_utils import SHORTHAND_MAPPINGS
from oumi.utils.logging import logger


def train(
    ctx: typer.Context,
    config: Annotated[
        str,
        typer.Option(
            *cli_utils.CONFIG_FLAGS, help="Path to the configuration file for training."
        ),
    ],
    # Add explicit shorthand options for common parameters
    model: Annotated[
        Optional[str],
        typer.Option("--model", help=SHORTHAND_MAPPINGS["model"]["help"]),
    ] = None,
    dataset: Annotated[
        Optional[str],
        typer.Option("--dataset", help=SHORTHAND_MAPPINGS["dataset"]["help"]),
    ] = None,
    dataset_path: Annotated[
        Optional[str],
        typer.Option("--dataset_path", help=SHORTHAND_MAPPINGS["dataset_path"]["help"]),
    ] = None,
    lr: Annotated[
        Optional[str],
        typer.Option("--lr", help=SHORTHAND_MAPPINGS["lr"]["help"]),
    ] = None,
    epochs: Annotated[
        Optional[str],
        typer.Option("--epochs", help=SHORTHAND_MAPPINGS["epochs"]["help"]),
    ] = None,
    batch_size: Annotated[
        Optional[str],
        typer.Option("--batch_size", help=SHORTHAND_MAPPINGS["batch_size"]["help"]),
    ] = None,
    gradient_accumulation: Annotated[
        Optional[str],
        typer.Option(
            "--gradient_accumulation",
            help=SHORTHAND_MAPPINGS["gradient_accumulation"]["help"],
        ),
    ] = None,
    lora_rank: Annotated[
        Optional[str],
        typer.Option("--lora_rank", help=SHORTHAND_MAPPINGS["lora_rank"]["help"]),
    ] = None,
    seed: Annotated[
        Optional[str],
        typer.Option("--seed", help=SHORTHAND_MAPPINGS["seed"]["help"]),
    ] = None,
    level: cli_utils.LOG_LEVEL_TYPE = None,
):
    """Train a model.

    Shorthand Arguments:
        --model VALUE: Shorthand for --model.model_name VALUE
        --dataset VALUE: Shorthand for --data.train.datasets[0].dataset_name VALUE
        --dataset_path VALUE: Shorthand for --data.train.datasets[0].dataset_path VALUE
        --lr VALUE: Shorthand for --training.learning_rate VALUE
        --epochs VALUE: Shorthand for --training.num_epochs VALUE
        --batch_size VALUE: Shorthand for --training.per_device_train_batch_size VALUE
        --gradient_accumulation VALUE: Shorthand for \
            --training.gradient_accumulation_steps VALUE
        --lora_rank VALUE: Shorthand for --peft.lora_rank VALUE
        --seed VALUE: Shorthand for --training.seed VALUE

    Examples:
        # Using shorthand arguments
        oumi train llama4-scout --dataset alpaca --lr 2e-5

        # Using full arguments (still supported)
        oumi train llama4-scout --data.train.datasets[0].dataset_name alpaca \
            --training.learning_rate 2e-5

    Args:
        ctx: The Typer context object.
        config: Path to the configuration file for training.
        batch_size: Shorthand for --training.per_device_train_batch_size VALUE.
        dataset: Shorthand for --data.train.datasets[0].dataset_name VALUE.
        dataset_path: Shorthand for --data.train.datasets[0].dataset_path VALUE.
        epochs: Shorthand for --training.num_epochs VALUE.
        gradient_accumulation: Shorthand for
            --training.gradient_accumulation_steps VALUE.
        lora_rank: Shorthand for --peft.lora_rank VALUE.
        lr: Shorthand for --training.learning_rate VALUE.
        model: Shorthand for --model.model_name VALUE.
        seed: Shorthand for --training.seed VALUE.
        level: The logging level for the specified command.
    """
    # Parse extra CLI args
    extra_args = cli_utils.parse_extra_cli_args(ctx)

    # Add shorthand arguments to extra_args if provided
    shorthand_args = {}
    if model is not None:
        shorthand_args["model.model_name"] = model
    if dataset is not None:
        shorthand_args["data.train.datasets[0].dataset_name"] = dataset
    if dataset_path is not None:
        shorthand_args["data.train.datasets[0].dataset_path"] = dataset_path
    if lr is not None:
        shorthand_args["training.learning_rate"] = lr
    if epochs is not None:
        shorthand_args["training.num_epochs"] = epochs
    if batch_size is not None:
        shorthand_args["training.per_device_train_batch_size"] = batch_size
    if gradient_accumulation is not None:
        shorthand_args["training.gradient_accumulation_steps"] = gradient_accumulation
    if lora_rank is not None:
        shorthand_args["peft.lora_rank"] = lora_rank
    if seed is not None:
        shorthand_args["training.seed"] = seed

    # Convert shorthand args to CLI format
    for key, value in shorthand_args.items():
        extra_args.append(f"{key}={value}")

    config = str(
        cli_utils.resolve_and_fetch_config(
            try_get_config_name_for_alias(config, AliasType.TRAIN),
        )
    )
    with cli_utils.CONSOLE.status(
        "[green]Loading configuration...[/green]", spinner="dots"
    ):
        # Delayed imports
        from oumi import train as oumi_train
        from oumi.core.configs import TrainingConfig
        from oumi.core.distributed import set_random_seeds
        from oumi.utils.torch_utils import (
            device_cleanup,
            limit_per_process_memory,
        )
        # End imports

    cli_utils.configure_common_env_vars()

    parsed_config: TrainingConfig = TrainingConfig.from_yaml_and_arg_list(
        config, extra_args, logger=logger
    )
    parsed_config.finalize_and_validate()

    limit_per_process_memory()
    device_cleanup()
    set_random_seeds(
        parsed_config.training.seed, parsed_config.training.use_deterministic
    )

    # Run training
    oumi_train(parsed_config)

    device_cleanup()
