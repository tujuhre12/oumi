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

from typing import Annotated, List

import typer
from typing_extensions import Annotated

import oumi.cli.cli_utils as cli_utils
from oumi.cli.alias import AliasType, try_get_config_name_for_alias
from oumi.core.registry import REGISTRY, RegistryType
from oumi.utils.logging import logger


# Get lists of available datasets and models for autocompletion
def get_datasets() -> List[str]:
    """Get a list of all available datasets for autocompletion."""
    return list(REGISTRY.get_all(RegistryType.DATASET).keys())


def get_models() -> List[str]:
    """Get a list of all available models for autocompletion."""
    return list(REGISTRY.get_all(RegistryType.MODEL).keys())


# Common training config parameters for autocompletion
def get_training_params() -> List[str]:
    """Get a list of common training parameters for autocompletion."""
    return [
        "data.train.dataset_name",
        "model.model_name",
        "training.trainer_type",
        "training.output_dir",
        "training.per_device_train_batch_size",
        "training.learning_rate",
        "training.num_train_epochs",
        "training.max_steps",
        "training.seed",
        "peft.lora_r",
        "fsdp.enable_fsdp",
    ]


def train(
    ctx: typer.Context,
    config: Annotated[
        str,
        typer.Option(
            *cli_utils.CONFIG_FLAGS, help="Path to the configuration file for training.",
            autocompletion=lambda: ["configs/recipes/llama3_1/sft/8b_lora/train.yaml", "configs/recipes/phi3/sft/lora_train.yaml"]
        ),
    ],
    model: Annotated[
        str,
        typer.Option(
            "--model", "-m", help="Model to train", autocompletion=get_models
        ),
    ] = None,
    dataset: Annotated[
        str,
        typer.Option(
            "--dataset", "-d", help="Dataset to train on", autocompletion=get_datasets
        ),
    ] = None,
    param: Annotated[
        List[str],
        typer.Option(
            "--param", "-p", help="Override config parameters (e.g. training.seed=42)", 
            autocompletion=get_training_params
        ),
    ] = None,
    level: cli_utils.LOG_LEVEL_TYPE = None,
):
    """Train a model.

    Args:
        ctx: The Typer context object.
        config: Path to the configuration file for training.
        model: Model to train (overrides config).
        dataset: Dataset to train on (overrides config).
        param: Override config parameters (e.g. training.seed=42).
        level: The logging level for the specified command.
    """
    extra_args = cli_utils.parse_extra_cli_args(ctx)

    # Add model and dataset to extra_args if provided
    if model:
        extra_args.append(f"model.model_name={model}")
    
    if dataset:
        extra_args.append(f"data.train.datasets.0.dataset_name={dataset}")
    
    # Add any additional parameters passed via --param
    if param:
        extra_args.extend(param)

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
