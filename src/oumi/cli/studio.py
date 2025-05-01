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

"""Studio command for launching Oumi Studio."""

from pathlib import Path

import typer

from oumi_tui import main as studio_main


def studio(
    dataset: str = typer.Argument(None, help="Dataset name or path to open"),
) -> None:
    """Launch Oumi Studio.

    This command launches an interactive terminal user interface for Oumi,
    providing a visual way to manage datasets, models, configurations,
    and monitor training runs.

    Args:
        dataset: Optional dataset name or path to open. If a path is provided,
                it should point to a JSON, JSONL, or YAML file containing the dataset.
                If a name is provided, it should match a dataset in the registry.
    """
    # Convert dataset path to absolute path if it's a file path
    dataset_path = None
    if dataset:
        path = Path(dataset)
        if path.exists():
            dataset_path = str(path.absolute())
        else:
            # If not a valid path, treat as dataset name
            dataset_path = dataset

    studio_main(dataset=dataset_path)
