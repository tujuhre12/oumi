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

"""Studio command for launching the Oumi Terminal UI."""

import typer

from oumi_tui import main as tui_main


def studio() -> None:
    """Launch the Oumi Terminal UI.

    This command launches an interactive terminal user interface for Oumi,
    providing a visual way to manage datasets, models, configurations,
    and monitor training runs.
    """
    tui_main()
