#!/usr/bin/env python3

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

import json
import os
import random
import time
from pathlib import Path

try:
    import yaml
except ImportError:
    import pyyaml as yaml

# Import only what we need
from rich.console import Console
from rich.panel import Panel
from rich.pretty import Pretty
from rich.syntax import Syntax
from rich.table import Table as RichTable
from rich.text import Text
from textual import on
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, ScrollableContainer, Vertical
from textual.screen import ModalScreen
from textual.widgets import (
    Button,
    DataTable,
    DirectoryTree,
    Footer,
    Header,
    Input,
    Label,
    ListItem,
    ListView,
    OptionList,
    Rule,
    Select,
    Static,
    TabPane,
    Tabs,
    TextArea,
)

LOGO = r"""
   ____  _    _ __  __ _____
  / __ \| |  | |  \/  |_   _|
 | |  | | |  | | \  / | | |
 | |  | | |  | | |\/| | | |
 | |__| | |__| | |  | |_| |_
  \____/ \____/|_|  |_|_____|"""

# Sample dataset format for demonstration
SAMPLE_DATASET = {
    "alpaca": [
        {
            "instruction": "Tell me about alpacas.",
            "input": "",
            "output": "Alpacas are domesticated versions of the vicuña, a South American camelid. They're smaller than llamas and are primarily kept for their fiber.",
        },
        {
            "instruction": "Write a brief poem about mountains.",
            "input": "",
            "output": "Majestic peaks touch the sky,\nStanding tall as time goes by.\nSilent guardians, ancient and strong,\nIn their presence, we belong.",
        },
    ],
    "oasst": [
        {
            "prompt": "How do I learn to code?",
            "response": "Learning to code involves several steps. First, choose a programming language to start with - Python is often recommended for beginners due to its readable syntax. Next, find resources like online courses, tutorials, or books.",
        }
    ],
}


class DatasetViewer(ScrollableContainer):
    """A component for visualizing dataset contents."""

    BINDINGS = [
        Binding("q", "close", "Close"),
        Binding("f", "toggle_file_browser", "Toggle File Browser"),
        Binding("r", "refresh", "Refresh"),
        Binding("j", "next_item", "Next Item"),
        Binding("k", "previous_item", "Previous Item"),
        Binding("s", "search", "Search"),
    ]

    def __init__(self, dataset_name=None, file_path=None, **kwargs):
        """Initialize the dataset viewer."""
        super().__init__(**kwargs)
        self.dataset_name = dataset_name
        self.file_path = file_path
        self.dataset = None
        self.current_index = 0
        self.total_items = 0
        self.show_file_browser = False
        self.search_term = ""

    def compose(self) -> ComposeResult:
        """Compose the dataset viewer."""
        yield Container(
            Horizontal(
                Button("Load File", id="load-file-button", variant="primary"),
                Button("View Registry", id="registry-button", variant="primary"),
                Button("Close", id="close-button", variant="error"),
                classes="button-row",
            ),
            Rule(),
            Label("Dataset Browser", id="dataset-title", classes="section-header"),
            Horizontal(
                Label("Search: ", classes="search-label"),
                Input(placeholder="Search in dataset...", id="dataset-search-input"),
                Button("Find", id="search-button", variant="primary"),
                classes="search-container",
            ),
            Container(id="file-browser-container", classes="file-browser"),
            Static(id="dataset-info"),
            Static(id="dataset-preview"),
            Static(id="dataset-navigation"),
            id="dataset-viewer-container",
        )

    def on_mount(self):
        """Initialize UI when mounted."""
        # Hide file browser by default
        self.query_one("#file-browser-container").display = False

        # Initialize with dataset if provided
        if self.dataset_name:
            self.load_registered_dataset(self.dataset_name)
        elif self.file_path:
            self.load_dataset_file(self.file_path)
        else:
            self.show_welcome_message()

    def show_welcome_message(self):
        """Show welcome message when no dataset is loaded."""
        info = self.query_one("#dataset-info", Static)
        preview = self.query_one("#dataset-preview", Static)
        navigation = self.query_one("#dataset-navigation", Static)

        info.update(
            Panel(
                Text.from_markup(
                    "[bold]Welcome to Dataset Viewer[/bold]\n\n"
                    "Use this tool to visualize and explore datasets in the Oumi format.\n\n"
                    "You can either:\n"
                    "• Load a registered dataset from the Oumi registry\n"
                    "• Load a dataset file from disk (JSONL, JSON, YAML)"
                ),
                title="Dataset Viewer",
                border_style="green",
            )
        )

        preview.update("")
        navigation.update("")

    def load_registered_dataset(self, dataset_name):
        """Load a dataset from the registry."""
        self.dataset_name = dataset_name
        self.file_path = None

        # For demonstration, we'll use the sample datasets
        if dataset_name in SAMPLE_DATASET:
            self.dataset = SAMPLE_DATASET[dataset_name]
            self.total_items = len(self.dataset)
            self.current_index = 0
            self.query_one("#dataset-title").update(
                f"Dataset: {dataset_name} ({self.total_items} items)"
            )
            self.update_dataset_view()
        else:
            # In a real implementation, you would load from the actual registry
            self.query_one("#dataset-info").update(
                Panel(
                    Text.from_markup(
                        f"[bold red]Dataset '{dataset_name}' not found in registry.[/bold red]"
                    )
                )
            )

    def load_dataset_file(self, file_path):
        """Load a dataset from a file."""
        self.file_path = file_path
        self.dataset_name = Path(file_path).stem

        # Check if file exists
        file = Path(file_path)
        if not file.exists():
            self.query_one("#dataset-info").update(
                Panel(
                    Text.from_markup(
                        f"[bold red]File not found: {file_path}[/bold red]"
                    )
                )
            )
            return

        try:
            # In a real implementation, this would parse the file
            # For demo, we'll use the sample dataset
            if "alpaca" in file_path.lower():
                self.dataset = SAMPLE_DATASET["alpaca"]
            elif "oasst" in file_path.lower():
                self.dataset = SAMPLE_DATASET["oasst"]
            else:
                # Default sample
                self.dataset = SAMPLE_DATASET["alpaca"]

            self.total_items = len(self.dataset)
            self.current_index = 0
            self.query_one("#dataset-title").update(
                f"Dataset File: {file.name} ({self.total_items} items)"
            )
            self.update_dataset_view()

        except Exception as e:
            self.query_one("#dataset-info").update(
                Panel(
                    Text.from_markup(
                        f"[bold red]Error loading file: {str(e)}[/bold red]"
                    )
                )
            )

    def update_dataset_view(self):
        """Update the dataset view with current item."""
        if not self.dataset or self.total_items == 0:
            return

        # Get current item
        item = self.dataset[self.current_index]

        # Update info panel
        info = self.query_one("#dataset-info", Static)
        preview = self.query_one("#dataset-preview", Static)
        navigation = self.query_one("#dataset-navigation", Static)

        # Format based on item structure
        if "instruction" in item:
            # Alpaca-style format
            info_text = Text.from_markup(
                f"[bold cyan]Item {self.current_index + 1} of {self.total_items}[/bold cyan]\n\n"
                f"[bold]Instruction:[/bold] {item['instruction']}\n\n"
            )

            if item.get("input"):
                info_text.append(
                    Text.from_markup(f"[bold]Input:[/bold] {item['input']}\n\n")
                )

            info.update(Panel(info_text, title="Dataset Item", border_style="blue"))

            # Show output in a separate panel
            preview.update(
                Panel(Text(item["output"]), title="Output", border_style="green")
            )
        elif "prompt" in item:
            # OASST-style format
            info.update(
                Panel(
                    Text.from_markup(
                        f"[bold cyan]Item {self.current_index + 1} of {self.total_items}[/bold cyan]\n\n"
                        f"[bold]Prompt:[/bold] {item['prompt']}"
                    ),
                    title="Dataset Item",
                    border_style="blue",
                )
            )

            preview.update(
                Panel(Text(item["response"]), title="Response", border_style="green")
            )
        else:
            # Generic format - show as JSON
            info.update(
                Panel(
                    Text.from_markup(
                        f"[bold cyan]Item {self.current_index + 1} of {self.total_items}[/bold cyan]"
                    ),
                    title="Dataset Item",
                    border_style="blue",
                )
            )

            # Create pretty-printed JSON display
            console = Console(width=80, file=None)
            console.begin_capture()
            console.print(Pretty(item))
            output = console.end_capture()

            preview.update(
                Panel(
                    Syntax(output, "json", theme="monokai"),
                    title="JSON Data",
                    border_style="green",
                )
            )

        # Update navigation
        navigation.update(
            Text.from_markup(
                "Navigation: [bold]j[/bold] Next Item • [bold]k[/bold] Previous Item • "
                f"[bold]s[/bold] Search • [bold]q[/bold] Close • "
                f"[bold]f[/bold] {'Hide' if self.show_file_browser else 'Show'} File Browser"
            )
        )

    def action_next_item(self):
        """Navigate to the next item."""
        if self.dataset and self.total_items > 0:
            self.current_index = (self.current_index + 1) % self.total_items
            self.update_dataset_view()

    def action_previous_item(self):
        """Navigate to the previous item."""
        if self.dataset and self.total_items > 0:
            self.current_index = (self.current_index - 1) % self.total_items
            self.update_dataset_view()

    def action_toggle_file_browser(self):
        """Toggle the file browser visibility."""
        container = self.query_one("#file-browser-container")
        self.show_file_browser = not self.show_file_browser

        if self.show_file_browser:
            container.display = True
            # Initialize file browser if it doesn't exist
            if not container.query("DirectoryTree"):
                # Start in the data directory or current directory
                start_path = Path("data") if Path("data").exists() else Path.cwd()
                container.mount(DirectoryTree(start_path, id="dataset-file-tree"))
                container.mount(
                    Button("Select File", id="select-file-button", variant="primary")
                )
        else:
            container.display = False

    def action_refresh(self):
        """Refresh the current dataset view."""
        if self.dataset_name:
            self.load_registered_dataset(self.dataset_name)
        elif self.file_path:
            self.load_dataset_file(self.file_path)

    def action_search(self):
        """Focus the search input."""
        self.query_one("#dataset-search-input").focus()

    def action_close(self):
        """Close the dataset viewer."""
        # This will be handled by parent container
        self.app.pop_screen()

    @on(Button.Pressed, "#close-button")
    def handle_close(self):
        """Close the dataset viewer."""
        self.action_close()

    @on(Button.Pressed, "#load-file-button")
    def handle_load_file(self):
        """Show file browser to load a dataset file."""
        self.action_toggle_file_browser()

    @on(Button.Pressed, "#registry-button")
    def handle_registry(self):
        """Show a list of registered datasets."""
        # In a real implementation, this would show a list of datasets from the registry
        # For demo purposes, we'll show a small set of options
        self.app.push_screen(DatasetRegistryScreen(self))

    @on(Button.Pressed, "#search-button")
    def handle_search(self):
        """Search in the dataset."""
        search_input = self.query_one("#dataset-search-input")
        self.search_term = search_input.value.strip().lower()

        if not self.search_term or not self.dataset:
            return

        # Simple search implementation
        for i, item in enumerate(self.dataset):
            # Search in all string values
            found = False
            for key, value in item.items():
                if isinstance(value, str) and self.search_term in value.lower():
                    found = True
                    break

            if found:
                self.current_index = i
                self.update_dataset_view()
                self.app.notify(f"Found search term at item {i + 1}")
                return

        self.app.notify(f"Search term '{self.search_term}' not found", severity="error")

    @on(Input.Submitted, "#dataset-search-input")
    def handle_search_input(self):
        """Handle Enter key in search input."""
        self.handle_search()

    @on(DirectoryTree.FileSelected, "#dataset-file-tree")
    def handle_file_selected(self, event: DirectoryTree.FileSelected):
        """Handle file selection from the file browser."""
        file_path = event.path
        if file_path.suffix.lower() in [".json", ".jsonl", ".yaml", ".yml"]:
            self.load_dataset_file(str(file_path))
            # Auto-hide file browser after selection
            self.action_toggle_file_browser()
        else:
            self.app.notify(
                "Please select a JSON, JSONL, or YAML file", severity="warning"
            )


class DatasetRegistryScreen(ModalScreen):
    """Screen for selecting a dataset from the registry."""

    BINDINGS = [
        Binding("escape", "dismiss", "Back"),
    ]

    def __init__(self, viewer=None):
        """Initialize the dataset registry screen."""
        super().__init__()
        self.viewer = viewer

    def compose(self) -> ComposeResult:
        """Compose the dataset registry screen."""
        yield Container(
            Label("Select Dataset from Registry", classes="modal-title"),
            ListView(
                ListItem(Label("alpaca (Instruction Tuning)"), id="dataset-alpaca"),
                ListItem(Label("oasst (Chat Dataset)"), id="dataset-oasst"),
                ListItem(Label("coco_captions (Image Captions)"), id="dataset-coco"),
                ListItem(Label("mmlu (Benchmark)"), id="dataset-mmlu"),
                id="dataset-registry-list",
            ),
            Button("Cancel", id="cancel-registry", variant="primary"),
            id="dataset-registry-dialog",
        )

    @on(ListView.Selected)
    def handle_selection(self, event: ListView.Selected):
        """Handle dataset selection."""
        if not self.viewer:
            self.dismiss()
            return

        label = event.item.query_one(Label)
        if not label:
            self.dismiss()
            return

        # Extract dataset name from label
        dataset_text = label.renderable
        if isinstance(dataset_text, str):
            dataset_name = dataset_text.split(" ")[0]
        else:
            dataset_name = str(dataset_text).split(" ")[0]

        # Update viewer with selected dataset
        self.viewer.load_registered_dataset(dataset_name)
        self.dismiss()

    @on(Button.Pressed, "#cancel-registry")
    def handle_cancel(self):
        """Cancel dataset selection."""
        self.dismiss()


class DatasetViewerScreen(ModalScreen):
    """Full-screen dataset viewer."""

    BINDINGS = [
        Binding("escape", "dismiss", "Close Viewer"),
    ]

    def __init__(self, dataset_name=None, file_path=None):
        """Initialize the dataset viewer screen."""
        super().__init__()
        self.dataset_name = dataset_name
        self.file_path = file_path

    def compose(self) -> ComposeResult:
        """Compose the dataset viewer screen."""
        yield DatasetViewer(
            dataset_name=self.dataset_name,
            file_path=self.file_path,
            id="full-dataset-viewer",
        )

    def action_dismiss(self) -> None:
        """Dismiss the screen."""
        self.dismiss()


class ConfigPanel(Static):
    """A panel for displaying and editing configuration."""

    def __init__(self, config: dict = None, **kwargs):
        """Initialize the config panel."""
        super().__init__(**kwargs)
        self.config = config or {}

    def compose(self) -> ComposeResult:
        """Compose the config panel."""
        # In Textual 3.x, language parameter is specified differently
        yield TextArea(self._config_to_yaml(), id="config-editor", language="yaml")

    def _config_to_yaml(self) -> str:
        """Convert the config dictionary to YAML."""
        return yaml.dump(self.config, default_flow_style=False)

    def update_config(self, config: dict):
        """Update the configuration."""
        self.config = config
        editor = self.query_one("#config-editor", TextArea)
        if editor:
            editor.text = self._config_to_yaml()


class DatasetBrowser(Container):
    """Browse and select datasets."""

    BINDINGS = [
        Binding("tab", "focus_next", "Next"),
        Binding("shift+tab", "focus_previous", "Previous"),
        Binding("c", "cycle_categories", "Change Category"),
        # Arrow key navigation
        Binding("up", "cursor_up", "Up"),
        Binding("down", "cursor_down", "Down"),
        Binding("left", "select_category", "Category"),
        Binding("right", "select_list", "List"),
        Binding("enter", "select_item", "Select"),
    ]

    def __init__(self, **kwargs):
        """Initialize the dataset browser."""
        super().__init__(**kwargs)
        self.dataset_categories = {
            "SFT": [],
            "Pretraining": [],
            "Vision-Language": [],
            "Evaluation": [],
        }
        self.selected_dataset = None
        self.category_index = 0

    def action_cycle_categories(self) -> None:
        """Cycle through dataset categories."""
        categories = list(self.dataset_categories.keys())
        self.category_index = (self.category_index + 1) % len(categories)
        category = categories[self.category_index]

        # Update the select widget and dataset list
        select = self.query_one("#dataset-category-select", Select)
        # In Textual 3.x, we need to use clear() and add_option
        select.value = category
        # Update the dataset list with the new category
        self.update_dataset_list(category)

    def action_cursor_up(self) -> None:
        """Navigate up in the current focused widget."""
        # Check which widget has focus and move up in that widget
        dataset_list = self.query_one("#dataset-list", ListView)
        if dataset_list.has_focus:
            dataset_list.action_cursor_up()

    def action_cursor_down(self) -> None:
        """Navigate down in the current focused widget."""
        # Check which widget has focus and move down in that widget
        dataset_list = self.query_one("#dataset-list", ListView)
        if dataset_list.has_focus:
            dataset_list.action_cursor_down()

    def action_select_category(self) -> None:
        """Focus on the category selector."""
        self.query_one("#dataset-category-select", Select).focus()

    def action_select_list(self) -> None:
        """Focus on the dataset list."""
        self.query_one("#dataset-list", ListView).focus()

    def action_select_item(self) -> None:
        """Select the current item under cursor."""
        dataset_list = self.query_one("#dataset-list", ListView)
        if dataset_list.has_focus and dataset_list.index is not None:
            # Trigger selection event
            dataset_list.action_select_cursor()

    def compose(self) -> ComposeResult:
        """Compose the dataset browser."""
        yield Label("Select Dataset Category", classes="section-header")
        yield Select(
            [(cat, cat) for cat in self.dataset_categories.keys()],
            id="dataset-category-select",
            prompt="Select a dataset category",
        )
        yield Label("Available Datasets", classes="section-header")
        yield ListView(id="dataset-list")
        yield Label("Dataset Details", classes="section-header")
        yield Static("Select a dataset to view details", id="dataset-details")
        yield Horizontal(
            Button(
                "View Dataset",
                id="view-dataset-button",
                variant="primary",
                disabled=True,
            ),
            classes="button-row",
        )

    def on_mount(self):
        """Initialize dataset information when mounted."""
        # Mock datasets for demonstration
        self.dataset_categories = {
            "SFT": ["alpaca", "ultrachat", "tulu3", "dolly", "oasst1", "orca"],
            "Pretraining": [
                "c4",
                "dolma",
                "wikipedia",
                "redpajama",
                "slimpajama",
                "the_stack",
            ],
            "Vision-Language": [
                "coco_captions",
                "llava",
                "flickr30k",
                "vqa",
                "imagenet_captions",
            ],
            "Evaluation": ["mmlu", "truthfulqa", "gsm8k", "hellaswag", "humaneval"],
        }

        # Update the UI with the first category's datasets
        self.update_dataset_list("SFT")

    def update_dataset_list(self, category: str):
        """Update the dataset list based on selected category."""
        dataset_list = self.query_one("#dataset-list", ListView)
        dataset_list.clear()

        for dataset in self.dataset_categories.get(category, []):
            list_item = ListItem(Label(dataset))
            dataset_list.append(list_item)

    @on(Select.Changed, "#dataset-category-select")
    def handle_category_change(self, event: Select.Changed):
        """Handle dataset category selection."""
        self.update_dataset_list(event.value)

    @on(ListView.Selected, "#dataset-list")
    def handle_dataset_selection(self, event: ListView.Selected):
        """Handle dataset selection."""
        dataset_name = event.item.query_one(Label).renderable
        self.selected_dataset = dataset_name

        # Display dataset details
        dataset_details = self.query_one("#dataset-details", Static)
        dataset_details.update(
            Panel(
                Text.from_markup(
                    f"[bold cyan]{dataset_name}[/bold cyan]\n\n"
                    f"Type: {self._get_dataset_category(dataset_name)}\n"
                    f"Description: Sample description for {dataset_name}\n"
                    f"Format: jsonl\n"
                    f"Size: Varies\n\n"
                    f"[green]Click to add to configuration[/green]"
                )
            )
        )

    def _get_dataset_category(self, dataset_name: str) -> str:
        """Get the category of a dataset."""
        for category, datasets in self.dataset_categories.items():
            if dataset_name in datasets:
                return category
        return "Unknown"

    @on(Button.Pressed, "#view-dataset-button")
    def handle_view_dataset(self):
        """Open the dataset viewer for the selected dataset."""
        if not self.selected_dataset:
            self.app.notify("Please select a dataset first", severity="warning")
            return

        # Open the dataset viewer as a modal screen
        self.app.push_screen(DatasetViewerScreen(dataset_name=self.selected_dataset))

    @on(ListView.Selected, "#dataset-list")
    def handle_dataset_selection(self, event: ListView.Selected):
        """Handle dataset selection."""
        dataset_name = event.item.query_one(Label).renderable
        self.selected_dataset = dataset_name

        # Enable the view dataset button
        view_button = self.query_one("#view-dataset-button", Button)
        view_button.disabled = False

        # Display dataset details
        dataset_details = self.query_one("#dataset-details", Static)
        dataset_details.update(
            Panel(
                Text.from_markup(
                    f"[bold cyan]{dataset_name}[/bold cyan]\n\n"
                    f"Type: {self._get_dataset_category(dataset_name)}\n"
                    f"Description: Sample description for {dataset_name}\n"
                    f"Format: jsonl\n"
                    f"Size: Varies\n\n"
                    f"[green]Click 'View Dataset' to explore the data[/green]"
                )
            )
        )


class ModelSelector(Container):
    """Browse and select models."""

    BINDINGS = [
        Binding("tab", "focus_next", "Next"),
        Binding("shift+tab", "focus_previous", "Previous"),
        Binding("c", "cycle_categories", "Change Category"),
        # Arrow key navigation
        Binding("up", "cursor_up", "Up"),
        Binding("down", "cursor_down", "Down"),
        Binding("left", "select_category", "Category"),
        Binding("right", "select_list", "List"),
        Binding("enter", "select_item", "Select"),
    ]

    def __init__(self, **kwargs):
        """Initialize the model selector."""
        super().__init__(**kwargs)
        self.model_categories = {
            "Base Models": [
                "Llama-3.1-8B",
                "Llama-3.1-70B",
                "Llama-3.2-1B",
                "Llama-3.2-3B",
                "Phi-3-Mini",
                "Qwen2-7B",
                "SmolLM-1.7B",
            ],
            "Instruct Models": [
                "Llama-3.1-8B-Instruct",
                "Llama-3.1-70B-Instruct",
                "Phi-3-Mini-Instruct",
                "Qwen2-7B-Instruct",
                "SmolLM-1.7B-Instruct",
            ],
            "Vision-Language Models": [
                "Llama-3.2-Vision",
                "LLaVA-1.5",
                "Phi-3-Vision",
                "Qwen2-VL",
                "SmolVLM",
            ],
        }
        self.selected_model = None
        self.category_index = 0

    def action_cycle_categories(self) -> None:
        """Cycle through model categories."""
        categories = list(self.model_categories.keys())
        self.category_index = (self.category_index + 1) % len(categories)
        category = categories[self.category_index]

        # Update the select widget and model list
        select = self.query_one("#model-category-select", Select)
        select.value = category
        # Update the model list with the new category
        self.update_model_list(category)

    def action_cursor_up(self) -> None:
        """Navigate up in the current focused widget."""
        # Check which widget has focus and move up in that widget
        model_list = self.query_one("#model-list", ListView)
        if model_list.has_focus:
            model_list.action_cursor_up()

    def action_cursor_down(self) -> None:
        """Navigate down in the current focused widget."""
        # Check which widget has focus and move down in that widget
        model_list = self.query_one("#model-list", ListView)
        if model_list.has_focus:
            model_list.action_cursor_down()

    def action_select_category(self) -> None:
        """Focus on the category selector."""
        self.query_one("#model-category-select", Select).focus()

    def action_select_list(self) -> None:
        """Focus on the model list."""
        self.query_one("#model-list", ListView).focus()

    def action_select_item(self) -> None:
        """Select the current item under cursor."""
        model_list = self.query_one("#model-list", ListView)
        if model_list.has_focus and model_list.index is not None:
            # Trigger selection event
            model_list.action_select_cursor()

    def compose(self) -> ComposeResult:
        """Compose the model selector."""
        yield Label("Select Model Category", classes="section-header")
        yield Select(
            [(cat, cat) for cat in self.model_categories.keys()],
            id="model-category-select",
            prompt="Select a model category",
        )
        yield Label("Available Models", classes="section-header")
        yield ListView(id="model-list")
        yield Label("Model Details", classes="section-header")
        yield Static("Select a model to view details", id="model-details")

    def on_mount(self):
        """Initialize with the first category's models."""
        self.update_model_list("Base Models")

    def update_model_list(self, category: str):
        """Update the model list based on selected category."""
        model_list = self.query_one("#model-list", ListView)
        model_list.clear()

        for model in self.model_categories.get(category, []):
            list_item = ListItem(Label(model))
            model_list.append(list_item)

    @on(Select.Changed, "#model-category-select")
    def handle_category_change(self, event: Select.Changed):
        """Handle model category selection."""
        self.update_model_list(event.value)

    @on(ListView.Selected, "#model-list")
    def handle_model_selection(self, event: ListView.Selected):
        """Handle model selection."""
        model_name = event.item.query_one(Label).renderable
        self.selected_model = model_name

        # Display model details
        model_details = self.query_one("#model-details", Static)
        model_details.update(
            Panel(
                Text.from_markup(
                    f"[bold cyan]{model_name}[/bold cyan]\n\n"
                    f"Type: {self._get_model_category(model_name)}\n"
                    f"Parameters: {self._get_model_size(model_name)}\n"
                    f"Description: Sample description for {model_name}\n\n"
                    f"[green]Click to add to configuration[/green]"
                )
            )
        )

    def _get_model_category(self, model_name: str) -> str:
        """Get the category of a model."""
        for category, models in self.model_categories.items():
            if model_name in models:
                return category
        return "Unknown"

    def _get_model_size(self, model_name: str) -> str:
        """Get the size of a model based on its name."""
        if "70B" in model_name:
            return "70 Billion"
        elif "8B" in model_name:
            return "8 Billion"
        elif "7B" in model_name:
            return "7 Billion"
        elif "3B" in model_name:
            return "3 Billion"
        elif "1.7B" in model_name:
            return "1.7 Billion"
        elif "1B" in model_name:
            return "1 Billion"
        elif "Mini" in model_name:
            return "3-4 Billion"
        return "Unknown"


class ConfigBuilder(Container):
    """Build configuration for Oumi commands."""

    BINDINGS = [
        Binding("t", "build_train", "Training Config"),
        Binding("e", "build_eval", "Eval Config"),
        Binding("i", "build_infer", "Infer Config"),
        Binding("s", "save", "Save Config"),
        Binding("r", "run", "Run Command"),
        # Arrow key navigation for text editor
        Binding("up", "cursor_up", "Up"),
        Binding("down", "cursor_down", "Down"),
        Binding("left", "cursor_left", "Left"),
        Binding("right", "cursor_right", "Right"),
    ]

    def compose(self) -> ComposeResult:
        """Compose the config builder."""
        yield Label("Configuration Builder", classes="section-header")
        yield Horizontal(
            Button("Training Config", id="build-train-config", variant="primary"),
            Button("Evaluation Config", id="build-eval-config", variant="primary"),
            Button("Inference Config", id="build-infer-config", variant="primary"),
            classes="button-row",
        )
        yield Label("Current Configuration", classes="section-header")
        yield ConfigPanel(id="config-panel")
        yield Horizontal(
            Button("Save Config", id="save-config", variant="success"),
            Button("Run Command", id="run-command", variant="success"),
            classes="button-row",
        )

    def action_build_train(self) -> None:
        """Build training config with keyboard shortcut."""
        self.build_train_config()

    def action_build_eval(self) -> None:
        """Build evaluation config with keyboard shortcut."""
        self.build_eval_config()

    def action_build_infer(self) -> None:
        """Build inference config with keyboard shortcut."""
        self.build_infer_config()

    def action_save(self) -> None:
        """Save config with keyboard shortcut."""
        self.save_config()

    def action_run(self) -> None:
        """Run command with keyboard shortcut."""
        self.run_command()

    def action_cursor_up(self) -> None:
        """Move cursor up in text editor."""
        editor = self.query_one("#config-editor", TextArea)
        if editor.has_focus:
            editor.action_cursor_up()

    def action_cursor_down(self) -> None:
        """Move cursor down in text editor."""
        editor = self.query_one("#config-editor", TextArea)
        if editor.has_focus:
            editor.action_cursor_down()

    def action_cursor_left(self) -> None:
        """Move cursor left in text editor."""
        editor = self.query_one("#config-editor", TextArea)
        if editor.has_focus:
            editor.action_cursor_left()

    def action_cursor_right(self) -> None:
        """Move cursor right in text editor."""
        editor = self.query_one("#config-editor", TextArea)
        if editor.has_focus:
            editor.action_cursor_right()

    @on(Button.Pressed, "#build-train-config")
    def build_train_config(self):
        """Build a training configuration."""
        base_config = {
            "model": {
                "model_name": "meta-llama/Llama-3.1-8B",
                "cache_dir": "cache",
                "trust_remote_code": True,
            },
            "data": {
                "train": {
                    "datasets": [
                        {
                            "dataset_name": "alpaca",
                            "mixture_proportion": 1.0,
                        }
                    ],
                    "batch_size": 8,
                    "shuffle": True,
                },
                "val": {
                    "datasets": [
                        {
                            "dataset_name": "alpaca",
                            "mixture_proportion": 1.0,
                        }
                    ],
                    "batch_size": 8,
                },
            },
            "training": {
                "output_dir": "output/llama-3.1-8b-alpaca",
                "num_epochs": 3,
                "learning_rate": 2e-5,
                "lr_scheduler": "cosine",
                "weight_decay": 0.01,
                "seed": 42,
                "fp16": True,
                "bf16": False,
                "fsdp": False,
            },
            "peft": {
                "lora_alpha": 16,
                "lora_dropout": 0.05,
                "lora_r": 8,
                "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
                "use_peft": True,
            },
        }
        self.query_one("#config-panel", ConfigPanel).update_config(base_config)

    @on(Button.Pressed, "#build-eval-config")
    def build_eval_config(self):
        """Build an evaluation configuration."""
        base_config = {
            "model": {
                "model_name": "meta-llama/Llama-3.1-8B",
                "cache_dir": "cache",
                "trust_remote_code": True,
            },
            "evaluation": {
                "output_dir": "eval_results",
                "batch_size": 8,
                "benchmarks": ["mmlu", "truthfulqa", "gsm8k"],
                "num_samples": 10,
                "seed": 42,
            },
        }
        self.query_one("#config-panel", ConfigPanel).update_config(base_config)

    @on(Button.Pressed, "#build-infer-config")
    def build_infer_config(self):
        """Build an inference configuration."""
        base_config = {
            "model": {
                "model_name": "meta-llama/Llama-3.1-8B-Instruct",
                "cache_dir": "cache",
                "trust_remote_code": True,
            },
            "inference": {
                "engine": "native_text",
                "max_tokens": 512,
                "temperature": 0.7,
                "top_p": 0.95,
                "repetition_penalty": 1.1,
            },
        }
        self.query_one("#config-panel", ConfigPanel).update_config(base_config)

    @on(Button.Pressed, "#save-config")
    def save_config(self):
        """Save the configuration to a file."""
        config_text = self.query_one("#config-editor", TextArea).text
        self.app.push_screen(SaveConfigScreen(config_text))

    @on(Button.Pressed, "#run-command")
    def run_command(self):
        """Prepare to run a command with the configuration."""
        config_text = self.query_one("#config-editor", TextArea).text
        self.app.push_screen(RunCommandScreen(config_text))


class MetricsVisualizer(Container):
    """Display training metrics in a visual format."""

    def __init__(self, **kwargs):
        """Initialize the metrics visualizer."""
        super().__init__(**kwargs)
        self.metrics = {}

    def compose(self) -> ComposeResult:
        """Compose the metrics visualizer."""
        yield Label("Training Metrics", classes="section-header")
        yield Static("No metrics available", id="metrics-display")

    def update_metrics(self, metrics: dict):
        """Update the displayed metrics."""
        if not metrics:
            return

        self.metrics = metrics
        display = self.query_one("#metrics-display", Static)

        # Format metrics for display
        metrics_text = ""
        for category, values in metrics.items():
            metrics_text += f"[bold green]{category.title()}[/bold green]\n"
            if isinstance(values, dict):
                for name, value in values.items():
                    if isinstance(value, float):
                        metrics_text += f"  {name}: [cyan]{value:.4f}[/cyan]\n"
                    else:
                        metrics_text += f"  {name}: [cyan]{value}[/cyan]\n"
            else:
                if isinstance(values, float):
                    metrics_text += f"  Value: [cyan]{values:.4f}[/cyan]\n"
                else:
                    metrics_text += f"  Value: [cyan]{values}[/cyan]\n"
            metrics_text += "\n"

        if not metrics_text:
            metrics_text = "No metrics available"

        display.update(Panel(Text.from_markup(metrics_text)))


class ConfigViewer(Container):
    """View training configuration details."""

    def __init__(self, **kwargs):
        """Initialize the config viewer."""
        super().__init__(**kwargs)
        self.config = {}

    def compose(self) -> ComposeResult:
        """Compose the config viewer."""
        yield Label("Configuration", classes="section-header")
        yield TextArea(id="config-viewer", language="yaml")

    def update_config(self, config_path: str):
        """Update the displayed configuration."""
        config_text = ""
        try:
            with open(config_path, "r") as f:
                config_text = f.read()
        except:
            config_text = f"Error loading configuration from {config_path}"

        self.query_one("#config-viewer", TextArea).text = config_text


class TrainingMonitor(Container):
    """Monitor training progress."""

    BINDINGS = [
        Binding("l", "view_logs", "View Logs"),
        Binding("c", "view_config", "View Config"),
        Binding("m", "view_metrics", "View Metrics"),
        Binding("x", "stop_run", "Stop Run"),
        Binding("f5", "refresh", "Refresh"),
        Binding("up", "previous_run", "Previous Run"),
        Binding("down", "next_run", "Next Run"),
        Binding("tab", "switch_view", "Switch View"),
    ]

    def __init__(self, **kwargs):
        """Initialize the training monitor."""
        super().__init__(**kwargs)
        self.runs = []
        self.selected_run_path = None
        self.current_view = "details"  # details, logs, config, metrics

    def compose(self) -> ComposeResult:
        """Compose the training monitor."""
        yield Label("Training Runs", classes="section-header")
        yield DataTable(id="runs-table")
        yield Tabs(
            TabPane("Details", Static(id="run-details"), id="tab-details"),
            TabPane("Logs", Static(id="log-viewer"), id="tab-logs"),
            TabPane("Config", ConfigViewer(id="config-viewer"), id="tab-config"),
            TabPane("Metrics", MetricsVisualizer(id="metrics-viz"), id="tab-metrics"),
            id="run-tabs",
        )
        yield Horizontal(
            Button("View Logs", id="view-logs", disabled=True),
            Button("View Config", id="view-config", disabled=True),
            Button("View Metrics", id="view-metrics", disabled=True),
            classes="button-row-top",
        )
        yield Horizontal(
            Button("Stop Run", id="stop-run", disabled=True, variant="error"),
            Button("Refresh", id="refresh-runs", variant="primary"),
            classes="button-row",
        )

    def on_mount(self):
        """Set up the runs table and load runs."""
        # Set up the table
        table = self.query_one("#runs-table", DataTable)
        table.add_columns(
            "Run Name", "Model", "Status", "Training Type", "Progress", "Last Updated"
        )
        # Make sure the datatable is selectable
        table.cursor_type = "row"

        # Initial load of runs
        self.load_runs()

    def load_runs(self):
        """Load training runs from the output directory."""
        # Clear the current runs
        self.runs = []

        # Get all directories in the output folder
        output_dir = Path("output")
        if not output_dir.exists() or not output_dir.is_dir():
            return

        for run_dir in output_dir.iterdir():
            if not run_dir.is_dir():
                continue

            run_path = str(run_dir)
            run_name = run_dir.name

            # Determine the model name and training type
            parts = run_name.split("_")
            model = parts[-1] if len(parts) > 1 else run_name
            training_type = "_".join(parts[:-1]) if len(parts) > 1 else "standard"

            # Determine status based on multiple indicators
            status = "Completed"

            # Check for trainer_state.json to determine real status
            trainer_state_path = run_dir / "trainer_state.json"
            if trainer_state_path.exists():
                try:
                    with open(trainer_state_path, "r") as f:
                        trainer_state = json.loads(f.read())

                    # Check if training is actually completed
                    if "log_history" in trainer_state and trainer_state["log_history"]:
                        log_history = trainer_state["log_history"]
                        if (
                            "global_step" in trainer_state
                            and "max_steps" in trainer_state
                        ):
                            if (
                                trainer_state["global_step"]
                                >= trainer_state["max_steps"]
                            ):
                                status = "Completed"
                            else:
                                # Check recent modification time for active status
                                log_files = list((run_dir / "logs").glob("*.log"))
                                if log_files:
                                    last_modified = max(
                                        f.stat().st_mtime for f in log_files
                                    )
                                    time_since = time.time() - last_modified
                                    if (
                                        time_since < 3600
                                    ):  # Modified within the last hour
                                        status = "Running"
                                    else:
                                        status = "Stopped"
                except:
                    # If there's an error, fall back to time-based detection
                    log_files = (
                        list((run_dir / "logs").glob("*.log"))
                        if (run_dir / "logs").exists()
                        else []
                    )
                    if log_files:
                        last_modified = max(f.stat().st_mtime for f in log_files)
                        time_since = time.time() - last_modified
                        if time_since < 3600:  # Within the last hour
                            status = "Running"
                        elif time_since < 86400:  # Within the last day
                            status = "Recently Completed"

            # Determine progress based on trainer state
            progress = "N/A"

            # Try to extract from trainer_state.json
            if trainer_state_path.exists():
                try:
                    with open(trainer_state_path, "r") as f:
                        trainer_state = json.loads(f.read())

                    if "global_step" in trainer_state:
                        if (
                            "max_steps" in trainer_state
                            and trainer_state["max_steps"] > 0
                        ):
                            progress_pct = (
                                trainer_state["global_step"]
                                / trainer_state["max_steps"]
                            ) * 100
                            progress = f"{progress_pct:.1f}%"
                        elif (
                            "num_train_epochs" in trainer_state
                            and "epoch" in trainer_state
                        ):
                            if trainer_state["num_train_epochs"] > 0:
                                progress_pct = (
                                    trainer_state["epoch"]
                                    / trainer_state["num_train_epochs"]
                                ) * 100
                                progress = f"{progress_pct:.1f}%"
                            else:
                                progress = f"Step {trainer_state['global_step']}"
                        else:
                            progress = f"Step {trainer_state['global_step']}"
                except:
                    # Fallback to status-based progress
                    if status == "Running":
                        progress = "In Progress"
                    elif status == "Completed" or status == "Recently Completed":
                        progress = "100%"
                    elif status == "Stopped":
                        progress = "Incomplete"

            # Last updated
            last_updated_timestamp = 0
            for root, dirs, files in os.walk(run_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    try:
                        last_modified = os.path.getmtime(file_path)
                        last_updated_timestamp = max(
                            last_updated_timestamp, last_modified
                        )
                    except:
                        pass

            last_updated = "Unknown"
            if last_updated_timestamp > 0:
                last_updated = time.strftime(
                    "%Y-%m-%d %H:%M", time.localtime(last_updated_timestamp)
                )

            # Add to runs list
            self.runs.append(
                {
                    "name": run_name,
                    "path": run_path,
                    "model": model,
                    "status": status,
                    "training_type": training_type,
                    "progress": progress,
                    "last_updated": last_updated,
                }
            )

        # Sort runs by last_updated (most recent first)
        self.runs = sorted(self.runs, key=lambda x: x["last_updated"], reverse=True)

        # Update the table
        table = self.query_one("#runs-table", DataTable)
        table.clear()

        for run in self.runs:
            table.add_row(
                run["name"],
                run["model"],
                run["status"],
                run["training_type"],
                run["progress"],
                run["last_updated"],
            )

    # In Textual 3.x, there are two approaches to handle selection events
    # First attempt: Use RowSelected event directly
    @on(DataTable.RowSelected, "#runs-table")
    def handle_row_selected(self, event: DataTable.RowSelected):
        """Handle row selection via RowSelected event."""
        # Provide debug feedback
        self.app.notify("RowSelected event triggered")

        try:
            row_index = event.cursor_row
            if row_index is None or row_index >= len(self.runs):
                row_index = 0  # Default to first run if index is invalid

            # Get the selected run
            run = self.runs[row_index]
            self.selected_run_path = run["path"]

            # Provide user feedback
            self.app.notify(f"Selected run: {run['name']}")

            # Enable buttons
            self.query_one("#view-logs", Button).disabled = False
            self.query_one("#view-config", Button).disabled = False
            self.query_one("#view-metrics", Button).disabled = False
            self.query_one("#stop-run", Button).disabled = run["status"] != "Running"

            # Set current view and update
            self.current_view = "details"
            self.update_current_view()
        except Exception as e:
            self.app.notify(f"Error in selection: {str(e)}")

    # Second approach: Use general DataTable event
    @on(DataTable.CellSelected, "#runs-table")
    def handle_cell_selected(self, event: DataTable.CellSelected):
        """Alternative handler for table interaction."""
        # Provide debug feedback
        self.app.notify("CellSelected event triggered")

        try:
            # Get table and current row
            table = self.query_one("#runs-table", DataTable)
            row_index = table.cursor_row

            if row_index is None or row_index >= len(self.runs):
                return

            # Get the selected run
            run = self.runs[row_index]
            self.selected_run_path = run["path"]

            # Provide user feedback
            self.app.notify(f"Selected run: {run['name']}")

            # Enable buttons
            self.query_one("#view-logs", Button).disabled = False
            self.query_one("#view-config", Button).disabled = False
            self.query_one("#view-metrics", Button).disabled = False
            self.query_one("#stop-run", Button).disabled = run["status"] != "Running"

            # Set current view and update
            self.current_view = "details"
            self.update_current_view()
        except Exception as e:
            self.app.notify(f"Error in cell selection: {str(e)}")

    def update_current_view(self):
        """Update the currently visible tab with data from the selected run."""
        if not self.selected_run_path:
            return

        run_path = Path(self.selected_run_path)
        run = next((r for r in self.runs if r["path"] == self.selected_run_path), None)
        if not run:
            return

        # Get the tabs widget
        tabs = self.query_one("#run-tabs", Tabs)

        # First activate the correct tab
        # This ensures the tab content is visible before updating
        tab_id = None
        if self.current_view == "details":
            tab_id = "tab-details"
        elif self.current_view == "logs":
            tab_id = "tab-logs"
        elif self.current_view == "config":
            tab_id = "tab-config"
        elif self.current_view == "metrics":
            tab_id = "tab-metrics"

        # Set a flag to prevent recursive tab activation
        self._programmatic_tab_change = True

        # Use proper tab activation method for Textual 3.x
        # Simple but effective approach: directly set the active tab
        # based on the view name instead of the tab ID
        if self.current_view == "details":
            tabs.active = 0  # First tab
        elif self.current_view == "logs":
            tabs.active = 1  # Second tab
        elif self.current_view == "config":
            tabs.active = 2  # Third tab
        elif self.current_view == "metrics":
            tabs.active = 3  # Fourth tab

        # Now update only the content for the active tab
        if self.current_view == "details":
            self.update_details_view(run)
        elif self.current_view == "logs":
            self.update_logs_view(run)
        elif self.current_view == "config":
            self.update_config_view(run)
        elif self.current_view == "metrics":
            self.update_metrics_view(run)

    def update_details_view(self, run):
        """Update the details view with run information."""
        run_details = self.query_one("#run-details", Static)

        # Make sure this widget is visible in the UI
        run_details.display = True

        # Get additional details from various files
        run_info = {
            "basic": {
                "run_name": run["name"],
                "status": run["status"],
                "model": run["model"],
                "training_type": run["training_type"],
                "progress": run["progress"],
                "last_updated": run["last_updated"],
                "path": run["path"],
            }
        }

        # Get config details
        config_file = Path(run["path"]) / "telemetry" / "training_config.yaml"
        if config_file.exists():
            try:
                with open(config_file, "r") as f:
                    config = yaml.safe_load(f)

                if config and isinstance(config, dict):
                    # Get dataset info
                    if (
                        "data" in config
                        and "train" in config["data"]
                        and "datasets" in config["data"]["train"]
                    ):
                        datasets = []
                        for dataset in config["data"]["train"]["datasets"]:
                            if "dataset_name" in dataset:
                                datasets.append(dataset["dataset_name"])

                        if datasets:
                            run_info["datasets"] = datasets

                    # Get model info
                    if "model" in config:
                        model_info = {}
                        model_config = config["model"]
                        for key in [
                            "model_name",
                            "trust_remote_code",
                            "torch_dtype_str",
                            "device_map",
                        ]:
                            if key in model_config and model_config[key] is not None:
                                model_info[key] = model_config[key]

                        if model_info:
                            run_info["model_config"] = model_info

                    # Get training settings
                    if "training" in config:
                        training_info = {}
                        training_config = config["training"]
                        for key in [
                            "learning_rate",
                            "num_train_epochs",
                            "max_steps",
                            "per_device_train_batch_size",
                            "optimizer",
                            "weight_decay",
                            "gradient_accumulation_steps",
                        ]:
                            if (
                                key in training_config
                                and training_config[key] is not None
                            ):
                                training_info[key] = training_config[key]

                        # Check if using PEFT
                        if "use_peft" in training_config:
                            training_info["use_peft"] = training_config["use_peft"]

                        if training_info:
                            run_info["training_config"] = training_info
            except Exception as e:
                run_info["config_error"] = str(e)

        # Get trainer state info
        trainer_state_path = Path(run["path"]) / "trainer_state.json"
        if trainer_state_path.exists():
            try:
                with open(trainer_state_path, "r") as f:
                    trainer_state = json.loads(f.read())

                state_info = {}
                for key in [
                    "epoch",
                    "global_step",
                    "max_steps",
                    "num_train_epochs",
                    "train_batch_size",
                ]:
                    if key in trainer_state:
                        state_info[key] = trainer_state[key]

                if state_info:
                    run_info["trainer_state"] = state_info

                # Get last metrics
                if "log_history" in trainer_state and trainer_state["log_history"]:
                    last_metrics = trainer_state["log_history"][-1]
                    if "train_loss" in last_metrics:
                        if "metrics" not in run_info:
                            run_info["metrics"] = {}
                        run_info["metrics"]["loss"] = last_metrics["train_loss"]
                    if "mean_token_accuracy" in last_metrics:
                        if "metrics" not in run_info:
                            run_info["metrics"] = {}
                        run_info["metrics"]["accuracy"] = last_metrics[
                            "mean_token_accuracy"
                        ]
            except Exception as e:
                run_info["state_error"] = str(e)

        # Check if checkpoint exists
        checkpoint_dirs = [
            d for d in Path(run["path"]).glob("checkpoint-*") if d.is_dir()
        ]
        if checkpoint_dirs:
            if "checkpoints" not in run_info:
                run_info["checkpoints"] = []
            for checkpoint_dir in checkpoint_dirs:
                run_info["checkpoints"].append(checkpoint_dir.name)

        # Construct the details panel
        details_text = f"[bold cyan]{run['name']}[/bold cyan]\n\n"

        # Basic info section
        details_text += f"[bold white]Run Information:[/bold white]\n"
        details_text += f"Status: [green]{run['status']}[/green]\n"
        details_text += f"Progress: {run['progress']}\n"
        details_text += f"Last Updated: {run['last_updated']}\n"

        # Dataset info
        if "datasets" in run_info:
            details_text += f"\n[bold magenta]Datasets:[/bold magenta]\n"
            for dataset in run_info["datasets"]:
                details_text += f"• [cyan]{dataset}[/cyan]\n"

        # Model info
        if "model_config" in run_info:
            details_text += f"\n[bold blue]Model:[/bold blue]\n"
            for key, value in run_info["model_config"].items():
                if key == "model_name":
                    details_text += f"• Name: [cyan]{value}[/cyan]\n"
                else:
                    details_text += f"• {key}: [cyan]{value}[/cyan]\n"

        # Training settings
        if "training_config" in run_info:
            details_text += f"\n[bold yellow]Training Settings:[/bold yellow]\n"
            for key, value in run_info["training_config"].items():
                display_name = key.replace("_", " ").title()
                details_text += f"• {display_name}: [cyan]{value}[/cyan]\n"

        # Current progress/metrics
        if "trainer_state" in run_info or "metrics" in run_info:
            details_text += f"\n[bold green]Progress:[/bold green]\n"

            if "trainer_state" in run_info:
                for key, value in run_info["trainer_state"].items():
                    if key in ["epoch", "global_step"]:
                        details_text += f"• {key.capitalize()}: [cyan]{value}[/cyan]\n"

            if "metrics" in run_info:
                for key, value in run_info["metrics"].items():
                    if isinstance(value, float):
                        details_text += (
                            f"• {key.capitalize()}: [cyan]{value:.4f}[/cyan]\n"
                        )
                    else:
                        details_text += f"• {key.capitalize()}: [cyan]{value}[/cyan]\n"

        # Checkpoints
        if "checkpoints" in run_info:
            details_text += f"\n[bold white]Checkpoints:[/bold white]\n"
            for checkpoint in run_info["checkpoints"]:
                details_text += f"• [cyan]{checkpoint}[/cyan]\n"

        # Create a panel with clear styling
        details_panel = Panel(
            Text.from_markup(details_text),
            title="Run Details",
            border_style="green",
            title_align="center",
        )

        # Update the details panel
        run_details.update(details_panel)

        # Ensure component is visible and refreshed
        run_details.display = True
        run_details.refresh()

        # Make sure parent container is updated
        tab_pane = self.query_one("#tab-details", TabPane)
        if hasattr(tab_pane, "refresh"):
            tab_pane.refresh()

    def update_logs_view(self, run):
        """Update the logs view with run logs."""
        log_viewer = self.query_one("#log-viewer", Static)

        # Make sure this widget is visible in the UI
        log_viewer.display = True

        # Find log files
        log_dir = Path(run["path"]) / "logs"
        log_content = ""

        if log_dir.exists() and log_dir.is_dir():
            log_files = list(log_dir.glob("*.log"))
            if log_files:
                # Use the first log file
                try:
                    with open(log_files[0], "r") as f:
                        # Get last 50 lines
                        lines = f.readlines()
                        log_content = "".join(lines[-50:])
                except:
                    log_content = f"Error reading log file: {log_files[0]}"
            else:
                log_content = "No log files found."
        else:
            log_content = f"No logs directory found in {run['path']}"

        log_viewer.update(
            Syntax(log_content, "log", theme="monokai", line_numbers=True)
        )

        # Ensure component is visible and refreshed
        log_viewer.display = True
        log_viewer.refresh()

        # Make sure parent container is updated
        tab_pane = self.query_one("#tab-logs", TabPane)
        if hasattr(tab_pane, "refresh"):
            tab_pane.refresh()

    def update_config_view(self, run):
        """Update the config view with run configuration."""
        config_viewer = self.query_one("#config-viewer", ConfigViewer)

        # Make sure the tab content is visible
        config_tab = self.query_one("#tab-config", TabPane)
        if hasattr(config_tab, "display"):
            config_tab.display = True

        # Find config file
        config_file = Path(run["path"]) / "telemetry" / "training_config.yaml"
        if config_file.exists():
            config_viewer.update_config(str(config_file))
        else:
            # Try alternative locations
            alt_config_file = Path(run["path"]) / "config.yaml"
            if alt_config_file.exists():
                config_viewer.update_config(str(alt_config_file))
            else:
                # No config file found
                config_viewer.query_one(
                    "#config-viewer", TextArea
                ).text = f"No configuration file found in {run['path']}"

        # Ensure the config viewer is refreshed
        if hasattr(config_viewer, "refresh"):
            config_viewer.refresh()

        # Make sure parent container is updated
        tab_pane = self.query_one("#tab-config", TabPane)
        if hasattr(tab_pane, "refresh"):
            tab_pane.refresh()

    def update_metrics_view(self, run):
        """Update the metrics view with training metrics."""
        metrics_viz = self.query_one("#metrics-viz", MetricsVisualizer)

        # Make sure the tab content is visible
        metrics_tab = self.query_one("#tab-metrics", TabPane)
        if hasattr(metrics_tab, "display"):
            metrics_tab.display = True

        # Initialize metrics dictionary
        training_metrics = {}

        # Try to get metrics from trainer_state.json
        trainer_state_path = Path(run["path"]) / "trainer_state.json"
        if trainer_state_path.exists():
            try:
                with open(trainer_state_path, "r") as f:
                    trainer_state = json.loads(f.read())

                # Extract metrics from trainer state
                if "log_history" in trainer_state and trainer_state["log_history"]:
                    # Get the latest log entry
                    latest_log = trainer_state["log_history"][-1]

                    # Performance metrics
                    performance_metrics = {}
                    if "train_samples_per_second" in latest_log:
                        performance_metrics["samples_per_second"] = latest_log[
                            "train_samples_per_second"
                        ]
                    if "train_steps_per_second" in latest_log:
                        performance_metrics["steps_per_second"] = latest_log[
                            "train_steps_per_second"
                        ]
                    if "total_flos" in latest_log:
                        performance_metrics["total_flos"] = (
                            f"{latest_log['total_flos'] / 1e12:.2f} TFLOPs"
                        )
                    if "train_runtime" in latest_log:
                        performance_metrics["runtime"] = (
                            f"{latest_log['train_runtime']:.2f} seconds"
                        )

                    if performance_metrics:
                        training_metrics["performance"] = performance_metrics

                    # Training metrics
                    train_metrics = {}
                    if "train_loss" in latest_log:
                        train_metrics["loss"] = latest_log["train_loss"]
                    if "epoch" in latest_log:
                        train_metrics["epoch"] = latest_log["epoch"]
                    if "step" in latest_log:
                        train_metrics["step"] = latest_log["step"]
                    if "mean_token_accuracy" in latest_log:
                        train_metrics["token_accuracy"] = latest_log[
                            "mean_token_accuracy"
                        ]
                    if "num_tokens" in latest_log:
                        train_metrics["tokens_processed"] = int(
                            latest_log["num_tokens"]
                        )

                    if train_metrics:
                        training_metrics["training"] = train_metrics

                # Global training info
                global_info = {}
                for key in ["train_batch_size", "max_steps", "num_train_epochs"]:
                    if key in trainer_state:
                        global_info[key] = trainer_state[key]

                if global_info:
                    training_metrics["configuration"] = global_info
            except Exception as e:
                training_metrics["error"] = {
                    "message": f"Error parsing trainer state: {str(e)}",
                    "file": str(trainer_state_path),
                }

        # Try to extract learning rate from config if not present in metrics
        if (
            "training" in training_metrics
            and "learning_rate" not in training_metrics["training"]
        ):
            config_file = Path(run["path"]) / "telemetry" / "training_config.yaml"
            if config_file.exists():
                try:
                    with open(config_file, "r") as f:
                        config = yaml.safe_load(f)
                    if (
                        config
                        and "training" in config
                        and "learning_rate" in config["training"]
                    ):
                        if "training" not in training_metrics:
                            training_metrics["training"] = {}
                        training_metrics["training"]["learning_rate"] = config[
                            "training"
                        ]["learning_rate"]
                except:
                    pass

        # Get model info
        model_info = {}
        # From config.json
        config_json_path = Path(run["path"]) / "config.json"
        if config_json_path.exists():
            try:
                with open(config_json_path, "r") as f:
                    model_config = json.loads(f.read())
                if "hidden_size" in model_config:
                    model_info["hidden_size"] = model_config["hidden_size"]
                if "num_hidden_layers" in model_config:
                    model_info["num_layers"] = model_config["num_hidden_layers"]
                if "num_attention_heads" in model_config:
                    model_info["attention_heads"] = model_config["num_attention_heads"]
                if "vocab_size" in model_config:
                    model_info["vocab_size"] = model_config["vocab_size"]
            except:
                pass

        # From training_config.yaml
        config_file = Path(run["path"]) / "telemetry" / "training_config.yaml"
        if config_file.exists():
            try:
                with open(config_file, "r") as f:
                    config = yaml.safe_load(f)
                if config and "model" in config and "model_name" in config["model"]:
                    model_info["model_name"] = config["model"]["model_name"]
            except:
                pass

        if model_info:
            training_metrics["model"] = model_info

        # If we didn't find any metrics, add some placeholders
        if not training_metrics:
            training_metrics = {
                "note": {
                    "message": "No detailed metrics found for this run",
                    "run_directory": run["path"],
                }
            }

        # Add log file info
        log_dir = Path(run["path"]) / "logs"
        if log_dir.exists() and log_dir.is_dir():
            log_files = list(log_dir.glob("*.log"))
            if log_files:
                if "files" not in training_metrics:
                    training_metrics["files"] = {}
                training_metrics["files"]["log_file"] = log_files[0].name
                training_metrics["files"]["log_size"] = (
                    f"{log_files[0].stat().st_size / 1024:.1f} KB"
                )

        # Update metrics visualization
        metrics_viz.update_metrics(training_metrics)

        # Ensure the metrics viz is refreshed
        if hasattr(metrics_viz, "refresh"):
            metrics_viz.refresh()

        # Make sure parent container is updated
        tab_pane = self.query_one("#tab-metrics", TabPane)
        if hasattr(tab_pane, "refresh"):
            tab_pane.refresh()

        # Reset programmatic tab change flag now that all updates are done
        self._programmatic_tab_change = False

    @on(Tabs.TabActivated, "#run-tabs")
    def handle_tab_change(self, event: Tabs.TabActivated):
        """Handle tab change events."""
        if not hasattr(event, "tab_index"):
            return

        # Direct mapping of tab index to view name
        tab_index = event.tab_index
        if tab_index == 0:
            self.current_view = "details"
        elif tab_index == 1:
            self.current_view = "logs"
        elif tab_index == 2:
            self.current_view = "config"
        elif tab_index == 3:
            self.current_view = "metrics"
        else:
            return

        # Skip if this is a programmatic tab change that we triggered
        # This prevents recursive updates when we change tabs in code
        if (
            not hasattr(self, "_programmatic_tab_change")
            or not self._programmatic_tab_change
        ):
            # Update the current view
            if self.selected_run_path:
                run = next(
                    (r for r in self.runs if r["path"] == self.selected_run_path), None
                )
                if run:
                    # Set flag to prevent recursive tab changes
                    self._programmatic_tab_change = True
                    self.update_current_view()
                    self._programmatic_tab_change = False

    def action_switch_view(self) -> None:
        """Cycle through views with tab key."""
        if not self.selected_run_path:
            return

        run = next((r for r in self.runs if r["path"] == self.selected_run_path), None)
        if not run:
            return

        views = ["details", "logs", "config", "metrics"]

        # Find current index and calculate next view
        current_index = views.index(self.current_view)
        next_index = (current_index + 1) % len(views)
        next_view = views[next_index]

        # Set the new view
        self.current_view = next_view

        # Use the update_current_view method to handle tab activation and content update
        self.update_current_view()

        # Provide user feedback
        self.app.notify(f"Switched to {next_view} view")

    @on(Button.Pressed, "#view-logs")
    def view_logs(self):
        """View logs for selected run."""
        # Call the keyboard action handler
        self.action_view_logs()

    @on(Button.Pressed, "#view-config")
    def view_config(self):
        """View configuration for selected run."""
        # Call the keyboard action handler
        self.action_view_config()

    @on(Button.Pressed, "#view-metrics")
    def view_metrics(self):
        """View metrics for selected run."""
        # Call the keyboard action handler
        self.action_view_metrics()

    def action_view_logs(self) -> None:
        """View logs with keyboard shortcut."""
        if not self.selected_run_path:
            return

        run = next((r for r in self.runs if r["path"] == self.selected_run_path), None)
        if not run:
            return

        try:
            # Set the current view
            self.current_view = "logs"

            # Get the tabs widget
            tabs = self.query_one("#run-tabs", Tabs)

            # Use direct index approach - simpler and more reliable
            # In the TrainingMonitor's compose method, tabs are in this order:
            # 0: Details, 1: Logs, 2: Config, 3: Metrics
            tabs.active = 1  # Logs tab is index 1

            # No need to search for tab index - we're using a direct approach

            # Update the logs view directly
            self.update_logs_view(run)

            # Provide user feedback
            self.app.notify("Showing logs view")
        except Exception as e:
            # Provide diagnostic information
            self.app.notify(f"Error showing logs: {str(e)}", severity="error")

    def action_view_config(self) -> None:
        """View config with keyboard shortcut."""
        if not self.selected_run_path:
            return

        run = next((r for r in self.runs if r["path"] == self.selected_run_path), None)
        if not run:
            return

        try:
            # Set the current view
            self.current_view = "config"

            # Get the tabs widget
            tabs = self.query_one("#run-tabs", Tabs)

            # Use direct index approach - simpler and more reliable
            # In the TrainingMonitor's compose method, tabs are in this order:
            # 0: Details, 1: Logs, 2: Config, 3: Metrics
            tabs.active = 2  # Config tab is index 2

            # Update the config view directly
            self.update_config_view(run)

            # Provide user feedback
            self.app.notify("Showing config view")
        except Exception as e:
            # Provide diagnostic information
            self.app.notify(f"Error showing config: {str(e)}", severity="error")

    def action_view_metrics(self) -> None:
        """View metrics with keyboard shortcut."""
        if not self.selected_run_path:
            return

        run = next((r for r in self.runs if r["path"] == self.selected_run_path), None)
        if not run:
            return

        try:
            # Set the current view
            self.current_view = "metrics"

            # Get the tabs widget
            tabs = self.query_one("#run-tabs", Tabs)

            # Use direct index approach - simpler and more reliable
            # In the TrainingMonitor's compose method, tabs are in this order:
            # 0: Details, 1: Logs, 2: Config, 3: Metrics
            tabs.active = 3  # Metrics tab is index 3

            # Update the metrics view directly
            self.update_metrics_view(run)

            # Provide user feedback
            self.app.notify("Showing metrics view")
        except Exception as e:
            # Provide diagnostic information
            self.app.notify(f"Error showing metrics: {str(e)}", severity="error")

    def action_stop_run(self) -> None:
        """Stop selected run with keyboard shortcut."""
        if not self.selected_run_path:
            return

        run = next((r for r in self.runs if r["path"] == self.selected_run_path), None)
        if run and run["status"] == "Running":
            # In a real implementation, this would stop the run
            self.app.notify(f"Stopping run: {run['name']}...")
            # Update status
            run["status"] = "Stopping"
            # Refresh the table
            self.refresh_runs()

    def action_refresh(self) -> None:
        """Refresh runs with keyboard shortcut."""
        self.refresh_runs()

    def action_previous_run(self) -> None:
        """Select previous run in the table."""
        table = self.query_one("#runs-table", DataTable)
        if table.row_count > 0:
            current_row = table.cursor_row or 0
            new_row = max(0, current_row - 1)
            table.move_cursor(row=new_row)
            # Trigger row selection to update details
            table.action_select_cursor()

    def action_next_run(self) -> None:
        """Select next run in the table."""
        table = self.query_one("#runs-table", DataTable)
        if table.row_count > 0:
            current_row = table.cursor_row or 0
            new_row = min(table.row_count - 1, current_row + 1)
            table.move_cursor(row=new_row)
            # Trigger row selection to update details
            table.action_select_cursor()

    @on(Button.Pressed, "#refresh-runs")
    def refresh_runs(self):
        """Refresh the list of runs."""
        # Reload runs from the output directory
        self.load_runs()

        # If there was a selected run, try to reselect it
        if self.selected_run_path:
            table = self.query_one("#runs-table", DataTable)
            for i, run in enumerate(self.runs):
                if run["path"] == self.selected_run_path:
                    table.move_cursor(row=i)
                    table.action_select_cursor()
                    break


class SaveConfigScreen(ModalScreen):
    """Screen for saving configuration to a file."""

    BINDINGS = [
        Binding("escape", "cancel", "Cancel"),
        Binding("enter", "save", "Save"),
    ]

    def __init__(self, config_text: str):
        """Initialize the save config screen."""
        super().__init__()
        self.config_text = config_text

    def action_cancel(self) -> None:
        """Cancel saving and close the modal."""
        self.dismiss()

    def action_save(self) -> None:
        """Save configuration when Enter is pressed."""
        self.save()

    def compose(self) -> ComposeResult:
        """Compose the save config screen."""
        yield Container(
            Label("Save Configuration", classes="modal-title"),
            Input(
                placeholder="Enter a filename (e.g., train_config.yaml)",
                id="filename-input",
            ),
            Horizontal(
                Button("Cancel", variant="error", id="cancel-button"),
                Button("Save", variant="success", id="save-button"),
                classes="button-row",
            ),
            id="save-dialog",
        )

    @on(Button.Pressed, "#cancel-button")
    def cancel(self):
        """Cancel saving and close the modal."""
        self.dismiss()

    @on(Button.Pressed, "#save-button")
    def save(self):
        """Save the configuration to a file."""
        filename = self.query_one("#filename-input", Input).value
        if not filename:
            filename = "config.yaml"

        if not filename.endswith((".yaml", ".yml")):
            filename += ".yaml"

        try:
            config_dir = Path("configs")
            config_dir.mkdir(exist_ok=True)

            file_path = config_dir / filename
            with open(file_path, "w") as f:
                f.write(self.config_text)

            self.app.notify(f"Configuration saved to {file_path}")
            self.dismiss()
        except Exception as e:
            self.app.notify(f"Error saving configuration: {e}", severity="error")
            self.dismiss()


class RunCommandScreen(ModalScreen):
    """Screen for running a command with the configuration."""

    BINDINGS = [
        Binding("escape", "cancel", "Cancel"),
        Binding("enter", "run", "Run"),
        Binding("1", "select_train", "Training"),
        Binding("2", "select_eval", "Evaluation"),
        Binding("3", "select_infer", "Inference"),
    ]

    def __init__(self, config_text: str):
        """Initialize the run command screen."""
        super().__init__()
        self.config_text = config_text

    def action_cancel(self) -> None:
        """Cancel running and close the modal."""
        self.dismiss()

    def action_run(self) -> None:
        """Run command when Enter is pressed."""
        self.run()

    def action_select_train(self) -> None:
        """Select training option."""
        option_list = self.query_one("#command-option", OptionList)
        option_list.highlighted = 0

    def action_select_eval(self) -> None:
        """Select evaluation option."""
        option_list = self.query_one("#command-option", OptionList)
        option_list.highlighted = 1

    def action_select_infer(self) -> None:
        """Select inference option."""
        option_list = self.query_one("#command-option", OptionList)
        option_list.highlighted = 2

    def compose(self) -> ComposeResult:
        """Compose the run command screen."""
        yield Container(
            Label("Run Command", classes="modal-title"),
            Static("What command would you like to run?"),
            OptionList(
                "Training (oumi train)",
                "Evaluation (oumi evaluate)",
                "Inference (oumi infer)",
                id="command-option",
            ),
            Horizontal(
                Button("Cancel", variant="error", id="cancel-button"),
                Button("Run", variant="success", id="run-button"),
                classes="button-row",
            ),
            id="run-dialog",
        )

    @on(Button.Pressed, "#cancel-button")
    def cancel(self):
        """Cancel running and close the modal."""
        self.dismiss()

    @on(Button.Pressed, "#run-button")
    def run(self):
        """Run the selected command."""
        command_options = {0: "train", 1: "evaluate", 2: "infer"}

        option_list = self.query_one("#command-option", OptionList)
        if option_list.highlighted is None:
            # Default to training if nothing selected
            command = "train"
        else:
            command = command_options.get(option_list.highlighted, "train")

        # Save the config to a temporary file
        config_path = Path("temp_config.yaml")
        with open(config_path, "w") as f:
            f.write(self.config_text)

        cmd = f"oumi {command} -c {config_path}"
        self.app.notify(f"Running: {cmd}")
        self.dismiss()

        # In a real implementation, we would actually run the command
        # and capture the output for display in the app
        # For now, we'll just print the command that would be run
        print(f"Would run: {cmd}")


class HelpScreen(ModalScreen):
    """Screen for showing keyboard shortcuts and help information."""

    BINDINGS = [
        Binding("escape", "close", "Close"),
    ]

    def compose(self) -> ComposeResult:
        """Compose the help screen."""
        yield Container(
            Label("Keyboard Shortcuts", classes="modal-title"),
            Static(
                """
# Global Navigation
- 1-4: Switch between main sections
- Q: Quit
- D: Toggle dark mode
- F1: Show this help
- Tab/Shift+Tab: Move between navigation and content
- Arrow keys: 
  * Left/Right: Move between navigation buttons
  * Up: Move focus to navigation bar
  * Down: Move focus to content area

# Dataset/Model Browser
- Tab/Shift+Tab: Move between widgets
- Up/Down: Navigate lists
- Left: Focus category selector
- Right: Focus list
- Enter: Select item
- C: Cycle categories

# Config Builder
- T: Build training config
- E: Build evaluation config
- I: Build inference config
- S: Save config
- R: Run command
- Arrow keys: Navigate in text editor

# Training Monitor
- Up/Down: Navigate between runs
- Tab: Switch between detail views
- L: View logs of selected run
- C: View config of selected run
- M: View metrics of selected run
- X: Stop selected run
- F5: Refresh view

# Modals
- Escape: Cancel/Close
- Enter: Confirm/Save
                """,
                classes="help-text",
            ),
            Button("Close", id="close-button", variant="primary"),
            id="help-dialog",
        )

    def action_close(self) -> None:
        """Close the help screen."""
        self.dismiss()

    @on(Button.Pressed, "#close-button")
    def close_help(self):
        """Close the help screen."""
        self.dismiss()


class LogViewerScreen(ModalScreen):
    """Screen for viewing logs from a training run."""

    BINDINGS = [
        Binding("escape", "close", "Close"),
        Binding("f5", "refresh", "Refresh"),
    ]

    def __init__(self, run_id: str):
        """Initialize the log viewer screen."""
        super().__init__()
        self.run_id = run_id
        self.log_content = ""

    def action_close(self) -> None:
        """Close the log viewer."""
        self.dismiss()

    def action_refresh(self) -> None:
        """Refresh logs with F5 key."""
        self.refresh()

    def compose(self) -> ComposeResult:
        """Compose the log viewer screen."""
        yield Container(
            Label(f"Log Viewer - Run {self.run_id}", classes="modal-title"),
            Static(id="log-content", classes="logs"),
            Horizontal(
                Button("Close", id="close-button"),
                Button("Refresh", variant="primary", id="refresh-logs"),
                classes="button-row",
            ),
            id="log-viewer",
        )

    def on_mount(self):
        """Load logs when the screen is mounted."""
        # In a real implementation, this would load actual logs
        # For now, we'll use mock data
        mock_logs = f"""
[2025-04-29 12:34:56] Starting training run {self.run_id}
[2025-04-29 12:35:01] Loaded model from cache
[2025-04-29 12:35:15] Loaded dataset with 50,000 examples
[2025-04-29 12:35:30] Starting epoch 1/3
[2025-04-29 12:40:45] Epoch 1/3 complete: loss=2.345, accuracy=0.675
[2025-04-29 12:40:50] Starting epoch 2/3
[2025-04-29 12:46:10] Epoch 2/3 complete: loss=1.876, accuracy=0.723
[2025-04-29 12:46:15] Starting epoch 3/3
[2025-04-29 12:51:30] Epoch 3/3 complete: loss=1.245, accuracy=0.768
[2025-04-29 12:51:45] Training complete
[2025-04-29 12:52:00] Saving model to output directory
[2025-04-29 12:52:30] Model saved successfully
"""
        self.log_content = mock_logs

        log_display = self.query_one("#log-content", Static)
        log_display.update(
            Syntax(self.log_content, "log", theme="monokai", line_numbers=True)
        )

    @on(Button.Pressed, "#close-button")
    def close(self):
        """Close the log viewer."""
        self.dismiss()

    @on(Button.Pressed, "#refresh-logs")
    def refresh(self):
        """Refresh the logs."""
        # In a real implementation, this would reload the actual logs
        # For the demo, we'll just append a new log line
        from datetime import datetime

        new_line = (
            f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
            f"Refreshed logs for run {self.run_id}"
        )
        self.log_content += new_line

        log_display = self.query_one("#log-content", Static)
        log_display.update(
            Syntax(self.log_content, "log", theme="monokai", line_numbers=True)
        )


class OumiTUI(App):
    """Oumi Terminal UI."""

    CSS_PATH = "oumi_tui.css"
    BINDINGS = [
        # Navigation
        Binding("q", "quit", "Quit"),
        Binding("d", "toggle_dark", "Toggle Dark Mode"),
        Binding("f1", "help", "Help"),
        # Tab navigation shortcuts
        Binding("1", "show_datasets", "Datasets"),
        Binding("2", "show_models", "Models"),
        Binding("3", "show_config", "Config"),
        Binding("4", "show_monitor", "Monitor"),
        # Common actions
        Binding("s", "save_config", "Save Config", show=False),
        Binding("r", "run_command", "Run Command", show=False),
        Binding("f5", "refresh", "Refresh", show=False),
        # Global arrow key navigation
        Binding("left", "global_left", "Left", show=False),
        Binding("right", "global_right", "Right", show=False),
        Binding("up", "global_up", "Up", show=False),
        Binding("down", "global_down", "Down", show=False),
        Binding("tab", "tab_to_content", "Content", show=False),
        Binding("shift+tab", "tab_to_nav", "Navigation", show=False),
    ]
    SCREENS = {
        "save_config": SaveConfigScreen,
        "run_command": RunCommandScreen,
        "log_viewer": LogViewerScreen,
        "help": HelpScreen,
        "dataset_viewer": DatasetViewerScreen,
        "dataset_registry": DatasetRegistryScreen,
    }

    def compose(self) -> ComposeResult:
        """Create the UI layout."""
        yield Header(classes="app-header")
        yield Static(LOGO, id="logo")
        yield Rule()
        yield Horizontal(
            Button("Dataset Explorer", id="nav-datasets", variant="primary"),
            Button("Model Explorer", id="nav-models", variant="primary"),
            Button("Config Builder", id="nav-config", variant="primary"),
            Button("Training Monitor", id="nav-monitor", variant="primary"),
            classes="nav-buttons",
        )
        yield Container(id="main-content")
        yield Footer()

    def on_mount(self):
        """Set up the initial UI state."""
        # Show the dataset browser by default
        self.show_dataset_browser()

        # Set initial focus to the navigation button
        self.set_focus(self.query_one("#nav-datasets", Button))

    def show_dataset_browser(self):
        """Show the dataset browser."""
        main_content = self.query_one("#main-content", Container)
        main_content.remove_children()
        # Create a unique ID each time to avoid duplicate ID error - using timestamp for true uniqueness
        # In Textual 3.x, we need to actually create an instance first
        dataset_browser = DatasetBrowser(id=f"dataset-browser-{time.time_ns()}")
        main_content.mount(dataset_browser)

    def show_model_explorer(self):
        """Show the model explorer."""
        main_content = self.query_one("#main-content", Container)
        main_content.remove_children()
        # Create a unique ID each time to avoid duplicate ID error - using timestamp for true uniqueness
        model_selector = ModelSelector(id=f"model-explorer-{time.time_ns()}")
        main_content.mount(model_selector)

    def show_config_builder(self):
        """Show the configuration builder."""
        main_content = self.query_one("#main-content", Container)
        main_content.remove_children()
        # Create a unique ID each time to avoid duplicate ID error - using timestamp for true uniqueness
        config_builder = ConfigBuilder(id=f"config-builder-{time.time_ns()}")
        main_content.mount(config_builder)

    def show_training_monitor(self):
        """Show the training monitor."""
        main_content = self.query_one("#main-content", Container)
        main_content.remove_children()
        # Create a unique ID each time to avoid duplicate ID error - using timestamp for true uniqueness
        training_monitor = TrainingMonitor(id=f"training-monitor-{time.time_ns()}")
        main_content.mount(training_monitor)

    @on(Button.Pressed, "#nav-datasets")
    def handle_datasets_nav(self):
        """Handle click on the Datasets navigation button."""
        self.show_dataset_browser()
        # Focus on content after navigation
        self._focus_main_content()

    @on(Button.Pressed, "#nav-models")
    def handle_models_nav(self):
        """Handle click on the Models navigation button."""
        self.show_model_explorer()
        # Focus on content after navigation
        self._focus_main_content()

    @on(Button.Pressed, "#nav-config")
    def handle_config_nav(self):
        """Handle click on the Config navigation button."""
        self.show_config_builder()
        # Focus on content after navigation
        self._focus_main_content()

    @on(Button.Pressed, "#nav-monitor")
    def handle_monitor_nav(self):
        """Handle click on the Monitor navigation button."""
        self.show_training_monitor()
        # Focus on content after navigation
        self._focus_main_content()

    def action_toggle_dark(self) -> None:
        """Toggle dark mode."""
        self.dark = not self.dark

    def action_show_datasets(self) -> None:
        """Navigate to dataset browser."""
        self.show_dataset_browser()
        # Focus content area after navigation
        self._focus_main_content()
        # Highlight the corresponding nav button
        self.query_one("#nav-datasets", Button).focus()

    def action_show_models(self) -> None:
        """Navigate to model explorer."""
        self.show_model_explorer()
        # Focus content area after navigation
        self._focus_main_content()
        # Highlight the corresponding nav button
        self.query_one("#nav-models", Button).focus()

    def action_show_config(self) -> None:
        """Navigate to config builder."""
        self.show_config_builder()
        # Focus content area after navigation
        self._focus_main_content()
        # Highlight the corresponding nav button
        self.query_one("#nav-config", Button).focus()

    def action_show_monitor(self) -> None:
        """Navigate to training monitor."""
        self.show_training_monitor()
        # Focus content area after navigation
        self._focus_main_content()
        # Highlight the corresponding nav button
        self.query_one("#nav-monitor", Button).focus()

    def action_save_config(self) -> None:
        """Trigger save configuration action if config builder is visible."""
        config_builder = self.query("ConfigBuilder")
        if config_builder:
            # Find the save config button and trigger a press
            save_button = self.query_one("#save-config", Button)
            if save_button:
                save_button.press()

    def action_run_command(self) -> None:
        """Trigger run command action if config builder is visible."""
        config_builder = self.query("ConfigBuilder")
        if config_builder:
            # Find the run command button and trigger a press
            run_button = self.query_one("#run-command", Button)
            if run_button:
                run_button.press()

    def action_refresh(self) -> None:
        """Refresh the current view."""
        # If in training monitor, refresh runs
        training_monitor = self.query("TrainingMonitor")
        if training_monitor:
            refresh_button = self.query_one("#refresh-runs", Button)
            if refresh_button:
                refresh_button.press()

    def action_help(self) -> None:
        """Show the help screen."""
        self.push_screen("help")

    def action_global_left(self) -> None:
        """Global left arrow navigation."""
        # Handle left arrow navigation across the UI
        nav_buttons = self.query(".nav-buttons Button")

        # Check if any navigation button has focus
        focused_button = None
        for i, button in enumerate(nav_buttons):
            if button.has_focus:
                focused_button = i
                break

        if focused_button is not None and focused_button > 0:
            # Move focus to the previous navigation button
            nav_buttons[focused_button - 1].focus()

    def action_global_right(self) -> None:
        """Global right arrow navigation."""
        # Handle right arrow navigation across the UI
        nav_buttons = self.query(".nav-buttons Button")

        # Check if any navigation button has focus
        focused_button = None
        for i, button in enumerate(nav_buttons):
            if button.has_focus:
                focused_button = i
                break

        if focused_button is not None and focused_button < len(nav_buttons) - 1:
            # Move focus to the next navigation button
            nav_buttons[focused_button + 1].focus()

    def action_global_up(self) -> None:
        """Global up arrow navigation."""
        # If we're in the main content and no widget has specific focus handling,
        # move focus up to the navigation buttons
        if self._content_has_focus() and not self._active_widget_handles_keys():
            self._focus_nav_buttons()

    def action_global_down(self) -> None:
        """Global down arrow navigation."""
        # If nav buttons have focus, move down to the content area
        nav_buttons = self.query(".nav-buttons Button")
        for button in nav_buttons:
            if button.has_focus:
                self._focus_main_content()
                return

    def action_tab_to_content(self) -> None:
        """Tab from navigation to content area."""
        # Only handle this if a navigation button has focus
        nav_buttons = self.query(".nav-buttons Button")
        for button in nav_buttons:
            if button.has_focus:
                self._focus_main_content()
                return

    def action_tab_to_nav(self) -> None:
        """Tab from content to navigation area."""
        # Only handle this if we're in the main content
        if self._content_has_focus():
            self._focus_nav_buttons()

    def _content_has_focus(self) -> bool:
        """Check if any widget in the main content has focus."""
        main_content = self.query_one("#main-content", Container)
        for widget in main_content.query("*"):
            if widget.has_focus:
                return True
        return False

    def _active_widget_handles_keys(self) -> bool:
        """Check if the active widget implements its own key handling."""
        # This is a simplification - in reality, we would need to check
        # if the focused widget implements specific actions
        focused = self.focused
        return focused and any(
            hasattr(focused, f"action_{key}")
            for key in ["cursor_up", "cursor_down", "cursor_left", "cursor_right"]
        )

    def _focus_nav_buttons(self) -> None:
        """Focus on the first navigation button."""
        nav_buttons = self.query(".nav-buttons Button")
        if nav_buttons:
            nav_buttons[0].focus()

    def _focus_main_content(self) -> None:
        """Focus on the first focusable widget in the main content."""
        main_content = self.query_one("#main-content", Container)
        # Try to find a focusable widget in main content
        # In Textual 3.x, we need to check if the widget can receive focus
        for widget in main_content.query("Input, Select, TextArea, ListView"):
            if hasattr(widget, "can_focus") and widget.can_focus:
                widget.focus()
                return
            # Fallback for widgets without can_focus attribute
            try:
                widget.focus()
                return
            except:
                continue

    def on_screen_resume(self, event) -> None:
        """Handle when a screen is dismissed and we return to the app."""
        # The screen functionality has been updated to use app.notify directly
        # rather than passing messages through dismiss


def main():
    """Run the Oumi Terminal UI."""
    app = OumiTUI()
    app.run()


if __name__ == "__main__":
    main()
