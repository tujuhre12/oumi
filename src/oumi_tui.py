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

import random
import time
from pathlib import Path

import yaml

# Import only what we need
from rich.panel import Panel
from rich.syntax import Syntax
from rich.text import Text
from textual import on
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal
from textual.screen import ModalScreen
from textual.widgets import (
    Button,
    DataTable,
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
    TextArea,
)

LOGO = r"""
   ____  _    _ __  __ _____
  / __ \| |  | |  \/  |_   _|
 | |  | | |  | | \  / | | |
 | |  | | |  | | |\/| | | |
 | |__| | |__| | |  | |_| |_
  \____/ \____/|_|  |_|_____|"""


class ConfigPanel(Static):
    """A panel for displaying and editing configuration."""

    def __init__(self, config: dict = None, **kwargs):
        """Initialize the config panel."""
        super().__init__(**kwargs)
        self.config = config or {}

    def compose(self) -> ComposeResult:
        """Compose the config panel."""
        yield TextArea(self._config_to_yaml(), language="yaml", id="config-editor")

    def _config_to_yaml(self) -> str:
        """Convert the config dictionary to YAML."""
        return yaml.dump(self.config, default_flow_style=False)

    def update_config(self, config: dict):
        """Update the configuration."""
        self.config = config
        if self.query_one("#config-editor", TextArea):
            self.query_one("#config-editor", TextArea).text = self._config_to_yaml()


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
        select.value = category
        
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


class TrainingMonitor(Container):
    """Monitor training progress."""
    
    BINDINGS = [
        Binding("l", "view_logs", "View Logs"),
        Binding("x", "stop_run", "Stop Run"),
        Binding("f5", "refresh", "Refresh"),
        Binding("up", "previous_run", "Previous Run"),
        Binding("down", "next_run", "Next Run"),
    ]

    def __init__(self, **kwargs):
        """Initialize the training monitor."""
        super().__init__(**kwargs)
        self.active_runs = []

    def compose(self) -> ComposeResult:
        """Compose the training monitor."""
        yield Label("Training Runs", classes="section-header")
        yield DataTable(id="runs-table")
        yield Label("Selected Run Details", classes="section-header")
        yield Static("Select a run to view details", id="run-details")
        yield Horizontal(
            Button("View Logs", id="view-logs", disabled=True),
            Button("Stop Run", id="stop-run", disabled=True, variant="error"),
            Button("Refresh", id="refresh-runs", variant="primary"),
            classes="button-row",
        )

    def on_mount(self):
        """Set up the runs table and load runs."""
        table = self.query_one("#runs-table", DataTable)
        table.add_columns(
            "ID", "Name", "Status", "Progress", "Time Elapsed", "Resources"
        )

        # Mock active training runs
        self.active_runs = [
            ["run-001", "llama-3.1-8b-alpaca", "Running", "76%", "2h 34m", "1 GPU"],
            [
                "run-002",
                "phi-3-mini-ultrachat",
                "Completed",
                "100%",
                "5h 12m",
                "2 GPUs",
            ],
            ["run-003", "qwen2-7b-lora", "Pending", "0%", "0m", "4 GPUs"],
        ]

        # Add runs to the table
        for run in self.active_runs:
            table.add_row(*run)

    @on(DataTable.RowSelected, "#runs-table")
    def handle_row_selection(self, event: DataTable.RowSelected):
        """Handle run selection."""
        run_id = self.active_runs[event.row_key.row_index][0]
        run_name = self.active_runs[event.row_key.row_index][1]
        run_status = self.active_runs[event.row_key.row_index][2]

        # Update run details
        run_details = self.query_one("#run-details", Static)
        run_details.update(
            Panel(
                Text.from_markup(
                    f"[bold cyan]{run_name}[/bold cyan] ([purple]{run_id}[/purple])\n\n"
                    f"Status: {run_status}\n"
                    f"Model: {run_name.split('-')[0]}\n"
                    f"Dataset: {run_name.split('-')[-1]}\n"
                    f"Learning Rate: 2e-5\n"
                    f"Batch Size: 8\n"
                    f"Current Loss: 1.245\n"
                    f"Training Speed: 12.3 samples/sec\n"
                )
            )
        )

        # Enable buttons
        self.query_one("#view-logs", Button).disabled = False
        self.query_one("#stop-run", Button).disabled = run_status != "Running"

    @on(Button.Pressed, "#view-logs")
    def view_logs(self):
        """View logs for selected run."""
        selected_row = self.query_one("#runs-table", DataTable).cursor_row
        if selected_row is not None:
            run_id = self.active_runs[selected_row][0]
            self.app.push_screen(LogViewerScreen(run_id))

    def action_view_logs(self) -> None:
        """View logs of selected run with keyboard shortcut."""
        selected_row = self.query_one("#runs-table", DataTable).cursor_row
        if selected_row is not None:
            self.view_logs()
    
    def action_stop_run(self) -> None:
        """Stop selected run with keyboard shortcut."""
        selected_row = self.query_one("#runs-table", DataTable).cursor_row
        if selected_row is not None:
            run_status = self.active_runs[selected_row][2]
            if run_status == "Running":
                # In a real implementation, this would stop the run
                self.app.notify("Stopping run...")
                # Update status to stopping
                self.active_runs[selected_row][2] = "Stopping"
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
        # In a real implementation, we would re-fetch the list of runs
        # For the demo, we'll just randomize progress percentages

        table = self.query_one("#runs-table", DataTable)
        table.clear()

        # Update the mock data with new progress
        for i, run in enumerate(self.active_runs):
            if run[2] == "Running":
                progress = f"{random.randint(10, 99)}%"
                self.active_runs[i][3] = progress

        # Re-add to table
        for run in self.active_runs:
            table.add_row(*run)


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
- L: View logs of selected run
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
        main_content.mount(DatasetBrowser(id=f"dataset-browser-{time.time_ns()}"))

    def show_model_explorer(self):
        """Show the model explorer."""
        main_content = self.query_one("#main-content", Container)
        main_content.remove_children()
        # Create a unique ID each time to avoid duplicate ID error - using timestamp for true uniqueness
        main_content.mount(ModelSelector(id=f"model-explorer-{time.time_ns()}"))

    def show_config_builder(self):
        """Show the configuration builder."""
        main_content = self.query_one("#main-content", Container)
        main_content.remove_children()
        # Create a unique ID each time to avoid duplicate ID error - using timestamp for true uniqueness
        main_content.mount(ConfigBuilder(id=f"config-builder-{time.time_ns()}"))

    def show_training_monitor(self):
        """Show the training monitor."""
        main_content = self.query_one("#main-content", Container)
        main_content.remove_children()
        # Create a unique ID each time to avoid duplicate ID error - using timestamp for true uniqueness
        main_content.mount(TrainingMonitor(id=f"training-monitor-{time.time_ns()}"))

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
        return focused and any(hasattr(focused, f"action_{key}") for key in 
                              ["cursor_up", "cursor_down", "cursor_left", "cursor_right"])
                              
    def _focus_nav_buttons(self) -> None:
        """Focus on the first navigation button."""
        nav_buttons = self.query(".nav-buttons Button")
        if nav_buttons:
            nav_buttons[0].focus()
            
    def _focus_main_content(self) -> None:
        """Focus on the first focusable widget in the main content."""
        main_content = self.query_one("#main-content", Container)
        # Try to find a focusable widget in main content
        for widget in main_content.query("Input, Select, TextArea, ListView"):
            widget.focus()
            return

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
