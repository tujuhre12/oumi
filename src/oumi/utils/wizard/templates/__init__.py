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

"""Configuration templates for the Oumi config wizard."""

import os
from pathlib import Path
from typing import Dict, List, Optional, Any

import jinja2

# Define the template directory paths
TEMPLATES_DIR = os.path.dirname(os.path.abspath(__file__))

# Set up Jinja2 environment
jinja_env = jinja2.Environment(
    loader=jinja2.FileSystemLoader(TEMPLATES_DIR),
    trim_blocks=True,
    lstrip_blocks=True,
)


def render_template(template_name: str, **kwargs: Any) -> str:
    """Render a Jinja template with the given context.
    
    Args:
        template_name: Name of the template file
        **kwargs: Context variables for the template
        
    Returns:
        The rendered template as a string
    """
    template = jinja_env.get_template(template_name)
    return template.render(**kwargs)


def get_training_template(model_name: str, training_type: str, dataset_name: Optional[str] = None) -> str:
    """Get a training template for a given model and training type.
    
    Args:
        model_name: Name or HF identifier of the model
        training_type: Type of training (full, lora, qlora)
        dataset_name: Optional dataset name
        
    Returns:
        YAML template string
    """
    if dataset_name is None:
        dataset_name = "yahma/alpaca-cleaned"  # Default dataset
    
    # Map training type to template file
    if training_type not in ["full", "lora", "qlora"]:
        training_type = "qlora"  # Default to QLora
    
    template_name = f"train/{training_type}.jinja"
    
    context = {
        "model_name": model_name,
        "dataset_name": dataset_name,
    }
    
    return render_template(template_name, **context)


def get_evaluation_template(model_name: str) -> str:
    """Get an evaluation template for a given model.
    
    Args:
        model_name: Name or HF identifier of the model
        
    Returns:
        YAML template string
    """
    context = {
        "model_name": model_name,
    }
    
    return render_template("eval/base.jinja", **context)


def get_inference_template(model_name: str) -> str:
    """Get an inference template for a given model.
    
    Args:
        model_name: Name or HF identifier of the model
        
    Returns:
        YAML template string
    """
    context = {
        "model_name": model_name,
    }
    
    return render_template("infer/base.jinja", **context)