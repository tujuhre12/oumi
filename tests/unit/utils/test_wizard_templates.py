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

"""Tests for the template system of the config wizard."""

import os
import pytest
import yaml

from oumi.utils.wizard import templates


def test_render_template():
    """Test rendering a Jinja template."""
    # Test basic rendering
    result = templates.render_template("base/model.jinja", model_name="test-model")
    assert "test-model" in result
    assert "model:" in result
    
    # Test with multiple variables
    result = templates.render_template(
        "base/model.jinja", 
        model_name="test-model", 
        model_max_length=4096, 
        trust_remote_code=True
    )
    assert "test-model" in result
    assert "4096" in result
    assert "trust_remote_code: True" in result
    
    # Make sure the result is valid YAML
    yaml_result = yaml.safe_load(result)
    assert isinstance(yaml_result, dict)
    assert "model" in yaml_result


def test_get_training_template():
    """Test getting a training template."""
    model_name = "test-model"
    dataset_name = "test-dataset"
    
    # Test full fine-tuning template
    template = templates.get_training_template(model_name, "full", dataset_name)
    assert model_name in template
    assert dataset_name in template
    assert "enable_fsdp: true" in template.lower()
    
    # Test LoRA template
    template = templates.get_training_template(model_name, "lora", dataset_name)
    assert model_name in template
    assert dataset_name in template
    assert "enable_peft: true" in template.lower()
    
    # Test QLoRA template
    template = templates.get_training_template(model_name, "qlora", dataset_name)
    assert model_name in template
    assert dataset_name in template
    assert "load_in_4bit: true" in template.lower()
    
    # Test default dataset
    template = templates.get_training_template(model_name, "lora")
    assert model_name in template
    assert "yahma/alpaca-cleaned" in template
    
    # Test invalid training type (should default to qlora)
    template = templates.get_training_template(model_name, "invalid", dataset_name)
    assert model_name in template
    assert dataset_name in template
    assert "load_in_4bit: true" in template.lower()
    
    # Ensure all templates are valid YAML
    for training_type in ["full", "lora", "qlora"]:
        template = templates.get_training_template(model_name, training_type, dataset_name)
        yaml_result = yaml.safe_load(template)
        assert isinstance(yaml_result, dict)


def test_get_evaluation_template():
    """Test getting an evaluation template."""
    model_name = "test-model"
    
    template = templates.get_evaluation_template(model_name)
    assert model_name in template
    assert "evaluation_backend" in template
    assert "lm_harness" in template
    
    # Make sure the result is valid YAML
    yaml_result = yaml.safe_load(template)
    assert isinstance(yaml_result, dict)
    assert "model" in yaml_result
    assert "tasks" in yaml_result


def test_get_inference_template():
    """Test getting an inference template."""
    model_name = "test-model"
    
    template = templates.get_inference_template(model_name)
    assert model_name in template
    assert "engine" in template
    assert "generation" in template
    
    # Make sure the result is valid YAML
    yaml_result = yaml.safe_load(template)
    assert isinstance(yaml_result, dict)
    assert "model" in yaml_result
    assert "generation" in yaml_result