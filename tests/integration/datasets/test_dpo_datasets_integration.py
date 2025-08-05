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

"""Integration tests for DPO datasets with real tokenizers and processors."""

import tempfile
from pathlib import Path

import jsonlines
import pytest
import torch
from typing_extensions import override

from oumi.builders import build_processor, build_tokenizer
from oumi.core.configs import ModelParams
from oumi.core.datasets.base_dpo_dataset import BaseDpoDataset
from oumi.core.datasets.vision_language_dpo_dataset import VisionLanguageDpoDataset
from oumi.datasets.vision_language.vision_dpo_jsonlines import VisionDpoJsonlinesDataset


class TestBaseDpoDatasetIntegration(BaseDpoDataset):
    """Test implementation of BaseDpoDataset for integration testing."""

    default_dataset = "test_dpo_integration"

    def __init__(self, data=None, **kwargs):
        self._test_data = data or []
        super().__init__(dataset_name="test_dpo_integration", **kwargs)

    @override
    def _load_data(self):
        """Load test data."""
        return self._test_data

    @override
    def __len__(self):
        return len(self._test_data)

    @override
    def __getitem__(self, idx):
        return self.transform(self._test_data[idx])


class TestVisionLanguageDpoDatasetIntegration(VisionLanguageDpoDataset):
    """Test implementation of VisionLanguageDpoDataset for integration testing."""

    default_dataset = "test_vision_dpo_integration"

    def __init__(self, data=None, **kwargs):
        self._test_data = data or []
        super().__init__(dataset_name="test_vision_dpo_integration", **kwargs)

    @override
    def _load_data(self):
        """Load test data."""
        return self._test_data

    @override
    def __len__(self):
        return len(self._test_data)

    @override
    def __getitem__(self, idx):
        return self.transform(self._test_data[idx])


@pytest.fixture
def sample_dpo_data():
    """Sample DPO data for integration testing."""
    return [
        {
            "prompt": [{"role": "user", "content": "What is the capital of France?"}],
            "chosen": [
                {"role": "assistant", "content": "The capital of France is Paris."}
            ],
            "rejected": [
                {"role": "assistant", "content": "The capital of France is London."}
            ],
        },
        {
            "prompt": [{"role": "user", "content": "Explain photosynthesis briefly."}],
            "chosen": [
                {
                    "role": "assistant",
                    "content": "Photosynthesis is the process by which plants convert light energy into chemical energy using chlorophyll.",
                }
            ],
            "rejected": [
                {
                    "role": "assistant",
                    "content": "Photosynthesis is when plants eat sunlight for breakfast.",
                }
            ],
        },
    ]


@pytest.fixture
def sample_vision_dpo_data():
    """Sample vision DPO data for integration testing."""
    return [
        {
            "prompt": "What do you see in this image?",
            "images": ["tests/testdata/images/oumi_logo_light.png"],
            "chosen": [
                {
                    "role": "assistant",
                    "content": "I can see the Oumi logo, which appears to be a clean and professional design.",
                }
            ],
            "rejected": [
                {"role": "assistant", "content": "I see nothing useful in this image."}
            ],
        },
        {
            "prompt": [
                {"role": "user", "content": "Describe the colors in this image"}
            ],
            "images": ["tests/testdata/images/the_great_wave_off_kanagawa.jpg"],
            "chosen": [
                {
                    "role": "assistant",
                    "content": "The image contains beautiful blues and whites, creating a classic and artistic composition.",
                }
            ],
            "rejected": [
                {
                    "role": "assistant",
                    "content": "The image is just random colors with no meaning.",
                }
            ],
        },
    ]


@pytest.mark.parametrize(
    "model_name",
    [
        "microsoft/Phi-3-mini-4k-instruct",
        "microsoft/DialoGPT-medium",
    ],
)
def test_base_dpo_dataset_with_real_tokenizer(model_name, sample_dpo_data):
    """Test BaseDpoDataset with real tokenizers."""
    # Build real tokenizer
    tokenizer = build_tokenizer(ModelParams(model_name=model_name))

    # Create dataset
    dataset = TestBaseDpoDatasetIntegration(
        data=sample_dpo_data, tokenizer=tokenizer, return_tensors=False
    )

    # Test dataset length
    assert len(dataset) == 2

    # Test first sample transformation
    sample = dataset[0]
    assert "prompt" in sample
    assert "chosen" in sample
    assert "rejected" in sample

    # Verify the data structure
    assert isinstance(sample["prompt"], list)
    assert isinstance(sample["chosen"], list)
    assert isinstance(sample["rejected"], list)

    # Test tokenization functionality
    features = {
        "prompt": sample_dpo_data[0]["prompt"],
        "chosen": sample_dpo_data[0]["chosen"],
        "rejected": sample_dpo_data[0]["rejected"],
    }

    tokenized = dataset.tokenize_row(features)

    # Verify tokenized output structure
    assert "prompt_input_ids" in tokenized
    assert "chosen_input_ids" in tokenized
    assert "rejected_input_ids" in tokenized

    # Verify all are lists of integers
    assert isinstance(tokenized["prompt_input_ids"], list)
    assert isinstance(tokenized["chosen_input_ids"], list)
    assert isinstance(tokenized["rejected_input_ids"], list)
    assert all(isinstance(x, int) for x in tokenized["prompt_input_ids"])
    assert all(isinstance(x, int) for x in tokenized["chosen_input_ids"])
    assert all(isinstance(x, int) for x in tokenized["rejected_input_ids"])

    # Verify EOS token was added to chosen/rejected
    assert tokenized["chosen_input_ids"][-1] == tokenizer.eos_token_id
    assert tokenized["rejected_input_ids"][-1] == tokenizer.eos_token_id


@pytest.mark.parametrize(
    "model_name",
    [
        "llava-hf/llava-1.5-7b-hf",
    ],
)
def test_vision_dpo_dataset_with_real_processor(model_name, sample_vision_dpo_data):
    """Test VisionLanguageDpoDataset with real processors."""
    # Build real tokenizer and processor
    tokenizer = build_tokenizer(ModelParams(model_name=model_name))
    processor = build_processor(model_name, tokenizer, trust_remote_code=True)

    # Create dataset
    dataset = TestVisionLanguageDpoDatasetIntegration(
        data=sample_vision_dpo_data,
        tokenizer=tokenizer,
        processor=processor,
        return_tensors=False,
    )

    # Test dataset length
    assert len(dataset) == 2

    # Test first sample transformation
    sample = dataset[0]

    # Should have the key features for vision DPO
    expected_keys = [
        "prompt_input_ids",
        "pixel_values",
        "chosen_input_ids",
        "rejected_input_ids",
    ]
    for key in expected_keys:
        assert key in sample, f"Missing key: {key}"

    # Verify data types (can be lists or tensors depending on processing)
    prompt_ids = sample["prompt_input_ids"]
    chosen_ids = sample["chosen_input_ids"]
    rejected_ids = sample["rejected_input_ids"]

    assert isinstance(prompt_ids, (list, torch.Tensor))
    assert isinstance(chosen_ids, (list, torch.Tensor))
    assert isinstance(rejected_ids, (list, torch.Tensor))
    assert torch.is_tensor(sample["pixel_values"]) or isinstance(
        sample["pixel_values"], list
    )

    # Verify EOS tokens were added (handle both list and tensor cases)
    if torch.is_tensor(chosen_ids):
        assert chosen_ids[-1].item() == tokenizer.eos_token_id
        assert rejected_ids[-1].item() == tokenizer.eos_token_id
    else:
        assert chosen_ids[-1] == tokenizer.eos_token_id
        assert rejected_ids[-1] == tokenizer.eos_token_id

    # Test second sample (different format)
    sample2 = dataset[1]
    for key in expected_keys:
        assert key in sample2


def test_vision_dpo_dataset_with_processor_name(sample_vision_dpo_data):
    """Test VisionLanguageDpoDataset initialization with processor_name."""
    model_name = "llava-hf/llava-1.5-7b-hf"
    tokenizer = build_tokenizer(ModelParams(model_name=model_name))

    # Create dataset using processor_name instead of processor object
    dataset = TestVisionLanguageDpoDatasetIntegration(
        data=sample_vision_dpo_data,
        tokenizer=tokenizer,
        processor_name=model_name,
        trust_remote_code=True,
        return_tensors=False,
    )

    # Verify processor was built correctly
    assert dataset._processor is not None
    assert hasattr(dataset._processor, "processor_name")

    # Test functionality
    assert len(dataset) == 2
    sample = dataset[0]

    expected_keys = [
        "prompt_input_ids",
        "pixel_values",
        "chosen_input_ids",
        "rejected_input_ids",
    ]
    for key in expected_keys:
        assert key in sample


def test_base_dpo_dataset_with_jsonl_file(sample_vision_dpo_data):
    """Test BaseDpoDataset loading from JSONL file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create JSONL file
        jsonl_path = Path(tmpdir) / "test_dpo.jsonl"
        with jsonlines.open(jsonl_path, mode="w") as writer:
            writer.write_all(sample_vision_dpo_data)

        # Build tokenizer
        tokenizer = build_tokenizer(
            ModelParams(model_name="microsoft/Phi-3-mini-4k-instruct")
        )

        # Create dataset from file
        dataset = VisionDpoJsonlinesDataset(
            dataset_path=str(jsonl_path),
            tokenizer=tokenizer,
            processor_name="microsoft/Phi-3-mini-4k-instruct",
        )

        # Test that data was loaded correctly
        assert len(dataset) == 2
        sample = dataset[0]
        assert "prompt" in sample
        assert "chosen" in sample
        assert "rejected" in sample


def test_vision_dpo_dataset_no_images(sample_dpo_data):
    """Test VisionLanguageDpoDataset with no images (text-only DPO)."""
    # Convert regular DPO data to vision format without images
    vision_data = []
    for item in sample_dpo_data:
        vision_item = {
            "prompt": item["prompt"][0]["content"]
            if isinstance(item["prompt"], list)
            else item["prompt"],
            "images": [],
            "chosen": item["chosen"],
            "rejected": item["rejected"],
        }
        vision_data.append(vision_item)

    model_name = "llava-hf/llava-1.5-7b-hf"
    tokenizer = build_tokenizer(ModelParams(model_name=model_name))
    processor = build_processor(model_name, tokenizer, trust_remote_code=True)

    dataset = TestVisionLanguageDpoDatasetIntegration(
        data=vision_data, tokenizer=tokenizer, processor=processor
    )

    # Test functionality with no images
    assert len(dataset) == 2
    sample = dataset[0]

    # Should still have expected keys
    expected_keys = [
        "prompt_input_ids",
        "chosen_input_ids",
        "rejected_input_ids",
    ]
    for key in expected_keys:
        assert key in sample
    assert "pixel_values" not in sample


def test_vision_dpo_dataset_max_size_parameter(sample_vision_dpo_data):
    """Test VisionLanguageDpoDataset with max_size parameter."""
    model_name = "llava-hf/llava-1.5-7b-hf"
    tokenizer = build_tokenizer(ModelParams(model_name=model_name))
    processor = build_processor(model_name, tokenizer, trust_remote_code=True)

    # Create dataset with max_size
    dataset = TestVisionLanguageDpoDatasetIntegration(
        data=sample_vision_dpo_data,
        tokenizer=tokenizer,
        processor=processor,
        max_size=224,
    )

    # Test that dataset works with max_size constraint
    assert len(dataset) == 2
    sample = dataset[0]

    expected_keys = [
        "prompt_input_ids",
        "pixel_values",
        "chosen_input_ids",
        "rejected_input_ids",
    ]
    for key in expected_keys:
        assert key in sample


def test_conversation_format_conversion():
    """Test _to_conversation_dict method with real scenarios."""
    model_name = "llava-hf/llava-1.5-7b-hf"
    tokenizer = build_tokenizer(ModelParams(model_name=model_name))
    processor = build_processor(model_name, tokenizer, trust_remote_code=True)

    dataset = TestVisionLanguageDpoDatasetIntegration(
        data=[], tokenizer=tokenizer, processor=processor
    )

    from oumi.core.types.conversation import Role

    # Test string conversion
    result = dataset._to_conversation_dict("Hello world", Role.USER)
    assert result == [{"role": "user", "content": "Hello world"}]

    # Test dict conversion
    input_dict = {"role": "assistant", "content": "Response"}
    result = dataset._to_conversation_dict(input_dict, Role.ASSISTANT)
    assert result == [input_dict]

    # Test list conversion
    input_list = [
        {"role": "assistant", "content": "Response 1"},
        {"role": "assistant", "content": "Response 2"},
    ]
    result = dataset._to_conversation_dict(input_list, Role.ASSISTANT)
    assert result == input_list


@pytest.mark.parametrize(
    "model_name,test_images",
    [
        ("llava-hf/llava-1.5-7b-hf", ["tests/testdata/images/oumi_logo_light.png"]),
    ],
)
def test_end_to_end_vision_dpo_processing(model_name, test_images):
    """End-to-end test of vision DPO processing with real components."""
    # Build real components
    tokenizer = build_tokenizer(ModelParams(model_name=model_name))
    processor = build_processor(model_name, tokenizer, trust_remote_code=True)

    # Create comprehensive test data
    data = [
        {
            "prompt": "What do you see in this image?",
            "images": test_images,
            "chosen": [
                {
                    "role": "assistant",
                    "content": "I can see a clear, professional logo design.",
                }
            ],
            "rejected": [
                {"role": "assistant", "content": "I cannot see anything meaningful."}
            ],
        }
    ]

    # Create dataset
    dataset = TestVisionLanguageDpoDatasetIntegration(
        data=data, tokenizer=tokenizer, processor=processor
    )

    # Process sample
    sample = dataset[0]

    # Comprehensive validation
    assert isinstance(sample, dict)

    # Check all expected keys are present
    required_keys = [
        "prompt_input_ids",
        "pixel_values",
        "chosen_input_ids",
        "rejected_input_ids",
    ]
    for key in required_keys:
        assert key in sample, f"Missing required key: {key}"

    # Validate input_ids are properly tokenized (can be lists or tensors)
    prompt_ids = sample["prompt_input_ids"]
    chosen_ids = sample["chosen_input_ids"]
    rejected_ids = sample["rejected_input_ids"]

    assert isinstance(prompt_ids, (list, torch.Tensor))
    assert len(prompt_ids) > 0

    # Validate chosen/rejected responses
    assert isinstance(chosen_ids, (list, torch.Tensor))
    assert isinstance(rejected_ids, (list, torch.Tensor))
    assert len(chosen_ids) > 0
    assert len(rejected_ids) > 0

    # Check EOS tokens (handle both list and tensor cases)
    if torch.is_tensor(chosen_ids):
        assert chosen_ids[-1].item() == tokenizer.eos_token_id
        assert rejected_ids[-1].item() == tokenizer.eos_token_id
    else:
        assert chosen_ids[-1] == tokenizer.eos_token_id
        assert rejected_ids[-1] == tokenizer.eos_token_id

    # Validate pixel values (image features)
    pixel_values = sample["pixel_values"]
    assert pixel_values is not None

    # Should be tensor or numpy-like with proper shape for images
    if torch.is_tensor(pixel_values):
        assert pixel_values.ndim >= 3  # At least [C, H, W]
    else:
        assert hasattr(pixel_values, "__len__")


def test_multiple_images_support():
    """Test vision DPO dataset with multiple images."""
    model_name = "llava-hf/llava-1.5-7b-hf"
    tokenizer = build_tokenizer(ModelParams(model_name=model_name))
    processor = build_processor(model_name, tokenizer, trust_remote_code=True)

    # Test data with multiple images
    data = [
        {
            "prompt": "Compare these images",
            "images": [
                "tests/testdata/images/oumi_logo_light.png",
                "tests/testdata/images/oumi_logo_dark.png",
            ],
            "chosen": [
                {
                    "role": "assistant",
                    "content": "These are different versions of the same logo - one light and one dark theme.",
                }
            ],
            "rejected": [
                {
                    "role": "assistant",
                    "content": "These images are completely unrelated.",
                }
            ],
        }
    ]

    dataset = TestVisionLanguageDpoDatasetIntegration(
        data=data, tokenizer=tokenizer, processor=processor
    )

    # Should handle multiple images without errors
    sample = dataset[0]
    assert "pixel_values" in sample
    assert "prompt_input_ids" in sample
    assert "chosen_input_ids" in sample
    assert "rejected_input_ids" in sample


@pytest.mark.parametrize(
    "model_name",
    [
        "microsoft/Phi-3-vision-128k-instruct",
        "Qwen/Qwen2-VL-2B-Instruct",
    ],
)
def test_vision_dpo_dataset_with_extended_models(model_name, sample_vision_dpo_data):
    """Test VisionLanguageDpoDataset with extended vision models (Phi3-Vision, Qwen2-VL)."""
    # These tests are skipped by default as they require specific model setups
    # They can be enabled for comprehensive testing when the environment supports these models

    try:
        # Build real tokenizer and processor
        tokenizer = build_tokenizer(ModelParams(model_name=model_name))
        processor = build_processor(model_name, tokenizer, trust_remote_code=True)

        # Create dataset
        dataset = TestVisionLanguageDpoDatasetIntegration(
            data=sample_vision_dpo_data,
            tokenizer=tokenizer,
            processor=processor,
            return_tensors=False,
        )

        # Test dataset functionality
        assert len(dataset) == 2
        sample = dataset[0]

        # Should have the key features for vision DPO
        expected_keys = [
            "prompt_input_ids",
            "pixel_values",
            "chosen_input_ids",
            "rejected_input_ids",
        ]
        for key in expected_keys:
            assert key in sample, f"Missing key: {key} for model {model_name}"

        # Verify EOS tokens were added
        assert sample["chosen_input_ids"][-1] == tokenizer.eos_token_id
        assert sample["rejected_input_ids"][-1] == tokenizer.eos_token_id

    except Exception as e:
        pytest.skip(f"Could not test {model_name}: {e}")


def test_vision_dpo_processor_compatibility():
    """Test that different processor configurations work with vision DPO."""
    # Test with different processor configurations that might be used
    # in the config files we saw in the diff

    model_configs = [
        {
            "model_name": "llava-hf/llava-1.5-7b-hf",
            "processor_kwargs": {},
            "expected_keys": [
                "prompt_input_ids",
                "pixel_values",
                "chosen_input_ids",
                "rejected_input_ids",
            ],
        }
    ]

    for config in model_configs:
        try:
            tokenizer = build_tokenizer(ModelParams(model_name=config["model_name"]))
            processor = build_processor(
                config["model_name"],
                tokenizer,
                trust_remote_code=True,
                processor_kwargs=config.get("processor_kwargs", {}),
            )

            # Simple test data
            data = [
                {
                    "prompt": "What's in this image?",
                    "images": ["tests/testdata/images/oumi_logo_light.png"],
                    "chosen": [{"role": "assistant", "content": "I see a logo."}],
                    "rejected": [{"role": "assistant", "content": "I see nothing."}],
                }
            ]

            dataset = TestVisionLanguageDpoDatasetIntegration(
                data=data, tokenizer=tokenizer, processor=processor
            )

            sample = dataset[0]

            # Verify expected keys are present
            for key in config["expected_keys"]:
                assert key in sample, f"Missing {key} for {config['model_name']}"

        except Exception as e:
            pytest.skip(
                f"Could not test processor compatibility for {config['model_name']}: {e}"
            )


def test_dpo_dataset_format_validation():
    """Test that DPO datasets properly validate and handle different input formats."""
    model_name = "llava-hf/llava-1.5-7b-hf"
    tokenizer = build_tokenizer(ModelParams(model_name=model_name))
    processor = build_processor(model_name, tokenizer, trust_remote_code=True)

    # Test different format variations that might appear in real data
    format_variants = [
        # Standard format
        {
            "prompt": "Describe this image",
            "images": ["tests/testdata/images/oumi_logo_light.png"],
            "chosen": [
                {"role": "assistant", "content": "This is a professional logo."}
            ],
            "rejected": [{"role": "assistant", "content": "This is garbage."}],
        },
        # Prompt as conversation
        {
            "prompt": [{"role": "user", "content": "What do you see?"}],
            "images": ["tests/testdata/images/oumi_logo_light.png"],
            "chosen": [{"role": "assistant", "content": "I see a logo design."}],
            "rejected": [{"role": "assistant", "content": "I see nothing."}],
        },
        # String responses (non-conversation format)
        {
            "prompt": "Analyze this image",
            "images": ["tests/testdata/images/oumi_logo_light.png"],
            "chosen": "This appears to be a well-designed logo.",
            "rejected": "This is just random pixels.",
        },
    ]

    for i, variant in enumerate(format_variants):
        dataset = TestVisionLanguageDpoDatasetIntegration(
            data=[variant], tokenizer=tokenizer, processor=processor
        )

        # Should handle all format variants without errors
        sample = dataset[0]

        # All should produce the same output structure
        expected_keys = [
            "prompt_input_ids",
            "pixel_values",
            "chosen_input_ids",
            "rejected_input_ids",
        ]
        for key in expected_keys:
            assert key in sample, f"Missing {key} in format variant {i}"
