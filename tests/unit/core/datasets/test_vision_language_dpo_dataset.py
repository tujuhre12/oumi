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

from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest
import transformers
from PIL import Image
from typing_extensions import override

from oumi.core.configs.internal.internal_model_config import (
    InternalFeatureFirstDimAction,
    InternalFeatureSpec,
    InternalModelConfig,
)
from oumi.core.datasets.vision_language_dpo_dataset import VisionLanguageDpoDataset
from oumi.core.tokenizers.base_tokenizer import BaseTokenizer
from oumi.core.types.conversation import Role


@pytest.fixture
def mock_tokenizer():
    """Mock tokenizer for testing."""
    tokenizer = MagicMock(spec=BaseTokenizer)
    tokenizer.eos_token_id = 2
    tokenizer.apply_chat_template = Mock(
        side_effect=lambda messages, tokenize=False: _mock_apply_chat_template(
            messages, tokenize
        )
    )
    tokenizer.side_effect = lambda text, add_special_tokens=False: _mock_tokenize(text)
    return tokenizer


@pytest.fixture
def mock_processor():
    """Mock processor for testing."""
    processor = Mock()
    processor.processor_name = "llava-hf/llava-1.5-7b-hf"
    processor.side_effect = (
        lambda images, text, add_special_tokens=False: _mock_processor_call(
            images, text
        )
    )
    return processor


@pytest.fixture
def mock_internal_model_config():
    """Mock internal model config."""
    config = Mock(spec=InternalModelConfig)
    config.model_input_features = {
        "input_ids": InternalFeatureSpec(
            name="input_ids",
            first_dim_action=InternalFeatureFirstDimAction.DROP_IF_DUMMY,
        ),
        "pixel_values": InternalFeatureSpec(
            name="pixel_values",
            first_dim_action=InternalFeatureFirstDimAction.DROP_IF_DUMMY,
        ),
        "pixel_attention_mask": InternalFeatureSpec(
            name="pixel_attention_mask",
            first_dim_action=InternalFeatureFirstDimAction.DROP_ALWAYS,
        ),
        "image_sizes": InternalFeatureSpec(
            name="image_sizes",
            first_dim_action=InternalFeatureFirstDimAction.DROP_ALWAYS,
        ),
    }
    return config


@pytest.fixture
def sample_vision_dpo_data():
    """Sample vision DPO data for testing."""
    return [
        {
            "prompt": "What do you see in this image?",
            "images": ["path/to/image1.jpg"],
            "chosen": [
                {
                    "role": "assistant",
                    "content": "I see a beautiful landscape with mountains.",
                }
            ],
            "rejected": [
                {"role": "assistant", "content": "I see nothing interesting."}
            ],
        },
        {
            "prompt": [{"role": "user", "content": "Describe this image"}],
            "images": ["path/to/image2.jpg", "path/to/image3.jpg"],
            "chosen": "This image shows a bustling city street.",
            "rejected": "This image is unclear.",
        },
    ]


def _mock_apply_chat_template(messages, tokenize=False):
    """Mock chat template application."""
    result = ""
    for msg in messages:
        if isinstance(msg["content"], str):
            result += f"<{msg['role']}>{msg['content']}</{msg['role']}>"
        else:
            # Handle list of content items
            text_parts = [
                item["content"] for item in msg["content"] if item.get("type") == "text"
            ]
            result += f"<{msg['role']}>{''.join(text_parts)}</{msg['role']}>"
    return result


def _mock_tokenize(text, add_special_tokens=False):
    """Mock tokenization."""
    return {"input_ids": list(range(len(text.split())))}


def _mock_processor_call(images, text, add_special_tokens=False):
    """Mock processor call."""
    return transformers.BatchEncoding(
        {
            "input_ids": [[1, 2, 3, 4] for _ in text],
            "pixel_values": [np.ones((3, 224, 224)) for _ in (images or [])],
            "pixel_attention_mask": [[1, 1, 1, 1] for _ in (images or [])],
            "image_sizes": [[224, 224] for _ in (images or [])],
        }
    )


def _create_test_image():
    """Create a test PIL image."""
    return Image.new("RGB", (100, 100), color="red")


class TestVisionLanguageDpoDataset(VisionLanguageDpoDataset):
    """Test implementation of VisionLanguageDpoDataset."""

    default_dataset = "test_vision_dpo"

    def __init__(self, data=None, **kwargs):
        self._test_data = data or []
        super().__init__(dataset_name="test_vision_dpo", **kwargs)

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


def test_vision_dpo_dataset_initialization(mock_tokenizer, mock_processor):
    """Test VisionLanguageDpoDataset initialization."""
    dataset = TestVisionLanguageDpoDataset(
        tokenizer=mock_tokenizer, processor=mock_processor, return_tensors=False
    )

    assert dataset._tokenizer == mock_tokenizer
    assert dataset._processor == mock_processor
    assert dataset._return_tensors is False


def test_vision_dpo_dataset_initialization_with_processor_name(mock_tokenizer):
    """Test initialization with processor_name."""
    with (
        patch(
            "oumi.core.datasets.vision_language_dpo_dataset.build_processor"
        ) as mock_build_processor,
        patch(
            "oumi.core.datasets.vision_language_dpo_dataset.find_internal_model_config_using_model_name"
        ) as mock_find_config,
        patch(
            "oumi.core.datasets.vision_language_dpo_dataset.get_default_vlm_model_config"
        ) as mock_get_default,
    ):
        mock_processor = Mock()
        mock_processor.processor_name = "test-processor"
        mock_build_processor.return_value = mock_processor
        mock_find_config.return_value = None
        mock_get_default.return_value = Mock()

        dataset = TestVisionLanguageDpoDataset(
            tokenizer=mock_tokenizer,
            processor_name="test-processor",
            return_tensors=False,
        )

        mock_build_processor.assert_called_once()
        assert dataset._processor == mock_processor


def test_vision_dpo_dataset_no_processor_error(mock_tokenizer):
    """Test error when no processor is provided."""
    with pytest.raises(ValueError, match="Processor is not set"):
        TestVisionLanguageDpoDataset(tokenizer=mock_tokenizer, return_tensors=False)


def test_to_conversation_dict_string(mock_tokenizer, mock_processor):
    """Test _to_conversation_dict with string input."""
    dataset = TestVisionLanguageDpoDataset(
        tokenizer=mock_tokenizer, processor=mock_processor
    )

    result = dataset._to_conversation_dict("Hello world", Role.USER)
    expected = [{"role": "user", "content": "Hello world"}]
    assert result == expected


def test_to_conversation_dict_dict(mock_tokenizer, mock_processor):
    """Test _to_conversation_dict with dict input."""
    dataset = TestVisionLanguageDpoDataset(
        tokenizer=mock_tokenizer, processor=mock_processor
    )

    input_dict = {"role": "assistant", "content": "Response"}
    result = dataset._to_conversation_dict(input_dict, Role.ASSISTANT)
    expected = [input_dict]
    assert result == expected


def test_to_conversation_dict_list(mock_tokenizer, mock_processor):
    """Test _to_conversation_dict with list input."""
    dataset = TestVisionLanguageDpoDataset(
        tokenizer=mock_tokenizer, processor=mock_processor
    )

    input_list = [
        {"role": "assistant", "content": "Response 1"},
        {"role": "assistant", "content": "Response 2"},
    ]
    result = dataset._to_conversation_dict(input_list, Role.ASSISTANT)
    assert result == input_list


def test_load_image_from_path(mock_tokenizer, mock_processor):
    """Test _load_image with image path."""
    with (
        patch(
            "oumi.core.datasets.vision_language_dpo_dataset.find_internal_model_config_using_model_name"
        ) as mock_find_config,
        patch(
            "oumi.core.datasets.vision_language_dpo_dataset.get_default_vlm_model_config"
        ) as mock_get_default,
        patch(
            "oumi.core.datasets.vision_language_dpo_dataset.load_pil_image_from_content_item"
        ) as mock_load,
    ):
        mock_find_config.return_value = None
        mock_get_default.return_value = Mock()
        mock_get_default.return_value.model_input_features = {}

        dataset = TestVisionLanguageDpoDataset(
            tokenizer=mock_tokenizer, processor=mock_processor
        )

        mock_image = _create_test_image()
        mock_load.return_value = mock_image

        result = dataset._load_image("path/to/image.jpg")

        mock_load.assert_called_once()
        assert result == mock_image


def test_resize_image_with_max_size(mock_tokenizer, mock_processor):
    """Test _resize_image with max_size specified."""
    dataset = TestVisionLanguageDpoDataset(
        tokenizer=mock_tokenizer, processor=mock_processor, max_size=50
    )

    # Create a large image
    large_image = Image.new("RGB", (200, 200), color="blue")

    with patch.object(large_image, "thumbnail") as mock_thumbnail:
        result = dataset._resize_image(large_image)

        mock_thumbnail.assert_called_once_with((50, 50))
        assert result == large_image


def test_resize_image_no_max_size(mock_tokenizer, mock_processor):
    """Test _resize_image without max_size."""
    dataset = TestVisionLanguageDpoDataset(
        tokenizer=mock_tokenizer, processor=mock_processor
    )

    image = _create_test_image()
    result = dataset._resize_image(image)

    # Should return the image unchanged
    assert result == image


def test_drop_first_dim_if_needed(
    mock_tokenizer, mock_processor, mock_internal_model_config
):
    """Test _drop_first_dim_if_needed method."""
    dataset = TestVisionLanguageDpoDataset(
        tokenizer=mock_tokenizer, processor=mock_processor
    )
    dataset._internal_model_config = mock_internal_model_config

    # Test DROP_ALWAYS
    result = dataset._drop_first_dim_if_needed("pixel_attention_mask", [[1, 2, 3]])
    assert result == [1, 2, 3]

    # Test DROP_IF_DUMMY with single item
    result = dataset._drop_first_dim_if_needed("input_ids", [[1, 2, 3]])
    assert result == [1, 2, 3]

    # Test DROP_IF_DUMMY with multiple items
    result = dataset._drop_first_dim_if_needed("input_ids", [[1, 2], [3, 4]])
    assert result == [[1, 2], [3, 4]]

    # Test unknown feature (defaults to DROP_IF_DUMMY)
    result = dataset._drop_first_dim_if_needed("unknown_feature", [[5, 6]])
    assert result == [5, 6]


def test_transform_preference_string_formats(
    mock_tokenizer, mock_processor, sample_vision_dpo_data
):
    """Test transform_preference with different input formats."""
    with (
        patch(
            "oumi.core.datasets.vision_language_dpo_dataset.find_internal_model_config_using_model_name"
        ) as mock_find_config,
        patch(
            "oumi.core.datasets.vision_language_dpo_dataset.get_default_vlm_model_config"
        ) as mock_get_default,
    ):
        mock_find_config.return_value = None
        mock_get_default.return_value = Mock()
        mock_get_default.return_value.model_input_features = {}

        dataset = TestVisionLanguageDpoDataset(
            data=sample_vision_dpo_data,
            tokenizer=mock_tokenizer,
            processor=mock_processor,
        )

        with (
            patch.object(dataset, "_load_image") as mock_load_image,
            patch.object(dataset, "_resize_image") as mock_resize_image,
            patch.object(dataset, "_process_sample") as mock_process_sample,
        ):
            mock_image = _create_test_image()
            mock_load_image.return_value = mock_image
            mock_resize_image.return_value = mock_image
            mock_process_sample.return_value = {"processed": "data"}

            # Test with string prompt
            sample = sample_vision_dpo_data[0]
            result = dataset.transform_preference(sample)

            mock_process_sample.assert_called_once()
            assert result == {"processed": "data"}


def test_process_sample(mock_tokenizer, mock_processor):
    """Test _process_sample method."""
    with (
        patch(
            "oumi.core.datasets.vision_language_dpo_dataset.find_internal_model_config_using_model_name"
        ) as mock_find_config,
        patch(
            "oumi.core.datasets.vision_language_dpo_dataset.get_default_vlm_model_config"
        ) as mock_get_default,
    ):
        mock_config = Mock()
        mock_config.model_input_features = {
            "input_ids": InternalFeatureSpec(
                name="input_ids",
                first_dim_action=InternalFeatureFirstDimAction.DROP_IF_DUMMY,
            ),
            "pixel_values": InternalFeatureSpec(
                name="pixel_values",
                first_dim_action=InternalFeatureFirstDimAction.DROP_IF_DUMMY,
            ),
        }
        mock_find_config.return_value = None
        mock_get_default.return_value = mock_config

        dataset = TestVisionLanguageDpoDataset(
            tokenizer=mock_tokenizer, processor=mock_processor
        )

        features = {
            "prompt": [{"role": "user", "content": "Test prompt"}],
            "chosen": [{"role": "assistant", "content": "Good response"}],
            "rejected": [{"role": "assistant", "content": "Bad response"}],
            "images": [_create_test_image()],
        }

        result = dataset._process_sample(features)

        # Verify expected keys are present
        expected_keys = [
            "prompt_input_ids",
            "pixel_values",
            "chosen_input_ids",
            "rejected_input_ids",
        ]
        for key in expected_keys:
            assert key in result

        # Verify eos_token_id was added to chosen/rejected
        assert result["chosen_input_ids"][-1] == mock_tokenizer.eos_token_id
        assert result["rejected_input_ids"][-1] == mock_tokenizer.eos_token_id


def test_process_sample_no_tokenizer_or_processor():
    """Test _process_sample raises error when tokenizer or processor missing."""
    with (
        patch(
            "oumi.core.datasets.vision_language_dpo_dataset.find_internal_model_config_using_model_name"
        ) as mock_find_config,
        patch(
            "oumi.core.datasets.vision_language_dpo_dataset.get_default_vlm_model_config"
        ) as mock_get_default,
    ):
        mock_find_config.return_value = None
        mock_get_default.return_value = Mock()
        mock_get_default.return_value.model_input_features = {}

        # Create a dataset with dummy processor to pass initialization, then set to None
        dummy_processor = Mock()
        dataset = TestVisionLanguageDpoDataset(
            tokenizer=Mock(), processor=dummy_processor
        )
        # Now simulate missing tokenizer/processor
        dataset._tokenizer = None
        dataset._processor = None

        features = {
            "prompt": [{"role": "user", "content": "Test"}],
            "chosen": [{"role": "assistant", "content": "Good"}],
            "rejected": [{"role": "assistant", "content": "Bad"}],
            "images": [],
        }

        with pytest.raises(ValueError, match="Tokenizer and processor are required"):
            dataset._process_sample(features)


def test_transform_preference_with_image_insertion(mock_tokenizer, mock_processor):
    """Test that images are properly inserted into prompt."""
    with (
        patch(
            "oumi.core.datasets.vision_language_dpo_dataset.find_internal_model_config_using_model_name"
        ) as mock_find_config,
        patch(
            "oumi.core.datasets.vision_language_dpo_dataset.get_default_vlm_model_config"
        ) as mock_get_default,
    ):
        mock_find_config.return_value = None
        mock_get_default.return_value = Mock()
        mock_get_default.return_value.model_input_features = {}

        dataset = TestVisionLanguageDpoDataset(
            tokenizer=mock_tokenizer, processor=mock_processor
        )

        sample = {
            "prompt": "What's in this image?",
            "images": ["path/to/image.jpg"],
            "chosen": [{"role": "assistant", "content": "A beautiful scene"}],
            "rejected": [{"role": "assistant", "content": "Nothing special"}],
        }

        with (
            patch.object(dataset, "_load_image") as mock_load_image,
            patch.object(dataset, "_resize_image") as mock_resize_image,
            patch.object(dataset, "_process_sample") as mock_process_sample,
        ):
            mock_image = _create_test_image()
            mock_load_image.return_value = mock_image
            mock_resize_image.return_value = mock_image
            mock_process_sample.return_value = {"test": "result"}

            result = dataset.transform_preference(sample)

            # Verify _process_sample was called with images added to prompt
            call_args = mock_process_sample.call_args[0][0]
            assert (
                len(call_args["prompt"]) == 2
            )  # Original user message + image message
            assert call_args["prompt"][1]["role"] == "user"
            assert call_args["prompt"][1]["content"][0]["type"] == "image_bytes"


def test_transform_preference_no_images(mock_tokenizer, mock_processor):
    """Test transform_preference with no images."""
    with (
        patch(
            "oumi.core.datasets.vision_language_dpo_dataset.find_internal_model_config_using_model_name"
        ) as mock_find_config,
        patch(
            "oumi.core.datasets.vision_language_dpo_dataset.get_default_vlm_model_config"
        ) as mock_get_default,
    ):
        mock_find_config.return_value = None
        mock_get_default.return_value = Mock()
        mock_get_default.return_value.model_input_features = {}

        dataset = TestVisionLanguageDpoDataset(
            tokenizer=mock_tokenizer, processor=mock_processor
        )

        sample = {
            "prompt": "Answer this question",
            "images": [],
            "chosen": [{"role": "assistant", "content": "Good answer"}],
            "rejected": [{"role": "assistant", "content": "Bad answer"}],
        }

        with patch.object(dataset, "_process_sample") as mock_process_sample:
            mock_process_sample.return_value = {"test": "result"}

            result = dataset.transform_preference(sample)

            # Verify _process_sample was called with only the original prompt
            call_args = mock_process_sample.call_args[0][0]
            assert len(call_args["prompt"]) == 1  # Only original user message
            assert call_args["images"] == []


def test_dataset_integration(mock_tokenizer, mock_processor, sample_vision_dpo_data):
    """Test full dataset integration."""
    with (
        patch(
            "oumi.core.datasets.vision_language_dpo_dataset.find_internal_model_config_using_model_name"
        ) as mock_find_config,
        patch(
            "oumi.core.datasets.vision_language_dpo_dataset.get_default_vlm_model_config"
        ) as mock_get_default,
    ):
        mock_find_config.return_value = None
        mock_get_default.return_value = Mock()
        mock_get_default.return_value.model_input_features = {}

        dataset = TestVisionLanguageDpoDataset(
            data=sample_vision_dpo_data,
            tokenizer=mock_tokenizer,
            processor=mock_processor,
        )

        assert len(dataset) == 2

        with (
            patch.object(dataset, "_load_image") as mock_load_image,
            patch.object(dataset, "_resize_image") as mock_resize_image,
        ):
            mock_image = _create_test_image()
            mock_load_image.return_value = mock_image
            mock_resize_image.return_value = mock_image

            # Test accessing first item
            first_item = dataset[0]
            assert isinstance(first_item, dict)

            # Test accessing second item
            second_item = dataset[1]
            assert isinstance(second_item, dict)
