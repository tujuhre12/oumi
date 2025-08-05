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

from unittest.mock import MagicMock, Mock

import pytest
from typing_extensions import override

from oumi.core.datasets.base_dpo_dataset import BaseDpoDataset
from oumi.core.tokenizers.base_tokenizer import BaseTokenizer


@pytest.fixture
def mock_tokenizer():
    """Mock tokenizer for testing."""
    tokenizer = MagicMock(spec=BaseTokenizer)
    tokenizer.eos_token_id = 2
    tokenizer.apply_chat_template = Mock(
        side_effect=lambda messages, tokenize=False: "User: "
        + messages[0]["content"]
        + "\nAssistant: "
        + (messages[1]["content"] if len(messages) > 1 else "")
    )
    tokenizer.side_effect = lambda text, add_special_tokens=False: {
        "input_ids": [1, 2, 3] if "User:" in text else [4, 5, 6]
    }
    return tokenizer


@pytest.fixture
def sample_dpo_data():
    """Sample DPO data for testing."""
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
            "prompt": [{"role": "user", "content": "Explain photosynthesis."}],
            "chosen": [
                {
                    "role": "assistant",
                    "content": "Photosynthesis is the process by which plants convert "
                    "light energy into chemical energy.",
                }
            ],
            "rejected": [
                {
                    "role": "assistant",
                    "content": "Photosynthesis is when plants eat sunlight.",
                }
            ],
        },
    ]


class TestBaseDpoDataset(BaseDpoDataset):
    """Test implementation of BaseDpoDataset."""

    default_dataset = "test_dpo"

    def __init__(self, data=None, **kwargs):
        self._test_data = data or []
        super().__init__(dataset_name="test_dpo", **kwargs)

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


def test_base_dpo_dataset_initialization(mock_tokenizer):
    """Test BaseDpoDataset initialization."""
    dataset = TestBaseDpoDataset(tokenizer=mock_tokenizer, return_tensors=False)

    assert dataset._tokenizer == mock_tokenizer
    assert dataset._return_tensors is False


def test_transform_preference_basic(mock_tokenizer, sample_dpo_data):
    """Test basic transform_preference functionality."""
    dataset = TestBaseDpoDataset(
        data=sample_dpo_data, tokenizer=mock_tokenizer, return_tensors=False
    )

    sample = sample_dpo_data[0]
    result = dataset.transform_preference(sample)

    assert "prompt" in result
    assert "chosen" in result
    assert "rejected" in result

    assert result["prompt"] == sample["prompt"]
    assert result["chosen"] == sample["chosen"]
    assert result["rejected"] == sample["rejected"]


def test_tokenize_row_functionality(mock_tokenizer):
    """Test tokenize_row method."""

    # Setup mock tokenizer behavior
    def mock_apply_chat_template(messages, tokenize=False):
        if len(messages) == 1:  # Prompt only
            return "User: " + messages[0]["content"]
        else:  # Prompt + response
            return (
                "User: "
                + messages[0]["content"]
                + "\nAssistant: "
                + messages[1]["content"]
            )

    def mock_tokenize(text, add_special_tokens=False):
        if "User: What is" in text and "Assistant:" not in text:
            return {"input_ids": [1, 2, 3]}  # Prompt only
        elif "Paris" in text:
            return {"input_ids": [4, 5]}  # Chosen response
        elif "London" in text:
            return {"input_ids": [6, 7]}  # Rejected response
        return {"input_ids": []}

    mock_tokenizer.apply_chat_template = Mock(side_effect=mock_apply_chat_template)
    mock_tokenizer.side_effect = mock_tokenize

    dataset = TestBaseDpoDataset(tokenizer=mock_tokenizer)

    features = {
        "prompt": [{"role": "user", "content": "What is the capital of France?"}],
        "chosen": [{"role": "assistant", "content": "Paris"}],
        "rejected": [{"role": "assistant", "content": "London"}],
    }

    result = dataset.tokenize_row(features)

    assert "prompt_input_ids" in result
    assert "chosen_input_ids" in result
    assert "rejected_input_ids" in result

    assert result["prompt_input_ids"] == [1, 2, 3]
    assert result["chosen_input_ids"] == [4, 5, 2]  # includes eos_token_id
    assert result["rejected_input_ids"] == [6, 7, 2]  # includes eos_token_id


def test_tokenize_row_no_tokenizer():
    """Test tokenize_row raises error when no tokenizer provided."""
    dataset = TestBaseDpoDataset(tokenizer=None)

    features = {
        "prompt": [{"role": "user", "content": "Test"}],
        "chosen": [{"role": "assistant", "content": "Response"}],
        "rejected": [{"role": "assistant", "content": "Bad response"}],
    }

    with pytest.raises(ValueError, match="Tokenizer is required to process a sample"):
        dataset.tokenize_row(features)


def test_transform_calls_transform_preference(mock_tokenizer, sample_dpo_data):
    """Test that transform method calls transform_preference."""
    dataset = TestBaseDpoDataset(data=sample_dpo_data, tokenizer=mock_tokenizer)

    sample = sample_dpo_data[0]

    # Mock transform_preference to verify it's called
    original_transform_preference = dataset.transform_preference
    dataset.transform_preference = Mock(return_value={"test": "result"})

    result = dataset.transform(sample)

    dataset.transform_preference.assert_called_once_with(sample)
    assert result == {"test": "result"}

    # Restore original method
    dataset.transform_preference = original_transform_preference


def test_dataset_iteration(mock_tokenizer, sample_dpo_data):
    """Test dataset iteration functionality."""
    dataset = TestBaseDpoDataset(data=sample_dpo_data, tokenizer=mock_tokenizer)

    assert len(dataset) == 2

    # Test accessing items by index
    first_item = dataset[0]
    assert "prompt" in first_item
    assert "chosen" in first_item
    assert "rejected" in first_item

    second_item = dataset[1]
    assert "prompt" in second_item
    assert "chosen" in second_item
    assert "rejected" in second_item


def test_empty_dataset(mock_tokenizer):
    """Test behavior with empty dataset."""
    dataset = TestBaseDpoDataset(data=[], tokenizer=mock_tokenizer)

    assert len(dataset) == 0


def test_tokenize_row_with_complex_chat_template(mock_tokenizer):
    """Test tokenize_row with more complex chat template behavior."""

    def mock_apply_chat_template(messages, tokenize=False):
        result = ""
        for msg in messages:
            result += f"<{msg['role']}>{msg['content']}</{msg['role']}>"
        return result

    def mock_tokenize(text, add_special_tokens=False):
        return {"input_ids": list(range(len(text.split())))}

    mock_tokenizer.apply_chat_template = Mock(side_effect=mock_apply_chat_template)
    mock_tokenizer.side_effect = mock_tokenize

    dataset = TestBaseDpoDataset(tokenizer=mock_tokenizer)

    features = {
        "prompt": [{"role": "user", "content": "Hello world"}],
        "chosen": [{"role": "assistant", "content": "Hi there"}],
        "rejected": [{"role": "assistant", "content": "Go away"}],
    }

    result = dataset.tokenize_row(features)

    # Verify that apply_chat_template was called correctly for each combination
    expected_calls = [
        ([{"role": "user", "content": "Hello world"}],),
        (
            [
                {"role": "user", "content": "Hello world"},
                {"role": "assistant", "content": "Hi there"},
            ],
        ),
        (
            [
                {"role": "user", "content": "Hello world"},
                {"role": "assistant", "content": "Go away"},
            ],
        ),
    ]

    assert mock_tokenizer.apply_chat_template.call_count == 3
    assert all(
        key in result
        for key in ["prompt_input_ids", "chosen_input_ids", "rejected_input_ids"]
    )


def test_dataset_with_return_tensors_false(mock_tokenizer, sample_dpo_data):
    """Test dataset behavior with return_tensors=False."""
    dataset = TestBaseDpoDataset(
        data=sample_dpo_data, tokenizer=mock_tokenizer, return_tensors=False
    )

    sample = dataset[0]

    # Should return the basic preference format without tokenization
    assert "prompt" in sample
    assert "chosen" in sample
    assert "rejected" in sample

    # Values should be the original conversation format
    assert isinstance(sample["prompt"], list)
    assert isinstance(sample["chosen"], list)
    assert isinstance(sample["rejected"], list)
