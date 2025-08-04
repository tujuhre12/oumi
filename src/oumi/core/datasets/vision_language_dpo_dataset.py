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

from typing import Any, Optional, Union

from PIL import Image
from typing_extensions import override

from oumi.builders.processors import build_processor
from oumi.core.configs.internal.internal_model_config import (
    InternalFeatureFirstDimAction,
    InternalModelConfig,
)
from oumi.core.configs.internal.supported_models import (
    find_internal_model_config_using_model_name,
    get_default_vlm_model_config,
)
from oumi.core.datasets.base_dpo_dataset import BaseDpoDataset
from oumi.core.tokenizers.base_tokenizer import BaseTokenizer
from oumi.core.types.conversation import ContentItem, Type
from oumi.utils.conversation_utils import load_pil_image_from_content_item

_PROMPT_KEY = "prompt"
_CHOSEN_KEY = "chosen"
_REJECTED_KEY = "rejected"
_IMAGES_KEY = "images"


class VisionLanguageDpoDataset(BaseDpoDataset):
    """Dataset for vision-language DPO (Direct Preference Optimization) models.

    This class extends BaseDpoDataset to provide functionality specific to
    vision-language preference optimization tasks. It handles the processing of
    both image and text data for preference learning.

    The dataset expects data in the format:
    {
        "prompt": "What's in this image?",
        "images": ["path/to/image.jpg", ...],  # Optional image paths/URLs
        "chosen": [{"role": "assistant", "content": "I see a cat"}],
        "rejected": [{"role": "assistant", "content": "I see a dog"}]
    }

    Example:
        >>> from oumi.builders import build_processor, build_tokenizer
        >>> from oumi.core.configs import ModelParams
        >>> from oumi.core.datasets import VisionLanguageDpoDataset
        >>> class MyVisionLanguageDpoDataset(VisionLanguageDpoDataset):
        ...     def transform_preference(self, example: dict):
        ...         # Implement the abstract method
        ...         # Convert the raw example into preference conversations
        ...         pass
        >>> tokenizer = build_tokenizer(
        ...     ModelParams(model_name="llava-hf/llava-1.5-7b-hf")
        ... )
        >>> dataset = MyVisionLanguageDpoDataset( # doctest: +SKIP
        ...     tokenizer=tokenizer,
        ...     processor_name="llava-hf/llava-1.5-7b-hf",
        ...     dataset_name="my_vision_dpo_dataset",
        ...     split="train"
        ... )
        >>> sample = next(iter(dataset))  # doctest: +SKIP
        >>> print(sample.keys()) # doctest: +SKIP
    """

    def __init__(
        self,
        *,
        dataset_name: Optional[str] = None,
        dataset_path: Optional[str] = None,
        split: Optional[str] = None,
        tokenizer: Optional[BaseTokenizer] = None,
        return_tensors: bool = False,
        processor: Optional[Any] = None,
        processor_name: Optional[str] = None,
        trust_remote_code: bool = False,
        processor_kwargs: Optional[dict[str, Any]] = None,
        **kwargs,
    ) -> None:
        """Initializes a new instance of the VisionLanguageDpoDataset class.

        The dataset will return dictionaries containing formatted preference data
        ready for DPO training with chat templates applied.

        Args:
            processor: The vision-language processor for applying chat templates
                and processing images.
            tokenizer: The tokenizer for encoding text data.
            return_tensors: Whether to return tensors instead of strings.
            dataset_name: The name of the dataset.
            dataset_path: The path to the dataset.
            split: The split of the dataset.
            **kwargs: Additional keyword arguments to pass to the base class.
        """
        super().__init__(
            dataset_name=dataset_name,
            dataset_path=dataset_path,
            split=split,
            tokenizer=tokenizer,
            return_tensors=return_tensors,
            **kwargs,
        )
        self._processor = build_processor(
            processor_name,
            tokenizer,
            trust_remote_code=True,
            processor_kwargs=processor_kwargs,
        )
        self._internal_model_config: InternalModelConfig = (
            find_internal_model_config_using_model_name(
                self._processor.processor_name, trust_remote_code=trust_remote_code
            )
            or get_default_vlm_model_config()
        )

    @override
    def transform_preference(self, sample: dict) -> dict:
        """Transform a DPO sample to the format expected by DPO trainer.

        Args:
            sample: Raw preference data sample

        Returns:
            Dict with prompt, chosen, and rejected conversations or features

        Transforms a raw DPO example into three Oumi Conversation objects.

        Args:
            example (dict): A dictionary representing a single DPO preference example.
                Expected format:
                {
                    "prompt": "What's in this image?",
                    "images": ["path/to/image.jpg", ...],  # Optional
                    "chosen": [{"role": "assistant", "content": "preferred response"}],
                    "rejected": [{"role": "assistant", "content": "rejected response"}]
                }

        Returns:
            Dict with prompt, chosen, and rejected conversations or features
        """
        prompt = sample[_PROMPT_KEY]
        chosen_chat = sample[_CHOSEN_KEY]
        rejected_chat = sample[_REJECTED_KEY]
        images = sample[_IMAGES_KEY] or []

        if images is not None:
            images = [self._resize_image(self._load_image(image)) for image in images]

        prompt_chat = [{"role": "user", "content": prompt}]
        for image in images:
            prompt_chat.append(
                {"role": "user", "content": [{"type": "image_bytes", "content": image}]}
            )

        return self.process_row(
            {
                _PROMPT_KEY: prompt_chat,
                _CHOSEN_KEY: chosen_chat,
                _REJECTED_KEY: rejected_chat,
                _IMAGES_KEY: images,
            }
        )

    def _load_image(self, image_path: Union[str, ContentItem]) -> Image.Image:
        """Load images from the given paths."""
        if isinstance(image_path, str):
            content_type = (
                Type.IMAGE_URL if image_path.startswith("http") else Type.IMAGE_PATH
            )
            image = ContentItem(type=content_type, content=image_path)
        else:
            image = image_path

        return load_pil_image_from_content_item(image)

    def _resize_image(self, image: Image.Image) -> Image.Image:
        if self._processor is None:
            return image

        # If the processor has an image processor, resize the image to the
        # longest edge of the image processor.
        if hasattr(self._processor, "image_processor") and hasattr(
            self._processor.image_processor, "size"
        ):
            max_size = self._processor.image_processor.size["longest_edge"]

            image.thumbnail((max_size, max_size))
        return image

    @staticmethod
    def tokenize_row(
        features,
        processing_class,
    ):
        """Tokenize a row of the dataset.

        Example:
        ```python
        >>> from transformers import GPT2Tokenizer
        >>> tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        >>> features = {"prompt": "The sky is", "chosen": " blue", "rejected": " green"}
        >>> DPOTrainer.tokenize_row(
        ...     features, tokenizer, max_prompt_length=3, max_completion_length=3, add_special_tokens=False
        ... )
        {'prompt_input_ids': [464, 6766, 318], 'chosen_input_ids': [4171, 50256], 'rejected_input_ids': [4077, 50256]}
        ```
        """
        tokenizer = processing_class  # the processing class is a tokenizer
        prompt_input_ids = tokenizer(features["prompt"], add_special_tokens=False)[
            "input_ids"
        ]
        chosen_input_ids = tokenizer(features["chosen"], add_special_tokens=False)[
            "input_ids"
        ]
        rejected_input_ids = tokenizer(features["rejected"], add_special_tokens=False)[
            "input_ids"
        ]

        chosen_input_ids = chosen_input_ids + [tokenizer.eos_token_id]
        rejected_input_ids = rejected_input_ids + [tokenizer.eos_token_id]

        return {
            "prompt_input_ids": prompt_input_ids,
            "chosen_input_ids": chosen_input_ids,
            "rejected_input_ids": rejected_input_ids,
        }

    def _drop_first_dim_if_needed(self, feature_name, value):
        """Drop the first dimension of the features."""
        feature_spec = self._internal_model_config.model_input_features.get(
            feature_name,
        )

        if feature_spec is None:
            action = InternalFeatureFirstDimAction.DROP_IF_DUMMY
        else:
            action = feature_spec.first_dim_action

        if action == InternalFeatureFirstDimAction.DROP_ALWAYS:
            return value[0]
        elif action == InternalFeatureFirstDimAction.DROP_IF_DUMMY:
            if len(value) == 1:
                return value[0]
            else:
                return value
        return value

    def process_row(
        self,
        features,
    ):
        """Process a row of the dataset."""
        processor, tokenizer = self._processor, self._tokenizer
        prompt = tokenizer.apply_chat_template(features["prompt"], tokenize=False)
        prompt_chosen = tokenizer.apply_chat_template(
            features["prompt"] + features["chosen"], tokenize=False
        )
        chosen = prompt_chosen[len(prompt) :]
        prompt_rejected = tokenizer.apply_chat_template(
            features["prompt"] + features["rejected"], tokenize=False
        )
        rejected = prompt_rejected[len(prompt) :]

        processed_features = processor(
            images=features["images"], text=prompt, add_special_tokens=False
        )

        prompt_input_ids = self._drop_first_dim_if_needed(
            "input_ids", processed_features["input_ids"]
        )

        pixel_values = self._drop_first_dim_if_needed(
            "pixel_values", processed_features["pixel_values"]
        )
        chosen_input_ids = tokenizer(chosen, add_special_tokens=False)["input_ids"]
        rejected_input_ids = tokenizer(rejected, add_special_tokens=False)["input_ids"]

        # Add special tokens (typically for encoder-decoder models)
        chosen_input_ids = chosen_input_ids + [tokenizer.eos_token_id]
        rejected_input_ids = rejected_input_ids + [tokenizer.eos_token_id]

        output = {
            "prompt_input_ids": prompt_input_ids,
            "pixel_values": pixel_values,
            "chosen_input_ids": chosen_input_ids,
            "rejected_input_ids": rejected_input_ids,
        }

        if "pixel_attention_mask" in processed_features:
            output["pixel_attention_mask"] = processed_features["pixel_attention_mask"][
                0
            ]
        if "image_sizes" in processed_features:
            output["image_sizes"] = processed_features["image_sizes"][0]

        return output
