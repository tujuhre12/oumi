"""Self-contained test script to verify Role enum serialization fix for vision models.

This script tests the fix for the issue where Message objects with Role enums
were not being properly serialized to dictionaries before being passed to
HuggingFace's apply_chat_template, causing role confusion in chat templates.

Usage:
    python test_role_serialization_fix.py
    python test_role_serialization_fix.py --model qwen2vl
    python test_role_serialization_fix.py --test-completions-only
    python test_role_serialization_fix.py --verbose
"""

import argparse
import io
import sys
import traceback

import torch
from PIL import Image

# Oumi imports
from oumi.builders import build_data_collator, build_tokenizer
from oumi.core.configs import ModelParams
from oumi.core.constants import LABEL_IGNORE_INDEX
from oumi.core.types import ContentItem, Conversation, Message, Role
from oumi.core.types.conversation import Type

# Model configurations extracted from configs/recipes/vision
VISION_MODELS = {
    "phi3": {
        "model_name": "microsoft/Phi-3-vision-128k-instruct",
        "chat_template": None,
        # "chat_template": "phi3-instruct",
        "trust_remote_code": True,
        "response_template": "<|assistant|>",
        "instruction_template": "<|user|>",
        "collator": "vision_language_sft",
    },
    "phi4": {
        "model_name": "microsoft/Phi-4-multimodal-instruct",
        "chat_template": None,
        "trust_remote_code": True,
        "response_template": "<|assistant|>",
        "instruction_template": "<|user|>",
        "collator": "vision_language_sft",
    },
    "llava": {
        "model_name": "llava-hf/llava-1.5-7b-hf",
        "chat_template": "llava",
        "trust_remote_code": True,
        "response_template": "ASSISTANT:",
        "instruction_template": "USER:",
        "collator": "vision_language_with_padding",
    },
    "qwen2vl": {
        "model_name": "Qwen/Qwen2-VL-2B-Instruct",
        "chat_template": "qwen2-vl-instruct",
        "trust_remote_code": True,
        "response_template": "<|im_start|>assistant\n",
        "instruction_template": "<|im_start|>user\n",
        "collator": "vision_language_sft",
    },
    "qwen2_5vl_3b": {
        "model_name": "Qwen/Qwen2.5-VL-3B-Instruct",
        "chat_template": "qwen2-vl-instruct",
        "trust_remote_code": True,
        "response_template": "<|im_start|>assistant\n",
        "instruction_template": "<|im_start|>user\n",
        "collator": "vision_language_with_padding",
    },
    "qwen2_5vl_7b": {
        "model_name": "Qwen/Qwen2.5-VL-7B-Instruct",
        "chat_template": "qwen2-vl-instruct",
        "trust_remote_code": True,
        "response_template": "<|im_start|>assistant\n",
        "instruction_template": "<|im_start|>user\n",
        "collator": "vision_language_with_padding",
    },
    "internvl3": {
        "model_name": "OpenGVLab/InternVL3-1B-hf",
        "chat_template": "internvl3",
        "trust_remote_code": True,
        "response_template": "<|im_start|>assistant\n",
        "instruction_template": "<|im_start|>user\n",
        "collator": "vision_language_sft",
    },
    "llama3_2_11b": {
        "model_name": "meta-llama/Llama-3.2-11B-Vision-Instruct",
        "chat_template": "llama3-instruct",
        "trust_remote_code": False,
        "response_template": "<|start_header_id|>assistant<|end_header_id|>\n\n",
        "instruction_template": "<|start_header_id|>user<|end_header_id|>\n\n",
        "collator": "vision_language_with_padding",
    },
    "llama3_2_90b": {
        "model_name": "meta-llama/Llama-3.2-90B-Vision-Instruct",
        "chat_template": "llama3-instruct",
        "trust_remote_code": False,
        "response_template": "<|start_header_id|>assistant<|end_header_id|>\n\n",
        "instruction_template": "<|start_header_id|>user<|end_header_id|>\n\n",
        "collator": "vision_language_with_padding",
    },
    "molmo_d": {
        "model_name": "oumi-ai/Molmo-7B-D-0924",
        "chat_template": "molmo",
        "trust_remote_code": True,
        "response_template": "Assistant:",
        "instruction_template": "User:",
        "collator": "vision_language_sft",
    },
    "molmo_o": {
        "model_name": "oumi-ai/Molmo-7B-O-0924",
        "chat_template": "molmo",
        "trust_remote_code": True,
        "response_template": "Assistant:",
        "instruction_template": "User:",
        "collator": "vision_language_sft",
    },
    "smolvlm": {
        "model_name": "HuggingFaceTB/SmolVLM-Instruct",
        "chat_template": "llava",
        "trust_remote_code": True,
        "response_template": "ASSISTANT:",
        "instruction_template": "USER:",
        "collator": "vision_language_with_padding",
    },
}


def create_test_image_bytes() -> bytes:
    """Create a simple test image as bytes."""
    image = Image.new("RGB", (100, 100), color="red")
    img_bytes = io.BytesIO()
    image.save(img_bytes, format="PNG")
    return img_bytes.getvalue()


def create_test_conversation() -> Conversation:
    """Create a test conversation."""
    return Conversation(
        messages=[
            Message(
                role=Role.USER,
                content=[
                    ContentItem(
                        type=Type.TEXT, content="Extract data from this image as JSON:"
                    ),
                    ContentItem(
                        type=Type.IMAGE_BINARY, binary=create_test_image_bytes()
                    ),
                ],
            ),
            Message(
                role=Role.ASSISTANT,
                content=[
                    ContentItem(
                        type=Type.TEXT,
                        content=(
                            '```json\n{\n  "color": "red",\n  "size": "100x100"\n}\n```'
                        ),
                    )
                ],
            ),
        ]
    )


def test_completions_only_masking(model_config: dict, verbose: bool = False) -> bool:
    """Test that completions-only training preserves JSON backticks."""
    print(f"Testing completions-only masking for {model_config['model_name']}...")

    try:
        # Build tokenizer
        model_params = ModelParams(
            model_name=model_config["model_name"],
            device_map="cpu",
            trust_remote_code=model_config["trust_remote_code"],
            chat_template=model_config["chat_template"],
        )

        tokenizer = build_tokenizer(model_params)

        # Create collator with completions-only training
        collator_kwargs = {
            "tokenizer": tokenizer,
            "processor_name": model_config["model_name"],
            "train_on_completions_only": True,
            "response_template": model_config["response_template"],
            "instruction_template": model_config["instruction_template"],
            "trust_remote_code": model_config["trust_remote_code"],
            "max_length": 512,
        }

        # Add model-specific configurations
        # if "molmo" in model_config["model_name"].lower():
        #     collator_kwargs["process_individually"] = True
        collator_kwargs["process_individually"] = True

        collator = build_data_collator(
            collator_name=model_config["collator"], **collator_kwargs
        )

        # Create test conversation
        conversation = create_test_conversation()
        batch = [{"conversation_json": conversation.to_json()}]
        print(batch)

        # Process batch
        result = collator(batch)

        # Verify result structure
        required_keys = ["input_ids", "labels", "attention_mask"]
        for key in required_keys:
            assert key in result, f"Missing key: {key}"
            assert isinstance(result[key], torch.Tensor), f"{key} is not a tensor"

        # Extract tensors
        input_ids = result["input_ids"][0]
        labels = result["labels"][0]

        # Verify shapes match
        assert input_ids.shape == labels.shape, "Input IDs and labels shape mismatch"

        # Find unmasked positions (assistant content)
        unmasked_positions = [
            i for i, label in enumerate(labels.tolist()) if label != LABEL_IGNORE_INDEX
        ]
        assert len(unmasked_positions) > 0, "No assistant response tokens are unmasked"

        # Decode only the unmasked (assistant) portion
        unmasked_labels = [labels[i].item() for i in unmasked_positions]
        try:
            # Filter out -1 (image) tokens for decoding
            filtered_labels = [label for label in unmasked_labels if label != -1]
            assistant_response = tokenizer.decode(
                filtered_labels, skip_special_tokens=False
            )

            # Add note about image tokens if present
            image_count = unmasked_labels.count(-1)
            if image_count > 0:
                assistant_response = (
                    f"{assistant_response} [Contains {image_count} image tokens]"
                )
        except (OverflowError, ValueError) as e:
            assistant_response = f"<DECODE_ERROR:{e}>"

        if verbose:
            print(f"  Assistant response: {repr(assistant_response)}")

            # Display chat template being used
            try:
                chat_template = tokenizer.chat_template
                if chat_template:
                    template_preview = (
                        f"{repr(chat_template[:200])}"
                        f"{'...' if len(chat_template) > 200 else ''}"
                    )
                    print(f"  Chat template: {template_preview}")
                else:
                    print("  Chat template: None (using default)")
            except AttributeError:
                print("  Chat template: <Not available>")

            # Log detailed token masking information
            print("  Token masking details:")
            input_ids_list = input_ids.tolist()
            labels_list = labels.tolist()

            print(f"    Total tokens: {len(input_ids_list)}")
            masked_count = sum(
                1 for label in labels_list if label == LABEL_IGNORE_INDEX
            )
            print(f"    Masked tokens: {masked_count}")
            print(f"    Unmasked tokens: {len(unmasked_positions)}")
            print(f"    Response template: {repr(model_config['response_template'])}")
            print(
                f"    Instruction template: "
                f"{repr(model_config['instruction_template'])}"
            )

            def safe_decode_token(token_id: int) -> str:
                """Safely decode a single token."""
                if token_id == -1:
                    return "<IMAGE_TOKEN>"
                try:
                    return tokenizer.decode([token_id], skip_special_tokens=False)
                except (OverflowError, ValueError) as e:
                    return f"<DECODE_ERROR:{e}>"

            def safe_decode_tokens(token_ids: list[int]) -> str:
                """Safely decode a list of tokens."""
                try:
                    # Filter out -1 (image) tokens for decoding, but preserve in output
                    filtered_tokens = [tid for tid in token_ids if tid != -1]
                    decoded = tokenizer.decode(
                        filtered_tokens, skip_special_tokens=False
                    )

                    # Replace -1 tokens with placeholders in the original sequence
                    result_parts = []
                    for tid in token_ids:
                        if tid == -1:
                            result_parts.append("<IMAGE_TOKEN>")
                        else:
                            # This is a simplification - in reality we'd need to map
                            # positions
                            pass

                    # For now, just return the decoded text with a note about image
                    # tokens
                    image_count = token_ids.count(-1)
                    if image_count > 0:
                        return f"{decoded} [Contains {image_count} image tokens]"
                    return decoded
                except (OverflowError, ValueError) as e:
                    return f"<DECODE_ERROR:{e}>"

            # Show ALL tokens with their masking status
            print("    Complete token breakdown:")
            for i in range(len(input_ids_list)):
                token_id = input_ids_list[i]
                token_text = safe_decode_token(token_id)
                mask_status = (
                    "MASKED" if labels_list[i] == LABEL_IGNORE_INDEX else "UNMASKED"
                )
                label_value = (
                    labels_list[i] if labels_list[i] != LABEL_IGNORE_INDEX else "IGNORE"
                )
                print(
                    f"      [{i:3d}] {mask_status:8s} {token_id:6d} -> "
                    f"{str(label_value):>6s} {repr(token_text)}"
                )

            print(f"    Summary of unmasked positions: {unmasked_positions}")

            # Show full conversation for context
            full_conversation = safe_decode_tokens(input_ids_list)
            print(f"    Full conversation: {repr(full_conversation)}")

        # Verify JSON backticks are preserved
        assert "```json" in assistant_response, (
            f"Opening JSON backticks missing from: {assistant_response}"
        )

        # Check for closing backticks (after removing the opening ones)
        remaining_text = assistant_response.replace("```json", "", 1)
        assert "```" in remaining_text, (
            f"Closing backticks missing from: {assistant_response}"
        )

        # Verify JSON content is preserved
        assert (
            '"color": "red"' in assistant_response or "color" in assistant_response
        ), "JSON content missing"

        print("  ✅ PASSED: JSON backticks preserved in completions-only training")
        return True

    except Exception as e:
        print(f"  ❌ FAILED: {str(e)}")
        if verbose:
            traceback.print_exc()
        return False


def main():
    """Main function to run the Role enum serialization tests for vision models."""
    parser = argparse.ArgumentParser(
        description="Test Role enum serialization fix for vision models"
    )
    parser.add_argument(
        "--model",
        choices=list(VISION_MODELS.keys()) + ["all"],
        default="all",
        help="Specific model to test (default: all)",
    )
    parser.add_argument(
        "--test-completions-only",
        action="store_true",
        help="Test completions-only training (requires model download)",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose output"
    )

    args = parser.parse_args()

    # Determine which models to test
    if args.model == "all":
        models_to_test = VISION_MODELS
    else:
        models_to_test = {args.model: VISION_MODELS[args.model]}

    print("=" * 80)
    print("Testing Role Enum Serialization Fix for Vision Models")
    print("=" * 80)
    print(f"Models to test: {list(models_to_test.keys())}")
    print(f"Test completions-only: {args.test_completions_only}")
    print(f"Verbose mode: {args.verbose}")
    print("=" * 80)

    total_tests = 0
    passed_tests = 0
    failed_models = []

    for model_key, model_config in models_to_test.items():
        print(f"\n🔍 Testing {model_key} ({model_config['model_name']})")
        print("-" * 60)

        # Test 1: Basic processor role serialization (lightweight)
        total_tests += 1
        # if test_processor_role_serialization(model_config, args.verbose):
        #     passed_tests += 1
        # else:
        #     failed_models.append(f"{model_key} (processor)")

        # Test 2: Completions-only masking (requires model download)
        if args.test_completions_only:
            total_tests += 1
            if test_completions_only_masking(model_config, args.verbose):
                passed_tests += 1
            else:
                failed_models.append(f"{model_key} (completions-only)")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total tests: {total_tests}")
    print(f"Passed: {passed_tests}")
    print(f"Failed: {total_tests - passed_tests}")

    if failed_models:
        print("\nFailed models:")
        for model in failed_models:
            print(f"  ❌ {model}")
    else:
        print("\n🎉 All tests passed!")

    # Exit with error code if any tests failed
    sys.exit(0 if passed_tests == total_tests else 1)


if __name__ == "__main__":
    main()
