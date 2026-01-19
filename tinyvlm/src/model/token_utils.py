"""Utilities for managing special tokens in TinyVLM post-training.

This module handles:
- Adding <think> and </think> tokens to the tokenizer
- Resizing model embeddings for new tokens
- Extracting answers from thinking-mode outputs
"""

import torch
import torch.nn as nn
from transformers import PreTrainedTokenizer
from typing import Tuple, Optional, List
import re


# Special token definitions
THINK_START_TOKEN = "<think>"
THINK_END_TOKEN = "</think>"
ANSWER_START_TOKEN = "<answer>"
ANSWER_END_TOKEN = "</answer>"

# Mode tags (regular text, not special tokens)
THINKING_MODE_TAG = "/think"
NON_THINKING_MODE_TAG = "/no_think"

# Format instruction for thinking mode prompts
FORMAT_INSTRUCTION = "Output your thinking process in <think></think> tags and your final answer in <answer></answer> tags."


def add_thinking_tokens(
    tokenizer: PreTrainedTokenizer,
    model: nn.Module,
) -> Tuple[PreTrainedTokenizer, nn.Module, dict]:
    """
    Add thinking and answer tokens to tokenizer and resize model embeddings.

    This function:
    1. Adds <think>, </think>, <answer>, </answer> as special tokens
    2. Resizes model embeddings to accommodate new tokens
    3. Initializes new embeddings with mean of existing embeddings

    Args:
        tokenizer: The tokenizer to modify
        model: The model (TinyVLM) to resize embeddings for

    Returns:
        Tuple of (updated tokenizer, updated model, token_info dict)
    """
    # Get original vocab size
    original_vocab_size = len(tokenizer)

    # Add special tokens (all 4 tokens)
    special_tokens = {
        "additional_special_tokens": [
            THINK_START_TOKEN,
            THINK_END_TOKEN,
            ANSWER_START_TOKEN,
            ANSWER_END_TOKEN,
        ]
    }
    num_added = tokenizer.add_special_tokens(special_tokens)

    print(f"Added {num_added} special tokens to tokenizer")
    print(f"  Vocabulary size: {original_vocab_size} -> {len(tokenizer)}")

    # Get new token IDs
    think_start_id = tokenizer.convert_tokens_to_ids(THINK_START_TOKEN)
    think_end_id = tokenizer.convert_tokens_to_ids(THINK_END_TOKEN)
    answer_start_id = tokenizer.convert_tokens_to_ids(ANSWER_START_TOKEN)
    answer_end_id = tokenizer.convert_tokens_to_ids(ANSWER_END_TOKEN)

    print(f"  {THINK_START_TOKEN}: ID {think_start_id}")
    print(f"  {THINK_END_TOKEN}: ID {think_end_id}")
    print(f"  {ANSWER_START_TOKEN}: ID {answer_start_id}")
    print(f"  {ANSWER_END_TOKEN}: ID {answer_end_id}")

    # Resize model embeddings
    if num_added > 0:
        model.resize_token_embeddings(len(tokenizer))
        print(f"Resized model embeddings to {len(tokenizer)}")

        # Initialize new embeddings with mean of existing embeddings
        _initialize_new_embeddings(model, original_vocab_size, len(tokenizer))

    token_info = {
        "think_start_token": THINK_START_TOKEN,
        "think_end_token": THINK_END_TOKEN,
        "answer_start_token": ANSWER_START_TOKEN,
        "answer_end_token": ANSWER_END_TOKEN,
        "think_start_id": think_start_id,
        "think_end_id": think_end_id,
        "answer_start_id": answer_start_id,
        "answer_end_id": answer_end_id,
        "original_vocab_size": original_vocab_size,
        "new_vocab_size": len(tokenizer),
    }

    return tokenizer, model, token_info


def _initialize_new_embeddings(
    model: nn.Module,
    original_size: int,
    new_size: int,
) -> None:
    """
    Initialize new embedding rows with mean of existing embeddings.

    This provides a reasonable starting point for the new tokens rather than
    random initialization.

    Args:
        model: The TinyVLM model
        original_size: Original vocabulary size
        new_size: New vocabulary size
    """
    # Get embedding layers
    # For TinyVLM, we need to access language_model.model.model.embed_tokens
    # and language_model.model.lm_head

    try:
        lm = model.language_model.model  # The HF model

        # Input embeddings
        embed_tokens = lm.get_input_embeddings()
        if embed_tokens is not None:
            with torch.no_grad():
                # Compute mean of original embeddings
                mean_embed = embed_tokens.weight[:original_size].mean(dim=0)
                # Initialize new tokens with mean
                for i in range(original_size, new_size):
                    embed_tokens.weight[i] = mean_embed.clone()
            print(f"  Initialized input embeddings for tokens {original_size}-{new_size-1}")

        # Output embeddings (lm_head)
        lm_head = lm.get_output_embeddings()
        if lm_head is not None:
            with torch.no_grad():
                mean_output = lm_head.weight[:original_size].mean(dim=0)
                for i in range(original_size, new_size):
                    lm_head.weight[i] = mean_output.clone()
            print(f"  Initialized output embeddings for tokens {original_size}-{new_size-1}")

    except AttributeError as e:
        print(f"  Warning: Could not initialize embeddings: {e}")


def get_think_token_ids(tokenizer: PreTrainedTokenizer) -> Tuple[int, int]:
    """
    Get token IDs for thinking tokens.

    Args:
        tokenizer: Tokenizer with thinking tokens added

    Returns:
        Tuple of (think_start_id, think_end_id)
    """
    think_start_id = tokenizer.convert_tokens_to_ids(THINK_START_TOKEN)
    think_end_id = tokenizer.convert_tokens_to_ids(THINK_END_TOKEN)
    return think_start_id, think_end_id


def get_answer_token_ids(tokenizer: PreTrainedTokenizer) -> Tuple[int, int]:
    """
    Get token IDs for answer tokens.

    Args:
        tokenizer: Tokenizer with answer tokens added

    Returns:
        Tuple of (answer_start_id, answer_end_id)
    """
    answer_start_id = tokenizer.convert_tokens_to_ids(ANSWER_START_TOKEN)
    answer_end_id = tokenizer.convert_tokens_to_ids(ANSWER_END_TOKEN)
    return answer_start_id, answer_end_id


def get_all_special_token_ids(tokenizer: PreTrainedTokenizer) -> dict:
    """
    Get all special token IDs for thinking and answer tokens.

    Args:
        tokenizer: Tokenizer with special tokens added

    Returns:
        Dict with think_start_id, think_end_id, answer_start_id, answer_end_id
    """
    return {
        "think_start_id": tokenizer.convert_tokens_to_ids(THINK_START_TOKEN),
        "think_end_id": tokenizer.convert_tokens_to_ids(THINK_END_TOKEN),
        "answer_start_id": tokenizer.convert_tokens_to_ids(ANSWER_START_TOKEN),
        "answer_end_id": tokenizer.convert_tokens_to_ids(ANSWER_END_TOKEN),
    }


def has_thinking_tokens(tokenizer: PreTrainedTokenizer) -> bool:
    """
    Check if tokenizer has thinking and answer tokens.

    Args:
        tokenizer: Tokenizer to check

    Returns:
        True if all special tokens are present
    """
    try:
        think_start_id = tokenizer.convert_tokens_to_ids(THINK_START_TOKEN)
        think_end_id = tokenizer.convert_tokens_to_ids(THINK_END_TOKEN)
        answer_start_id = tokenizer.convert_tokens_to_ids(ANSWER_START_TOKEN)
        answer_end_id = tokenizer.convert_tokens_to_ids(ANSWER_END_TOKEN)
        # If tokens are unknown, they'll be mapped to unk_token_id
        unk_id = tokenizer.unk_token_id
        return (
            think_start_id != unk_id
            and think_end_id != unk_id
            and answer_start_id != unk_id
            and answer_end_id != unk_id
        )
    except Exception:
        return False


def extract_answer_after_think(
    text: str,
    default: Optional[str] = None,
) -> str:
    """
    Extract the final answer that appears after </think> tag.

    NOTE: This is a legacy function. Prefer extract_answer_from_tags() for new format.

    Args:
        text: Generated text potentially containing <think>...</think>answer
        default: Default value if no answer found

    Returns:
        The answer portion after </think>, or default if not found
    """
    # First try to extract from <answer> tags (new format)
    answer = extract_answer_from_tags(text)
    if answer:
        return answer

    # Fallback: Find </think> and get everything after it (old format)
    match = re.search(r'</think>\s*(.*)$', text, re.DOTALL)
    if match:
        answer = match.group(1).strip()
        return answer if answer else default
    return default or text.strip()


def extract_answer_from_tags(
    text: str,
    default: Optional[str] = None,
) -> str:
    """
    Extract the answer from <answer>...</answer> tags.

    Args:
        text: Generated text containing <answer>final_answer</answer>
        default: Default value if no answer tags found

    Returns:
        The content inside <answer>...</answer>, or default if not found
    """
    match = re.search(r'<answer>(.*?)</answer>', text, re.DOTALL)
    if match:
        answer = match.group(1).strip()
        return answer if answer else default
    return default


def extract_thinking_content(
    text: str,
) -> Tuple[Optional[str], Optional[str]]:
    """
    Extract thinking content from <think>...</think> tags.

    NOTE: This is a legacy function. Prefer extract_thinking_and_answer() for new format.

    Args:
        text: Generated text with <think>...</think> format

    Returns:
        Tuple of (thinking_content, remaining_text)
        If no thinking format found, returns (None, text)
    """
    # Pattern: <think>content</think>remaining
    match = re.search(r'<think>(.*?)</think>\s*(.*)', text, re.DOTALL)
    if match:
        thinking = match.group(1).strip()
        remaining = match.group(2).strip()
        return thinking, remaining
    return None, text.strip()


def extract_thinking_and_answer(
    text: str,
) -> Tuple[Optional[str], Optional[str]]:
    """
    Extract thinking and answer from the new format.

    Expected format: <think>reasoning</think><answer>final_answer</answer>

    Args:
        text: Generated text with thinking and answer tags

    Returns:
        Tuple of (thinking_content, answer_content)
        If tags not found, returns (None, None)
    """
    thinking = None
    answer = None

    # Extract thinking content
    think_match = re.search(r'<think>(.*?)</think>', text, re.DOTALL)
    if think_match:
        thinking = think_match.group(1).strip()

    # Extract answer content
    answer_match = re.search(r'<answer>(.*?)</answer>', text, re.DOTALL)
    if answer_match:
        answer = answer_match.group(1).strip()

    return thinking, answer


def format_thinking_response(
    thinking: str,
    answer: str,
) -> str:
    """
    Format a response with thinking and answer tokens.

    Args:
        thinking: The reasoning/thinking content
        answer: The final answer

    Returns:
        Formatted string: <think>thinking</think><answer>answer</answer>
    """
    return f"{THINK_START_TOKEN}{thinking}{THINK_END_TOKEN}{ANSWER_START_TOKEN}{answer}{ANSWER_END_TOKEN}"


def has_correct_tag_order(text: str) -> bool:
    """
    Check if text has all 4 tags in the correct order.

    Expected order: <think> ... </think> ... <answer> ... </answer>

    Args:
        text: Generated text to check

    Returns:
        True if all tags present in correct order
    """
    think_start_pos = text.find(THINK_START_TOKEN)
    think_end_pos = text.find(THINK_END_TOKEN)
    answer_start_pos = text.find(ANSWER_START_TOKEN)
    answer_end_pos = text.find(ANSWER_END_TOKEN)

    # All tags must be present
    if -1 in (think_start_pos, think_end_pos, answer_start_pos, answer_end_pos):
        return False

    # Tags must be in correct order
    return (
        think_start_pos < think_end_pos < answer_start_pos < answer_end_pos
    )


def add_mode_tag(
    prompt: str,
    thinking_mode: bool = True,
) -> str:
    """
    Add mode tag to a prompt.

    Args:
        prompt: The user prompt
        thinking_mode: If True, add /think; if False, add /no_think

    Returns:
        Prompt with mode tag prepended
    """
    tag = THINKING_MODE_TAG if thinking_mode else NON_THINKING_MODE_TAG
    return f"{tag} {prompt}"


def get_mode_from_prompt(prompt: str) -> Tuple[str, bool]:
    """
    Extract mode from prompt and return cleaned prompt.

    Args:
        prompt: Prompt potentially containing mode tag

    Returns:
        Tuple of (cleaned_prompt, is_thinking_mode)
    """
    if prompt.startswith(THINKING_MODE_TAG):
        return prompt[len(THINKING_MODE_TAG):].strip(), True
    elif prompt.startswith(NON_THINKING_MODE_TAG):
        return prompt[len(NON_THINKING_MODE_TAG):].strip(), False
    # Default to thinking mode if no tag
    return prompt, True


def find_think_end_position(
    input_ids: torch.Tensor,
    think_end_id: int,
) -> int:
    """
    Find position of </think> token in input_ids.

    Args:
        input_ids: Token IDs (1D tensor)
        think_end_id: Token ID for </think>

    Returns:
        Position index of </think> token, or -1 if not found
    """
    input_ids_list = input_ids.tolist()
    if isinstance(input_ids_list[0], list):
        input_ids_list = input_ids_list[0]

    try:
        return input_ids_list.index(think_end_id)
    except ValueError:
        return -1


def create_loss_mask_for_thinking(
    input_ids: torch.Tensor,
    think_start_id: int,
    think_end_id: int,
    answer_start_id: int = None,
    answer_end_id: int = None,
    mask_thinking: bool = False,
) -> torch.Tensor:
    """
    Create loss mask for thinking-mode training.

    Args:
        input_ids: Token IDs (B, seq_len)
        think_start_id: Token ID for <think>
        think_end_id: Token ID for </think>
        answer_start_id: Token ID for <answer> (optional)
        answer_end_id: Token ID for </answer> (optional)
        mask_thinking: If True, mask loss on thinking content (only predict answer)

    Returns:
        Boolean mask (B, seq_len) where True = compute loss
    """
    batch_size, seq_len = input_ids.shape
    mask = torch.ones_like(input_ids, dtype=torch.bool)

    if mask_thinking:
        # Mask everything between <think> and </think>
        for b in range(batch_size):
            in_thinking = False
            for i in range(seq_len):
                token = input_ids[b, i].item()
                if token == think_start_id:
                    in_thinking = True
                    mask[b, i] = False  # Don't predict <think>
                elif token == think_end_id:
                    mask[b, i] = False  # Don't predict </think>
                    in_thinking = False
                elif in_thinking:
                    mask[b, i] = False  # Don't predict thinking content

    return mask


def compute_format_reward(generated_text: str) -> float:
    """
    Compute format correctness reward for GRPO training.

    Rewards:
    - 0.05 for each tag present (<think>, </think>, <answer>, </answer>)
    - 0.10 bonus for correct tag order

    Args:
        generated_text: Model-generated text

    Returns:
        Format reward between 0.0 and 0.3
    """
    score = 0.0

    # Tag presence rewards (0.05 each)
    if THINK_START_TOKEN in generated_text:
        score += 0.05
    if THINK_END_TOKEN in generated_text:
        score += 0.05
    if ANSWER_START_TOKEN in generated_text:
        score += 0.05
    if ANSWER_END_TOKEN in generated_text:
        score += 0.05

    # Correct order bonus (0.10)
    if has_correct_tag_order(generated_text):
        score += 0.10

    return score
