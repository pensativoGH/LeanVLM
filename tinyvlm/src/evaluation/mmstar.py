"""MMStar benchmark evaluation for TinyVLM.

MMStar is a vision-indispensable multi-modal benchmark with 1,500 samples
testing 6 core capabilities across 18 detailed dimensions.

Dataset: https://huggingface.co/datasets/Lin-Chen/MMStar
"""

import re
import json
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict

import torch
from torch.utils.data import Dataset
from datasets import load_dataset, load_from_disk
from PIL import Image
from tqdm import tqdm

# Import format instruction for thinking mode
from ..model.token_utils import FORMAT_INSTRUCTION, extract_answer_from_tags


# MMStar category mapping (normalized names)
CATEGORY_MAPPING = {
    "coarse perception": "CP",
    "fine-grained perception": "FP",
    "instance reasoning": "IR",
    "logical reasoning": "LR",
    "science & technology": "ST",
    "math": "Math",
}


@dataclass
class EvalResult:
    """Single evaluation result."""
    index: int
    question: str
    predicted: str
    ground_truth: str
    correct: bool
    category: str
    l2_category: str


@dataclass
class EvalSummary:
    """Evaluation summary with accuracy metrics."""
    total: int = 0
    correct: int = 0
    accuracy: float = 0.0
    per_category: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    results: List[EvalResult] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "total": self.total,
            "correct": self.correct,
            "accuracy": round(self.accuracy * 100, 2),
            "per_category": {
                cat: {
                    "total": info["total"],
                    "correct": info["correct"],
                    "accuracy": round(info["accuracy"] * 100, 2),
                }
                for cat, info in self.per_category.items()
            },
        }


class MMStarDataset(Dataset):
    """
    MMStar benchmark dataset.

    Each sample contains:
    - image: PIL Image
    - question: Multiple choice question with options A-D
    - answer: Correct answer letter (A/B/C/D)
    - category: High-level category
    - l2_category: Detailed category
    """

    def __init__(
        self,
        data_dir: Optional[str] = None,
        processor=None,
    ):
        """
        Initialize MMStar dataset.

        Args:
            data_dir: Path to local data (if downloaded). If None, streams from HF.
            processor: VLMProcessor for image/text processing
        """
        self.processor = processor
        self.data_dir = Path(data_dir) if data_dir else None

        # Try loading from local disk first
        if self.data_dir and (self.data_dir / "data").exists():
            print(f"Loading MMStar from local: {self.data_dir}")
            self._dataset = load_from_disk(str(self.data_dir / "data"))
        else:
            print("Loading MMStar from HuggingFace...")
            self._dataset = load_dataset("Lin-Chen/MMStar", split="val")

        print(f"Loaded {len(self._dataset)} samples")

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a single sample."""
        sample = self._dataset[idx]

        # Get image
        image = sample["image"]
        if not isinstance(image, Image.Image):
            image = Image.open(image).convert("RGB")
        else:
            image = image.convert("RGB")

        result = {
            "index": sample.get("index", idx),
            "image": image,
            "question": sample["question"],
            "answer": sample["answer"].strip().upper(),
            "category": sample.get("category", "unknown"),
            "l2_category": sample.get("l2_category", "unknown"),
        }

        # Process image if processor available
        if self.processor is not None:
            result["pixel_values"] = self.processor.process_image(image).squeeze(0)

        return result


class MMStarEvaluator:
    """
    Evaluator for MMStar benchmark.

    Handles model inference and accuracy computation.
    Supports both base models (SmolLM2-135M) and instruct models (SmolLM2-360M-Instruct).

    Prompt format options:
    - "raw": Use question as-is (like nanoVLM/lmms-eval)
    - "instruct": Add "Answer with the letter..." instruction
    """

    def __init__(
        self,
        model,
        processor,
        device: torch.device,
        max_new_tokens: int = 32,
        use_chat_template: bool = False,
        debug: bool = False,
        prompt_style: str = "raw",  # "raw" (nanoVLM style) or "instruct"
        mode: str = None,  # "think", "no_think", or None (for thinking-trained models)
    ):
        """
        Initialize evaluator.

        Args:
            model: TinyVLM model
            processor: VLMProcessor
            device: torch device
            max_new_tokens: Max tokens to generate for answer
            use_chat_template: Whether to use chat template (for instruct models)
            debug: Print debug info for first few samples
            prompt_style: "raw" (nanoVLM/lmms-eval style) or "instruct" (explicit instruction)
            mode: "think", "no_think", or None (for thinking-trained models)
        """
        self.model = model
        self.processor = processor
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.use_chat_template = use_chat_template
        self.debug = debug
        self.prompt_style = prompt_style
        self.mode = mode
        self._debug_count = 0

    def extract_answer(self, response: str) -> str:
        """
        Extract answer letter (A/B/C/D) from model response.

        Handles:
        - New format: <think>...</think><answer>X</answer>
        - Legacy format: <think>...</think>X
        - Plain responses

        Tries multiple patterns to extract the answer robustly.
        """
        response = response.strip()

        # First, try to extract from <answer>...</answer> tags (new format)
        answer_content = extract_answer_from_tags(response, default=None)
        if answer_content:
            # Extract letter from answer content
            response = answer_content.strip()

        # If response contains </think>, extract content after it (legacy format)
        elif "</think>" in response:
            parts = response.split("</think>")
            if len(parts) > 1:
                response = parts[-1].strip()

        # Pattern 1: Direct letter at start
        if response and response[0].upper() in "ABCD":
            return response[0].upper()

        # Pattern 2: "The answer is X" or "Answer: X"
        patterns = [
            r"(?:the\s+)?answer\s*(?:is|:)\s*([A-Da-d])",
            r"^([A-Da-d])[\.\)\s]",
            r"\b([A-Da-d])\s*(?:is\s+)?(?:correct|right)",
            r"(?:option|choice)\s*([A-Da-d])",
        ]

        for pattern in patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                return match.group(1).upper()

        # Pattern 3: Last single letter that's A-D
        letters = re.findall(r"\b([A-Da-d])\b", response)
        if letters:
            return letters[-1].upper()

        # Default: return first character if it could be an answer
        if response:
            first_char = response[0].upper()
            if first_char in "ABCD":
                return first_char

        return ""

    def _format_question(self, question: str) -> str:
        """Format question based on prompt style and mode."""
        q = question.strip()

        # Add mode prefix if specified
        if self.mode == "think":
            q = f"/think {q}"
        elif self.mode == "no_think":
            q = f"/no_think {q}"

        if self.prompt_style == "raw":
            # Add format instruction for think mode
            if self.mode == "think":
                return f"{q}\n\n{FORMAT_INSTRUCTION}"
            return q
        else:
            base_prompt = f"{q}\nAnswer with the letter of the correct option (A, B, C, or D)."
            # Add format instruction for think mode
            if self.mode == "think":
                return f"{base_prompt}\n{FORMAT_INSTRUCTION}"
            return base_prompt

    @torch.no_grad()
    def generate_answer(
        self,
        pixel_values: torch.Tensor,
        question: str,
    ) -> str:
        """
        Generate answer for a single sample.

        Args:
            pixel_values: Processed image tensor
            question: Question text with options

        Returns:
            Generated response string
        """
        full_question = self._format_question(question)
        prompt = self.processor.format_prompt(
            full_question, use_chat_template=self.use_chat_template
        )

        # Debug output
        if self.debug and self._debug_count < 3:
            print(f"\n[DEBUG] Sample {self._debug_count}")
            print(f"[DEBUG] Question: {question[:100]}...")
            print(f"[DEBUG] Formatted prompt: {repr(prompt[:200])}...")

        # Tokenize
        encoded = self.processor.process_text(
            prompt,
            add_special_tokens=True,
            padding=False,
            truncation=True,
        )
        input_ids = encoded["input_ids"].to(self.device)

        # Prepare pixel values
        pixel_values = pixel_values.unsqueeze(0).to(self.device)
        if self.model.language_model.model.dtype == torch.bfloat16:
            pixel_values = pixel_values.to(torch.bfloat16)

        # Create attention mask
        num_image_tokens = self.model.num_image_tokens
        text_len = input_ids.shape[1]
        attention_mask = torch.ones(
            (1, num_image_tokens + text_len),
            dtype=torch.long,
            device=self.device,
        )

        # Generate
        output_ids = self.model.generate(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=self.max_new_tokens,
            temperature=0.0,  # Greedy for evaluation
            do_sample=False,
            pad_token_id=self.processor.pad_token_id,
            eos_token_id=self.processor.eos_token_id,
        )

        # Decode only the generated tokens (exclude prompt)
        input_length = input_ids.shape[1]
        output_length = output_ids.shape[1]

        if output_length > input_length:
            generated_ids = output_ids[0, input_length:]
        else:
            generated_ids = output_ids[0]

        # Keep special tokens to detect <think>, <answer> tags
        response = self.processor.tokenizer.decode(
            generated_ids,
            skip_special_tokens=False,
        ).strip()

        # Clean up EOS tokens for display but preserve thinking/answer tags
        response = response.replace("<|im_end|>", "").replace("<|endoftext|>", "").strip()

        # Debug output
        if self.debug and self._debug_count < 3:
            print(f"[DEBUG] Input IDs shape: {input_ids.shape}")
            print(f"[DEBUG] Output IDs shape: {output_ids.shape}")
            print(f"[DEBUG] Generated IDs: {generated_ids.tolist()}")
            print(f"[DEBUG] Raw response: {repr(response)}")
            full_response = self.processor.tokenizer.decode(
                output_ids[0], skip_special_tokens=False
            )
            print(f"[DEBUG] Full output (with special): {repr(full_response)}")
            self._debug_count += 1

        return response

    @torch.no_grad()
    def generate_answers_batch(
        self,
        pixel_values_list: List[torch.Tensor],
        questions: List[str],
    ) -> List[str]:
        """
        Generate answers for a batch of samples.

        Args:
            pixel_values_list: List of processed image tensors
            questions: List of question texts

        Returns:
            List of generated response strings
        """
        batch_size = len(questions)
        if batch_size == 0:
            return []

        # Format all prompts
        prompts = []
        for q in questions:
            full_question = self._format_question(q)
            prompt = self.processor.format_prompt(
                full_question, use_chat_template=self.use_chat_template
            )
            prompts.append(prompt)

        # Tokenize all prompts
        encoded_list = []
        for prompt in prompts:
            encoded = self.processor.process_text(
                prompt,
                add_special_tokens=True,
                padding=False,
                truncation=True,
            )
            encoded_list.append(encoded["input_ids"].squeeze(0))

        # Pad to same length
        max_len = max(ids.shape[0] for ids in encoded_list)
        padded_input_ids = torch.full(
            (batch_size, max_len),
            self.processor.pad_token_id,
            dtype=torch.long,
        )
        text_attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long)

        input_lengths = []
        for i, ids in enumerate(encoded_list):
            seq_len = ids.shape[0]
            padded_input_ids[i, :seq_len] = ids
            text_attention_mask[i, :seq_len] = 1
            input_lengths.append(seq_len)

        padded_input_ids = padded_input_ids.to(self.device)
        text_attention_mask = text_attention_mask.to(self.device)

        # Stack pixel values
        pixel_values = torch.stack(pixel_values_list).to(self.device)
        if self.model.language_model.model.dtype == torch.bfloat16:
            pixel_values = pixel_values.to(torch.bfloat16)

        # Create full attention mask (image + text)
        num_image_tokens = self.model.num_image_tokens
        image_attention = torch.ones(
            (batch_size, num_image_tokens),
            dtype=torch.long,
            device=self.device,
        )
        full_attention_mask = torch.cat([image_attention, text_attention_mask], dim=1)

        # Generate for batch
        output_ids = self.model.generate(
            pixel_values=pixel_values,
            input_ids=padded_input_ids,
            attention_mask=full_attention_mask,
            max_new_tokens=self.max_new_tokens,
            temperature=0.0,
            do_sample=False,
            pad_token_id=self.processor.pad_token_id,
            eos_token_id=self.processor.eos_token_id,
        )

        # Decode each response (only generated tokens, keep special tokens for tag detection)
        responses = []
        eos_id = self.processor.eos_token_id
        for i in range(batch_size):
            output_length = output_ids.shape[1]

            # Use per-sample input_length instead of max_len
            if output_length > input_lengths[i]:
                generated_ids = output_ids[i, input_lengths[i]:]
            else:
                generated_ids = output_ids[i]

            # Skip leading EOS/PAD tokens (from padding in batched input)
            start_idx = 0
            while start_idx < len(generated_ids) and generated_ids[start_idx].item() == eos_id:
                start_idx += 1
            generated_ids = generated_ids[start_idx:]

            # Truncate at first EOS token (end of real generation)
            if len(generated_ids) > 0:
                eos_positions = (generated_ids == eos_id).nonzero(as_tuple=True)[0]
                if len(eos_positions) > 0:
                    first_eos = eos_positions[0].item()
                    generated_ids = generated_ids[:first_eos]

            # Keep special tokens to detect <think>, <answer> tags
            response = self.processor.tokenizer.decode(
                generated_ids,
                skip_special_tokens=False,
            ).strip()

            # Clean up EOS tokens but preserve thinking/answer tags
            response = response.replace("<|im_end|>", "").replace("<|endoftext|>", "").strip()
            responses.append(response)

        return responses

    def evaluate(
        self,
        dataset: MMStarDataset,
        batch_size: int = 32,
        num_samples: Optional[int] = None,
        verbose: bool = True,
    ) -> EvalSummary:
        """
        Run evaluation on MMStar dataset.

        Args:
            dataset: MMStarDataset instance
            batch_size: Batch size for evaluation (higher = faster on GPU)
            num_samples: Limit number of samples (for testing)
            verbose: Show progress bar

        Returns:
            EvalSummary with accuracy metrics
        """
        self.model.eval()

        summary = EvalSummary()
        category_stats = defaultdict(lambda: {"correct": 0, "total": 0})

        # Determine number of samples
        n_samples = len(dataset)
        if num_samples:
            n_samples = min(num_samples, n_samples)

        # Process in batches
        n_batches = (n_samples + batch_size - 1) // batch_size

        if verbose:
            pbar = tqdm(total=n_samples, desc="Evaluating MMStar")

        for batch_idx in range(n_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, n_samples)
            current_batch_size = end_idx - start_idx

            # Collect batch samples
            batch_samples = []
            pixel_values_list = []
            questions = []

            for idx in range(start_idx, end_idx):
                sample = dataset[idx]
                batch_samples.append(sample)

                if "pixel_values" in sample:
                    pixel_values_list.append(sample["pixel_values"])
                else:
                    pixel_values_list.append(
                        self.processor.process_image(sample["image"]).squeeze(0)
                    )
                questions.append(sample["question"])

            # Generate answers for batch
            if batch_size == 1:
                responses = [self.generate_answer(pixel_values_list[0], questions[0])]
            else:
                responses = self.generate_answers_batch(pixel_values_list, questions)

            # Process results
            for i, (sample, response) in enumerate(zip(batch_samples, responses)):
                predicted = self.extract_answer(response)
                ground_truth = sample["answer"]
                correct = predicted == ground_truth

                # Get category
                category = sample["category"].lower()
                cat_abbrev = CATEGORY_MAPPING.get(category, category)

                # Create result
                result = EvalResult(
                    index=sample["index"],
                    question=sample["question"][:100] + "...",
                    predicted=predicted,
                    ground_truth=ground_truth,
                    correct=correct,
                    category=cat_abbrev,
                    l2_category=sample["l2_category"],
                )
                summary.results.append(result)

                # Update stats
                summary.total += 1
                if correct:
                    summary.correct += 1
                category_stats[cat_abbrev]["total"] += 1
                if correct:
                    category_stats[cat_abbrev]["correct"] += 1

            # Update progress bar after each batch
            if verbose:
                pbar.update(current_batch_size)
                acc = summary.correct / summary.total * 100 if summary.total > 0 else 0
                pbar.set_postfix({"acc": f"{acc:.1f}%"})

        # Close progress bar
        if verbose:
            pbar.close()

        # Compute final metrics
        summary.accuracy = summary.correct / summary.total if summary.total > 0 else 0

        for cat, stats in category_stats.items():
            stats["accuracy"] = stats["correct"] / stats["total"] if stats["total"] > 0 else 0
            summary.per_category[cat] = stats

        return summary


def save_results(
    summary: EvalSummary,
    output_path: Path,
    include_details: bool = False,
):
    """
    Save evaluation results to JSON.

    Args:
        summary: EvalSummary object
        output_path: Path to save JSON file
        include_details: Include per-sample results
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    results = summary.to_dict()

    if include_details:
        results["details"] = [
            {
                "index": r.index,
                "predicted": r.predicted,
                "ground_truth": r.ground_truth,
                "correct": r.correct,
                "category": r.category,
            }
            for r in summary.results
        ]

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to {output_path}")
