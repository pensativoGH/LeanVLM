#!/usr/bin/env python3
"""Evaluation script for TinyVLM on MMMU, MMBench, and MathVista."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TINYVLM_ROOT = PROJECT_ROOT.parent / "TinyVLM"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(TINYVLM_ROOT))

import argparse
import torch
from tqdm import tqdm
import re
import json
from collections import defaultdict
from datasets import load_from_disk
from PIL import Image
from transformers import AutoTokenizer, SiglipImageProcessor


# Import TinyVLM
from src.model.vlm import TinyVLM


def format_mcq_prompt(question: str, choices: list) -> str:
    """Format MCQ with choices."""
    prompt = question.strip()
    if choices:
        prompt += "\nChoices:"
        for i, choice in enumerate(choices):
            letter = chr(65 + i)
            prompt += f"\n{letter}. {choice}"
    prompt += "\nAnswer with the letter directly."
    return prompt


def format_prompt(tokenizer, question: str) -> str:
    """Format prompt in chat style."""
    return f"<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\nAnswer:"


def extract_answer(response: str, num_choices: int = 4) -> str:
    """Extract answer letter from response."""
    response = response.strip()
    valid_letters = [chr(65 + i) for i in range(num_choices)]
    
    if response and response[0].upper() in valid_letters:
        return response[0].upper()
    
    patterns = [
        r"(?:the\s+)?answer\s*(?:is|:)\s*([A-Za-z])",
        r"^([A-Za-z])[\.)\s]",
        r"\b([A-Za-z])\s*(?:is\s+)?(?:correct|right)",
        r"(?:option|choice)\s*([A-Za-z])",
    ]
    
    for pattern in patterns:
        match = re.search(pattern, response, re.IGNORECASE)
        if match:
            letter = match.group(1).upper()
            if letter in valid_letters:
                return letter
    
    letters = re.findall(r"\b([A-Za-z])\b", response)
    for letter in reversed(letters):
        if letter.upper() in valid_letters:
            return letter.upper()
    
    return ""


def load_mmmu_dataset(data_dir: str):
    """Load MMMU validation set."""
    samples = []
    mmmu_dir = Path(data_dir) / "mmmu"
    
    subjects = [d.name for d in mmmu_dir.iterdir() if d.is_dir()]
    print(f"Loading MMMU from {len(subjects)} subjects...")
    
    for subject in sorted(subjects):
        subject_dir = mmmu_dir / subject
        try:
            ds = load_from_disk(str(subject_dir))
            if "validation" in ds:
                for item in ds["validation"]:
                    image = item.get("image")
                    if image is None:
                        for key in ["image_1", "image_2", "image_3", "image_4", "image_5", "image_6", "image_7"]:
                            if item.get(key) is not None:
                                image = item[key]
                                break
                    
                    if image is not None:
                        choices = [item.get(f"option_{i}", "") for i in range(ord('a'), ord('e'))]
                        choices = [c for c in choices if c]
                        
                        samples.append({
                            "image": image,
                            "question": item["question"],
                            "choices": choices,
                            "answer": item["answer"].strip().upper(),
                            "subject": subject,
                        })
        except Exception as e:
            print(f"  Warning: Could not load {subject}: {e}")
    
    print(f"Loaded {len(samples)} MMMU validation samples")
    return samples


def load_mmbench_dataset(data_dir: str):
    """Load MMBench dev set."""
    samples = []
    mmbench_dir = Path(data_dir) / "mmbench" / "en"
    
    print(f"Loading MMBench from {mmbench_dir}...")
    ds = load_from_disk(str(mmbench_dir))
    
    if "dev" in ds:
        for item in ds["dev"]:
            image = item.get("image")
            if image is None:
                continue
            
            choices = []
            for key in ["A", "B", "C", "D"]:
                if item.get(key):
                    choices.append(item[key])
            
            samples.append({
                "image": image,
                "question": item["question"],
                "choices": choices,
                "answer": item["answer"].strip().upper(),
                "category": item.get("category", "unknown"),
            })
    
    print(f"Loaded {len(samples)} MMBench dev samples")
    return samples


def load_mathvista_dataset(data_dir: str):
    """Load MathVista testmini set."""
    samples = []
    mathvista_dir = Path(data_dir) / "mathvista"
    
    print(f"Loading MathVista from {mathvista_dir}...")
    ds = load_from_disk(str(mathvista_dir))
    
    if "testmini" in ds:
        for item in ds["testmini"]:
            image = item.get("decoded_image") or item.get("image")
            if image is None:
                continue
            
            question = item["question"]
            choices = item.get("choices", [])
            answer = str(item["answer"]).strip()
            
            if choices:
                answer_letter = ""
                for i, choice in enumerate(choices):
                    if str(choice).strip() == answer:
                        answer_letter = chr(65 + i)
                        break
                if not answer_letter and answer in "ABCD":
                    answer_letter = answer
                samples.append({
                    "image": image,
                    "question": question,
                    "choices": choices,
                    "answer": answer_letter,
                    "is_mcq": True,
                    "category": item.get("metadata", {}).get("task", "unknown") if isinstance(item.get("metadata"), dict) else "unknown",
                })
            else:
                samples.append({
                    "image": image,
                    "question": question,
                    "choices": [],
                    "answer": answer,
                    "is_mcq": False,
                    "category": item.get("metadata", {}).get("task", "unknown") if isinstance(item.get("metadata"), dict) else "unknown",
                })
    
    print(f"Loaded {len(samples)} MathVista testmini samples")
    return samples


def evaluate_dataset(
    model,
    tokenizer,
    image_processor,
    samples,
    dataset_name: str,
    batch_size: int = 64,
    max_new_tokens: int = 32,
    device: torch.device = None,
):
    """Evaluate TinyVLM on dataset."""
    print(f"\n{'='*50}")
    print(f"Evaluating {dataset_name}")
    print(f"Samples: {len(samples)}, Batch size: {batch_size}")
    print(f"{'='*50}")
    
    correct = 0
    total = 0
    per_category = defaultdict(lambda: {"correct": 0, "total": 0})
    
    pbar = tqdm(range(0, len(samples), batch_size), desc=f"Eval {dataset_name}")
    
    for batch_start in pbar:
        batch_end = min(batch_start + batch_size, len(samples))
        batch_samples = samples[batch_start:batch_end]
        
        # Prepare batch
        prompts = []
        pixel_values_list = []
        
        for sample in batch_samples:
            # Format question
            if sample.get("choices"):
                q = format_mcq_prompt(sample["question"], sample["choices"])
            else:
                q = sample["question"] + "\nAnswer directly with a short response."
            
            prompts.append(format_prompt(tokenizer, q))
            
            # Process image
            image = sample["image"]
            if not isinstance(image, Image.Image):
                image = Image.open(image).convert("RGB")
            else:
                image = image.convert("RGB")
            
            pv = image_processor(images=image, return_tensors="pt")["pixel_values"].squeeze(0)
            pixel_values_list.append(pv)
        
        # Tokenize
        encoded = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048,
        )
        input_ids = encoded["input_ids"].to(device)
        text_attention_mask = encoded["attention_mask"].to(device)
        pixel_values = torch.stack(pixel_values_list).to(device, dtype=torch.bfloat16)
        
        # Create full attention mask (image tokens + text tokens)
        batch_size_curr = pixel_values.shape[0]
        image_attention = torch.ones(batch_size_curr, model.num_image_tokens, device=device, dtype=text_attention_mask.dtype)
        attention_mask = torch.cat([image_attention, text_attention_mask], dim=1)
        
        # Generate
        with torch.no_grad():
            output_ids = model.generate(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        
        # Evaluate responses
        for i, sample in enumerate(batch_samples):
            response = tokenizer.decode(output_ids[i], skip_special_tokens=True)
            
            ground_truth = sample["answer"]
            
            if sample.get("choices"):
                predicted = extract_answer(response, len(sample["choices"]))
                is_correct = predicted == ground_truth
            else:
                predicted = response.strip()
                is_correct = ground_truth.lower() in predicted.lower()
            
            total += 1
            if is_correct:
                correct += 1
            
            category = sample.get("category") or sample.get("subject", "unknown")
            per_category[category]["total"] += 1
            if is_correct:
                per_category[category]["correct"] += 1
        
        pbar.set_postfix(acc=f"{correct/total*100:.1f}%")
    
    accuracy = correct / total if total > 0 else 0
    
    return {
        "dataset": dataset_name,
        "total": total,
        "correct": correct,
        "accuracy": round(accuracy * 100, 2),
        "per_category": {
            cat: {
                "total": stats["total"],
                "correct": stats["correct"],
                "accuracy": round(stats["correct"]/stats["total"]*100, 2) if stats["total"] > 0 else 0
            }
            for cat, stats in per_category.items()
        }
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--datasets", nargs="+", default=["mmmu", "mmbench", "mathvista"])
    parser.add_argument("--data-dir", type=str, default="data/eval_datasets")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--output-dir", type=str, default="results/multimodal_tinyvlm")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load checkpoint to get config
    print(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = checkpoint.get("config", {})
    model_config = config.get("model", {})
    
    vision_model = model_config.get("vision_encoder", "google/siglip-base-patch16-384")
    lm_model = model_config.get("language_model", "HuggingFaceTB/SmolLM2-360M-Instruct")
    
    print(f"Vision encoder: {vision_model}")
    print(f"Language model: {lm_model}")
    
    # Load model
    model = TinyVLM(
        vision_encoder_name=vision_model,
        language_model_name=lm_model,
        freeze_vision=True,
        freeze_lm=False,
    )
    
    # Load weights from checkpoint (nested state dicts)
    model_sd = checkpoint["model_state_dict"]
    model.vision_encoder.load_state_dict(model_sd["vision_encoder"])
    model.projector.load_state_dict(model_sd["projector"])
    model.language_model.model.load_state_dict(model_sd["language_model"])
    
    model = model.to(device=device, dtype=torch.bfloat16)
    model.eval()
    
    # Load tokenizer and image processor
    tokenizer = AutoTokenizer.from_pretrained(lm_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    image_processor = SiglipImageProcessor.from_pretrained(vision_model)
    
    print(f"\nModel loaded. Image tokens: {model.num_image_tokens}")
    
    # Evaluate each dataset
    all_results = {}
    
    for dataset_name in args.datasets:
        if dataset_name == "mmmu":
            samples = load_mmmu_dataset(args.data_dir)
        elif dataset_name == "mmbench":
            samples = load_mmbench_dataset(args.data_dir)
        elif dataset_name == "mathvista":
            samples = load_mathvista_dataset(args.data_dir)
        else:
            print(f"Unknown dataset: {dataset_name}")
            continue
        
        if not samples:
            print(f"No samples loaded for {dataset_name}")
            continue
        
        results = evaluate_dataset(
            model, tokenizer, image_processor, samples, dataset_name,
            batch_size=args.batch_size,
            max_new_tokens=args.max_new_tokens,
            device=device,
        )
        all_results[dataset_name] = results
        
        print(f"\n{dataset_name} Accuracy: {results['accuracy']:.2f}%")
    
    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    ckpt_name = Path(args.checkpoint).stem
    output_file = output_dir / f"eval_{ckpt_name}.json"
    
    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n{'='*50}")
    print("SUMMARY")
    print(f"{'='*50}")
    for ds, res in all_results.items():
        print(f"{ds}: {res['accuracy']:.2f}% ({res['correct']}/{res['total']})")
    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
