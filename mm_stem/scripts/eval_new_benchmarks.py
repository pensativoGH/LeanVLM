#!/usr/bin/env python3
"""Evaluation script for RealWorldQA, HallusionBench, OCRBench, MM-Vet with timing."""

import sys
from pathlib import Path
import time

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import torch
from tqdm import tqdm
import re
import json
from collections import defaultdict
from datasets import load_from_disk
from PIL import Image

from src.model.stem_vlm import STEMVLM, STEMVLMConfig
from src.data.processor import MultimodalProcessor


def format_prompt(question: str, choices: list = None) -> str:
    """Format prompt with optional choices."""
    prompt = question.strip()
    if choices:
        prompt += "\nChoices:"
        for i, choice in enumerate(choices):
            letter = chr(65 + i)
            prompt += f"\n{letter}. {choice}"
        prompt += "\nAnswer with the letter directly."
    return f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\nAnswer:"


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
    ]
    
    for pattern in patterns:
        match = re.search(pattern, response, re.IGNORECASE)
        if match:
            letter = match.group(1).upper()
            if letter in valid_letters:
                return letter
    
    return response.strip()[:50]  # Return raw for free-form


def load_realworldqa(data_dir: str):
    """Load RealWorldQA dataset."""
    ds = load_from_disk(str(Path(data_dir) / "realworldqa"))
    samples = []
    for item in ds["test"]:
        samples.append({
            "image": item["image"],
            "question": item["question"],
            "answer": item["answer"].strip(),
            "is_mcq": False,  # Free-form answers
        })
    print(f"Loaded {len(samples)} RealWorldQA samples")
    return samples


def load_hallusionbench(data_dir: str):
    """Load HallusionBench dataset (image split only)."""
    ds = load_from_disk(str(Path(data_dir) / "hallusionbench"))
    samples = []
    for item in ds["image"]:
        samples.append({
            "image": item["image"],
            "question": item["question"],
            "answer": item["gt_answer"].strip().lower(),  # yes/no
            "category": item["category"],
            "is_mcq": False,
        })
    print(f"Loaded {len(samples)} HallusionBench samples")
    return samples


def load_ocrbench(data_dir: str):
    """Load OCRBench dataset."""
    ds = load_from_disk(str(Path(data_dir) / "ocrbench"))
    samples = []
    for item in ds["test"]:
        samples.append({
            "image": item["image"],
            "question": item["question"],
            "answer": str(item["answer"]).strip(),
            "question_type": item["question_type"],
            "is_mcq": False,
        })
    print(f"Loaded {len(samples)} OCRBench samples")
    return samples


def load_mmvet(data_dir: str):
    """Load MM-Vet dataset."""
    ds = load_from_disk(str(Path(data_dir) / "mmvet"))
    samples = []
    for item in ds["test"]:
        samples.append({
            "image": item["image"],
            "question": item["question"],
            "answer": item["answer"].strip(),
            "capability": item["capability"],
            "is_mcq": False,
        })
    print(f"Loaded {len(samples)} MM-Vet samples")
    return samples


def evaluate_dataset(
    model,
    processor,
    samples,
    dataset_name: str,
    batch_size: int = 64,
    max_new_tokens: int = 32,
    device: torch.device = None,
):
    """Evaluate model on dataset with timing."""
    print(f"\n{'='*50}")
    print(f"Evaluating {dataset_name}")
    print(f"Samples: {len(samples)}, Batch size: {batch_size}")
    print(f"{'='*50}")
    
    correct = 0
    total = 0
    per_category = defaultdict(lambda: {"correct": 0, "total": 0})
    
    # Timing
    total_inference_time = 0.0
    sample_times = []
    
    dataset_start_time = time.time()
    
    pbar = tqdm(range(0, len(samples), batch_size), desc=f"Eval {dataset_name}")
    
    for batch_start in pbar:
        batch_end = min(batch_start + batch_size, len(samples))
        batch_samples = samples[batch_start:batch_end]
        current_batch_size = len(batch_samples)
        
        # Prepare batch
        prompts = []
        pixel_values_list = []
        
        for sample in batch_samples:
            prompts.append(format_prompt(sample["question"]))
            
            image = sample["image"]
            if not isinstance(image, Image.Image):
                image = Image.open(image).convert("RGB")
            else:
                image = image.convert("RGB")
            
            pv = processor.process_image(image)
            if pv.dim() == 4:
                pv = pv.squeeze(0)
            pixel_values_list.append(pv)
        
        # Tokenize
        encoded = processor.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048,
        )
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)
        pixel_values = torch.stack(pixel_values_list).to(device, dtype=torch.bfloat16)
        
        # Time inference
        torch.cuda.synchronize()
        batch_start_time = time.time()
        
        with torch.no_grad():
            output_ids = model.generate(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=processor.tokenizer.pad_token_id,
                eos_token_id=processor.tokenizer.eos_token_id,
            )
        
        torch.cuda.synchronize()
        batch_inference_time = time.time() - batch_start_time
        total_inference_time += batch_inference_time
        
        # Per-sample time
        per_sample_time = batch_inference_time / current_batch_size
        sample_times.extend([per_sample_time] * current_batch_size)
        
        # Evaluate responses
        for i, sample in enumerate(batch_samples):
            new_tokens = output_ids[i, input_ids.shape[1]:]
            response = processor.decode(new_tokens, skip_special_tokens=True)
            
            ground_truth = sample["answer"]
            predicted = extract_answer(response)
            
            # Flexible matching
            is_correct = (
                predicted.lower() == ground_truth.lower() or
                ground_truth.lower() in predicted.lower() or
                predicted.lower() in ground_truth.lower()
            )
            
            total += 1
            if is_correct:
                correct += 1
            
            category = sample.get("category") or sample.get("question_type") or sample.get("capability", "unknown")
            per_category[category]["total"] += 1
            if is_correct:
                per_category[category]["correct"] += 1
        
        pbar.set_postfix(acc=f"{correct/total*100:.1f}%", ms_per_sample=f"{per_sample_time*1000:.1f}")
    
    dataset_total_time = time.time() - dataset_start_time
    accuracy = correct / total if total > 0 else 0
    avg_sample_time = total_inference_time / total if total > 0 else 0
    
    return {
        "dataset": dataset_name,
        "total": total,
        "correct": correct,
        "accuracy": round(accuracy * 100, 2),
        "timing": {
            "total_time_sec": round(dataset_total_time, 2),
            "inference_time_sec": round(total_inference_time, 2),
            "avg_sample_time_ms": round(avg_sample_time * 1000, 2),
            "throughput_samples_per_sec": round(total / total_inference_time, 2) if total_inference_time > 0 else 0,
        },
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
    parser.add_argument("--datasets", nargs="+", default=["realworldqa", "hallusionbench", "ocrbench", "mmvet"])
    parser.add_argument("--data-dir", type=str, default="data/eval_datasets")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--output-dir", type=str, default="results/new_benchmarks")
    parser.add_argument("--model-name", type=str, default="mm-stem", help="Model name for output file")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config_dict = checkpoint.get("config", {})
    
    config = STEMVLMConfig(
        vision_model_name=config_dict.get("vision_model_name", "google/siglip-base-patch16-384"),
        pixel_shuffle_scale=config_dict.get("pixel_shuffle_scale", 2),
        freeze_vision_encoder=config_dict.get("freeze_vision_encoder", True),
        image_size=config_dict.get("image_size", 384),
        projector_type=config_dict.get("projector_type", "mlp"),
        projector_num_layers=config_dict.get("projector_num_layers", 2),
        projector_dropout=config_dict.get("projector_dropout", 0.0),
        lm_model_name=config_dict.get("lm_model_name", "HuggingFaceTB/SmolLM2-360M-Instruct"),
        vocab_size=config_dict.get("vocab_size", 49152),
        hidden_size=config_dict.get("hidden_size", 960),
        intermediate_size=config_dict.get("intermediate_size", 2560),
        num_hidden_layers=config_dict.get("num_hidden_layers", 32),
        num_attention_heads=config_dict.get("num_attention_heads", 15),
        num_key_value_heads=config_dict.get("num_key_value_heads", 5),
        max_position_embeddings=config_dict.get("max_position_embeddings", 8192),
        rope_theta=config_dict.get("rope_theta", 100000.0),
        rms_norm_eps=config_dict.get("rms_norm_eps", 1e-5),
        hidden_act=config_dict.get("hidden_act", "silu"),
        tie_word_embeddings=config_dict.get("tie_word_embeddings", True),
    )
    
    model = STEMVLM(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device=device, dtype=torch.bfloat16)
    model.eval()
    model.enable_stem()
    
    processor = MultimodalProcessor(
        tokenizer_name=config.lm_model_name,
        image_size=config.image_size,
        max_length=2048,
    )
    
    print(f"\nModel loaded. Image tokens: {model.num_image_tokens}")
    
    # Dataset loaders
    loaders = {
        "realworldqa": load_realworldqa,
        "hallusionbench": load_hallusionbench,
        "ocrbench": load_ocrbench,
        "mmvet": load_mmvet,
    }
    
    # Evaluate each dataset
    all_results = {}
    total_eval_time = 0
    
    for dataset_name in args.datasets:
        if dataset_name not in loaders:
            print(f"Unknown dataset: {dataset_name}")
            continue
        
        samples = loaders[dataset_name](args.data_dir)
        if not samples:
            continue
        
        results = evaluate_dataset(
            model, processor, samples, dataset_name,
            batch_size=args.batch_size,
            max_new_tokens=args.max_new_tokens,
            device=device,
        )
        all_results[dataset_name] = results
        total_eval_time += results["timing"]["total_time_sec"]
        
        print(f"\n{dataset_name}: {results['accuracy']:.2f}% | {results['timing']['avg_sample_time_ms']:.1f}ms/sample")
    
    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / f"eval_{args.model_name}.json"
    
    # Add summary
    all_results["_summary"] = {
        "model": args.model_name,
        "checkpoint": args.checkpoint,
        "batch_size": args.batch_size,
        "total_eval_time_sec": round(total_eval_time, 2),
    }
    
    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"SUMMARY - {args.model_name}")
    print(f"{'='*60}")
    print(f"{'Dataset':<20} {'Accuracy':>10} {'Time(s)':>10} {'ms/sample':>12} {'samples/s':>12}")
    print("-" * 64)
    for ds, res in all_results.items():
        if ds.startswith("_"):
            continue
        t = res["timing"]
        print(f"{ds:<20} {res['accuracy']:>9.2f}% {t['total_time_sec']:>10.1f} {t['avg_sample_time_ms']:>12.1f} {t['throughput_samples_per_sec']:>12.1f}")
    print("-" * 64)
    print(f"Total evaluation time: {total_eval_time:.1f}s")
    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
