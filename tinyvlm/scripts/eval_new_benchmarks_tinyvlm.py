#!/usr/bin/env python3
"""Evaluation script for TinyVLM on new benchmarks with timing."""

import sys
from pathlib import Path
import time

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

from src.model.vlm import TinyVLM


def format_prompt(question: str) -> str:
    return f"<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\nAnswer:"


def extract_answer(response: str) -> str:
    response = response.strip()
    valid_letters = "ABCD"
    if response and response[0].upper() in valid_letters:
        return response[0].upper()
    patterns = [
        r"(?:the\s+)?answer\s*(?:is|:)\s*([A-Za-z])",
        r"^([A-Za-z])[\.)\s]",
    ]
    for pattern in patterns:
        match = re.search(pattern, response, re.IGNORECASE)
        if match:
            return match.group(1).upper()
    return response.strip()[:50]


def load_realworldqa(data_dir: str):
    ds = load_from_disk(str(Path(data_dir) / "realworldqa"))
    samples = []
    for item in ds["test"]:
        samples.append({
            "image": item["image"],
            "question": item["question"],
            "answer": item["answer"].strip(),
        })
    print(f"Loaded {len(samples)} RealWorldQA samples")
    return samples


def load_hallusionbench(data_dir: str):
    ds = load_from_disk(str(Path(data_dir) / "hallusionbench"))
    samples = []
    for item in ds["image"]:
        samples.append({
            "image": item["image"],
            "question": item["question"],
            "answer": item["gt_answer"].strip().lower(),
            "category": item["category"],
        })
    print(f"Loaded {len(samples)} HallusionBench samples")
    return samples


def load_ocrbench(data_dir: str):
    ds = load_from_disk(str(Path(data_dir) / "ocrbench"))
    samples = []
    for item in ds["test"]:
        samples.append({
            "image": item["image"],
            "question": item["question"],
            "answer": str(item["answer"]).strip(),
            "question_type": item["question_type"],
        })
    print(f"Loaded {len(samples)} OCRBench samples")
    return samples


def load_mmvet(data_dir: str):
    ds = load_from_disk(str(Path(data_dir) / "mmvet"))
    samples = []
    for item in ds["test"]:
        samples.append({
            "image": item["image"],
            "question": item["question"],
            "answer": item["answer"].strip(),
            "capability": item["capability"],
        })
    print(f"Loaded {len(samples)} MM-Vet samples")
    return samples


def evaluate_dataset(
    model, tokenizer, image_processor, samples, dataset_name,
    batch_size=64, max_new_tokens=32, device=None,
):
    print(f"\n{'='*50}")
    print(f"Evaluating {dataset_name}")
    print(f"Samples: {len(samples)}, Batch size: {batch_size}")
    print(f"{'='*50}")
    
    correct = 0
    total = 0
    per_category = defaultdict(lambda: {"correct": 0, "total": 0})
    total_inference_time = 0.0
    
    dataset_start_time = time.time()
    pbar = tqdm(range(0, len(samples), batch_size), desc=f"Eval {dataset_name}")
    
    for batch_start in pbar:
        batch_end = min(batch_start + batch_size, len(samples))
        batch_samples = samples[batch_start:batch_end]
        current_batch_size = len(batch_samples)
        
        prompts = []
        pixel_values_list = []
        
        for sample in batch_samples:
            prompts.append(format_prompt(sample["question"]))
            image = sample["image"]
            if not isinstance(image, Image.Image):
                image = Image.open(image).convert("RGB")
            else:
                image = image.convert("RGB")
            pv = image_processor(images=image, return_tensors="pt")["pixel_values"].squeeze(0)
            pixel_values_list.append(pv)
        
        encoded = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=2048)
        input_ids = encoded["input_ids"].to(device)
        text_attention_mask = encoded["attention_mask"].to(device)
        pixel_values = torch.stack(pixel_values_list).to(device, dtype=torch.bfloat16)
        
        image_attention = torch.ones(current_batch_size, model.num_image_tokens, device=device, dtype=text_attention_mask.dtype)
        attention_mask = torch.cat([image_attention, text_attention_mask], dim=1)
        
        torch.cuda.synchronize()
        batch_start_time = time.time()
        
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
        
        torch.cuda.synchronize()
        batch_inference_time = time.time() - batch_start_time
        total_inference_time += batch_inference_time
        per_sample_time = batch_inference_time / current_batch_size
        
        for i, sample in enumerate(batch_samples):
            response = tokenizer.decode(output_ids[i], skip_special_tokens=True)
            ground_truth = sample["answer"]
            predicted = extract_answer(response)
            
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
            cat: {"total": s["total"], "correct": s["correct"],
                  "accuracy": round(s["correct"]/s["total"]*100, 2) if s["total"] > 0 else 0}
            for cat, s in per_category.items()
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
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    print(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = checkpoint.get("config", {})
    model_config = config.get("model", {})
    
    vision_model = model_config.get("vision_encoder", "google/siglip-base-patch16-384")
    lm_model = model_config.get("language_model", "HuggingFaceTB/SmolLM2-360M-Instruct")
    
    print(f"Vision: {vision_model}, LM: {lm_model}")
    
    model = TinyVLM(vision_encoder_name=vision_model, language_model_name=lm_model, freeze_vision=True, freeze_lm=False)
    model_sd = checkpoint["model_state_dict"]
    model.vision_encoder.load_state_dict(model_sd["vision_encoder"])
    model.projector.load_state_dict(model_sd["projector"])
    model.language_model.model.load_state_dict(model_sd["language_model"])
    model = model.to(device=device, dtype=torch.bfloat16)
    model.eval()
    
    tokenizer = AutoTokenizer.from_pretrained(lm_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    image_processor = SiglipImageProcessor.from_pretrained(vision_model)
    
    print(f"\nModel loaded. Image tokens: {model.num_image_tokens}")
    
    loaders = {
        "realworldqa": load_realworldqa,
        "hallusionbench": load_hallusionbench,
        "ocrbench": load_ocrbench,
        "mmvet": load_mmvet,
    }
    
    all_results = {}
    total_eval_time = 0
    
    for dataset_name in args.datasets:
        if dataset_name not in loaders:
            continue
        samples = loaders[dataset_name](args.data_dir)
        if not samples:
            continue
        
        results = evaluate_dataset(
            model, tokenizer, image_processor, samples, dataset_name,
            batch_size=args.batch_size, max_new_tokens=args.max_new_tokens, device=device,
        )
        all_results[dataset_name] = results
        total_eval_time += results["timing"]["total_time_sec"]
        print(f"\n{dataset_name}: {results['accuracy']:.2f}% | {results['timing']['avg_sample_time_ms']:.1f}ms/sample")
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "eval_tinyvlm.json"
    
    all_results["_summary"] = {
        "model": "tinyvlm",
        "checkpoint": args.checkpoint,
        "batch_size": args.batch_size,
        "total_eval_time_sec": round(total_eval_time, 2),
    }
    
    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n{'='*60}")
    print("SUMMARY - TinyVLM")
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
