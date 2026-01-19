# LeanVLM

> **Note:** This code was prepared with assistance from Claude and Codex.

This repository contains two Vision-Language Model implementations optimized for efficiency.

## 1. TinyVLM (~450M parameters)

A compact VLM built from scratch using:
- **Vision Encoder**: SigLIP (google/siglip-base-patch16-384, 86M params)
- **Language Model**: SmolLM2-360M-Instruct
- **Training**: Two-stage (alignment + instruction tuning)

## 2. MM-STEM (~360M parameters + 7.7GB STEM embeddings)

Extension of STEM (Scaling Transformers with Embedding Modules) to multimodal settings:
- **Vision Encoder**: SigLIP-so400m-patch14-384
- **Language Model**: SmolLM2-360M-Instruct with STEM
- **Key Innovation**: Replaces FFN up-projection with embedding lookup for text tokens

Reference: [STEM Paper (arXiv:2601.10639)](https://arxiv.org/abs/2601.10639)

## Evaluation Results Comparison

| Benchmark | TinyVLM | MM-STEM | Notes |
|-----------|---------|---------|-------|
| **MMMU** | 32.89% | **56.11%** | MM-STEM +23% |
| **MMBench** | **26.38%** | 26.06% | Similar |
| **MathVista** | **23.70%** | 18.90% | TinyVLM +5% |
| **MMStar** | 26.60% | **26.87%** | Similar |
| **OCRBench** | 25.60% | **35.90%** | MM-STEM +10% |
| **MM-Vet** | 42.20% | **45.41%** | MM-STEM +3% |
| **RealWorldQA** | **35.69%** | 15.82% | TinyVLM +20% |

### Key Insights

- **MM-STEM excels at knowledge-intensive tasks** (MMMU: 56% vs 33%)
- **MM-STEM has superior OCR capabilities** (+10% on OCRBench)
- **TinyVLM better at real-world understanding** (RealWorldQA: 36% vs 16%)
- **TinyVLM better at mathematical reasoning** (MathVista: 24% vs 19%)

## Quick Start

### TinyVLM

```bash
cd tinyvlm

# Install
pip install -r requirements.txt

# Download data
python scripts/download_data.py --dataset all

# Train
python scripts/train.py --config configs/base.yaml --stage 1
python scripts/train.py --config configs/base.yaml --stage 2 --resume checkpoints/stage1_final.pt

# Evaluate
python scripts/eval_mmstar.py --checkpoint checkpoints/stage2_final.pt
python scripts/eval_multimodal_tinyvlm.py --checkpoint checkpoints/stage2_final.pt
python scripts/eval_new_benchmarks_tinyvlm.py --checkpoint checkpoints/stage2_final.pt

# Inference
python scripts/inference.py --checkpoint checkpoints/stage2_final.pt --image path/to/image.jpg
```

### MM-STEM

```bash
cd mm_stem

# Install
pip install -r requirements.txt

# Download data
python scripts/download_data.py --dataset all

# Train (two-stage, automatic)
python scripts/train_stem.py --config configs/stem_config.yaml

# Evaluate
python scripts/eval_mmstar.py --checkpoint checkpoints/stage2_final.pt
python scripts/eval_multimodal.py --checkpoint checkpoints/stage2_final.pt
python scripts/eval_new_benchmarks.py --checkpoint checkpoints/stage2_final.pt

# Inference
python scripts/eval_stem.py --checkpoint checkpoints/stage2_final.pt --image path/to/image.jpg
```

## Project Structure

```
LeanVLM/
├── README.md
├── tinyvlm/                  # TinyVLM implementation
│   ├── configs/
│   ├── src/
│   │   ├── model/
│   │   ├── data/
│   │   ├── training/
│   │   └── evaluation/
│   └── scripts/
│
└── mm_stem/                  # MM-STEM implementation
    ├── configs/
    ├── src/
    │   ├── model/
    │   ├── data/
    │   └── training/
    └── scripts/
```

## Hardware Requirements

Both models require an A100 40GB GPU for training.

## References

- [STEM Paper (arXiv:2601.10639)](https://arxiv.org/abs/2601.10639)
- [LLaVA](https://llava-vl.github.io/)
- [SigLIP](https://arxiv.org/abs/2303.15343)
- [SmolLM2](https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct)

## License

MIT License
