# ScalingRL

GRPO training framework for investigating LoRA rank sensitivity across model families on math reasoning.

**Experiment grid**: 5 model families (7-9B base) x 6 LoRA ranks x AdamW = 30 experiments
**Training data**: GSM8K (openai/gsm8k)
**Evaluation**: GSM8K test (pass@1), AIME, data contamination audit

| Family  | Model ID                         |
| ------- | -------------------------------- |
| qwen2.5 | `Qwen/Qwen2.5-7B`               |
| qwen3   | `Qwen/Qwen3-8B`                 |
| olmo3   | `allenai/OLMo-3-1025-7B`        |
| mistral | `mistralai/Ministral-8B-Base-2410` |
| gemma2  | `google/gemma-2-9b`              |

## Setup

```bash
uv sync
wandb login  # optional, or use --no-wandb
```

## Training

```bash
# Smoke test
bash smoke_test.sh

# Single run
python -m scripts.train --model-family qwen2.5 --lora-rank 8 --no-wandb

# Full sweep (30 experiments) — dry-run first
python -m scripts.sweep --phase phase1 --dry-run
python -m scripts.sweep --phase phase1

# Custom subset
python -m scripts.sweep --phase custom --model-families qwen2.5 mistral --lora-ranks 4 8 --dry-run
```

Rollout generation uses vLLM in colocate mode by default. Adjust GPU memory fraction with `--vllm-gpu-memory` (default 0.3).

## Evaluation

```bash
# GSM8K pass@1
python -m scripts.evaluate --checkpoint ./outputs/qwen2.5_lora8 --datasets gsm8k

# AIME
python -m scripts.evaluate --checkpoint ./outputs/qwen2.5_lora8 --datasets aime2025

# Data contamination (completion @ 60%, per "Reasoning or Memorization?" paper)
python -m scripts.evaluate_contamination --model-name Qwen/Qwen2.5-7B-Instruct
python -m scripts.evaluate_contamination --all-families
python -m scripts.evaluate_contamination --all-families --output-json contamination.json
```

## Tests

```bash
uv run pytest tests/ -v
uv run pytest tests/test_data.py -v          # single file
uv run pytest tests/test_data.py::test_load_gsm8k_dataset -v  # single test
```

## Reproducing Morris et al. (GSM8K, Qwen2.5-7B)

Config defaults now match the paper: batch 8 x grad_accum 8 = effective batch 64, 4 generations, max_completion_length 4096, beta 0, 3 epochs.

```bash
# Single Qwen2.5 run
python -m scripts.train --model-family qwen2.5 --lora-rank 8

# Sweep LoRA ranks for Qwen2.5 (use --wandb-group to separate from other runs)
python -m scripts.sweep --phase custom \
  --model-families qwen2.5 \
  --lora-ranks 1 2 4 8 16 64 \
  --wandb-group qwen2.5-lora-sweep
```

Note: paper used instruct models and swept LR per rank ({1e-7, 5e-7, 1e-6, 5e-6, 1e-5, 1e-4, 2e-4}).

## Configuration

All defaults are in `scalingrl/config.py`. Override via CLI args — see `python -m scripts.train --help`.
