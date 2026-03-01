"""Evaluate data contamination across model families via partial-prompt completion.

Implements the "completion @ 60%" metric from "Reasoning or Memorization?"
(Wu et al. 2025). For each test problem, truncate to 60% of its text,
let the model greedily complete the rest (no chat template), and measure
ROUGE-L, exact-match, and answer accuracy.

Evaluates on both GSM8K and MATH-500 unconditionally.

Usage:
    # Single base model
    python scripts/evaluate_contamination.py --model-name Qwen/Qwen2.5-7B-Instruct

    # All 5 families
    python scripts/evaluate_contamination.py --all-families

    # Trained checkpoint
    python scripts/evaluate_contamination.py --checkpoint ./outputs/qwen2.5_lora8

    # Custom prefix ratio
    python scripts/evaluate_contamination.py --model-name Qwen/Qwen2.5-7B-Instruct --prefix-ratio 0.4
"""

import argparse
import json
import os

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from scalingrl.evaluation.contamination import SUPPORTED_DATASETS, ContaminationEvaluator
from scalingrl.utils import set_seed

MODEL_FAMILIES = {
    "qwen2.5": "Qwen/Qwen2.5-7B",
    "qwen3": "Qwen/Qwen3-8B",
    "olmo3": "allenai/OLMo-3-1025-7B",
    "mistral": "mistralai/Ministral-8B-Base-2410",
    "gemma2": "google/gemma-2-9b",
}


def load_base_model(model_name: str):
    """Load a base model (no checkpoint / no adapter)."""
    print(f"Loading base model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    return model, tokenizer


def load_checkpoint(checkpoint_path: str):
    """Load a trained checkpoint (with optional PEFT adapter)."""
    print(f"Loading checkpoint: {checkpoint_path}")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)

    adapter_config_path = os.path.join(checkpoint_path, "adapter_config.json")
    if os.path.exists(adapter_config_path):
        with open(adapter_config_path) as f:
            adapter_config = json.load(f)
            base_model_name = adapter_config.get("base_model_name_or_path")
        if base_model_name is None:
            raise ValueError("Could not determine base model from adapter config")
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        model = PeftModel.from_pretrained(base_model, checkpoint_path)
        model = model.merge_and_unload()
    else:
        model = AutoModelForCausalLM.from_pretrained(
            checkpoint_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )

    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    return model, tokenizer


def evaluate_model_datasets(model, tokenizer, prefix_ratio, batch_size, max_samples, datasets=None):
    """Run contamination evaluation on a single model across selected datasets."""
    results = {}
    for ds_name in datasets or SUPPORTED_DATASETS:
        evaluator = ContaminationEvaluator(
            model=model,
            tokenizer=tokenizer,
            batch_size=batch_size,
            max_new_tokens=4096,
            prefix_ratio=prefix_ratio,
            dataset_name=ds_name,
        )
        dataset = evaluator.load_dataset()
        if max_samples is not None:
            dataset = dataset.select(range(min(max_samples, len(dataset))))
        results[ds_name] = evaluator.evaluate(dataset)
    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate data contamination via partial-prompt completion")

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--model-name", type=str, help="HuggingFace model ID (base model)")
    group.add_argument("--checkpoint", type=str, help="Path to trained checkpoint")
    group.add_argument(
        "--all-families",
        action="store_true",
        help="Evaluate all 5 model families",
    )

    parser.add_argument(
        "--prefix-ratio",
        type=float,
        default=0.6,
        help="Fraction of question text to keep as prompt (default: 0.6)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        nargs="+",
        choices=list(SUPPORTED_DATASETS),
        default=None,
        help="Dataset(s) to evaluate (default: all)",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-samples", type=int, default=None, help="Limit test set size")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-json", type=str, default=None, help="Save results to JSON")

    args = parser.parse_args()
    set_seed(args.seed)

    all_results = {}

    if args.all_families:
        for family, model_id in MODEL_FAMILIES.items():
            print("\n" + "=" * 60)
            print(f"  {family}: {model_id}")
            print("=" * 60)
            model, tokenizer = load_base_model(model_id)
            ds_results = evaluate_model_datasets(
                model, tokenizer, args.prefix_ratio, args.batch_size, args.max_samples, args.dataset
            )
            for ds_name, res in ds_results.items():
                all_results[f"{family}/{ds_name}"] = res
            del model, tokenizer
            torch.cuda.empty_cache()
    else:
        if args.model_name:
            model, tokenizer = load_base_model(args.model_name)
            label = args.model_name
        else:
            model, tokenizer = load_checkpoint(args.checkpoint)
            label = args.checkpoint

        ds_results = evaluate_model_datasets(
            model, tokenizer, args.prefix_ratio, args.batch_size, args.max_samples, args.dataset
        )
        for ds_name, res in ds_results.items():
            all_results[f"{label}/{ds_name}"] = res

    # Summary table
    print("\n" + "=" * 60)
    print(f"  Contamination Summary (prefix={int(args.prefix_ratio * 100)}%)")
    print("=" * 60)
    print(f"  {'Model':<35} {'ROUGE-L':>8} {'EM':>8} {'Ans Acc':>8}")
    print("  " + "-" * 61)
    for name, res in all_results.items():
        print(f"  {name:<35} {res['rouge_l']:>7.2%} {res['em']:>7.2%} {res['answer_accuracy']:>7.2%}")
    print("=" * 60)

    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nResults saved to {args.output_json}")


if __name__ == "__main__":
    main()
