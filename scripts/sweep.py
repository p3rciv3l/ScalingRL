"""Launch experiment sweeps for model family x LoRA rank investigation."""

import argparse
import subprocess
from itertools import product

# 5 model families (all ~7-9B base models)
MODEL_FAMILIES = {
    "qwen2.5": "Qwen/Qwen2.5-7B",
    "qwen3": "Qwen/Qwen3-8B",
    "olmo3": "allenai/OLMo-3-1025-7B",
    "mistral": "mistralai/Ministral-8B-Base-2410",
    "gemma2": "google/gemma-2-9b",
}

LORA_RANKS = [1, 2, 4, 8, 16, 64]


def launch_experiment(
    family,
    model_name,
    lora_rank,
    adapter_type="lora",
    tiny_lora_u=None,
    tiny_lora_n_tie=None,
    extra_flags=None,
    dry_run=False,
):
    """Launch a single experiment."""
    cmd = [
        "python",
        "-m",
        "scripts.train",
        "--model-name",
        model_name,
        "--model-family",
        family,
        "--lora-rank",
        str(lora_rank),
        "--adapter-type",
        adapter_type,
    ]
    if tiny_lora_u is not None:
        cmd.extend(["--tiny-lora-u", str(tiny_lora_u)])
    if tiny_lora_n_tie is not None:
        cmd.extend(["--tiny-lora-n-tie", str(tiny_lora_n_tie)])
    if extra_flags:
        cmd.extend(extra_flags)

    if dry_run:
        print(f"  Command: {' '.join(cmd)}")
    else:
        subprocess.run(cmd)


def main():
    """Main sweep launcher."""
    parser = argparse.ArgumentParser(description="Launch experiment sweeps")
    parser.add_argument(
        "--phase",
        type=str,
        choices=["phase1", "custom", "smoke"],
        default="phase1",
        help="Which phase to run",
    )
    parser.add_argument(
        "--model-families",
        type=str,
        nargs="+",
        choices=list(MODEL_FAMILIES.keys()),
        help="Model families (for custom)",
    )
    parser.add_argument(
        "--lora-ranks",
        type=int,
        nargs="+",
        help="LoRA ranks (for custom)",
    )
    parser.add_argument(
        "--adapter-type",
        type=str,
        choices=["lora", "lora_xs", "tiny_lora"],
        default="lora",
        help="Adapter type (default: lora)",
    )
    parser.add_argument("--tiny-lora-u", type=int, default=None, help="TinyLoRA projection dimension")
    parser.add_argument("--tiny-lora-n-tie", type=int, default=None, help="TinyLoRA weight tying factor")
    parser.add_argument("--vllm-gpu-memory", type=float, default=0.3, help="vLLM GPU memory utilization")
    parser.add_argument("--batch-size", type=int, default=None, help="Per-device train batch size")
    parser.add_argument("--grad-accum", type=int, default=None, help="Gradient accumulation steps")
    parser.add_argument("--num-generations", type=int, default=None, help="GRPO num generations")
    parser.add_argument("--epochs", type=int, default=None, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate")
    parser.add_argument("--max-samples", type=int, default=None, help="Limit dataset size")
    parser.add_argument("--wandb-group", type=str, default=None, help="Wandb group name")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without running",
    )

    args = parser.parse_args()

    if args.phase == "smoke":
        # Quick smoke test
        print("Smoke Test")
        print("=" * 60)
        families = ["qwen2.5"]
        lora_ranks = [8]

    elif args.phase == "phase1":
        # Phase 1: Full grid — 5 families x 6 ranks = 30 experiments
        print("Phase 1: Model Family x LoRA Rank Sweep")
        print("=" * 60)
        families = list(MODEL_FAMILIES.keys())
        lora_ranks = LORA_RANKS

    elif args.phase == "custom":
        # Custom sweep
        if not (args.model_families and args.lora_ranks):
            parser.error("Custom phase requires --model-families and --lora-ranks")
        families = args.model_families
        lora_ranks = args.lora_ranks

    else:
        parser.error(f"Unknown phase: {args.phase}")

    # Build flags to forward to train.py
    forward_flags = []
    if args.vllm_gpu_memory != 0.3:
        forward_flags.extend(["--vllm-gpu-memory", str(args.vllm_gpu_memory)])
    if args.batch_size is not None:
        forward_flags.extend(["--batch-size", str(args.batch_size)])
    if args.grad_accum is not None:
        forward_flags.extend(["--grad-accum", str(args.grad_accum)])
    if args.num_generations is not None:
        forward_flags.extend(["--num-generations", str(args.num_generations)])
    if args.epochs is not None:
        forward_flags.extend(["--epochs", str(args.epochs)])
    if args.lr is not None:
        forward_flags.extend(["--lr", str(args.lr)])
    if args.max_samples is not None:
        forward_flags.extend(["--max-samples", str(args.max_samples)])
    if args.wandb_group is not None:
        forward_flags.extend(["--wandb-group", args.wandb_group])

    # Generate experiments
    experiments = list(product(families, lora_ranks))
    print(f"Total experiments: {len(experiments)}\n")

    # Launch experiments
    for i, (family, rank) in enumerate(experiments, 1):
        model_name = MODEL_FAMILIES[family]
        print(f"[{i}/{len(experiments)}] {family} ({model_name}) | {args.adapter_type} r={rank}")
        launch_experiment(
            family,
            model_name,
            rank,
            adapter_type=args.adapter_type,
            tiny_lora_u=args.tiny_lora_u,
            tiny_lora_n_tie=args.tiny_lora_n_tie,
            extra_flags=forward_flags or None,
            dry_run=args.dry_run,
        )
        print()


if __name__ == "__main__":
    main()
