"""Sweep: Qwen3-8B and Ministral-3-8B x LoRA(r=1,8) + TinyLoRA(r=2,u=100).

Tracks completed experiments in a state file so re-running after a failure
skips already-finished experiments. Each experiment runs training then a
post-training GSM8K pass@1 eval, with state saved between each step.

Usage:
    python -m scripts.sweep_rl               # run all pending
    python -m scripts.sweep_rl --dry-run     # print commands only
    python -m scripts.sweep_rl --no-wandb    # disable wandb
"""

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from subprocess import run as sp_run

OUTPUTS_DIR = Path("outputs")
STATE_FILE = Path("sweep_rl_state.json")

MODEL_FAMILIES = {
    "qwen3": "Qwen/Qwen3-8B",
    "mistral": "mistralai/Ministral-8B-Base-2410",
}


@dataclass
class Experiment:
    family: str
    adapter_type: str
    lora_rank: int
    tiny_lora_u: int | None = None
    tiny_lora_n_tie: int | None = None

    @property
    def run_id(self) -> str:
        s = f"{self.family}_{self.adapter_type}_r{self.lora_rank}"
        if self.adapter_type == "tiny_lora":
            s += f"_u{self.tiny_lora_u}_ntie{self.tiny_lora_n_tie or 'none'}"
        return s

    @property
    def checkpoint_dir(self) -> Path:
        # Matches the run_name logic in train.py
        adapter_label = {"lora": "lora", "lora_xs": "loraxs", "tiny_lora": "tinylora"}[self.adapter_type]
        return OUTPUTS_DIR / f"{self.family}_{adapter_label}{self.lora_rank}"

    @property
    def eval_json(self) -> Path:
        return OUTPUTS_DIR / f"{self.run_id}_eval.json"

    def train_cmd(self, extra_flags: list[str]) -> list[str]:
        cmd = [
            "python",
            "-m",
            "scripts.train",
            "--model-name",
            MODEL_FAMILIES[self.family],
            "--model-family",
            self.family,
            "--lora-rank",
            str(self.lora_rank),
            "--adapter-type",
            self.adapter_type,
        ]
        if self.tiny_lora_u is not None:
            cmd += ["--tiny-lora-u", str(self.tiny_lora_u)]
        if self.tiny_lora_n_tie is not None:
            cmd += ["--tiny-lora-n-tie", str(self.tiny_lora_n_tie)]
        cmd += extra_flags
        return cmd

    def eval_cmd(self) -> list[str]:
        return [
            "python",
            "-m",
            "scripts.evaluate",
            "--checkpoint",
            str(self.checkpoint_dir),
            "--datasets",
            "gsm8k",
            "--output-json",
            str(self.eval_json),
        ]


# n_tie=1000 >> number of LoRA modules (~252 for 7-9B models), so all modules
# share a single v vector → total trainable params = u = 100.
EXPERIMENTS: list[Experiment] = [
    # TinyLoRA first (cheapest, both models)
    Experiment("qwen3", "tiny_lora", lora_rank=2, tiny_lora_u=100, tiny_lora_n_tie=1000),
    Experiment("mistral", "tiny_lora", lora_rank=2, tiny_lora_u=100, tiny_lora_n_tie=1000),
    # LoRA r=1, both models
    Experiment("qwen3", "lora", lora_rank=1),
    Experiment("mistral", "lora", lora_rank=1),
    # LoRA r=8, both models
    Experiment("qwen3", "lora", lora_rank=8),
    Experiment("mistral", "lora", lora_rank=8),
]


def load_state() -> dict[str, set[str]]:
    if STATE_FILE.exists():
        data = json.loads(STATE_FILE.read_text())
        return {
            "train": set(data.get("train", [])),
            "eval": set(data.get("eval", [])),
        }
    return {"train": set(), "eval": set()}


def save_state(state: dict[str, set[str]]) -> None:
    STATE_FILE.write_text(json.dumps({"train": sorted(state["train"]), "eval": sorted(state["eval"])}, indent=2) + "\n")


def run_step(label: str, cmd: list[str], dry_run: bool) -> bool:
    """Run a subprocess step. Returns True on success."""
    print(f"  {label}: {' '.join(cmd)}")
    if dry_run:
        return True
    result = sp_run(cmd)
    if result.returncode != 0:
        print(f"  Failed (exit {result.returncode})")
        return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep Qwen3 + Ministral x LoRA/TinyLoRA on GSM8K")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running")
    parser.add_argument("--no-wandb", action="store_true", help="Disable wandb logging")
    parser.add_argument("--wandb-group", type=str, default="rl-sweep", help="Wandb group name")
    parser.add_argument("--vllm-gpu-memory", type=float, default=0.3, help="vLLM GPU memory utilization")
    args = parser.parse_args()

    train_flags: list[str] = []
    if args.no_wandb:
        train_flags.append("--no-wandb")
    else:
        train_flags += ["--wandb-group", args.wandb_group]
    if args.vllm_gpu_memory != 0.3:
        train_flags += ["--vllm-gpu-memory", str(args.vllm_gpu_memory)]

    OUTPUTS_DIR.mkdir(exist_ok=True)

    state = load_state()
    n_done = sum(1 for e in EXPERIMENTS if e.run_id in state["train"] and e.run_id in state["eval"])
    print(f"Sweep: {len(EXPERIMENTS)} experiments, {n_done} fully complete")
    print(f"State file: {STATE_FILE}")
    print()

    for i, exp in enumerate(EXPERIMENTS, 1):
        train_done = exp.run_id in state["train"]
        eval_done = exp.run_id in state["eval"]

        if train_done and eval_done:
            print(f"[{i}/{len(EXPERIMENTS)}] {exp.run_id} — already complete, skipping")
            continue

        print(f"[{i}/{len(EXPERIMENTS)}] {exp.run_id}")

        if not train_done:
            ok = run_step("train", exp.train_cmd(train_flags), args.dry_run)
            if not ok:
                print(f"  Training failed — {n_done} experiments fully complete, state saved.")
                sys.exit(1)
            state["train"].add(exp.run_id)
            save_state(state)

        if not eval_done:
            ok = run_step("eval", exp.eval_cmd(), args.dry_run)
            if not ok:
                print("  Eval failed — training is saved, re-run to retry eval only.")
                sys.exit(1)
            state["eval"].add(exp.run_id)
            save_state(state)
            if not args.dry_run:
                print(f"  Results saved to: {exp.eval_json}")

        n_done += 1
        print()

    if not args.dry_run:
        print(f"All {len(EXPERIMENTS)} experiments complete.")
        print(f"Eval results in: {OUTPUTS_DIR}/<run_id>_eval.json")


if __name__ == "__main__":
    main()
