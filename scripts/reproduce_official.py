from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run(command: list[str]) -> str:
    print("$ " + " ".join(command), flush=True)
    completed = subprocess.run(command, cwd=ROOT, check=True, text=True, capture_output=True)
    if completed.stdout:
        print(completed.stdout, end="")
    if completed.stderr:
        print(completed.stderr, end="", file=sys.stderr)
    return completed.stdout


def checkpoint_from_output(output: str) -> str:
    for line in output.splitlines():
        if line.startswith("checkpoint:"):
            return line.split(":", 1)[1].strip()
    raise RuntimeError("Pretraining completed without reporting a checkpoint path.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Reproduce the five-seed official Sup and CTP experiments.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--seeds", default="1,2,3,4,5")
    parser.add_argument("--prepare-data", action="store_true")
    args = parser.parse_args()

    python = sys.executable
    seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
    prefix = [python, "dq.py", "--config", args.config]

    if args.prepare_data:
        run(prefix + ["init"])
        run(prefix + ["fetch", "--dataset", "protac8k"])
        run(prefix + ["audit", "--dataset", "protac8k", "--check-leakage"])
        run(prefix + ["prep", "--dataset", "protac8k"])
        run(prefix + ["fetchseq", "--dataset", "protac8k", "--source", "auto"])

    for seed in seeds:
        seeded = [python, "dq.py", "--config", args.config, "--seed", str(seed)]
        run(
            seeded
            + [
                "train", "--dataset", "protac8k", "--mode", "supervised",
                "--epochs", "120", "--batch-size", "64", "--lr", "0.0001",
                "--loss", "focal", "--focal-gamma", "1.0",
                "--protein-encoder", "aac", "--run-label", f"paper_sup_s{seed}",
            ]
        )
        pretrain_output = run(
            seeded
            + [
                "pretrain", "--dataset", "protac8k", "--tasks", "tuple_contrast,mol_contrast",
                "--epochs", "60", "--batch-size", "64", "--tuple-negative-policy", "full",
                "--temperature-tuple", "0.1", "--temperature-mol", "0.1",
                "--lambda-tuple", "1.0", "--lambda-mol", "0.1",
                "--protein-encoder", "aac", "--run-label", f"paper_ctp_pretrain_s{seed}",
            ]
        )
        checkpoint = checkpoint_from_output(pretrain_output)
        run(
            seeded
            + [
                "train", "--dataset", "protac8k", "--mode", "supervised",
                "--ablation", "ctp_finetune", "--init", checkpoint,
                "--epochs", "120", "--batch-size", "64", "--lr", "0.0001",
                "--loss", "focal", "--focal-gamma", "1.0",
                "--protein-encoder", "aac", "--run-label", f"paper_ctp_s{seed}",
            ]
        )


if __name__ == "__main__":
    main()
