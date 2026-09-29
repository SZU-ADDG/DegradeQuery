from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from dq_data import build_analysis_split_overlay, load_config, load_normalized_dataset


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument(
        "--output-dir",
        default="splits/target_e3",
    )
    parser.add_argument("--seeds", default="1,2,3,4,5,6,7,8,9,10")
    parser.add_argument("--test-frac", type=float, default=0.20)
    args = parser.parse_args()

    config = load_config(args.config)
    frame = load_normalized_dataset(config, "protac8k")
    root = Path(config["project"]["root"])
    output_dir = root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
    reports: Dict[str, Any] = {}

    for seed in seeds:
        overlay, report = build_analysis_split_overlay(
            config,
            frame,
            "target_e3_holdout",
            seed,
            test_frac=args.test_frac,
        )
        path = output_dir / f"target_e3_holdout_seed{seed}.csv"
        overlay.to_csv(path, index=False)
        reports[str(seed)] = {
            **report,
            "path": str(path),
            "sha256": sha256(path),
        }

    report_path = output_dir / "repeated_target_e3_split_report.json"
    report_path.write_text(
        json.dumps(
            {
                "scheme": "target_e3_holdout",
                "test_frac": args.test_frac,
                "seeds": seeds,
                "splits": reports,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    print(f"report: {report_path}")


if __name__ == "__main__":
    main()
